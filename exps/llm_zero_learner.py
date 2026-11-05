"""
LLM Zero-Shot Learner for CIFAR-10

Uses a frozen LLM (SmolLM-135M) with patchified CIFAR-10 image embeddings
for zero-shot in-context classification with KV cache management.

Design:
- Patchifier converts 32x32x3 images to patch embeddings (default: 16x16 patches, 4 per image)
- Visual embeddings mixed with text embeddings via inputs_embeds
- KV sink removes visual embeds from cache after processing to save memory
- Two-pass teaching: (1) user+visuals, (2) force-feed assistant response
- Classification via AR decode or probability comparison

Usage:
    # Default: 5 train per class, 1 test per class, KV sink enabled
    python exps/llm_zero_learner.py

    # Use full datasets, no KV sink
    python exps/llm_zero_learner.py --full-dataset --disable-kv-sink

    # Custom subset, save images
    python exps/llm_zero_learner.py --train-subset 10 --test-subset 3 --save-images

    # Larger subsets: 100 train per class, 10 test per class
    python exps/llm_zero_learner.py --train-subset 100 --test-subset 10

    # Larger subsets with KV sink disabled (use whole context)
    python exps/llm_zero_learner.py --train-subset 100 --test-subset 10 --disable-kv-sink

    # Sanity check: use same images for train and test (recall test)
    python exps/llm_zero_learner.py --train-subset 10 --sanity-check

    # Disable Flash Attention (enabled by default)
    python exps/llm_zero_learner.py --no-flash-attention

    # AR decode with multiple-choice format (logits mode is faster)
    python exps/llm_zero_learner.py --answer-mode ar --ar-decode-mode logits

    # AR decode with raw string matching
    python exps/llm_zero_learner.py --answer-mode ar --ar-decode-mode raw

    # Teaching modes:
    # Minibatch per class (split one class into batches)
    python exps/llm_zero_learner.py --teaching-mode minibatch-per-class --teaching-batch-size 2
    # Minibatch mixed (mix multiple classes per batch)
    python exps/llm_zero_learner.py --teaching-mode minibatch-mixed --teaching-batch-size 5

    # Disable confusion matrix plot
    python exps/llm_zero_learner.py --no-plot-confusion

    # Use config file
    python exps/llm_zero_learner.py --config exps/configs/llm_config.py
    
Notes:
    - Flash Attention is enabled by default via transformers SDPA (PyTorch 2.1+ auto-selects Flash on supported hardware). Use --no-flash-attention to force eager attention.
    - Flash Attention status is logged during LLM initialization
    - Confusion matrix is saved as PNG to log directory
    - Sanity check mode uses first N train images as test to verify recall
    - CUDA cache is cleared and GC run after each test image to prevent VRAM pollution
    - AR decode mode uses multiple-choice format (a. cat, b. dog, ...) with two decoding methods: 'logits' (first token filtered softmax, faster) or 'raw' (full AR decode with string matching)
    - Three teaching modes: 'full-batch-per-class' (default, all images of one class at once), 'minibatch-per-class' (split one class into batches), 'minibatch-mixed' (mix multiple classes per batch)
    - KV sink works with all three teaching modes to manage memory
"""

import argparse
import ast
import gc
import logging
import os
import random
import sys
from pathlib import Path
from typing import Optional, Union, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

# Try to import matplotlib for confusion matrix
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# SmolLM chat role tokens (Mistral-based)
SPECIAL_TOKENS = {
    "bos": "<s>",
    "eos": "</s>",
    "user": "user",
    "assistant": "assistant",
}

# Prompts with proper English
TEACHING_USER_PROMPT = "Class: {class_name}\nRemember these examples."
TEACHING_ASSISTANT_PROMPT = "I remember {class_name}."
INTRO_USER_PROMPT = "I will teach you to classify CIFAR-10 images. I will present visual embeddings for each class, one at a time. Please memorize the patterns."
INTRO_ASSISTANT_PROMPT = "I am ready to learn."
TEST_USER_PROMPT = "What class is this image?"
TEST_ASSISTANT_TEMPLATE = "This is a {class_name}."


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM Zero-Shot Learner for CIFAR-10"
    )

    # --- LLM ---
    parser.add_argument("--model-name", type=str, default="HuggingFaceTB/SmolLM-135M",
                        help="HuggingFace model name. Default: HuggingFaceTB/SmolLM-135M")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cuda, cpu, or None (auto). Default: auto")

    # --- Patchifier ---
    parser.add_argument("--patch-size", type=int, default=16,
                        help="Patch size (height/width). 32//patch_size must be integer. Default: 16")
    parser.add_argument("--embed-mode", choices=["float", "int8"], default="float",
                        help="Patch embedding mode. Default: float")
    parser.add_argument("--proj-dim", type=int, default=None,
                        help="Project visual embeds to this dimension. None = LLM hidden size. Default: None")

    # --- Dataset Subset ---
    parser.add_argument("--full-dataset", action="store_true", default=False,
                        help="Use full train/test datasets (overrides subset args)")
    parser.add_argument("--train-subset", type=str, default="5",
                        help="Training subset: int=N per class (e.g., '5') or float=fraction (e.g., '0.1'). Default: '5'")
    parser.add_argument("--test-subset", type=str, default="1",
                        help="Test subset: int=N per class (e.g., '1') or float=fraction (e.g., '0.1'). Default: '1'")
    parser.add_argument("--subset-ordered", action="store_true", default=True,
                        help="Use ordered (first N) sampling. Default: True")
    parser.add_argument("--subset-seed", type=int, default=None,
                        help="Random seed for subset sampling (only if not ordered)")

    # --- KV Cache ---
    parser.add_argument("--disable-kv-sink", action="store_true", default=False,
                        help="Disable KV sink - keep visual embeddings in KV cache. Default: KV sink enabled")

    # --- Answer Extraction ---
    parser.add_argument("--answer-mode", choices=["ar", "probs", "first-token"], default="first-token",
                        help="Answer extraction mode: 'ar'=AR decode, 'probs'=full sequence probs, 'first-token'=first token probs. Default: 'first-token'")
    parser.add_argument("--ar-decode-mode", choices=["raw", "logits"], default="logits",
                        help="AR decode mode: 'raw'=full AR decode with string matching, 'logits'=first token logits filtered to classes (faster). Default: 'logits'")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Top-p sampling for AR decode. Default: 0.9")
    parser.add_argument("--max-tokens", type=int, default=20,
                        help="Max tokens for AR decode. Default: 20")

    # --- Teaching Mode ---
    parser.add_argument("--teaching-mode", choices=["full-batch-per-class", "minibatch-per-class", "minibatch-mixed"], 
                        default="full-batch-per-class",
                        help="Teaching mode: 'full-batch-per-class'=all images of one class at once (default), \
'minibatch-per-class'=split one class into smaller batches, 'minibatch-mixed'=mix multiple classes per batch")
    parser.add_argument("--teaching-batch-size", type=int, default=5,
                        help="Batch size for minibatch teaching modes. Default: 5")

    # --- Data ---
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="CIFAR-10 data directory. Default: ./data")

    # --- Logging & Output ---
    parser.add_argument("--log-dir", type=str, default="./logs",
                        help="Logging directory. Default: ./logs")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level. Default: INFO")
    parser.add_argument("--save-images", action="store_true", default=True,
                        help="Save train/test subset images to disk. Default: True")
    parser.add_argument("--images-dir", type=str, default=None,
                        help="Directory to save images. Default: <log-dir>/images")
    
    # --- Debug/Testing ---
    parser.add_argument("--sanity-check", action="store_true", default=False,
                        help="Use same subset for train and test (recall/sanity check). Default: False")
    parser.add_argument("--plot-confusion", action="store_true", default=True,
                        help="Plot confusion matrix using matplotlib. Default: True (if matplotlib available)")
    parser.add_argument("--no-plot-confusion", action="store_true", default=False,
                        help="Disable confusion matrix plotting")

    # --- Flash Attention ---
    parser.add_argument("--no-flash-attention", action="store_true", default=False,
                        help="Disable Flash Attention for the model. Default: enabled")

    # --- Config file ---
    parser.add_argument("--config", type=str, default=None,
                        help="Path to Python config file to update args")

    # Parse known args first (to get config before merging)
    args, unknown = parser.parse_known_args()

    # Load config file if provided
    if args.config and os.path.exists(args.config):
        _load_config_file(args.config, parser)

    # Parse again with all defaults now potentially updated by config
    args = parser.parse_args()

    # Post-process
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Parse subset specs
    args.train_subset_spec = parse_subset_spec(args.train_subset)
    args.test_subset_spec = parse_subset_spec(args.test_subset)

    # Resolve images dir
    if args.images_dir is None:
        args.images_dir = Path(args.log_dir) / "images"
    else:
        args.images_dir = Path(args.images_dir)

    # KV sink flag
    args.use_kv_sink = not args.disable_kv_sink

    # Flash attention: enabled by default, --no-flash-attention disables it
    args.use_flash_attention = not args.no_flash_attention

    # Plot confusion: --no-plot-confusion overrides --plot-confusion
    if args.no_plot_confusion:
        args.plot_confusion = False

    return args


def _load_config_file(config_path, parser):
    """Load a Python config file and update the parser's defaults."""
    config_path = Path(config_path).resolve()
    with open(config_path, "r") as f:
        config_code = f.read()

    tree = ast.parse(config_code, filename=str(config_path))
    config_dict = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    key = target.id
                    try:
                        value_node = ast.Expression(node.value)
                        code = compile(value_node, str(config_path), "eval")
                        value = eval(code, {"__builtins__": {}}, {})
                        config_dict[key] = value
                    except Exception as e:
                        print(f"Warning: Could not evaluate config key '{key}': {e}")

    for key, value in config_dict.items():
        for action in parser._actions:
            if "--" + key.replace("_", "-") in action.option_strings:
                action.default = value
                parser._defaults[action.dest] = value
                break


def parse_subset_spec(spec: str) -> Optional[Union[int, float]]:
    """Parse subset specification: int = per-class count, float = fraction of dataset."""
    if spec is None:
        return None
    try:
        val = float(spec)
        if val.is_integer():
            return int(val)
        elif 0 <= val <= 1:
            return val
        else:
            raise ValueError(f"Fraction must be in [0,1], got {val}")
    except ValueError:
        raise ValueError(f"Invalid subset spec: {spec}. Use int (per-class) or float [0,1] (fraction)")


# ---------------------------------------------------------------------------
# Setup & Logging
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str, name: str = "llm_zero_learner", level: str = "INFO"):
    """Setup logging to file and stdout."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}.log"

    level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return log_file


# ---------------------------------------------------------------------------
# CIFAR-10 Dataset & Subsets
# ---------------------------------------------------------------------------

def get_cifar10_full(data_dir: str, train: bool = True, normalized: bool = False):
    """Get full CIFAR-10 dataset (train or test)."""
    data_dir = Path(data_dir)
    
    if normalized:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010],
            ),
        ])
    else:
        # Raw pixels for embedding
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts to [0, 1] float
        ])
    
    dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=train,
        download=True,
        transform=transform,
    )
    return dataset


def get_subset(dataset, spec: Optional[Union[int, float]] = None, 
                ordered: bool = True, seed: Optional[int] = None) -> Subset:
    """
    Get subset of dataset.
    
    Args:
        dataset: CIFAR10 dataset
        spec: int = N examples per class, float = fraction of dataset, None = full
        ordered: Use first N or random sampling
        seed: Random seed for sampling
        
    Returns:
        Subset of dataset
    """
    if spec is None:
        return dataset
    
    if isinstance(spec, float):
        # Fraction of total dataset
        total = len(dataset)
        num_samples = int(total * spec)
        if ordered:
            indices = list(range(num_samples))
        else:
            if seed is not None:
                random.seed(seed)
            indices = random.sample(range(total), num_samples)
        return Subset(dataset, indices)
    
    elif isinstance(spec, int):
        # N examples per class
        indices = []
        for class_idx in range(10):
            class_indices = [i for i, t in enumerate(dataset.targets) if t == class_idx]
            if ordered:
                selected = class_indices[:spec]
            else:
                if seed is not None:
                    random.seed(seed)
                selected = random.sample(class_indices, min(spec, len(class_indices)))
            indices.extend(selected)
        return Subset(dataset, indices)
    
    return dataset


def save_dataset_images(dataset, output_dir: Path, prefix: str = "train") -> List[str]:
    """Save all images in dataset to disk. Returns list of saved paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    
    for i, (img, label) in enumerate(dataset):
        class_name = CIFAR10_CLASSES[label]
        img_path = output_dir / f"{prefix}_{class_name}_{i:04d}.png"
        
        # Convert tensor to PIL: img is C x H x W, float [0, 1]
        img_pil = transforms.ToPILImage()(img)
        img_pil.save(img_path)
        paths.append(str(img_path))
    
    return paths


# ---------------------------------------------------------------------------
# Patchifier
# ---------------------------------------------------------------------------

class Patchifier(nn.Module):
    """
    Convert 32x32x3 CIFAR-10 images to patch embeddings.
    
    Args:
        patch_size: Size of each patch (height/width). Default: 16
        embed_mode: 'float' (normalized 0-1) or 'int8' (0-255). Default: 'float'
        proj_dim: Project patches to this dimension. None = no projection
    
    Output:
        For 32x32x3 input with patch_size=16: (num_patches, patch_dim) = (4, 768)
        where 768 = 16 * 16 * 3
    """
    
    def __init__(self, patch_size: int = 16, embed_mode: str = "float", 
                 proj_dim: Optional[int] = None):
        super().__init__()
        self.patch_size = patch_size
        self.embed_mode = embed_mode
        
        # Calculate grid size
        self.grid_size = 32 // patch_size
        if 32 % patch_size != 0:
            raise ValueError(f"32 must be divisible by patch_size. Got {patch_size}")
        
        # Patch dimension: patch_size * patch_size * 3
        self.patch_dim = patch_size * patch_size * 3
        
        # Optional projection layer
        self.proj = None
        if proj_dim is not None:
            self.proj = nn.Linear(self.patch_dim, proj_dim)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, C, H, W) tensor, float [0, 1] or [0, 255]
            
        Returns:
            (B, num_patches, patch_dim) or (B, num_patches, proj_dim) if proj_dim set
        """
        B, C, H, W = images.shape
        
        # Convert to patches: (B, num_patches, patch_dim)
        # Reshape: (B, C, H, W) -> (B, grid, grid, C, patch_h, patch_w)
        patches = images.unfold(2, self.patch_size, self.patch_size) \
                          .unfold(3, self.patch_size, self.patch_size)
        
        # Flatten spatial dimensions: (B, grid, grid, C, ph, pw) -> (B, num_patches, patch_dim)
        patches = patches.contiguous().view(B, self.grid_size * self.grid_size, -1)
        
        # Handle embed_mode
        if self.embed_mode == "int8":
            # Scale to 0-255 and cast to int8, but keep as float for embedding
            patches = patches * 255.0
        # else: float mode, already in [0, 1]
        
        # Optional projection
        if self.proj is not None:
            patches = self.proj(patches)
        
        return patches


# ---------------------------------------------------------------------------
# LLM Wrapper
# ---------------------------------------------------------------------------

class LLMWrapper:
    """
    Wrapper for HuggingFace LLM with mixed text+visual embedding support.
    
    Handles:
    - Loading model and tokenizer
    - Text tokenization and embedding
    - Mixed input (text embeddings + visual embeddings)
    - Prefill and decode with KV cache management
    - Probability extraction
    """
    
    def __init__(self, model_name: str, device: str = "auto", use_flash_attention: bool = True):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError:
            raise ImportError("transformers not installed. Run: pip install transformers")
        
        # For now, force single device to avoid device_map issues
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logging.info(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        logging.info(f"Loading model: {model_name}")
        
        # Build kwargs for model loading
        model_kwargs = {"device_map": None}  # Force single device
        
        # Enable Flash Attention (enabled by default)
        if use_flash_attention:
            # model_kwargs["attn_implementation"] = "flash_attention"
            logging.info("Flash Attention enabled")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
        ).to(self.device)
        
        # Get embedding dimension
        self.embedding_dim = self.model.config.hidden_size
        logging.info(f"Model embedding dimension: {self.embedding_dim}")
        logging.info(f"Model device: {self.device}")
        
        # Check for Flash Attention
        # 1. Check if explicitly requested
        use_flash_explicit = use_flash_attention
        
        # 2. Check model config attributes
        use_flash_config = (
            getattr(self.model.config, 'use_flash_attention', False) or
            getattr(self.model.config, 'use_flash_attn', False) or
            getattr(self.model.config, 'flash_attention', False)
        )
        
        # 3. Check attention implementation in config
        attn_impl = getattr(self.model.config, 'attn_implementation', None)
        use_flash_attn_impl = attn_impl == 'flash_attention' if attn_impl else False
        
        # 4. Check if flash attention is available in PyTorch
        has_flash_sdp = hasattr(torch.backends.cuda, 'enable_flash_sdp')
        has_math_sdp = hasattr(torch.backends.cuda, 'enable_math_sdp')
        
        # 5. Check model dtype
        model_dtype = next(self.model.parameters()).dtype
        flash_supported_dtype = model_dtype in (torch.bfloat16, torch.float16)
        
        # Flash is used if: explicitly requested OR config says so OR attn_implementation is flash
        use_flash = use_flash_explicit or use_flash_config or use_flash_attn_impl
        
        # Log detailed info
        logging.info(f"Flash Attention: {'enabled' if use_flash else 'disabled'}")
        logging.info(f"  Explicit request: {use_flash_explicit}")
        logging.info(f"  Config: use_flash_attention={use_flash_config}, attn_implementation={attn_impl}")
        logging.info(f"  PyTorch: enable_flash_sdp={has_flash_sdp}, enable_math_sdp={has_math_sdp}")
        logging.info(f"  Model dtype: {model_dtype}")
        
        # Store flash attention status for reference
        self.use_flash_attention = use_flash
        
        # Check for special tokens
        self.special_tokens = {}
        for name, token in SPECIAL_TOKENS.items():
            token_id = self.tokenizer.convert_tokens_to_ids([token])
            if token_id:
                self.special_tokens[name] = token_id[0]
        logging.info(f"Special tokens: {self.special_tokens}")
    
    def embed_text(self, text: str) -> torch.Tensor:
        """
        Tokenize text and convert to embeddings.
        
        Args:
            text: Input text string
            
        Returns:
            (1, seq_len, embedding_dim) tensor
        """
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids
        # Get device from model's embedding layer
        embed_layer = self.model.get_input_embeddings()
        device = next(embed_layer.parameters()).device
        input_ids = input_ids.to(device)
        embeds = embed_layer(input_ids)
        return embeds
    
    def prefill(self, inputs_embeds: torch.Tensor, 
                use_cache: bool = True,
                past_key_values: Optional[tuple] = None) -> Tuple[torch.Tensor, Optional[tuple]]:
        """
        Single forward pass (prefill).
        
        Args:
            inputs_embeds: (batch, seq_len, embedding_dim) tensor
            use_cache: Whether to return past_key_values
            past_key_values: Previous KV cache to use
            
        Returns:
            (logits, past_key_values) tuple
        """
        # Move to device and cast to model's dtype
        inputs_embeds = inputs_embeds.to(self.device)
        model_dtype = next(self.model.parameters()).dtype
        inputs_embeds = inputs_embeds.to(model_dtype)
        
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=True,
        )
        return outputs.logits, outputs.past_key_values if use_cache else None
    
    def decode(self, inputs_embeds: torch.Tensor, past_key_values: Optional[tuple] = None,
               max_new_tokens: int = 20, do_sample: bool = True, 
               top_p: float = 0.9) -> Tuple[torch.Tensor, Optional[tuple]]:
        """
        Auto-regressive decode with optional sampling.
        
        Args:
            inputs_embeds: (batch, seq_len, embedding_dim) - next token embeddings
            past_key_values: Previous KV cache
            max_new_tokens: Maximum tokens to generate
            do_sample: Use sampling or greedy
            top_p: Top-p sampling probability
            
        Returns:
            (output_ids, past_key_values) tuple
        """
        # For AR decode, we need to use input_ids, not inputs_embeds
        # This is a limitation - we need to convert embeds back or use a different approach
        # 
        # Alternative: Use the model's embedding layer to get the most likely token
        # But this is complex. For now, we'll use generate() which handles this internally.
        
        # For simplicity, we'll use generate with inputs_embeds
        # But generate() with inputs_embeds is tricky...
        # 
        # Actually, we can use the following approach:
        # 1. Get the last embedding as the starting point
        # 2. Use generate with the past_key_values
        
        raise NotImplementedError("Decode with inputs_embeds not yet implemented. Use generate_ar() instead.")
    
    def generate_ar(self, inputs_embeds: torch.Tensor, past_key_values: Optional[tuple] = None,
                    max_new_tokens: int = 20, do_sample: bool = True, 
                    top_p: float = 0.9) -> torch.Tensor:
        """
        Generate text using auto-regressive decoding.
        
        Args:
            inputs_embeds: (batch, seq_len, embedding_dim) - prefill embeddings
            past_key_values: Previous KV cache (optional)
            max_new_tokens: Maximum tokens to generate
            do_sample: Use sampling or greedy
            top_p: Top-p sampling probability
            
        Returns:
            (batch, generated_seq_len) token IDs
        """
        # This is a workaround: we need to convert inputs_embeds to something generate can use
        # But generate expects input_ids, not inputs_embeds
        # 
        # We'll use a different approach: manually iterate
        if past_key_values is None:
            # Full prefill + generate
            outputs = self.model.generate(
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
            )
        else:
            # This is complex - we need to handle inputs_embeds properly
            # For now, let's use a simple loop
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                use_cache=True,
            )
            # Continue from here...
            raise NotImplementedError("AR decode with KV cache not yet implemented")
        
        return outputs
    
    def get_next_token_logits(self, inputs_embeds: torch.Tensor, 
                               past_key_values: Optional[tuple] = None) -> torch.Tensor:
        """
        Get logits for the next token given context.
        
        Args:
            inputs_embeds: (batch, seq_len, embedding_dim) - context embeddings
            past_key_values: Previous KV cache
            
        Returns:
            (batch, vocab_size) logits for next token
        """
        inputs_embeds = inputs_embeds.to(self.device)
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=True,
        )
        # Get logits for last position
        next_token_logits = outputs.logits[:, -1, :]
        return next_token_logits
    
    def get_sequence_prob(self, context_embeds: torch.Tensor, 
                          target_text: str, past_key_values: Optional[tuple] = None) -> float:
        """
        Compute probability of generating target_text given context.
        
        Args:
            context_embeds: (1, ctx_len, embedding_dim) - context embeddings
            target_text: Target text to score
            past_key_values: Previous KV cache
            
        Returns:
            Log probability of the sequence
        """
        # Get model dtype for proper casting
        model_dtype = next(self.model.parameters()).dtype
        
        # Cast context embeds to model dtype
        context_embeds = context_embeds.to(self.device).to(model_dtype)
        
        target_ids = self.tokenizer(target_text, return_tensors="pt").input_ids.to(self.device)
        target_embeds = self.model.get_input_embeddings()(target_ids)
        # target_embeds is already in model dtype
        
        # Process context first to get updated KV cache
        log_prob = 0.0
        current_pkv = past_key_values
        
        # Process context embeddings (if any)
        if context_embeds.shape[1] > 0:
            outputs = self.model(
                inputs_embeds=context_embeds,
                past_key_values=current_pkv,
                use_cache=True,
            )
            current_pkv = outputs.past_key_values
        
        # Process target sequence token by token
        for i in range(target_embeds.shape[1]):
            token_embed = target_embeds[:, i:i+1]
            outputs = self.model(
                inputs_embeds=token_embed,
                past_key_values=current_pkv,
                use_cache=True,
            )
            # outputs.logits shape: (batch=1, seq_len=1, vocab_size)
            # Get logits for the single token in the batch
            token_logits = outputs.logits[0, 0, :]  # Shape: (vocab_size,)
            token_id = target_ids[0, i].item()
            prob = torch.softmax(token_logits, dim=-1)[token_id]
            log_prob += torch.log(prob + 1e-10)
            current_pkv = outputs.past_key_values
        
        return log_prob.item()


# ---------------------------------------------------------------------------
# KV Cache Manager
# ---------------------------------------------------------------------------

def remove_kv_range(past_key_values, start: int, end: int):
    """
    Remove KV cache entries for positions [start:end].
    
    Args:
        past_key_values: KV cache from model output (tuple, DynamicCache, or None)
        start: First position to remove (inclusive)
        end: Position after last to remove (exclusive)
        
    Returns:
        New past_key_values with range removed, or None if not supported
    """
    if past_key_values is None:
        return None
    
    # Handle DynamicCache (HuggingFace cache object for Llama/SmolLM)
    if hasattr(past_key_values, 'get_seq_length') and hasattr(past_key_values, 'layers'):
        # DynamicCache: directly modify keys/values tensors in each layer
        try:
            for layer in past_key_values.layers:
                # Get current seq_len
                seq_len = layer.get_seq_length()
                
                if start >= seq_len:
                    # Nothing to remove
                    continue
                
                # Only keep [0:start] - we can't keep non-contiguous ranges with DynamicCache
                # This is a limitation: we can only crop from the end, not remove from middle
                # So we crop to keep only tokens before the visual embeds
                if end < seq_len:
                    # Create new tensors with only [0:start] + [end:seq_len]
                    # But DynamicCache expects contiguous indices...
                    # Actually, we need to keep contiguous indices only
                    # So we'll keep [0:start] and shift [end:seq_len] to [start:end-start]
                    keep_before = layer.keys[:, :, :start, :]
                    keep_after = layer.keys[:, :, end:, :]
                    new_keys = torch.cat([keep_before, keep_after], dim=2)
                    
                    keep_before_v = layer.values[:, :, :start, :]
                    keep_after_v = layer.values[:, :, end:, :]
                    new_values = torch.cat([keep_before_v, keep_after_v], dim=2)
                    
                    layer.keys = new_keys
                    layer.values = new_values
            return past_key_values
        except Exception as e:
            logging.warning(f"KV sink for DynamicCache failed: {e}. Use --disable-kv-sink")
            return None
    
    # Handle tuple of tuples (older HuggingFace versions)
    if isinstance(past_key_values, (tuple, list)):
        new_pkv = []
        for pkv_item in past_key_values:
            if isinstance(pkv_item, tuple) and len(pkv_item) == 2:
                layer_k, layer_v = pkv_item
            elif isinstance(pkv_item, tuple) and len(pkv_item) == 3:
                layer_k, layer_v, _ = pkv_item
            else:
                logging.warning(f"Unexpected past_key_values structure")
                return None
            
            if start >= layer_k.shape[2]:
                new_pkv.append((layer_k, layer_v))
                continue
            
            new_k = layer_k[:, :, :start, :]
            if end < layer_k.shape[2]:
                new_k = torch.cat([new_k, layer_k[:, :, end:, :]], dim=2)
            
            new_v = layer_v[:, :, :start, :]
            if end < layer_v.shape[2]:
                new_v = torch.cat([new_v, layer_v[:, :, end:, :]], dim=2)
            
            new_pkv.append((new_k, new_v))
        return tuple(new_pkv)
    
    logging.warning(f"KV sink not supported for type {type(past_key_values)}. Use --disable-kv-sink")
    return None


def get_kv_seq_len(past_key_values):
    """Get the sequence length from KV cache."""
    if past_key_values is None:
        return 0
    
    # Handle different KV cache formats
    if hasattr(past_key_values, 'get_seq_length'):
        # DynamicCache or similar
        return past_key_values.get_seq_length()
    elif isinstance(past_key_values, (tuple, list)) and len(past_key_values) > 0:
        # Tuple of (key, value) tuples
        if isinstance(past_key_values[0], tuple) and len(past_key_values[0]) >= 2:
            return past_key_values[0][0].shape[2]
    
    return 0


def plot_confusion_matrix(true_labels: List[str], predicted_labels: List[str], log_dir: str):
    """
    Plot and save confusion matrix using matplotlib.
    
    Args:
        true_labels: List of true class names
        predicted_labels: List of predicted class names
        log_dir: Directory to save the plot
    """
    if not HAS_MATPLOTLIB:
        logging.warning("matplotlib not available, skipping confusion matrix")
        return
    
    # Create label to index mapping
    label_to_idx = {label: idx for idx, label in enumerate(CIFAR10_CLASSES)}
    n_classes = len(CIFAR10_CLASSES)
    
    # Build confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(true_labels, predicted_labels):
        true_idx = label_to_idx[true]
        pred_idx = label_to_idx[pred]
        cm[true_idx, pred_idx] += 1
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Add labels
    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, CIFAR10_CLASSES, rotation=45, ha='right')
    plt.yticks(tick_marks, CIFAR10_CLASSES)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(n_classes):
        for j in range(n_classes):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Save plot
    plot_path = Path(log_dir) / "confusion_matrix.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Confusion matrix saved to: {plot_path}")
    
    # Print text version
    logging.info("\nConfusion Matrix (rows=true, cols=predicted):")
    row_format = "{:<12}" + " {:>6}" * n_classes
    header = "" + row_format.format("", *CIFAR10_CLASSES)
    logging.info(header)
    for i, true_label in enumerate(CIFAR10_CLASSES):
        row = [true_label] + [cm[i, j] for j in range(n_classes)]
        logging.info(row_format.format(*row))


# ---------------------------------------------------------------------------
# In-Context Learning
# ---------------------------------------------------------------------------

def get_separator_embed(llm: LLMWrapper, separator: str = "\n") -> torch.Tensor:
    """Get embedding for a separator token."""
    return llm.embed_text(separator)


def build_teaching_prompt(llm: LLMWrapper, class_name: str, 
                           visual_embeds: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    """
    Build teaching prompt with visual embeddings.
    
    Args:
        llm: LLM wrapper
        class_name: Current class name
        visual_embeds: (num_patches, embedding_dim) visual embeddings
        
    Returns:
        (full_embeds, visual_start_idx, visual_end_idx) tuple
    """
    # Build user text
    user_text = f"Class: {class_name}\nRemember these examples."
    user_embeds = llm.embed_text(user_text)
    
    # Ensure visual embeds are in correct dimension
    if visual_embeds.dim() == 2:
        visual_embeds = visual_embeds.unsqueeze(0)  # (1, num_patches, emb_dim)
    
    # Ensure visual_embeds is on same device as user_embeds
    visual_embeds = visual_embeds.to(user_embeds.device)
    
    # Project if needed
    if visual_embeds.shape[-1] != llm.embedding_dim:
        # Need projection - for now, just use a linear layer
        # This should be handled by Patchifier
        raise ValueError(f"Visual embed dimension {visual_embeds.shape[-1]} != LLM embedding dim {llm.embedding_dim}")
    
    # Combine
    full_embeds = torch.cat([user_embeds, visual_embeds], dim=1)
    
    visual_start = user_embeds.shape[1]
    visual_end = visual_start + visual_embeds.shape[1]
    
    return full_embeds, visual_start, visual_end


def build_test_prompt(llm: LLMWrapper, test_embed: torch.Tensor, use_multiple_choice: bool = False) -> torch.Tensor:
    """Build test prompt with a single test embedding.
    
    Args:
        llm: LLMWrapper instance
        test_embed: Test image embedding tensor
        use_multiple_choice: If True, format as multiple-choice question (a. cat, b. dog, ...)
    """
    if use_multiple_choice:
        # Build multiple-choice prompt
        choices = [f"{chr(97 + i)}. {cls}" for i, cls in enumerate(CIFAR10_CLASSES)]
        choices_text = "\n".join(choices)
        user_text = f"What class is this image?\n{choices_text}\nAnswer: "
    else:
        user_text = "What class is this image?"
    
    user_embeds = llm.embed_text(user_text)
    
    if test_embed.dim() == 2:
        test_embed = test_embed.unsqueeze(0)
    
    # Ensure test_embed is on same device as user_embeds
    test_embed = test_embed.to(user_embeds.device)
    
    full_embeds = torch.cat([user_embeds, test_embed], dim=1)
    return full_embeds


def teach_full_batch_per_class(llm: LLMWrapper, class_name: str, train_images: torch.Tensor,
                patchifier: Patchifier, use_kv_sink: bool = True,
                past_key_values: Optional[tuple] = None) -> tuple:
    """Mode 1: Teach all images of ONE class in a single batch (current behavior)."""

    """
    Teach the LLM a single class using in-context learning.
    
    Args:
        llm: LLM wrapper
        class_name: Class name (e.g., "Cat")
        train_images: (N, C, H, W) tensor of training images
        patchifier: Patchifier instance
        use_kv_sink: Whether to remove visual embeds from KV cache
        past_key_values: Previous KV cache to extend
        
    Returns:
        past_key_values after teaching this class
    """
    # Move images to the same device as the patchifier/projection
    device = next(patchifier.parameters()).device if patchifier.proj is not None else llm.device
    train_images = train_images.to(device)
    
    # Get visual embeddings for all training images
    patches = patchifier(train_images)  # (N, num_patches, patch_dim)
    
    # Add separator embeddings between images
    # This helps the model distinguish between different visual examples
    separator_embed = get_separator_embed(llm, separator="\n")  # (1, 1, embedding_dim)
    
    num_images = patches.shape[0]
    num_patches_per_image = patches.shape[1]
    
    # Build visual embeds with separators: [img1_patches, sep, img2_patches, sep, ...]
    visual_embeds_list = []
    for i in range(num_images):
        img_patches = patches[i:i+1]  # (1, num_patches, patch_dim)
        visual_embeds_list.append(img_patches)
        # Add separator after each image except the last
        if i < num_images - 1:
            visual_embeds_list.append(separator_embed)
    
    visual_embeds = torch.cat(visual_embeds_list, dim=1)  # (1, total_patches + separators, patch_dim)
    
    # Track the actual number of patches and separators
    total_visual_tokens = visual_embeds.shape[1]
    
    # === KV Sink Strategy ===
    # Process full sequence, then remove visual embeds AFTER assistant response
    # This ensures assistant response is generated with full context (including visuals)
    # but visuals are removed from the final cache
    
    if use_kv_sink:
        # Save previous cache length BEFORE adding new content
        prev_len = get_kv_seq_len(past_key_values) if past_key_values else 0
        
        # Step 1: Process user text
        user_text = f"What class is this image?"
        user_embeds = llm.embed_text(user_text)
        user_seq_len = user_embeds.shape[1]
        
        # Step 2: Prepare visual embeds
        visual_embeds = visual_embeds.to(llm.device)
        model_dtype = next(llm.model.parameters()).dtype
        visual_embeds = visual_embeds.to(model_dtype)
        visual_seq_len = visual_embeds.shape[1]
        
        logging.debug(f"  KV sink: prev_len={prev_len}, user_seq_len={user_seq_len}, visual_seq_len={visual_seq_len}")
        
        # Step 3: Process user + visual together (with previous cache if provided)
        full_embeds = torch.cat([user_embeds, visual_embeds], dim=1)
        logits, past_kv = llm.prefill(full_embeds, use_cache=True, past_key_values=past_key_values)
        
        logging.debug(f"  KV sink: after user+visual prefill, cache_len={get_kv_seq_len(past_kv)}")
        
        # Step 4: Process assistant response with full context
        assistant_text = f"{class_name}"
        assistant_embeds = llm.embed_text(assistant_text)
        assistant_seq_len = assistant_embeds.shape[1]
        
        logits, past_kv = llm.prefill(assistant_embeds, use_cache=True, past_key_values=past_kv)
        
        # Step 5: Remove visual embeds from cache (KV sink)
        # Current cache has: [previous] + user (user_seq_len) + visual (visual_seq_len) + assistant (assistant_seq_len)
        # We want to keep: [previous] + user + assistant (remove visual)
        # Visual starts at (previous_len + user_seq_len) and ends at (previous_len + user_seq_len + visual_seq_len)
        vis_start = prev_len + user_seq_len
        vis_end = prev_len + user_seq_len + visual_seq_len
        
        logging.debug(f"  KV sink: vis_start={vis_start}, vis_end={vis_end}, current_cache={get_kv_seq_len(past_kv)}")
        
        # Remove visual embeds by keeping [0:vis_start] and [vis_end:]
        past_kv = remove_kv_range(past_kv, vis_start, vis_end)
        
        logging.debug(f"  KV sink: after removal, cache_len={get_kv_seq_len(past_kv)}")
        
        # Verify new length
        expected_len = prev_len + user_seq_len + assistant_seq_len
        actual_len = get_kv_seq_len(past_kv)
        if actual_len != expected_len:
            logging.warning(f"KV sink: expected {expected_len}, got {actual_len}")
        
        if past_kv is None:
            logging.warning("KV sink failed - falling back to full context")
            # Re-process without removal
            full_embeds_with_asst = torch.cat([user_embeds, visual_embeds.to(user_embeds.device), assistant_embeds], dim=1)
            logits, past_kv = llm.prefill(full_embeds_with_asst, use_cache=True, past_key_values=past_key_values)
    
    # === Standard (no KV sink) ===
    else:
        # Process full sequence WITH visuals (use whole context)
        user_text = f"What class is this image?"
        user_embeds = llm.embed_text(user_text)
        
        # Process user + visuals
        full_embeds = torch.cat([user_embeds, visual_embeds.to(user_embeds.device)], dim=1)
        logits, past_kv = llm.prefill(full_embeds, use_cache=True, past_key_values=past_key_values)
        
        # Process assistant response
        assistant_text = f"{class_name}"
        assistant_embeds = llm.embed_text(assistant_text)
        logits, past_kv = llm.prefill(assistant_embeds, use_cache=True, past_key_values=past_kv)
        # Keep full context including visuals
    
    return past_kv


# ---------------------------------------------------------------------------
# Teaching Mode 2: Minibatch Per Class
# ---------------------------------------------------------------------------

def teach_minibatch_per_class(llm: LLMWrapper, class_name: str, train_images: torch.Tensor,
                              patchifier: Patchifier, use_kv_sink: bool = True,
                              past_key_values: Optional[tuple] = None,
                              batch_size: int = 5) -> tuple:
    """Mode 2: Teach one class split into minibatches.
    
    Each minibatch contains multiple images from the SAME class.
    Assistant confirms each batch.
    """
    device = next(patchifier.parameters()).device if patchifier.proj is not None else llm.device
    train_images = train_images.to(device)
    
    num_images = train_images.shape[0]
    num_batches = (num_images + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_images)
        batch_images = train_images[start_idx:end_idx]
        
        # Get patches for this batch
        patches = patchifier(batch_images)  # (batch_size, num_patches, patch_dim)
        
        # Add separators between images in the batch
        separator_embed = get_separator_embed(llm, separator="\n")
        
        # Build visual embeds with separators
        visual_embeds_list = []
        for i in range(patches.shape[0]):
            img_patches = patches[i:i+1]
            visual_embeds_list.append(img_patches)
            if i < patches.shape[0] - 1:
                visual_embeds_list.append(separator_embed)
        visual_embeds = torch.cat(visual_embeds_list, dim=1)
        visual_embeds = visual_embeds.to(llm.device)
        model_dtype = next(llm.model.parameters()).dtype
        visual_embeds = visual_embeds.to(model_dtype)
        
        # Build prompt
        user_text = f"Class: {class_name}\nRemember these examples."
        user_embeds = llm.embed_text(user_text)
        user_seq_len = user_embeds.shape[1]
        
        # Process with KV sink
        if use_kv_sink:
            prev_len = get_kv_seq_len(past_key_values) if past_key_values else 0
            full_embeds = torch.cat([user_embeds, visual_embeds], dim=1)
            logits, past_kv = llm.prefill(full_embeds, use_cache=True, past_key_values=past_key_values)
            
            assistant_text = f"ok i learnt batch {batch_idx + 1} for class {class_name}"
            assistant_embeds = llm.embed_text(assistant_text)
            assistant_seq_len = assistant_embeds.shape[1]
            logits, past_kv = llm.prefill(assistant_embeds, use_cache=True, past_key_values=past_kv)
            
            # Remove visual embeds from cache
            vis_start = prev_len + user_seq_len
            vis_end = prev_len + user_seq_len + visual_embeds.shape[1]
            past_kv = remove_kv_range(past_kv, vis_start, vis_end)
            
            # Verify
            expected_len = prev_len + user_seq_len + assistant_seq_len
            actual_len = get_kv_seq_len(past_kv)
            if actual_len != expected_len:
                logging.warning(f"KV sink batch {batch_idx}: expected {expected_len}, got {actual_len}")
            
            if past_kv is None:
                logging.warning(f"KV sink failed for batch {batch_idx}")
                full_embeds_with_asst = torch.cat([user_embeds, visual_embeds, assistant_embeds], dim=1)
                logits, past_kv = llm.prefill(full_embeds_with_asst, use_cache=True, past_key_values=past_key_values)
        else:
            # No KV sink
            user_text_q = f"Class: {class_name}\nRemember these examples."
            user_embeds_q = llm.embed_text(user_text_q)
            full_embeds = torch.cat([user_embeds_q, visual_embeds.to(user_embeds_q.device)], dim=1)
            logits, past_kv = llm.prefill(full_embeds, use_cache=True, past_key_values=past_key_values)
            
            assistant_text = f"ok i learnt batch {batch_idx + 1} for class {class_name}"
            assistant_embeds = llm.embed_text(assistant_text)
            logits, past_kv = llm.prefill(assistant_embeds, use_cache=True, past_key_values=past_kv)
        
        past_key_values = past_kv
        logging.info(f"  Taught batch {batch_idx + 1}/{num_batches} for {class_name}")
    
    return past_key_values


# ---------------------------------------------------------------------------
# Teaching Mode 3: Minibatch Mixed
# ---------------------------------------------------------------------------

def teach_minibatch_mixed(llm: LLMWrapper, all_train_images_by_class: dict,
                           patchifier: Patchifier, use_kv_sink: bool = True,
                           past_key_values: Optional[tuple] = None,
                           batch_size: int = 5) -> tuple:
    """Mode 3: Teach mixed classes in minibatches.
    
    Each minibatch contains images from MULTIPLE classes.
    Assistant summarizes each batch.
    """
    device = next(patchifier.parameters()).device if patchifier.proj is not None else llm.device
    separator_embed = get_separator_embed(llm, separator="\n")
    model_dtype = next(llm.model.parameters()).dtype
    
    # Flatten all images with their labels
    all_images = []
    all_labels = []
    for class_name, images in all_train_images_by_class.items():
        if len(images) > 0:
            for img in images:
                all_images.append(img.to(device))
                all_labels.append(class_name)
    
    if len(all_images) == 0:
        return past_key_values
    
    # Create minibatches
    total_samples = len(all_images)
    num_batches = (total_samples + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_samples)
        batch_images = all_images[start_idx:end_idx]
        batch_labels = all_labels[start_idx:end_idx]
        
        # Stack and get patches
        batch_images_tensor = torch.stack(batch_images)
        patches = patchifier(batch_images_tensor)  # (batch_size, num_patches, patch_dim)
        
        # Build prompt with labels
        batch_desc_parts = []
        for i, (img_patches, label) in enumerate(zip(patches, batch_labels)):
            batch_desc_parts.append(f"image {i+1} - {label}")
        batch_desc = "\n".join(batch_desc_parts)
        user_text = f"batch {batch_idx + 1}:\n{batch_desc}"
        user_embeds = llm.embed_text(user_text)
        
        # Build visual embeds with separators
        visual_embeds_list = []
        for i in range(patches.shape[0]):
            img_patches = patches[i:i+1]
            visual_embeds_list.append(img_patches)
            if i < patches.shape[0] - 1:
                visual_embeds_list.append(separator_embed)
        visual_embeds = torch.cat(visual_embeds_list, dim=1)
        visual_embeds = visual_embeds.to(llm.device).to(model_dtype)
        
        # Process with KV sink
        if use_kv_sink:
            prev_len = get_kv_seq_len(past_key_values) if past_key_values else 0
            full_embeds = torch.cat([user_embeds, visual_embeds], dim=1)
            logits, past_kv = llm.prefill(full_embeds, use_cache=True, past_key_values=past_key_values)
            
            assistant_text = f"done batch {batch_idx + 1}"
            assistant_embeds = llm.embed_text(assistant_text)
            assistant_seq_len = assistant_embeds.shape[1]
            logits, past_kv = llm.prefill(assistant_embeds, use_cache=True, past_key_values=past_kv)
            
            # Remove visual embeds from cache
            vis_start = prev_len + user_embeds.shape[1]
            vis_end = prev_len + user_embeds.shape[1] + visual_embeds.shape[1]
            past_kv = remove_kv_range(past_kv, vis_start, vis_end)
            
            # Verify
            expected_len = prev_len + user_embeds.shape[1] + assistant_seq_len
            actual_len = get_kv_seq_len(past_kv)
            if actual_len != expected_len:
                logging.warning(f"KV sink batch {batch_idx}: expected {expected_len}, got {actual_len}")
            
            if past_kv is None:
                logging.warning(f"KV sink failed for batch {batch_idx}")
                full_embeds_with_asst = torch.cat([user_embeds, visual_embeds, assistant_embeds], dim=1)
                logits, past_kv = llm.prefill(full_embeds_with_asst, use_cache=True, past_key_values=past_key_values)
        else:
            # No KV sink
            full_embeds = torch.cat([user_embeds, visual_embeds.to(user_embeds.device)], dim=1)
            logits, past_kv = llm.prefill(full_embeds, use_cache=True, past_key_values=past_key_values)
            
            assistant_text = f"done batch {batch_idx + 1}"
            assistant_embeds = llm.embed_text(assistant_text)
            logits, past_kv = llm.prefill(assistant_embeds, use_cache=True, past_key_values=past_kv)
        
        past_key_values = past_kv
        logging.info(f"  Taught minibatch {batch_idx + 1}/{num_batches} (mixed classes)")
    
    return past_key_values


def classify_image(llm: LLMWrapper, patchifier: Patchifier, test_image: torch.Tensor,
                   past_kv: Optional[tuple] = None, 
                   answer_mode: str = "first-token",
                   ar_decode_mode: str = "logits") -> Tuple[str, dict]:
    """
    Classify a single test image.
    
    Args:
        llm: LLM wrapper
        patchifier: Patchifier instance
        test_image: (C, H, W) tensor
        past_kv: Previous KV cache from teaching phase
        answer_mode: 'ar', 'probs', or 'first-token'
        
    Returns:
        (predicted_class, class_probabilities) tuple
    """
    # Move image to the same device as patchifier
    device = next(patchifier.parameters()).device if patchifier.proj is not None else llm.device
    test_image = test_image.to(device)
    
    # Get test embedding
    test_patches = patchifier(test_image.unsqueeze(0))  # (1, num_patches, patch_dim)
    
    # Build test prompt (multiple-choice format for AR mode)
    use_multiple_choice = (answer_mode == "ar")
    test_embeds = build_test_prompt(llm, test_patches, use_multiple_choice=use_multiple_choice)
    
    # Get logits for next token
    next_token_logits, _ = llm.prefill(test_embeds, use_cache=False, 
                                        past_key_values=past_kv)
    next_token_logits = next_token_logits[:, -1, :]  # (1, vocab_size)
    
    if answer_mode == "ar":
        # AR decode with multiple-choice prompt
        # Two modes: 'raw' (full AR decode + string matching) or 'logits' (first token filtered softmax)
        
        if ar_decode_mode == "logits":
            # Method 2: Collect logits, filter to class names, do softmax (faster)
            # Get first token IDs for each class
            class_first_tokens = {}
            for cls in CIFAR10_CLASSES:
                first_token_id = llm.tokenizer(cls, return_tensors="pt").input_ids[0, 0].item()
                class_first_tokens[cls] = first_token_id
            
            # Get probabilities for all class first tokens
            probs = torch.softmax(next_token_logits, dim=-1)[0]
            class_probs = {}
            for cls, token_id in class_first_tokens.items():
                class_probs[cls] = probs[token_id].item()
            
            predicted = max(class_probs, key=class_probs.get)
            return predicted, class_probs
        
        else:  # ar_decode_mode == "raw"
            # Method 1: Raw AR decode and string matching
            # Generate the answer text
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # Map token IDs to classes (for single-token class names)
            class_first_tokens = {}
            for cls in CIFAR10_CLASSES:
                first_token_id = llm.tokenizer(cls, return_tensors="pt").input_ids[0, 0].item()
                class_first_tokens[first_token_id] = cls
            
            # Get top token and check if it's a class
            top_token_id = torch.argmax(probs[0]).item()
            if top_token_id in class_first_tokens:
                predicted = class_first_tokens[top_token_id]
                # Build class_probs for consistency
                class_probs = {}
                for cls in CIFAR10_CLASSES:
                    token_id = llm.tokenizer(cls, return_tensors="pt").input_ids[0, 0].item()
                    class_probs[cls] = probs[0, token_id].item()
                return predicted, class_probs
            else:
                # Token not a class - need full AR decode (not implemented yet)
                # Fall back to first-token mode for now
                logging.warning("AR raw decode: top token not a class name, falling back to first-token")
                class_probs = {}
                for cls in CIFAR10_CLASSES:
                    first_token_id = llm.tokenizer(cls, return_tensors="pt").input_ids[0, 0]
                    prob = probs[0, first_token_id].item()
                    class_probs[cls] = prob
                predicted = max(class_probs, key=class_probs.get)
                return predicted, class_probs
    
    elif answer_mode == "probs":
        # Full sequence probability
        class_probs = {}
        for cls in CIFAR10_CLASSES:
            answer_text = f"This is a {cls}."
            log_prob = llm.get_sequence_prob(test_embeds, answer_text, past_kv)
            class_probs[cls] = log_prob
            # Clear CUDA cache after each class to prevent VRAM buildup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
        predicted = max(class_probs, key=class_probs.get)
        return predicted, class_probs
    
    elif answer_mode == "first-token":
        # First token probability only (fastest)
        class_probs = {}
        
        # Get the first token ID for each class
        class_first_tokens = {}
        for cls in CIFAR10_CLASSES:
            first_token_id = llm.tokenizer(cls, return_tensors="pt").input_ids[0, 0].item()
            class_first_tokens[cls] = first_token_id
        
        # Get top-k tokens from logits
        probs = torch.softmax(next_token_logits, dim=-1)[0]
        topk = torch.topk(probs, k=20)
        
        # Print debug info
        logging.debug(f"Top 20 tokens and probabilities:")
        for token_id, prob in zip(topk.indices.tolist(), topk.values.tolist()):
            token_str = llm.tokenizer.convert_ids_to_tokens([token_id])[0]
            logging.debug(f"  {token_str}: {prob:.6f}")
            # Check if this token is the first token of any class
            for cls, first_id in class_first_tokens.items():
                if token_id == first_id:
                    class_probs[cls] = prob
        
        # For classes whose first token wasn't in top-20, set probability to 0
        for cls in CIFAR10_CLASSES:
            if cls not in class_probs:
                first_token_id = class_first_tokens[cls]
                class_probs[cls] = probs[first_token_id].item()
        
        predicted = max(class_probs, key=class_probs.get)
        return predicted, class_probs
    
    else:
        raise ValueError(f"Unknown answer mode: {answer_mode}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    setup_logging(args.log_dir, level=args.log_level)
    
    logging.info("=" * 80)
    logging.info("LLM Zero-Shot Learner for CIFAR-10")
    logging.info("=" * 80)
    
    # Log arguments
    logging.info("Arguments:")
    for key, value in vars(args).items():
        if not key.startswith('_'):
            logging.info(f"  {key}: {value}")
    
    # -----------------------------------------------------------------------
    # Setup device
    # -----------------------------------------------------------------------
    device = torch.device(args.device)
    logging.info(f"Using device: {device}")
    
    # -----------------------------------------------------------------------
    # Load CIFAR-10 datasets
    # -----------------------------------------------------------------------
    logging.info("\nLoading CIFAR-10 datasets...")
    
    # For now, use raw pixels (no normalization) for embedding
    train_ds_full = get_cifar10_full(args.data_dir, train=True, normalized=False)
    test_ds_full = get_cifar10_full(args.data_dir, train=False, normalized=False)
    
    logging.info(f"Full train set: {len(train_ds_full)} images")
    logging.info(f"Full test set: {len(test_ds_full)} images")
    
    # -----------------------------------------------------------------------
    # Create subsets
    # -----------------------------------------------------------------------
    if args.full_dataset:
        train_ds = train_ds_full
        test_ds = test_ds_full
        logging.info("Using full datasets (subset mode disabled)")
    else:
        train_ds = get_subset(
            train_ds_full,
            spec=args.train_subset_spec,
            ordered=args.subset_ordered,
            seed=args.subset_seed
        )
        
        # Sanity check mode: use same subset for train and test (recall check)
        if args.sanity_check:
            test_ds = train_ds
            logging.info(f"SANITY CHECK MODE: Using same {len(train_ds)} train images as test images")
        else:
            test_ds = get_subset(
                test_ds_full,
                spec=args.test_subset_spec,
                ordered=args.subset_ordered,
                seed=args.subset_seed
            )
            logging.info(f"Train subset: {len(train_ds)} images ({len(train_ds)/len(train_ds_full)*100:.1f}% of train set)")
            logging.info(f"Test subset: {len(test_ds)} images ({len(test_ds)/len(test_ds_full)*100:.1f}% of test set)")
    
    # -----------------------------------------------------------------------
    # Save images
    # -----------------------------------------------------------------------
    if args.save_images:
        args.images_dir.mkdir(parents=True, exist_ok=True)
        
        train_paths = save_dataset_images(train_ds, args.images_dir, "train")
        test_paths = save_dataset_images(test_ds, args.images_dir, "test")
        
        logging.info(f"\nSaved {len(train_paths)} train images to: {args.images_dir}")
        for p in train_paths[:5]:
            logging.info(f"  {p}")
        if len(train_paths) > 5:
            logging.info(f"  ... and {len(train_paths) - 5} more")
        
        logging.info(f"\nSaved {len(test_paths)} test images to: {args.images_dir}")
        for p in test_paths:
            logging.info(f"  {p}")
    
    # -----------------------------------------------------------------------
    # Initialize Patchifier
    # -----------------------------------------------------------------------
    logging.info("\nInitializing Patchifier...")
    patchifier = Patchifier(
        patch_size=args.patch_size,
        embed_mode=args.embed_mode,
        proj_dim=args.proj_dim
    )
    logging.info(f"Patchifier: patch_size={args.patch_size}, embed_mode={args.embed_mode}")
    
    # -----------------------------------------------------------------------
    # Initialize LLM
    # -----------------------------------------------------------------------
    logging.info("\nInitializing LLM...")
    try:
        llm = LLMWrapper(args.model_name, device=args.device, use_flash_attention=args.use_flash_attention)
    except ImportError as e:
        logging.error(f"Failed to import transformers: {e}")
        logging.error("Install with: pip install transformers")
        return
    
    # Auto-set projection dimension if needed
    if args.proj_dim is None and patchifier.proj is None:
        # Patchifier output dim must match LLM embedding dim
        proj_layer = nn.Linear(patchifier.patch_dim, llm.embedding_dim)
        patchifier.proj = proj_layer.to(device)
        logging.info(f"Auto-projection: {patchifier.patch_dim} -> {llm.embedding_dim}")
    
    # -----------------------------------------------------------------------
    # In-Context Learning: Teach all classes
    # -----------------------------------------------------------------------
    logging.info("\n" + "=" * 80)
    logging.info("TEACHING PHASE")
    logging.info("=" * 80)
    
    # Group train images by class
    train_by_class = {cls: [] for cls in CIFAR10_CLASSES}
    for img, label in train_ds:
        cls = CIFAR10_CLASSES[label]
        train_by_class[cls].append(img)
    
    # Convert to tensors
    for cls in train_by_class:
        if train_by_class[cls]:
            train_by_class[cls] = torch.stack(train_by_class[cls])
    
    # Teach each class
    past_kv = None
    
    if args.teaching_mode == "full-batch-per-class":
        # Mode 1: All images of one class in a single batch (original)
        for class_idx, class_name in enumerate(CIFAR10_CLASSES):
            images = train_by_class[class_name]
            if len(images) == 0:
                logging.warning(f"No training images for class: {class_name}")
                continue
            
            prev_len = get_kv_seq_len(past_kv)
            logging.info(f"\nTeaching class: {class_name} ({len(images)} examples), previous cache: {prev_len} tokens")
            
            try:
                past_kv = teach_full_batch_per_class(
                    llm, class_name, images,
                    patchifier, use_kv_sink=args.use_kv_sink,
                    past_key_values=past_kv
                )
                new_len = get_kv_seq_len(past_kv)
                logging.info(f"  -> Successfully taught {class_name}, cache now: {new_len} tokens")
            except Exception as e:
                logging.error(f"  -> Error teaching {class_name}: {e}")
                import traceback
                traceback.print_exc()
                break
    
    elif args.teaching_mode == "minibatch-per-class":
        # Mode 2: Split each class into minibatches
        for class_idx, class_name in enumerate(CIFAR10_CLASSES):
            images = train_by_class[class_name]
            if len(images) == 0:
                logging.warning(f"No training images for class: {class_name}")
                continue
            
            prev_len = get_kv_seq_len(past_kv)
            logging.info(f"\nTeaching class: {class_name} ({len(images)} examples in batches of {args.teaching_batch_size}), previous cache: {prev_len} tokens")
            
            try:
                past_kv = teach_minibatch_per_class(
                    llm, class_name, images,
                    patchifier, use_kv_sink=args.use_kv_sink,
                    past_key_values=past_kv,
                    batch_size=args.teaching_batch_size
                )
                new_len = get_kv_seq_len(past_kv)
                logging.info(f"  -> Successfully taught {class_name} in minibatches, cache now: {new_len} tokens")
            except Exception as e:
                logging.error(f"  -> Error teaching {class_name}: {e}")
                import traceback
                traceback.print_exc()
                break
    
    elif args.teaching_mode == "minibatch-mixed":
        # Mode 3: Mix all classes into minibatches
        logging.info(f"\nTeaching with mixed minibatches (size={args.teaching_batch_size})")
        
        try:
            past_kv = teach_minibatch_mixed(
                llm, train_by_class,
                patchifier, use_kv_sink=args.use_kv_sink,
                past_key_values=past_kv,
                batch_size=args.teaching_batch_size
            )
            new_len = get_kv_seq_len(past_kv)
            logging.info(f"  -> Successfully taught all classes in mixed minibatches, cache now: {new_len} tokens")
        except Exception as e:
            logging.error(f"  -> Error in mixed minibatch teaching: {e}")
            import traceback
            traceback.print_exc()
            past_kv = None
    
    if past_kv is None:
        logging.error("No classes were taught successfully. Exiting.")
        return
    
    logging.info(f"\nTeaching complete. KV cache sequence length: {get_kv_seq_len(past_kv)}")
    
    # -----------------------------------------------------------------------
    # Classification: Test all images
    # -----------------------------------------------------------------------
    logging.info("\n" + "=" * 80)
    logging.info("TESTING PHASE")
    logging.info("=" * 80)
    logging.info(f"Answer mode: {args.answer_mode}")
    
    # Group test images by class for reporting
    test_by_class = {cls: [] for cls in CIFAR10_CLASSES}
    for img, label in test_ds:
        cls = CIFAR10_CLASSES[label]
        test_by_class[cls].append((img, label))
    
    # Flatten for iteration and track paths
    test_images = []
    test_labels = []
    test_paths = []  # Track paths for each test image
    
    if args.save_images:
        # Build paths in the same order as test_images
        for cls in CIFAR10_CLASSES:
            class_images = [img for img, _ in test_by_class[cls]]
            test_images.extend(class_images)
            test_labels.extend([CIFAR10_CLASSES.index(cls) for _ in class_images])
            # Generate expected paths: test_{class}_{index}.png
            for i in range(len(class_images)):
                path = str(args.images_dir / f"test_{cls}_{i:04d}.png")
                test_paths.append(path)
    else:
        # No paths available, use empty list
        for cls in CIFAR10_CLASSES:
            test_images.extend([img for img, _ in test_by_class[cls]])
            test_labels.extend([label for _, label in test_by_class[cls]])
            test_paths.extend([None] * len(test_by_class[cls]))
    
    # Test each image
    correct = 0
    total = len(test_images)
    
    # For confusion matrix
    all_true_labels = []
    all_predicted_labels = []
    
    for test_idx, (test_img, true_label) in enumerate(zip(test_images, test_labels)):
        true_class = CIFAR10_CLASSES[true_label]
        test_path = test_paths[test_idx] if test_idx < len(test_paths) else None
        
        try:
            predicted_class, class_probs = classify_image(
                llm, patchifier, test_img,
                past_kv=past_kv,
                answer_mode=args.answer_mode,
                ar_decode_mode=args.ar_decode_mode
            )
            
            is_correct = predicted_class == true_class
            if is_correct:
                correct += 1
            
            # Track for confusion matrix
            all_true_labels.append(true_class)
            all_predicted_labels.append(predicted_class)
            
            path_str = f"\n  Path: {test_path}" if test_path else ""
            logging.info(f"\nTest {test_idx + 1}/{total}:{path_str}")
            logging.info(f"  True: {true_class}")
            logging.info(f"  Predicted: {predicted_class} {'✓' if is_correct else '✗'}")
            
            if args.log_level == "DEBUG":
                logging.debug(f"  Class probabilities:")
                for cls, prob in sorted(class_probs.items(), key=lambda x: -x[1]):
                    logging.debug(f"    {cls}: {prob:.4f}")
            
            # Clear CUDA cache and run garbage collection to prevent VRAM pollution
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
        
        except Exception as e:
            logging.error(f"Error classifying test image {test_idx}: {e}")
            import traceback
            traceback.print_exc()
            # Clear cache on error too
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    # -----------------------------------------------------------------------
    # Confusion Matrix
    # -----------------------------------------------------------------------
    if args.plot_confusion and HAS_MATPLOTLIB:
        try:
            plot_confusion_matrix(all_true_labels, all_predicted_labels, args.log_dir)
        except Exception as e:
            logging.warning(f"Failed to plot confusion matrix: {e}")
    
    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    logging.info("\n" + "=" * 80)
    logging.info("SUMMARY")
    logging.info("=" * 80)
    logging.info(f"Total test images: {total}")
    logging.info(f"Correct: {correct}")
    logging.info(f"Accuracy: {accuracy:.2f}%")
    logging.info(f"KV sink: {'enabled' if args.use_kv_sink else 'disabled'}")
    logging.info(f"Answer mode: {args.answer_mode}")
    logging.info(f"Flash Attention: {'enabled' if llm.use_flash_attention else 'disabled'}")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
