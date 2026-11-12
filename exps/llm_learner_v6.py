"""
LLM Learner - Simulated In-Context Learning via KV Cache

Experiment: simulate SGD-like few-shot learning without backprop using a frozen
pretrained LLM. Training examples are "taught" by prefilling the KV cache;
test images are classified by querying against that accumulated context.

Design pillars
--------------
1. Any HuggingFace causal LLM (Qwen3, Llama3, SmolLM, etc.)
2. Chat-tag embedding injection:
   - Chat turn opening/closing tags are extracted from the HF tokenizer's template
   - Visual (image) embeds are manually spliced BETWEEN those tags
   - No automatic apply_chat_template -- we do it ourselves embed-by-embed
3. Image tokenizer with two modes:
   - 'floats': patch pixels -> linear project to LLM hidden dim
     Reductions: 'random_proj' (learnable linear), 'sum', 'avg'
   - 'byte_embeds': each pixel channel (0-255) -> LLM vocab embedding vector,
     treat patch as bag-of-vectors, then reduce
     Reductions: 'sum' (default), 'avg', 'random_proj'
4. KV cache compression (pluggable base class):
   - Noop: identity, keep everything
   - StreamingLLM: retain fixed sink positions + last K tokens
   - AttentionQuery: retain top-K tokens by attention score from "attractor" query positions
5. Teaching: the model predicts one image at a time and receives feedback.
    Subsets default to 100 images per class; training shuffles with seed 0.
    Set --seed None for ordered training; integer and fractional subsets work.

Usage
-----
Qwen/Qwen3-0.6B
Qwen/Qwen3-1.7B
Qwen/Qwen3-4B
Qwen/Qwen3-8B
HuggingFaceTB/SmolLM-135M
HuggingFaceTB/SmolLM3-3B

python exps/llm_learner.py --generic-class-names --model-name HuggingFaceTB/SmolLM3-3B \
--img-mode byte_embeds --img-reduction sum --patch-size 16 \
--kv-policy attn_query --kv-sink 512 --kv-topk 2048 --neighbor-window 1 1 \
--attractor-round turn_tail --attractor-temporary \
--teaching-epochs 2 \
--eval-every 1 --train-eval-samples 100 --answer-mode full_prob --rank-k 3 --no-sample-ranking --reasoning --reasoning-max-new-tokens 24 --reasoning-temperature 0.7 --reasoning-top-p 0.9 --guess-prob-gt 0.0 --guess-prob-random 0.0
python exps/llm_learner.py --img-mode byte_embeds --img-reduction sum
python exps/llm_learner.py --kv-policy streaming --kv-sink 4 --kv-last 128
python exps/llm_learner.py --kv-policy attn_query --kv-topk 128 \
    --attractor-range -10 -1 --neighbor-window 1 1
python exps/llm_learner.py --train-samples 5
python exps/llm_learner.py --train-samples 0.1 --test-samples 0.1 --seed None
python exps/llm_learner.py --generic-class-names
python exps/llm_learner.py --config exps/configs/llm_learner_config.py
"""

import argparse
import ast
import copy
import csv
import gc
import logging
import math
import random
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_REASONING_MAX_NEW_TOKENS = 128

# CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer","dog", "frog", "horse", "ship", "truck",]
CIFAR10_CLASSES = ["A", "B", "C", "D", "E","F", "G", "H", "I", "J",]
# CIFAR10_CLASSES = ["P", "Q", "R", "S", "T","U", "V", "W", "X", "Y",]
# CIFAR10_CLASSES = ["1", "2", "3", "4", "5","6", "7", "8", "9", "10",]

# Sentinels for chat template parsing (chosen to be unlikely in real templates)
_SENT_U1 = "\x01USR1\x01"
_SENT_A1 = "\x01AST1\x01"
_SENT_U2 = "\x01USR2\x01"
_SENT_A2 = "\x01AST2\x01"


# ---------------------------------------------------------------------------
# 1. Chat Template Extractor
# ---------------------------------------------------------------------------

@dataclass
class ChatSegments:
    """
    Raw text segments of a model's chat template.

    For the FIRST turn of a conversation:
        full = global_prefix + <user content> + user_to_asst + <asst content> + asst_tail

    For SUBSEQUENT turns (turn N > 1):
        full += turn_separator + <user content> + user_to_asst + <asst content> + asst_tail

    When building a query (no assistant response yet):
        full = (global_prefix | turn_separator) + <user content> + asst_open
        where asst_open is the opening half of user_to_asst if splittable,
        otherwise the full user_to_asst (model will continue from there).
    """
    global_prefix: str    # BOS + system block + user opening tag
    user_to_asst: str     # user closing tag + assistant opening tag
    asst_tail: str        # assistant closing tag + EOS
    turn_separator: str   # asst_tail_between + user opening tag (between turns)

    # Optionally split user_to_asst -> (user_close, asst_open)
    # These may be empty strings if the template doesn't have an explicit split
    user_close: str = ""
    asst_open: str = ""


class ChatTemplateExtractor:
    """
    Extract chat-turn text segments from any HuggingFace tokenizer.

    Approach: call apply_chat_template with known sentinel strings, then
    parse the resulting text to identify where each segment lives.
    Uses two full turns so the SECOND turn is free from BOS/global-prefix noise.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._segments: Optional[ChatSegments] = None

    @property
    def segments(self) -> ChatSegments:
        if self._segments is None:
            self._segments = self._extract()
        return self._segments

    def _apply_chat_template(self, messages, **kwargs):
        """Use visible-answer mode where supported (Qwen3: disable thinking).

        Tokenizers without this option, including SmolLM-family templates,
        transparently retain their original chat-template behavior.
        """
        try:
            return self.tokenizer.apply_chat_template(
                messages, enable_thinking=False, **kwargs
            )
        except (TypeError, ValueError):
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def _extract(self) -> ChatSegments:
        tok = self.tokenizer

        # -- Two-turn conversation with 4 unique sentinels --
        msgs = [
            {"role": "user",      "content": _SENT_U1},
            {"role": "assistant", "content": _SENT_A1},
            {"role": "user",      "content": _SENT_U2},
            {"role": "assistant", "content": _SENT_A2},
        ]

        try:
            full = self._apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            # Some tokenizers won't allow this; try single-turn fallback
            return self._extract_single_turn_fallback()

        try:
            # Positions of each sentinel in the full string
            u1s = full.index(_SENT_U1)
            a1s = full.index(_SENT_A1)
            u2s = full.index(_SENT_U2)
            a2s = full.index(_SENT_A2)
        except ValueError as e:
            logging.warning(f"Chat template extraction sentinel not found: {e}. "
                            "Falling back to minimal template.")
            return self._extract_single_turn_fallback()

        u1e = u1s + len(_SENT_U1)
        a1e = a1s + len(_SENT_A1)
        u2e = u2s + len(_SENT_U2)
        a2e = a2s + len(_SENT_A2)

        # global_prefix: everything before the first user content
        global_prefix = full[:u1s]

        # user_to_asst: between end of user2 content and start of asst2 content
        # (This is clean because both sentinels are on the same second turn)
        user_to_asst = full[u2e:a2s]

        # asst_tail: after the last assistant content
        asst_tail = full[a2e:]

        # turn_separator: from end of asst1 content to start of user2 content
        # = asst close tags + whatever separates turns + user open tags
        turn_separator = full[a1e:u2s]

        # Try to split user_to_asst into (user_close, asst_open).
        user_close, asst_open = self._split_user_to_asst(user_to_asst)

        segs = ChatSegments(
            global_prefix=global_prefix,
            user_to_asst=user_to_asst,
            asst_tail=asst_tail,
            turn_separator=turn_separator,
            user_close=user_close,
            asst_open=asst_open,
        )
        logging.info("Chat template segments extracted:")
        logging.info(f"  global_prefix : {repr(segs.global_prefix)}")
        logging.info(f"  user_to_asst  : {repr(segs.user_to_asst)}")
        logging.info(f"  asst_tail     : {repr(segs.asst_tail)}")
        logging.info(f"  turn_separator: {repr(segs.turn_separator)}")
        return segs

    def _split_user_to_asst(self, user_to_asst: str) -> Tuple[str, str]:
        """
        Try to split user_to_asst into (user_close_tag, asst_open_tag).
        Uses add_generation_prompt=True to find where the assistant start token is.
        """
        tok = self.tokenizer
        msgs2 = [{"role": "user", "content": _SENT_U2}]
        try:
            gen_prompt = self._apply_chat_template(
                msgs2, tokenize=False, add_generation_prompt=True
            )
            u2s_gp = gen_prompt.index(_SENT_U2)
            u2e_gp = u2s_gp + len(_SENT_U2)
            asst_open = gen_prompt[u2e_gp:]
            if asst_open:
                if user_to_asst.endswith(asst_open):
                    return user_to_asst[:-len(asst_open)], asst_open
                # A completed-message template may append an empty reasoning
                # channel after the generation prefix. Do not carry that
                # completed suffix into an open assistant-generation prompt.
                marker = user_to_asst.rfind(asst_open)
                if marker >= 0:
                    return user_to_asst[:marker], asst_open
        except Exception:
            pass
        # Can't split, treat everything as the separator
        return "", user_to_asst

    def _extract_single_turn_fallback(self) -> ChatSegments:
        """Minimal fallback: single-turn extraction."""
        tok = self.tokenizer
        msgs = [{"role": "user", "content": _SENT_U1}]
        try:
            full = self._apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            u1s = full.index(_SENT_U1)
            u1e = u1s + len(_SENT_U1)
            global_prefix = full[:u1s]
            asst_open = full[u1e:]
            return ChatSegments(
                global_prefix=global_prefix,
                user_to_asst=asst_open,
                asst_tail="",
                turn_separator=asst_open,
                user_close="",
                asst_open=asst_open,
            )
        except Exception:
            logging.warning("Could not extract chat template. Using empty tags.")
            return ChatSegments(
                global_prefix="", user_to_asst="\n",
                asst_tail="", turn_separator="\n",
            )


# ---------------------------------------------------------------------------
# 2. Image Tokenizer
# ---------------------------------------------------------------------------

class ImageTokenizer(nn.Module):
    """
    Convert images to sequences of LLM-dimension embedding vectors.

    Parameters
    ----------
    patch_size : int
        Height/width of each patch. Image width/height must be divisible by this.
    img_size : int
        Assumed square image size (default 32 for CIFAR-10).
    mode : 'floats' | 'byte_embeds'
        floats     - flatten patch pixels (float 0-1) -> reduce -> llm_dim
        byte_embeds- each pixel channel (0-255 int) looked up in LLM's vocab
                     embedding table, treat patch as bag of vectors -> reduce -> llm_dim
    reduction : 'sum' | 'avg' | 'random_proj'
        How to reduce a patch's bag of vectors to a single llm_dim vector.
        random_proj uses a learnable Linear layer.
        sum / avg do element-wise pooling; for 'floats' mode a fixed (frozen)
        per-position linear maps scalars to llm_dim first.
    llm_dim : int
        Target embedding dimension (must match LLM hidden size).
    vocab_embeds : Optional[Tensor]
        (vocab_size, llm_dim) embedding matrix, required for byte_embeds mode.
        Pass the LLM's input embedding weights (detached).
    """

    def __init__(
        self,
        patch_size: int = 16,
        img_size: int = 32,
        mode: str = "floats",
        reduction: str = "random_proj",
        llm_dim: int = 512,
        vocab_embeds: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        assert img_size % patch_size == 0, \
            f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"
        assert mode in ("floats", "byte_embeds"), f"Unknown mode: {mode}"
        assert reduction in ("sum", "avg", "random_proj"), f"Unknown reduction: {reduction}"

        self.patch_size = patch_size
        self.img_size   = img_size
        self.mode       = mode
        self.reduction  = reduction
        self.llm_dim    = llm_dim
        self.grid        = img_size // patch_size               # patches per row/col
        self.num_patches = self.grid * self.grid
        self.patch_channels = patch_size * patch_size * 3      # floats per patch (3 channels)

        # -- byte_embeds: register byte lookup table (256 entries) --
        if mode == "byte_embeds":
            if vocab_embeds is None:
                raise ValueError("byte_embeds mode requires vocab_embeds")
            # Byte table: embeddings for token IDs 0-255
            # shape: (256, llm_dim) -- detached, not trained
            byte_table = vocab_embeds[:256].detach().clone().float()
            self.register_buffer("byte_table", byte_table)   # (256, llm_dim)
        else:
            self.byte_table = None

        # -- Projection / pooling weights --
        if reduction == "random_proj":
            if mode == "floats":
                # Simple linear: patch_channels floats -> llm_dim
                self.proj = nn.Linear(self.patch_channels, llm_dim)
            else:
                # byte_embeds: learnable scalar weight per patch position,
                # producing a weighted sum of the byte embed vectors.
                # proj.weight: (llm_dim, patch_channels), proj.bias: None
                self.proj = nn.Linear(self.patch_channels, llm_dim, bias=False)
        else:
            # sum / avg
            if mode == "floats":
                # Fixed (frozen) per-scalar linear: each of patch_channels float
                # values gets mapped to llm_dim via a shared random basis.
                # weight: (patch_channels, llm_dim) -- frozen after init
                weight = torch.randn(self.patch_channels, llm_dim) * (llm_dim ** -0.5)
                self.register_buffer("fixed_weight", weight)
            self.proj = None

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, H, W) float tensor in [0, 1]
        Returns:
            (B, num_patches, llm_dim) embedding tensor
        """
        B = images.shape[0]

        # -- Patchify: (B, 3, H, W) -> (B, num_patches, patch_channels) --
        # Layout: scan patches row-major, flatten (C, ph, pw) per patch
        patches = (
            images
            .unfold(2, self.patch_size, self.patch_size)   # (B, 3, grid, pw)
            .unfold(3, self.patch_size, self.patch_size)   # (B, 3, grid, grid, ph, pw)
            .contiguous()
            .view(B, 3, self.grid, self.grid, -1)          # (B, 3, g, g, ph*pw)
            .permute(0, 2, 3, 1, 4)                        # (B, g, g, 3, ph*pw)
            .contiguous()
            .view(B, self.num_patches, self.patch_channels) # (B, N, patch_channels)
        )

        if self.mode == "byte_embeds":
            return self._forward_byte(patches, B)
        else:
            return self._forward_floats(patches, B)

    def _forward_floats(self, patches: torch.Tensor, B: int) -> torch.Tensor:
        """floats mode: (B, N, patch_channels) -> (B, N, llm_dim)"""
        if self.reduction == "random_proj":
            return self.proj(patches)   # learnable linear

        # sum / avg: use fixed_weight matrix (patch_channels, llm_dim)
        # out[b, n, :] = patches[b, n, :] @ fixed_weight
        out = patches @ self.fixed_weight               # (B, N, llm_dim)
        if self.reduction == "avg":
            out = out / self.patch_channels
        return out

    def _forward_byte(self, patches: torch.Tensor, B: int) -> torch.Tensor:
        """
        byte_embeds mode: each pixel channel value -> vocab embedding vector.

        patches: (B, N, patch_channels) float in [0, 1]
        """
        # Quantize to 0-255
        byte_vals = (patches * 255.0).long().clamp(0, 255)  # (B, N, patch_channels)

        # Lookup byte table: (B, N, patch_channels, llm_dim)
        vecs = self.byte_table[byte_vals]

        if self.reduction == "sum":
            return vecs.sum(dim=2)                       # (B, N, llm_dim)

        elif self.reduction == "avg":
            return vecs.mean(dim=2)                      # (B, N, llm_dim)

        elif self.reduction == "random_proj":
            # proj.weight: (llm_dim, patch_channels)
            # vecs:        (B, N, patch_channels, llm_dim)
            # out[b,n,d] = sum_c(proj.weight[d, c] * vecs[b,n,c,d])
            w = self.proj.weight    # (llm_dim, patch_channels)
            out = torch.einsum("dc, bncd -> bnd", w, vecs)  # (B, N, llm_dim)
            return out

    def extra_repr(self):
        return (f"patch_size={self.patch_size}, grid={self.grid}x{self.grid}, "
                f"mode={self.mode}, reduction={self.reduction}, llm_dim={self.llm_dim}")


# ---------------------------------------------------------------------------
# 3. LLM Wrapper
# ---------------------------------------------------------------------------

class LLMWrapper:
    """
    Thin wrapper around any HuggingFace causal LLM.

    Responsibilities
    ----------------
    - Load model + tokenizer from any HF model name
    - Provide embed_text(): text -> (1, seq_len, hidden_dim) embedding
    - Provide prefill(): (1, T, hidden_dim) embeds -> logits, past_kv
      Optionally returns attention maps (output_attentions=True)
    - Cache dtype and embedding layer references
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        attn_implementation: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logging.info(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        logging.info(f"Loading model: {model_name} -> {self.device}")
        load_kwargs: Dict = {}
        if attn_implementation is not None:
            load_kwargs["attn_implementation"] = attn_implementation
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype
        # load_kwargs["attn_implementation"] = "eager"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, **load_kwargs
        ).to(self.device)
        self.model.eval()

        self.hidden_dim: int = self.model.config.hidden_size
        self.model_dtype: torch.dtype = next(self.model.parameters()).dtype
        self.attn_impl = attn_implementation

        logging.info(f"  hidden_dim={self.hidden_dim}, dtype={self.model_dtype}, "
                     f"attn={attn_implementation or 'default'}")

    # -- Embedding helpers --

    def get_vocab_embeds(self) -> torch.Tensor:
        """Return the full (vocab_size, hidden_dim) embedding weight (detached)."""
        return self.model.get_input_embeddings().weight.detach()

    @torch.no_grad()
    def embed_text(self, text: str, add_special_tokens: bool = True) -> torch.Tensor:
        """
        Tokenize text and embed it.

        Returns:
            (1, seq_len, hidden_dim) float tensor on model's device.
            Returns empty tensor (1, 0, hidden_dim) for empty string.
        """
        if not text:
            return torch.zeros(1, 0, self.hidden_dim,
                               device=self.device, dtype=self.model_dtype)
        ids = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
        ).input_ids.to(self.device)
        emb = self.model.get_input_embeddings()(ids)  # (1, T, D)
        return emb

    @torch.no_grad()
    def prefill(
        self,
        embeds: torch.Tensor,
        past_kv=None,
        use_cache: bool = True,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[object], Optional[Tuple]]:
        """
        Single forward pass (prefill step).

        Args:
            embeds: (1, T, hidden_dim) input embeddings (any dtype; auto-cast)
            past_kv: HF past_key_values cache object or None
            use_cache: whether to return updated KV cache
            output_attentions: whether to return per-layer attention maps.
                NOTE: incompatible with flash_attention_2; will raise or silently
                return None attentions if unsupported.

        Returns:
            (logits, past_kv, attentions)
            - logits: (1, T, vocab_size)
            - past_kv: updated KV cache (or None)
            - attentions: tuple of (1, heads, T, T) per layer, or None
        """
        embeds = embeds.to(device=self.device, dtype=self.model_dtype)
        try:
            out = self.model(
                inputs_embeds=embeds,
                past_key_values=past_kv,
                use_cache=use_cache,
                output_attentions=output_attentions,
                return_dict=True,
            )
        except Exception as e:
            if output_attentions:
                # Flash Attention doesn't support returning attention maps
                logging.warning(f"output_attentions=True failed ({e}); retrying without.")
                out = self.model(
                    inputs_embeds=embeds,
                    past_key_values=past_kv,
                    use_cache=use_cache,
                    output_attentions=False,
                    return_dict=True,
                )
                return out.logits, out.past_key_values if use_cache else None, None
            raise

        attentions = out.attentions if output_attentions else None
        return (
            out.logits,
            out.past_key_values if use_cache else None,
            attentions,
        )


# ---------------------------------------------------------------------------
# 4. KV Cache Utilities
# ---------------------------------------------------------------------------

def kv_seq_len(past_kv) -> int:
    """Return the current sequence length stored in a KV cache object."""
    if past_kv is None:
        return 0
    # DynamicCache (transformers >= 4.38)
    if hasattr(past_kv, "get_seq_length"):
        return past_kv.get_seq_length()
    # Tuple-of-tuples (older transformers)
    if isinstance(past_kv, (tuple, list)) and len(past_kv) > 0:
        layer0 = past_kv[0]
        if isinstance(layer0, (tuple, list)) and len(layer0) >= 2:
            return layer0[0].shape[2]   # (batch, heads, seq, head_dim)
    return 0


def kv_select(past_kv, indices: torch.Tensor):
    """
    Select specific sequence positions from a KV cache.

    Args:
        past_kv: HF past_key_values (DynamicCache or tuple-of-tuples)
        indices: (K,) long tensor of positions to keep (can be unsorted)

    Returns:
        New KV cache with only the selected positions, same format as input.
    """
    if past_kv is None:
        return None

    indices = indices.to(dtype=torch.long)

    # -- DynamicCache (new API: .key_cache, .value_cache lists) --
    if hasattr(past_kv, "key_cache") and hasattr(past_kv, "value_cache"):
        try:
            from transformers import DynamicCache
            new_cache = DynamicCache()
            for k, v in zip(past_kv.key_cache, past_kv.value_cache):
                # k, v: (batch, heads, seq, head_dim)
                new_cache.key_cache.append(k[:, :, indices, :])
                new_cache.value_cache.append(v[:, :, indices, :])
            return new_cache
        except Exception as e:
            logging.warning(f"kv_select DynamicCache failed: {e}")

    # -- Legacy DynamicCache (has .layers attribute) --
    if hasattr(past_kv, "layers"):
        for layer in past_kv.layers:
            if hasattr(layer, "keys"):
                layer.keys   = layer.keys[:, :, indices, :]
                layer.values = layer.values[:, :, indices, :]
        return past_kv

    # -- Tuple-of-tuples --
    if isinstance(past_kv, (tuple, list)):
        new_layers = []
        for item in past_kv:
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                k, v = item[0], item[1]
                new_layers.append((k[:, :, indices, :], v[:, :, indices, :]))
            else:
                new_layers.append(item)
        return tuple(new_layers)

    logging.warning(f"kv_select: unsupported KV type {type(past_kv)}, returning as-is.")
    return past_kv


def kv_remove_range(past_kv, start: int, end: int):
    """
    Remove positions [start, end) from a KV cache (keep the rest).
    Convenience wrapper around kv_select.
    """
    if past_kv is None:
        return None
    seq = kv_seq_len(past_kv)
    if start >= seq:
        return past_kv
    end = min(end, seq)
    keep = torch.cat([
        torch.arange(0, start),
        torch.arange(end, seq),
    ])
    return kv_select(past_kv, keep)


# ---------------------------------------------------------------------------
# 5. KV Cache Compression Policies
# ---------------------------------------------------------------------------

class KVCacheCompressor(ABC):
    """
    Base class for KV cache compression policies.

    Subclasses implement `compress()` which takes the current full KV cache
    (and optionally attention maps from the last prefill) and returns a
    (possibly smaller) KV cache.

    The KV cache passed in reflects the FULL context AFTER the latest prefill,
    including positions we might want to prune.
    """

    @abstractmethod
    def compress(
        self,
        past_kv,
        attention_maps: Optional[Tuple] = None,
        data_range: Optional[Tuple[int, int]] = None,
    ):
        """
        Compress the KV cache.

        Args:
            past_kv: Full KV cache after prefill.
            attention_maps: Optional tuple of per-layer attention tensors
                (1, num_heads, seq_len, seq_len). Used by AttentionQuery policy.
            data_range: Optional (start, end) positions within the full sequence
                that contain the data we want to potentially prune from.
                If None, the whole cache is eligible.

        Returns:
            Compressed past_kv (same format, fewer sequence positions).
        """
        ...

    def __repr__(self):
        return self.__class__.__name__


class NoopCompressor(KVCacheCompressor):
    """Identity compressor: keeps the entire KV cache unchanged."""

    def compress(self, past_kv, attention_maps=None, data_range=None):
        return past_kv


class StreamingLLMCompressor(KVCacheCompressor):
    """
    StreamingLLM-style KV cache: keep the first `num_sink` positions (attention
    sinks) and the last `keep_last` positions (recency).

    When a new batch is prefilled and the cache exceeds the budget, positions
    in the middle (after sinks, before the recent window) are dropped.

    Reference: Xiao et al., "Efficient Streaming Language Models with
    Attention Sinks" (2023).
    """

    def __init__(self, num_sink: int = 4, keep_last: int = 64):
        self.num_sink  = num_sink
        self.keep_last = keep_last

    def compress(self, past_kv, attention_maps=None, data_range=None):
        seq = kv_seq_len(past_kv)
        budget = self.num_sink + self.keep_last

        if seq <= budget:
            return past_kv  # nothing to do

        sink_indices = torch.arange(0, self.num_sink)
        last_indices = torch.arange(seq - self.keep_last, seq)
        keep = torch.unique(torch.cat([sink_indices, last_indices]))

        return kv_select(past_kv, keep)

    def __repr__(self):
        return f"StreamingLLMCompressor(sink={self.num_sink}, last={self.keep_last})"


class AttentionQueryCompressor(KVCacheCompressor):
    """
    Attention-based KV compression: keep positions with the highest cumulative
    attention score from a designated set of "attractor" query positions.

    The attractor queries are tokens whose attention back over the data we
    want to use to identify which data positions were most useful.

    Parameters
    ----------
    top_k : int
        Number of positions to keep (not counting always-kept sinks/attractors).
    attractor_range : Tuple[int, int]
        (start, end) positions within the FULL sequence that act as the
        query/attractor. Negative indices supported (e.g. (-5, -1) = last 5).
        end is INCLUSIVE for negative indexing convenience.
    neighbor_window : Tuple[int, int]
        (before, after) extra neighbors to include for each selected position.
        E.g. (1, 1) keeps pos-1 and pos+1 alongside each top-k position.
    agg : str
        How to aggregate attention across heads and layers: 'mean' or 'max'.
    num_sink : int
        Always-keep first N positions (attention sinks, like in StreamingLLM).
    """

    def __init__(
        self,
        top_k: int = 128,
        attractor_range: Tuple[int, int] = (-10, -1),
        neighbor_window: Tuple[int, int] = (0, 0),
        agg: str = "mean",
        num_sink: int = 0,
        attractor_round: str = "assistant",
        attractor_temporary: bool = False,
    ):
        self.top_k               = top_k
        self.attractor_range     = attractor_range
        self.neighbor_window     = neighbor_window
        self.agg                 = agg
        self.num_sink            = num_sink
        self.attractor_round     = attractor_round
        self.attractor_temporary = attractor_temporary

    def _resolve_range(self, start: int, end: int, seq_len: int) -> Tuple[int, int]:
        """Resolve possibly-negative attractor range to positive [start, end) indices."""
        s = start if start >= 0 else seq_len + start
        # end is inclusive for user convenience when negative
        e = (end if end >= 0 else seq_len + end) + 1
        s = max(0, min(s, seq_len))
        e = max(s, min(e, seq_len))
        return s, e

    def compress(
        self,
        past_kv,
        attention_maps: Optional[Tuple] = None,
        data_range: Optional[Tuple[int, int]] = None,
    ):
        if attention_maps is None:
            logging.warning(
                "AttentionQueryCompressor: no attention maps provided "
                "(Flash Attention returns None). Falling back to Noop."
            )
            return past_kv

        seq = kv_seq_len(past_kv)

        # -- Build scores: (seq_len,) mean attention from attractor to each position --
        scores = self._score_positions(attention_maps, seq)

        # -- Attractor range: these positions are always kept --
        a_start, a_end = self._resolve_range(
            self.attractor_range[0], self.attractor_range[1], seq
        )

        # -- Data range: positions eligible for top-k selection --
        d_start, d_end = data_range if data_range is not None else (0, seq)

        # Mask: only score positions inside data_range, excluding attractor range and sinks
        eligible = torch.zeros(seq, dtype=torch.bool)
        eligible[d_start:d_end] = True
        eligible[a_start:a_end] = False
        eligible[:self.num_sink] = False

        eligible_scores = scores.clone()
        eligible_scores[~eligible] = -float("inf")

        # Top-k among eligible
        actual_k = min(self.top_k, int(eligible.sum().item()))
        if actual_k > 0:
            topk_vals, topk_idx = torch.topk(eligible_scores, actual_k)
            selected = topk_idx[topk_vals > -float("inf")]
        else:
            selected = torch.tensor([], dtype=torch.long)

        # Expand with neighbor window
        if self.neighbor_window != (0, 0):
            before, after = self.neighbor_window
            expanded = set(selected.tolist())
            for pos in selected.tolist():
                for nb in range(pos - before, pos + after + 1):
                    if 0 <= nb < seq:
                        expanded.add(nb)
            selected = torch.tensor(sorted(expanded), dtype=torch.long)

        # Always keep: sinks + attractor range
        always_keep = set(range(self.num_sink))
        always_keep.update(range(a_start, a_end))
        always_tensor = torch.tensor(sorted(always_keep), dtype=torch.long)

        keep = torch.unique(torch.cat([always_tensor, selected])).sort().values

        return kv_select(past_kv, keep)

    def _score_from_query_attn(
        self,
        attn_maps: Tuple,
        pre_len: int,
        q_start: int,
        q_end: int,
    ) -> torch.Tensor:
        """
        Aggregate attention from the query rows [q_start, q_end) back to the pre_len keys.
        """
        layer_scores = []
        for attn in attn_maps:
            if attn is None:
                continue
            # attn: (1, num_heads, seq_q, seq_k)
            # seq_q = query_len, seq_k = pre_len + query_len
            if self.agg == "mean":
                attn_avg = attn[0].mean(dim=0)        # (seq_q, seq_k)
            else:
                attn_avg = attn[0].max(dim=0).values  # (seq_q, seq_k)

            q_start_c = max(0, min(q_start, attn_avg.shape[0]))
            q_end_c = max(q_start_c, min(q_end, attn_avg.shape[0]))

            if q_end_c > q_start_c:
                query_rows = attn_avg[q_start_c:q_end_c, :pre_len]  # (Q, pre_len)
                score = query_rows.mean(dim=0)                       # (pre_len,)
            else:
                score = attn_avg[:, :pre_len].mean(dim=0)
            layer_scores.append(score)

        if not layer_scores:
            return torch.zeros(pre_len)

        all_scores = torch.stack(layer_scores, dim=0)   # (num_layers, pre_len)
        result = all_scores.mean(dim=0) if self.agg == "mean" else all_scores.max(dim=0).values
        return result.float().cpu()

    def compress_with_query(
        self,
        past_kv,
        llm: "LLMWrapper",
        builder: "EmbedBuilder",
        class_naming: "ClassNaming",
        data_range: Optional[Tuple[int, int]] = None,
        cfg: "PromptConfig" = None,
        in_turn_attentions: Optional[Tuple] = None,
    ):
        """
        Run attractor query prefill (user query or assistant forced response),
        score prior KV tokens, and prune according to policy.
        """
        seq = kv_seq_len(past_kv)
        if seq == 0:
            return past_kv

        attractor_round = getattr(cfg, "attractor_round", self.attractor_round) if cfg else self.attractor_round
        attractor_temporary = getattr(cfg, "attractor_temporary", self.attractor_temporary) if cfg else self.attractor_temporary

        # Mode A: 'turn_tail' (legacy in-turn attention)
        if attractor_round == "turn_tail":
            return self.compress(past_kv, attention_maps=in_turn_attentions, data_range=data_range)

        # Mode B: separate attractor query prefill ('user' or 'assistant')
        class_names_list = _class_choices_list(class_naming) if class_naming else ""

        if attractor_round == "user":
            prompt_tmpl = getattr(cfg, "attractor_query_user_prompt", cfg.attractor_user_prompt) if cfg else "What class is this image, is it {class_names_list}?"
            user_text = _format_choices_template(
                prompt_tmpl,
                class_naming,
                "Respond with exactly one listed class label and no extra text.",
            )
            if cfg and cfg.verbose:
                _log_prompt_exchange(
                    "KV compression attractor",
                    user_text,
                    "<no assistant turn; user query only>",
                )
            query_emb, positions = builder.build_query_turn(
                user_text=user_text, visual_embeds=None, is_first_turn=False
            )
            q_start, q_end = positions.get("user_text", (0, query_emb.shape[1]))
        elif attractor_round == "assistant":
            user_prompt_tmpl = getattr(cfg, "attractor_user_prompt", "Now you need to consolidate now.") if cfg else "Now you need to consolidate now."
            asst_prompt_tmpl = getattr(cfg, "attractor_asst_prompt", "Understood, I consolidate {class_names_list}.") if cfg else "Understood, I consolidate {class_names_list}."
            user_text = _format_choices_template(
                user_prompt_tmpl,
                class_naming,
                "Keep these as the only allowed class labels.",
            )
            asst_text = _format_choices_template(
                asst_prompt_tmpl,
                class_naming,
                ""#"Use only one of these listed class labels.",
            )
            if cfg and cfg.verbose:
                _log_prompt_exchange(
                    "KV compression attractor",
                    user_text,
                    asst_text,
                )
            query_emb, positions = builder.build_teaching_turn(
                user_text=user_text, visual_embeds=None, assistant_text=asst_text, is_first_turn=False
            )
            q_start, q_end = positions.get("assistant", (0, query_emb.shape[1]))
        else:
            raise ValueError(f"Unknown attractor_round: {attractor_round}")

        pre_len = seq
        logits, query_kv, attn_maps = llm.prefill(
            query_emb, past_kv=past_kv, use_cache=True, output_attentions=True
        )
        post_len = kv_seq_len(query_kv)

        if attn_maps is None:
            logging.warning("AttentionQueryCompressor: attention maps returned None (e.g. Flash Attention).")
            if attractor_temporary:
                return past_kv
            return query_kv

        scores = self._score_from_query_attn(
            attn_maps=attn_maps,
            pre_len=pre_len,
            q_start=q_start,
            q_end=q_end,
        )

        d_start, d_end = data_range if data_range is not None else (0, pre_len)
        eligible = torch.zeros(pre_len, dtype=torch.bool)
        eligible[d_start:d_end] = True
        eligible[:self.num_sink] = False

        eligible_scores = scores.clone()
        eligible_scores[~eligible] = -float("inf")

        actual_k = min(self.top_k, int(eligible.sum().item()))
        if actual_k > 0:
            topk_vals, topk_idx = torch.topk(eligible_scores, actual_k)
            selected = topk_idx[topk_vals > -float("inf")]
        else:
            selected = torch.tensor([], dtype=torch.long)

        # Expand with neighbor window
        if self.neighbor_window != (0, 0):
            before, after = self.neighbor_window
            expanded = set(selected.tolist())
            for pos in selected.tolist():
                for nb in range(pos - before, pos + after + 1):
                    if 0 <= nb < pre_len:
                        expanded.add(nb)
            selected = torch.tensor(sorted(expanded), dtype=torch.long)

        always_keep = set(range(self.num_sink))
        always_tensor = torch.tensor(sorted(always_keep), dtype=torch.long)

        if attractor_temporary:
            # Temporary prefill: query tokens are NOT kept!
            keep = torch.unique(torch.cat([always_tensor, selected])).sort().values
            return kv_select(past_kv, keep)
        else:
            # Appended prefill: keep selected pre-tokens + all query tokens
            query_tokens = torch.arange(pre_len, post_len, dtype=torch.long)
            keep = torch.unique(torch.cat([always_tensor, selected, query_tokens])).sort().values
            return kv_select(query_kv, keep)

    def _score_positions(self, attention_maps: Tuple, seq: int) -> torch.Tensor:
        """
        Aggregate attention maps to a per-position score.

        attention_maps: tuple of (1, num_heads, seq, seq) per layer.
        Returns: (seq,) float tensor, scores for each key position.
        """
        a_start, a_end = self._resolve_range(
            self.attractor_range[0], self.attractor_range[1], seq
        )

        layer_scores = []
        for attn in attention_maps:
            if attn is None:
                continue
            # attn: (1, heads, seq_q, seq_k)
            if self.agg == "mean":
                attn_avg = attn[0].mean(dim=0)        # (seq_q, seq_k)
            else:  # max
                attn_avg = attn[0].max(dim=0).values  # (seq_q, seq_k)

            # Attention FROM attractor positions TO each key position
            if a_end > a_start:
                attn_from = attn_avg[a_start:a_end, :]  # (Q, seq_k)
                score = attn_from.mean(dim=0)            # (seq_k,)
            else:
                score = attn_avg.mean(dim=0)
            layer_scores.append(score)

        if not layer_scores:
            return torch.zeros(seq)

        all_scores = torch.stack(layer_scores, dim=0)   # (num_layers, seq)
        result = all_scores.mean(dim=0) if self.agg == "mean" else all_scores.max(dim=0).values
        return result.float().cpu()

    def __repr__(self):
        return (f"AttentionQueryCompressor(top_k={self.top_k}, "
                f"round={self.attractor_round}, temp={self.attractor_temporary}, "
                f"neighbor={self.neighbor_window})")


# ---------------------------------------------------------------------------
# 6. Embed Builder
# ---------------------------------------------------------------------------

class EmbedBuilder:
    """
    Build text + visual embedding sequences using extracted chat tags.

    All returned tensors are on the LLM's device with the LLM's dtype.
    Shape convention: (1, T, hidden_dim) -- single batch dimension.
    """

    def __init__(self, llm: LLMWrapper, chat: ChatSegments):
        self.llm  = llm
        self.chat = chat

    def _cast(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(device=self.llm.device, dtype=self.llm.model_dtype)

    def build_teaching_turn(
        self,
        user_text: str,
        visual_embeds: Optional[torch.Tensor],  # (1, N_vis, D) or None
        assistant_text: str,
        is_first_turn: bool = True,
        include_assistant_tail: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, Tuple[int, int]]]:
        """
        Build one complete user->assistant turn embedding.

        Layout:
            [turn_prefix] [user_text] [visual_embeds] [user_to_asst] [asst_text] [asst_tail]

        Args:
            user_text: Text part of the user message (can be "").
            visual_embeds: (1, N, D) continuous visual embeddings to inject.
            assistant_text: Text of the assistant response.
            is_first_turn: If True, use global_prefix; else use turn_separator.

        Returns:
            (embeds, positions) where positions maps segment name ->
            (start, end) token indices in the returned sequence.
        """
        prefix = self.chat.global_prefix if is_first_turn else self.chat.turn_separator

        parts: List[torch.Tensor] = []
        positions: Dict[str, Tuple[int, int]] = {}

        def _add(name: Optional[str], x: torch.Tensor):
            x = self._cast(x)
            if x.shape[1] == 0:
                return
            if name:
                start = sum(p.shape[1] for p in parts)
                positions[name] = (start, start + x.shape[1])
            parts.append(x)

        _add("prefix",      self.llm.embed_text(prefix))
        _add("user_text",   self.llm.embed_text(user_text))

        if visual_embeds is not None and visual_embeds.shape[1] > 0:
            _add("visual", visual_embeds)

        _add("user_to_asst", self.llm.embed_text(self.chat.user_to_asst))
        _add("assistant",    self.llm.embed_text(assistant_text))
        if include_assistant_tail:
            _add("asst_tail", self.llm.embed_text(self.chat.asst_tail))

        return torch.cat(parts, dim=1), positions

    def build_query_turn(
        self,
        user_text: str,
        visual_embeds: Optional[torch.Tensor],
        is_first_turn: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Tuple[int, int]]]:
        """
        Build a user query turn WITHOUT an assistant response.
        Ends with the assistant opening tag so the model is primed to respond.

        Returns:
            (embeds, positions)
        """
        prefix = self.chat.global_prefix if is_first_turn else self.chat.turn_separator
        # End with asst_open (generation prompt side); fallback to full user_to_asst
        asst_open = self.chat.asst_open if self.chat.asst_open else self.chat.user_to_asst

        parts: List[torch.Tensor] = []
        positions: Dict[str, Tuple[int, int]] = {}

        def _add(name: Optional[str], x: torch.Tensor):
            x = self._cast(x)
            if x.shape[1] == 0:
                return
            if name:
                start = sum(p.shape[1] for p in parts)
                positions[name] = (start, start + x.shape[1])
            parts.append(x)

        _add("prefix",     self.llm.embed_text(prefix))
        _add("user_text",  self.llm.embed_text(user_text))

        if visual_embeds is not None and visual_embeds.shape[1] > 0:
            _add("visual", visual_embeds)

        _add("user_close", self.llm.embed_text(self.chat.user_close))
        _add("asst_open",  self.llm.embed_text(asst_open))

        return torch.cat(parts, dim=1), positions


# ---------------------------------------------------------------------------
# 7. Class Naming
# ---------------------------------------------------------------------------

class ClassNaming:
    """
    Maps dataset class indices to display names.

    Parameters
    ----------
    classes : list of str
        Original class names (e.g. CIFAR10_CLASSES).
    use_generic : bool
        If True, map to 'class1', 'class2', ... instead of real names.
        Useful to reduce implicit LLM bias toward known categories.
    """

    def __init__(
        self,
        classes: List[str],
        use_generic: bool = False,
        label_mode: str = "direct",
    ):
        aliases = {"mc_letters": "letters", "mc_digits": "digits"}
        label_mode = aliases.get(label_mode, label_mode)
        if use_generic:
            label_mode = "generic"
        if label_mode not in ("direct", "letters", "digits", "generic"):
            raise ValueError(f"Unknown class label mode: {label_mode}")
        if label_mode == "letters" and len(classes) > 26:
            raise ValueError("Letter labels support at most 26 classes")

        self.classes      = classes
        self.label_mode   = label_mode
        self.use_generic  = label_mode == "generic"
        self._id_to_name  = {
            i: (
                cls if label_mode == "direct" else
                chr(ord("A") + i) if label_mode == "letters" else
                str(i + 1) if label_mode == "digits" else
                f"class{i+1}"
            )
            for i, cls in enumerate(classes)
        }
        self._name_to_id = {v: k for k, v in self._id_to_name.items()}

    def name(self, class_id: int) -> str:
        return self._id_to_name[class_id]

    def original(self, class_id: int) -> str:
        return self.classes[class_id]

    def id(self, name: str) -> int:
        return self._name_to_id[name]

    def display_name(self, name_or_id: Union[str, int]) -> str:
        """Return the label and, for encoded modes, its original class name."""
        if isinstance(name_or_id, int):
            cid = name_or_id
            cname = self.name(cid)
        else:
            cname = name_or_id
            cid = self._name_to_id.get(cname, None)

        if self.label_mode != "direct" and cid is not None and 0 <= cid < len(self.classes):
            return f"{cname} ({self.classes[cid]})"
        return cname

    @property
    def names(self) -> List[str]:
        return [self._id_to_name[i] for i in range(len(self.classes))]


def _class_choices_list(class_naming: ClassNaming) -> str:
    """Return the allowed labels, mapping encoded labels to their class names."""
    if class_naming.label_mode == "direct":
        return ", ".join(class_naming.names)
    # return ", ".join(
    #     f"{name}: {class_naming.original(index)}"
    #     for index, name in enumerate(class_naming.names)
    # )
    # if class_naming.use_generic:
    #     # return ", ".join(class_naming.names)
    #     return ", ".join(
    #         f"{name}: {}"
    #         for index, name in enumerate(class_naming.names)
    #     )
    return ", ".join(
        f"{name}: {class_naming.original(index)}"
        for index, name in enumerate(class_naming.names)
    )


def _format_class_prompt(
    template: str,
    class_naming: ClassNaming,
    class_name: Optional[str] = None,
    include_options: bool = False,
    verbose: bool = False,
    context: str = "class prompt",
    rank_k: int = 1,
    sample_ranking: bool = False,
    ranking_instruction: Optional[str] = None,
    sampled_ranking_instruction: Optional[str] = None,
    single_ranking_instruction: Optional[str] = None,
) -> str:
    """Format an image query with explicit choices and ranked-answer rules."""
    if rank_k < 1:
        raise ValueError("rank_k must be at least 1")
    # include_options = False
    class_names_list = (
        _class_choices_list(class_naming)
        if include_options else ", ".join(class_naming.names)
    )
    if include_options:
        if "{class_names_list}" not in template:
            template = f"{template} Options: {{class_names_list}}."
    formatted = template.format(
        class_name=class_name or "",
        class_names_list=class_names_list,
        rank_k=rank_k,
    )
    if rank_k == 1:
        rule_template = single_ranking_instruction or (
            "Return exactly the single most likely listed label."
        )
    elif sample_ranking:
        rule_template = sampled_ranking_instruction or (
            "Sample {rank_k} distinct listed labels without replacement and return "
            "them in sampled order, separated by commas."
        )
    else:
        rule_template = ranking_instruction or (
            "Return the top {rank_k} distinct listed labels in descending likelihood "
            "order, separated by commas."
        )
    ranking_rule = rule_template.format(rank_k=rank_k)
    formatted = f"{formatted} {ranking_rule} Use only the listed labels."
    if verbose:
        logging.info("Formatted prompt text (%s):\n%s", context, formatted or "<empty>")
    return formatted


def _format_choices_template(
    template: str,
    class_naming: ClassNaming,
    required_constraint: str,
    rank_k: int = 1,
) -> str:
    """Format prompt fields and ensure label choices and constraints are present."""
    if "{class_names_list}" not in template:
        template = f"{template} Allowed class labels: {{class_names_list}}."
    if "class_names_list" not in template.lower():
        template = f"{template} {required_constraint}"
    elif not any(
        phrase in template.lower()
        for phrase in ("only valid labels", "exactly one", "only allowed labels")
    ):
        template = f"{template} {required_constraint}"
    return template.format(
        class_names_list=_class_choices_list(class_naming),
        rank_k=rank_k,
    )


def _render_embedded_images(
    user_text: str,
    patch_count: int,
) -> str:
    """Render an image's continuous patch embedding sequence readably."""
    if patch_count <= 0:
        return user_text or "<empty>"
    image_marker = f"<image embeddings: {patch_count} patch vectors>"
    return f"{user_text}\n    {image_marker}" if user_text else image_marker


def _log_prompt_exchange(
    context: str,
    user_prompt: str,
    assistant_prompt: str,
) -> None:
    """Log the human-readable user/assistant content used for one exchange."""
    logging.info(
        "Prompt (%s):\n  User: %s\n  Assistant: %s",
        context,
        user_prompt or "<empty>",
        assistant_prompt or "<empty>",
    )


def _log_query_and_extraction(
    context: str,
    query_text: str,
    patch_count: int,
    chat: ChatSegments,
    is_first_turn: bool,
    answer_mode: str,
    temperature: float,
    scores: Dict[str, float],
    selected: List[str],
) -> None:
    """Log the exact image query and how its assistant answer was extracted."""
    logging.info("Runtime image-query trace (%s):", context)
    prefix = chat.global_prefix if is_first_turn else chat.turn_separator
    logging.info("  Embed flow: prefix=%r", prefix)
    logging.info("  User content:\n%s", _render_embedded_images(query_text, patch_count))
    logging.info(
        "  Chat boundary before answer: user_close=%r assistant_open=%r",
        chat.user_close,
        chat.asst_open or chat.user_to_asst,
    )
    logging.info(
        "  Answer extraction: mode=%s, temperature=%s, candidates=%s",
        answer_mode,
        temperature,
        ", ".join(scores),
    )
    score_kind = (
        "complete-label sequence log-probability"
        if answer_mode == "full_prob"
        else "first-token logit (complete sequence is used automatically on token-prefix collisions)"
    )
    logging.info("  Candidate %s scores:", score_kind)
    for label, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        logging.info("    %-20s %.8g", label, score)
    logging.info("  Selected assistant ranking: %s", ", ".join(selected))


def _log_prompt_catalog(
    cfg: "PromptConfig",
    class_naming: ClassNaming,
    chat: ChatSegments,
    answer_mode: str,
    temperature: float,
    attractor_round: str,
    patch_count: int,
    rank_k: int,
    sample_ranking: bool,
    reasoning_enabled: bool,
    reasoning_max_new_tokens: int,
) -> None:
    """Log all configured prompt templates and representative rendered turns."""
    if not cfg.verbose:
        return

    query = _format_class_prompt(
        cfg.guess_user_tmpl,
        class_naming,
        include_options=(cfg.include_class_options or answer_mode == "full_prob"),
        verbose=True,
        context="startup image-query template",
        rank_k=rank_k,
        sample_ranking=sample_ranking,
        ranking_instruction=cfg.ranking_instruction,
        sampled_ranking_instruction=cfg.sampled_ranking_instruction,
        single_ranking_instruction=cfg.single_ranking_instruction,
    )
    sample_label = class_naming.names[0]
    reasoning_instruction = (
        cfg.reasoning_prompt
        if reasoning_enabled else ""
    )
    feedback_log_template = (
        (cfg.feedback_user_prompt or "").strip()
        or (cfg.feedback_user_tmpl or "").strip()
        or "Ground-truth label: {class_name}. Ranked prediction: {ranked_labels}. "
           "{rank_feedback} {reasoning_instruction}"
    )
    feedback_user_correct = feedback_log_template.format(
        class_name=sample_label,
        ranked_labels=sample_label,
        true_rank=1,
        rank_k=rank_k,
        rank_feedback=cfg.rank_feedback_top1,
        reasoning_instruction=reasoning_instruction,
    )
    feedback_user_wrong = feedback_log_template.format(
        class_name=sample_label,
        ranked_labels=f"{class_naming.names[1] if len(class_naming.names) > 1 else sample_label}, {sample_label}",
        true_rank=2,
        rank_k=rank_k,
        rank_feedback=cfg.rank_feedback_in_topk.format(true_rank=2, rank_k=rank_k),
        reasoning_instruction=reasoning_instruction,
    )
    class_names_list = ", ".join(class_naming.names)
    attractor_query = cfg.attractor_query_user_prompt.format(
        class_names_list=class_names_list
    )
    attractor_user = cfg.attractor_user_prompt.format(
        class_names_list=class_names_list
    )
    attractor_asst = cfg.attractor_asst_prompt.format(
        class_names_list=class_names_list
    )

    logging.info("=" * 78)
    logging.info("VERBOSE PROMPT CATALOG — templates and representative rendered turns")
    logging.info("=" * 78)
    logging.info("Class labels: %s", ", ".join(class_naming.names))
    logging.info("Image insertion format: user text, then one image embedding sequence")
    logging.info("  %s", _render_embedded_images(query, patch_count=patch_count))
    logging.info("Answer extraction: mode=%s, rank_k=%d, sample=%s, temperature=%s",
                 answer_mode, rank_k, sample_ranking, temperature)
    logging.info("AR reasoning: enabled=%s, max_new_tokens=%d, top_p=%s",
                 reasoning_enabled, reasoning_max_new_tokens, "configured via CLI")
    logging.info("  Candidate labels: %s", ", ".join(class_naming.names))
    logging.info("  At temperature 0, select the maximum candidate score; otherwise sample from candidate scores.")
    logging.info("Chat-template boundaries used by the embed builder:")
    logging.info("  global prefix: %r", chat.global_prefix)
    logging.info("  user close:    %r", chat.user_close)
    logging.info("  assistant open:%r", chat.asst_open)
    logging.info("  user->assistant fallback: %r", chat.user_to_asst)
    logging.info("  assistant tail:%r", chat.asst_tail)
    logging.info("  next-turn separator: %r", chat.turn_separator)

    _log_prompt_exchange(
        "configured instruction turn",
        cfg.instruction_user_prompt,
        cfg.instruction_asst_prompt,
    )
    _log_prompt_exchange(
        "configured image query template (used in teaching and test)",
        _render_embedded_images(query, patch_count=patch_count),
        "<sampled from candidate labels: " + ", ".join(class_naming.names) + ">",
    )
    _log_prompt_exchange(
        "feedback user round (true label + ranking status)",
        feedback_user_correct,
        "<assistant generation starts here>" if reasoning_enabled else cfg.feedback_acknowledgement,
    )
    _log_prompt_exchange(
        "feedback user round (true label + ranking status; non-top-1 example)",
        feedback_user_wrong,
        "<assistant generation starts here>" if reasoning_enabled else cfg.feedback_acknowledgement,
    )
    _log_prompt_exchange(
        "configured attractor query prompt (user round)",
        attractor_query,
        "<no assistant turn; user query only>",
    )
    _log_prompt_exchange(
        "configured attractor prompt (assistant round)",
        attractor_user,
        attractor_asst,
    )
    logging.info("Active KV attractor mode: %s", attractor_round)
    logging.info("Instruction turn: %s", "enabled" if cfg.use_instruction_turn else "disabled")
    if attractor_round == "turn_tail":
        logging.info("Separate attractor user/assistant prompts are configured but inactive in turn_tail mode.")
    logging.info("Raw template formats:")
    for template_name, template_value in (
        ("instruction_user_prompt", cfg.instruction_user_prompt),
        ("instruction_asst_prompt", cfg.instruction_asst_prompt),
        ("guess_user_tmpl", cfg.guess_user_tmpl),
        ("ranking_instruction", cfg.ranking_instruction),
        ("sampled_ranking_instruction", cfg.sampled_ranking_instruction),
        ("single_ranking_instruction", cfg.single_ranking_instruction),
        ("reasoning_prompt", cfg.reasoning_prompt),
        ("feedback_user_prompt", cfg.feedback_user_prompt),
        ("feedback_acknowledgement", cfg.feedback_acknowledgement),
        ("rank_feedback_top1", cfg.rank_feedback_top1),
        ("rank_feedback_in_topk", cfg.rank_feedback_in_topk),
        ("rank_feedback_missed", cfg.rank_feedback_missed),
        ("attractor_query_user_prompt", cfg.attractor_query_user_prompt),
        ("attractor_user_prompt", cfg.attractor_user_prompt),
        ("attractor_asst_prompt", cfg.attractor_asst_prompt),
    ):
        logging.info("  %s = %r", template_name, template_value)
    logging.info("=" * 78)


from dataclasses import dataclass


@dataclass
class PromptConfig:
    """
    Text templates and conversation settings.

    Template variables:
        {class_name}, {class_names_list}, {rank_k}, {ranked_labels},
        {true_rank}, {rank_feedback}, {reasoning_instruction}

    Reasoning generation length is controlled separately by the runtime
    argument --reasoning-max-new-tokens. No word-budget setting belongs here.
    """

    # Initial instruction turn
    instruction_user_prompt: str = ""
    instruction_asst_prompt: str = ""
    use_instruction_turn: bool = True

    # Attractor query settings
    attractor_round: str = "assistant"
    attractor_temporary: bool = False
    attractor_user_prompt: str = ""
    attractor_asst_prompt: str = ""
    attractor_query_user_prompt: str = ""

    # Prediction turn
    guess_user_tmpl: str = ""
    prediction_asst_tmpl: str = ""

    # Ranking instructions
    ranking_instruction: str = ""
    sampled_ranking_instruction: str = ""
    single_ranking_instruction: str = ""

    # Feedback turn
    feedback_user_prompt: str = ""
    feedback_user_tmpl: str = ""
    feedback_asst_tmpl: str = ""
    feedback_acknowledgement: str = ""

    # Rank feedback
    rank_feedback_top1: str = ""
    rank_feedback_in_topk: str = ""
    rank_feedback_missed: str = ""

    # Visible, user-facing explanation prompt
    reasoning_prompt: str = ""
    reasoning_enabled: bool = False
    reasoning_only_on_error: bool = False

    # Ranking configuration
    rank_k: int = 3
    sample_ranking: bool = False

    # Other options
    include_class_options: bool = False
    verbose: bool = False


# =============================================================
# 1. Default configuration: original long prompt
# =============================================================

long_prompt_config = PromptConfig(
    instruction_user_prompt=(
        "You are a large language model. You were never trained on images directly, "
        "but in this session you will perform in-context learning analogous to deep neural network "
        "SGD classification training, assisted by a KV cache compression algorithm.\n"
        "Because earlier KV cache entries are dynamically removed, your self-attention at recent "
        "timesteps may not attend back to all earlier inputs.\n\n"
        "You will see one image's patch vectors at a time. The valid class labels "
        "are {class_names_list}. For each image, return up to {rank_k} distinct "
        "labels, separated by commas (for example: cat, dog, frog), following "
        "the ranking rule specified for this experiment. Use only the listed "
        "labels; do not add explanations to the ranked answer. The user will "
        "reveal the correct class; use that feedback to learn. After the assistant "
        "gives correctness feedback, if reasoning mode is enabled, provide one "
        "concise, user-facing visual explanation. Do not repeat the ranked labels "
        "or invent visual details.\n\n"
        "At the end of training, there will be test images unseen during in-context learning."
    ),
    instruction_asst_prompt="Understood.",
    use_instruction_turn=True,

    # Attractor
    attractor_round="assistant",
    attractor_temporary=False,
    attractor_user_prompt=(
        "Consolidate prior examples for these classes: {class_names_list}. "
        "Keep these as the only allowed class labels."
    ),
    attractor_asst_prompt=(
        "Understood. I will consolidate using only these class labels: {class_names_list}."
    ),
    attractor_query_user_prompt=(
        "What class is this image? Choose exactly one label from: {class_names_list}. "
        "Answer with only that label."
    ),

    # Prediction
    guess_user_tmpl="What class is this image?",
    ranking_instruction=(
        "Return the top {rank_k} distinct listed labels in descending likelihood "
        "order, separated by commas."
    ),
    sampled_ranking_instruction=(
        "Sample {rank_k} distinct listed labels without replacement and return "
        "them in sampled order, separated by commas."
    ),
    single_ranking_instruction=(
        "Return exactly the single most likely listed label."
    ),

    # Visible reasoning
    # reasoning_prompt=(
    #     "Briefly explain the visual evidence supporting the true class and the "
    #     "ranking outcome. Do not repeat the ranked labels or invent visual details."
    # ),
    reasoning_prompt=(
            "Give full reasoning why you are correct or wrong."
    ),
    reasoning_enabled=False,
    reasoning_only_on_error=False,

    # Feedback
    rank_feedback_top1="The true class is ranked first.",
    rank_feedback_in_topk=(
        "The true class is ranked {true_rank} of {rank_k}."
    ),
    rank_feedback_missed=(
        "The true class is outside the top {rank_k}."
    ),
    feedback_user_prompt=(
        "Ground-truth label: {class_name}. Ranked prediction: {ranked_labels}. "
        "{rank_feedback} {reasoning_instruction}"
    ),
    feedback_acknowledgement="Feedback received.",

    # Other
    include_class_options=False,
    verbose=False,
)


long_prompt_config2 = PromptConfig(
    instruction_user_prompt=(
        "You are a large language model. In this session, you will perform "
        "in-context learning analogous to deep neural network SGD classification "
        "training, assisted by a KV cache compression algorithm.\n"
        "Because earlier KV cache entries are dynamically removed, your self-attention "
        "at recent timesteps may not attend back to all earlier inputs.\n\n"
        "You will be given a features of an object. The valid class labels are {class_names_list}. For each example, "
        "return up to {rank_k} distinct labels, separated by commas, following "
        "the ranking rule specified for this experiment. Use only the listed "
        "labels and do not add explanations to the ranked answer. The user will "
        "reveal the correct class; use that feedback to learn. After giving "
        "correctness feedback, if reasoning mode is enabled, provide one concise, "
        "user-facing explanation. Do not repeat the ranked labels or invent details.\n\n"
        "At the end of training, there will be test examples unseen during "
        "in-context learning."
    ),
    instruction_asst_prompt="Understood.",
    use_instruction_turn=True,

    # Attractor
    attractor_round="assistant",
    attractor_temporary=False,
    attractor_user_prompt=(
        "Consolidate prior examples for these classes: {class_names_list}. "
        "Keep these as the only allowed class labels."
    ),
    attractor_asst_prompt=(
        "Understood. I will consolidate using only these class labels: {class_names_list}."
    ),
    attractor_query_user_prompt=(
        "What class is this example? Choose exactly one label from: {class_names_list}. "
        "Answer with only that label."
    ),

    # Prediction
    guess_user_tmpl="Classify this example.",
    ranking_instruction=(
        "Return the top {rank_k} distinct listed labels in descending likelihood "
        "order, separated by commas."
    ),
    sampled_ranking_instruction=(
        "Sample {rank_k} distinct listed labels without replacement and return "
        "them in sampled order, separated by commas."
    ),
    single_ranking_instruction="Return exactly the single most likely listed label.",

    # Visible reasoning
    reasoning_prompt=(
        "Briefly explain the evidence supporting the true class and the ranking "
        "outcome. Do not repeat the ranked labels or invent details."
    ),
    reasoning_enabled=False,
    reasoning_only_on_error=False,

    # Feedback
    rank_feedback_top1="The true class is ranked first.",
    rank_feedback_in_topk="The true class is ranked {true_rank} of {rank_k}.",
    rank_feedback_missed="The true class is outside the top {rank_k}.",
    feedback_user_prompt=(
        "Ground-truth label: {class_name}. Ranked prediction: {ranked_labels}. "
        "{rank_feedback} {reasoning_instruction}"
    ),
    feedback_acknowledgement="Feedback received.",

    # Other
    include_class_options=False,
    verbose=False,
)


prompt_cfg1 = PromptConfig(
    instruction_user_prompt=(
        "Rank the {rank_k} most likely classes for each features: "
        "{class_names_list}. Return labels only, comma-separated. "
        "After each prediction, I will provide the true label. Then state "
        "whether your prediction was correct and briefly around 30 words explain and update your description relevant classes. "
        "Remember these rules."
    ),
    instruction_asst_prompt="Understood.",
    use_instruction_turn=True,

    guess_user_tmpl="Classify this features",
    prediction_asst_tmpl="{ranked_labels}",

    # User provides the ground-truth label; the runtime adds the explanation
    # request when autoregressive visible reasoning is enabled.
    feedback_user_tmpl="{class_name}",
    feedback_asst_tmpl="{feedback}",

    ranking_instruction="Return the top {rank_k} labels, ranked by likelihood.",
    sampled_ranking_instruction=(
        "Sample {rank_k} distinct labels, ordered by likelihood."
    ),
    single_ranking_instruction="Return only the most likely label",

    rank_feedback_top1="The true label was ranked first.",
    rank_feedback_in_topk="The true label was ranked {true_rank}.",
    rank_feedback_missed="I was wrong because",

    reasoning_prompt="Revise and explain your memory on all classes different from previous turn",
    reasoning_enabled=True,
    reasoning_only_on_error=False,
)


prompt_cfg2 = PromptConfig(
    instruction_user_prompt=(
            "You are a large language model. In this session, you will perform "
            "in-context learning analogous to deep neural network SGD classification "
            "training, assisted by a KV cache compression algorithm.\n"
            "Because earlier KV cache entries are dynamically removed, your self-attention "
            "at recent timesteps may not attend back to all earlier inputs.\n\n"
            "You will be given a foreign-language description of an object as its "
            "features. The valid class labels are {class_names_list}. For each example, "
            "return up to {rank_k} distinct labels, separated by commas, following "
            "the ranking rule specified for this experiment. Use only the listed "
            "labels and do not add explanations to the ranked answer. The user will "
            "reveal the correct class; use that feedback to learn. After giving "
            "correctness feedback, if reasoning mode is enabled, provide one concise, "
            "user-facing explanation. Do not repeat the ranked labels or invent details.\n\n"
            "At the end of training, there will be test examples unseen during "
            "in-context learning."
        ),
    instruction_asst_prompt="Understood.",
    use_instruction_turn=True,

    guess_user_tmpl="Classify this.",
    prediction_asst_tmpl="",

    # User provides only the ground-truth label.
    feedback_user_tmpl="",

    # Used by non-autoregressive feedback paths.
    feedback_asst_tmpl="",

    single_ranking_instruction="Return only the most likely label.",

    rank_feedback_top1="The true label was ranked first.",
    rank_feedback_in_topk="The true label was ranked {true_rank}.",
    rank_feedback_missed="The true label was not in the ranking.",

    reasoning_prompt="Update and explain each classes",
    # reasoning_prompt="Update and explain each classes visual features",
    reasoning_enabled=True,
    reasoning_only_on_error=False,
)


# =============================================================
# 4. Active prompt configuration
# =============================================================

# DEFAULT_PROMPT_CFG = long_prompt_config
# DEFAULT_PROMPT_CFG = long_prompt_config2
DEFAULT_PROMPT_CFG = prompt_cfg1
# DEFAULT_PROMPT_CFG = prompt_cfg2

# ---------------------------------------------------------------------------
# 9. Teaching Functions
# ---------------------------------------------------------------------------

def compress_kv(
    past_kv,
    compressor: KVCacheCompressor,
    llm: LLMWrapper,
    builder: EmbedBuilder,
    class_naming: Optional[ClassNaming] = None,
    data_range: Optional[Tuple[int, int]] = None,
    cfg: PromptConfig = DEFAULT_PROMPT_CFG,
    in_turn_attentions: Optional[Tuple] = None,
):
    """
    Unified entry point for KV compression.
    Handles Noop, StreamingLLM, and AttentionQuery (both in-turn and external query probe).
    """
    if past_kv is None or compressor is None:
        return past_kv

    if isinstance(compressor, NoopCompressor):
        return past_kv

    if isinstance(compressor, StreamingLLMCompressor):
        return compressor.compress(past_kv, data_range=data_range)

    if isinstance(compressor, AttentionQueryCompressor):
        return compressor.compress_with_query(
            past_kv=past_kv,
            llm=llm,
            builder=builder,
            class_naming=class_naming,
            data_range=data_range,
            cfg=cfg,
            in_turn_attentions=in_turn_attentions,
        )

    return compressor.compress(past_kv, data_range=data_range)


@torch.no_grad()
def prefill_instruction(
    llm: LLMWrapper,
    builder: EmbedBuilder,
    past_kv,
    cfg: PromptConfig = DEFAULT_PROMPT_CFG,
    class_naming: Optional[ClassNaming] = None,
    rank_k: int = 3,
    reasoning_enabled: bool = False,
    sample_ranking: bool = False,
) -> Tuple[object, Dict]:
    """
    Prefill the first-round instruction turn (user instruction + assistant response).
    This establishes task context before any image training data is ingested.
    """
    user_text = cfg.instruction_user_prompt
    if class_naming is not None:
        user_text = _format_choices_template(
            user_text,
            class_naming,
            (
                (
                    cfg.sampled_ranking_instruction.format(rank_k=rank_k)
                    if sample_ranking else
                    cfg.ranking_instruction.format(rank_k=rank_k)
                ) + " Use only listed labels and include no explanation in the ranked answer."
            ),
            rank_k=rank_k,
        )
        if reasoning_enabled:
            user_text += " After feedback: " + cfg.reasoning_prompt
    if cfg.verbose:
        _log_prompt_exchange(
            "instruction",
            user_text,
            cfg.instruction_asst_prompt,
        )
    turn_emb, positions = builder.build_teaching_turn(
        user_text=user_text,
        visual_embeds=None,
        assistant_text=cfg.instruction_asst_prompt,
        is_first_turn=(past_kv is None or kv_seq_len(past_kv) == 0),
    )
    prev_len = kv_seq_len(past_kv)
    logits, past_kv, _ = llm.prefill(turn_emb, past_kv=past_kv, use_cache=True)
    full_len = kv_seq_len(past_kv)
    return past_kv, {
        "tokens_added": full_len - prev_len,
    }


@torch.no_grad()
def teach_guess_correct(
    llm: LLMWrapper,
    builder: EmbedBuilder,
    img_tok: ImageTokenizer,
    image: torch.Tensor,       # (C, H, W) single image
    true_class_name: str,
    class_naming: ClassNaming,
    past_kv,
    compressor: Optional[KVCacheCompressor] = None,
    cfg: PromptConfig = DEFAULT_PROMPT_CFG,
    answer_mode: str = "first_token",
    temperature: float = 0.0,
    rank_k: int = 3,
    sample_ranking: bool = False,
    reasoning_enabled: bool = False,
    reasoning_max_new_tokens: int = 32,
    reasoning_temperature: float = 0.7,
    reasoning_top_p: float = 0.9,
    guess_prob_gt: float = 0.0,
    guess_prob_random: float = 0.0,
) -> Tuple[object, bool, List[str]]:
    """
    'guess-correct' mode for a single image.

    Dialogue (all within one accumulated KV):
        [turn 1] user: <image query + visual embeds>
             assistant: <ranked model labels>
        [turn 2] user: <true label + ranking status + explanation request>
             assistant: autoregressively generated explanation (when enabled).

    In reasoning mode, turn 2 ends at the assistant-open chat tag. No assistant
    text is force-prefilled before decoding.

    Guess mode flags:
        guess_prob_gt: legacy probability of replacing the sampled prediction with ground truth
        guess_prob_random: legacy probability of replacing the sampled prediction with a random label

    Returns:
        (past_kv, top1_was_correct, ranked_class_names)
    """
    ranked_names, _, past_kv_q = predict_image(
        llm, builder, img_tok, image, class_naming, past_kv,
        cfg=cfg, answer_mode=answer_mode, temperature=temperature,
        rank_k=rank_k, sample_ranking=sample_ranking,
        keep_query_cache=True,
        prompt_context="teaching guess-correct / prediction",
    )

    # Legacy experiment overrides; both are disabled by default.
    if guess_prob_gt > 0.0 and random.random() < guess_prob_gt:
        ranked_names = [true_class_name] + [
            name for name in ranked_names if name != true_class_name
        ][:max(0, rank_k - 1)]
    elif guess_prob_random > 0.0 and random.random() < guess_prob_random:
        replacement = random.choice(class_naming.names)
        ranked_names = [replacement] + [
            name for name in ranked_names if name != replacement
        ][:max(0, rank_k - 1)]

    if cfg.verbose and (guess_prob_gt > 0.0 or guess_prob_random > 0.0):
        query_text = _format_class_prompt(
            cfg.guess_user_tmpl,
            class_naming,
            include_options=cfg.include_class_options,
            verbose=cfg.verbose,
            context="legacy teaching override query",
            rank_k=rank_k,
            ranking_instruction=cfg.ranking_instruction,
            sampled_ranking_instruction=cfg.sampled_ranking_instruction,
            single_ranking_instruction=cfg.single_ranking_instruction,
        )
        _log_prompt_exchange(
            "legacy teaching guess override",
            _render_embedded_images(query_text, patch_count=img_tok.num_patches),
            ", ".join(ranked_names),
        )

    # Append guessed_name as assistant response
    device = llm.device
    dtype = llm.model_dtype
    pred_emb = llm.embed_text(
        " " + ", ".join(ranked_names) + builder.chat.asst_tail,
        add_special_tokens=False,
    )
    pred_emb = pred_emb.to(device=device, dtype=dtype)
    _, past_kv_q, _ = llm.prefill(pred_emb, past_kv=past_kv_q, use_cache=True)

    # -- Turn 2: user reveals the true label and asks for feedback --
    # The explanation request belongs to the USER message. The constructed turn
    # ends at the assistant-open tag, so autoregressive decoding starts as an
    # actual assistant response rather than continuing a forced assistant text.
    true_rank = ranked_names.index(true_class_name) + 1 if true_class_name in ranked_names else None
    was_correct = ranked_names[0] == true_class_name
    rank_feedback = (
        cfg.rank_feedback_top1
        if true_rank == 1 else
        cfg.rank_feedback_in_topk.format(true_rank=true_rank, rank_k=len(ranked_names))
        if true_rank is not None else
        cfg.rank_feedback_missed.format(rank_k=len(ranked_names), true_rank="outside top-k")
    )
    reasoning_instruction = (
        cfg.reasoning_prompt
        if reasoning_enabled else ""
    )
    feedback_template = (getattr(cfg, "feedback_user_prompt", "") or "").strip()
    if not feedback_template:
        # Support concise PromptConfig variants that use feedback_user_tmpl.
        feedback_template = (getattr(cfg, "feedback_user_tmpl", "") or "").strip()
    if not feedback_template:
        # Never silently append an empty feedback user turn.
        feedback_template = (
            "Ground-truth label: {class_name}. Ranked prediction: {ranked_labels}. "
            "{rank_feedback} {reasoning_instruction}"
        )

    feedback_user_text = feedback_template.format(
        class_name=true_class_name,
        ranked_labels=", ".join(ranked_names),
        true_rank=true_rank if true_rank is not None else "outside top-k",
        rank_k=len(ranked_names),
        rank_feedback=rank_feedback,
        reasoning_instruction=reasoning_instruction,
    ).strip()
    if reasoning_enabled and reasoning_instruction and "{reasoning_instruction}" not in feedback_template:
        feedback_user_text = f"{feedback_user_text} {reasoning_instruction}".strip()
    if not feedback_user_text:
        raise RuntimeError("Feedback prompt resolved to an empty user message")

    if reasoning_enabled:
        # User round -> assistant-open. Do not prefill any assistant answer text.
        fb_emb, _ = builder.build_query_turn(
            user_text=feedback_user_text,
            visual_embeds=None,
            is_first_turn=False,
        )
        feedback_logits, past_kv_q, _ = llm.prefill(
            fb_emb, past_kv=past_kv_q, use_cache=True
        )
        next_logits = feedback_logits[0, -1]
        if cfg.verbose:
            logging.info("Feedback user round: %s", feedback_user_text)
            logging.info(
                "Feedback assistant-generation boundary: user_close=%r, assistant_open=%r",
                builder.chat.user_close,
                builder.chat.asst_open or builder.chat.user_to_asst,
            )
            top_values, top_ids = torch.topk(next_logits.float(), k=min(8, next_logits.numel()))
            top_tokens = [
                (int(token_id), repr(llm.tokenizer.decode([int(token_id)], skip_special_tokens=False)), float(value))
                for token_id, value in zip(top_ids.tolist(), top_values.tolist())
            ]
            logging.info("Next-token candidates at assistant-open: %s", top_tokens)

        reasoning_text, past_kv_q = autoregressive_decode(
            llm,
            past_kv_q,
            next_logits,
            max_new_tokens=reasoning_max_new_tokens,
            temperature=reasoning_temperature,
            top_p=reasoning_top_p,
        )
        # Close the assistant turn only AFTER autoregressive generation.
        tail_emb = llm.embed_text(builder.chat.asst_tail, add_special_tokens=False)
        if tail_emb.shape[1] > 0:
            _, past_kv_q, _ = llm.prefill(
                tail_emb, past_kv=past_kv_q, use_cache=True
            )
        if cfg.verbose:
            logging.info(
                "\n >> Generated assistant reasoning (%d-token limit):\n%s\n <<",
                reasoning_max_new_tokens,
                reasoning_text or "<empty>",
            )
    else:
        # Without AR reasoning, retain a complete user->assistant feedback turn.
        fb_emb, _ = builder.build_teaching_turn(
            user_text=feedback_user_text,
            visual_embeds=None,
            assistant_text=cfg.feedback_acknowledgement,
            is_first_turn=False,
            include_assistant_tail=True,
        )
        _, past_kv_q, _ = llm.prefill(
            fb_emb, past_kv=past_kv_q, use_cache=True
        )
        if cfg.verbose:
            _log_prompt_exchange(
                "teaching feedback (reasoning disabled)",
                feedback_user_text,
                cfg.feedback_acknowledgement,
            )

    if compressor is not None and not isinstance(compressor, NoopCompressor):
        past_kv_q = compress_kv(
            past_kv_q, compressor, llm, builder, class_naming,
            data_range=None, cfg=cfg, in_turn_attentions=None
        )

    return past_kv_q, was_correct, ranked_names


# ---------------------------------------------------------------------------
# 10. Classification
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_image(
    llm: LLMWrapper,
    builder: EmbedBuilder,
    img_tok: ImageTokenizer,
    image: torch.Tensor,       # (C, H, W)
    class_naming: ClassNaming,
    past_kv,
    cfg: PromptConfig = DEFAULT_PROMPT_CFG,
    answer_mode: str = "first_token",
    temperature: float = 0.0,
    rank_k: int = 3,
    sample_ranking: bool = False,
    keep_query_cache: bool = False,
    prompt_context: str = "test/evaluation",
) -> Tuple[List[str], Dict[str, float], object]:
    """
    Build the shared image query, then sample a class from its next-token scores.
    Teaching retains the query cache to append the answer and feedback; evaluation
    discards it after using the identical prompt and sampling path.
    """
    if answer_mode not in ("first_token", "full_prob"):
        raise ValueError(f"Unknown answer_mode: {answer_mode}")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if rank_k < 1:
        raise ValueError("rank_k must be at least 1")
    rank_k = min(rank_k, len(class_naming.names))
    device = llm.device
    dtype  = llm.model_dtype

    patches = img_tok(image.unsqueeze(0).to(device)).to(dtype)  # (1, N, D)

    query_text = _format_class_prompt(
        cfg.guess_user_tmpl, class_naming,
        include_options=True,
        verbose=cfg.verbose,
        context=prompt_context,
        rank_k=min(rank_k, len(class_naming.names)),
        sample_ranking=sample_ranking,
        ranking_instruction=cfg.ranking_instruction,
        sampled_ranking_instruction=cfg.sampled_ranking_instruction,
        single_ranking_instruction=cfg.single_ranking_instruction,
    )
    is_first_turn = past_kv is None or kv_seq_len(past_kv) == 0
    q_emb, _ = builder.build_query_turn(
        user_text=query_text,
        visual_embeds=patches,
        is_first_turn=is_first_turn,
    )

    label_tokens = [_class_token_ids(llm, name) for name in class_naming.names]
    has_shared_prefix = len({tokens[0] for tokens in label_tokens}) < len(label_tokens)
    needs_sequence_scores = answer_mode == "full_prob" or has_shared_prefix
    retain_query_cache = keep_query_cache or needs_sequence_scores

    # Cache is mutable in some Transformers implementations. Avoid mutating the
    # taught context during full-sequence evaluation.
    query_context = past_kv
    if needs_sequence_scores and not keep_query_cache:
        query_context = copy.deepcopy(past_kv)
    logits, query_kv, _ = llm.prefill(
        q_emb, past_kv=query_context, use_cache=retain_query_cache
    )
    next_logits = logits[0, -1, :]
    ranked_names, scores = sample_class(
        llm, next_logits, class_naming, answer_mode=answer_mode,
        temperature=temperature,
        ctx_kv=query_kv,
        rank_k=rank_k,
        sample_ranking=sample_ranking,
    )
    if cfg.verbose:
        _log_query_and_extraction(
            prompt_context,
            query_text,
            patch_count=patches.shape[1],
            chat=builder.chat,
            is_first_turn=is_first_turn,
            answer_mode=answer_mode,
            temperature=temperature,
            scores=scores,
            selected=ranked_names,
        )
    return ranked_names, scores, query_kv


def classify_image(
    llm: LLMWrapper,
    builder: EmbedBuilder,
    img_tok: ImageTokenizer,
    image: torch.Tensor,
    class_naming: ClassNaming,
    past_kv,
    cfg: PromptConfig = DEFAULT_PROMPT_CFG,
    answer_mode: str = "first_token",
    temperature: float = 0.0,
    rank_k: int = 3,
    sample_ranking: bool = False,
) -> Tuple[List[str], Dict[str, float]]:
    """Evaluation wrapper around the same predictor used during teaching."""
    ranked_names, scores, _ = predict_image(
        llm, builder, img_tok, image, class_naming, past_kv,
        cfg=cfg, answer_mode=answer_mode, temperature=temperature,
        rank_k=rank_k, sample_ranking=sample_ranking,
        keep_query_cache=False,
    )
    return ranked_names, scores


def _class_token_ids(llm: LLMWrapper, name: str) -> List[int]:
    """Tokenize a generated label without BOS/EOS, including its leading space."""
    ids = llm.tokenizer(" " + name, add_special_tokens=False).input_ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    if not ids:
        raise ValueError(f"Class label {name!r} produced no tokens")
    return [int(token_id) for token_id in ids]


def sample_class(
    llm: LLMWrapper,
    logits: torch.Tensor,
    class_naming: ClassNaming,
    answer_mode: str = "first_token",
    temperature: float = 0.0,
    ctx_kv=None,
    rank_k: int = 3,
    sample_ranking: bool = False,
) -> Tuple[List[str], Dict[str, float]]:
    """Rank up to ``rank_k`` candidates; optionally sample a distinct ordering."""
    if answer_mode not in ("first_token", "full_prob"):
        raise ValueError(f"Unknown answer_mode: {answer_mode}")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if rank_k < 1:
        raise ValueError("rank_k must be at least 1")
    token_ids = [_class_token_ids(llm, name) for name in class_naming.names]
    first_ids = [ids[0] for ids in token_ids]
    if answer_mode == "full_prob" or len(set(first_ids)) != len(first_ids):
        if ctx_kv is None:
            raise ValueError("Full-label sampling requires the query KV cache")
        ranking_logits = torch.tensor([
            _sequence_log_prob(llm, name, ctx_kv, first_logits=logits)
            for name in class_naming.names
        ], device=logits.device, dtype=torch.float32)
    else:
        ranking_logits = torch.log_softmax(logits.float(), dim=-1)[
            torch.tensor(first_ids, device=logits.device)
        ]

    probabilities = torch.softmax(ranking_logits, dim=0)
    scores = {
        name: float(probability)
        for name, probability in zip(class_naming.names, probabilities.tolist())
    }
    candidate_count = min(rank_k, len(class_naming.names))

    if sample_ranking and temperature > 0.0:
        sample_weights = torch.softmax(ranking_logits / temperature, dim=0)
        selected_indices = torch.multinomial(
            sample_weights,
            num_samples=candidate_count,
            replacement=False,
        ).tolist()
    elif sample_ranking and temperature == 0.0:
        # Temperature zero is deterministic greedy ordering, even if sampling
        # is enabled, matching conventional decoding semantics.
        selected_indices = torch.argsort(ranking_logits, descending=True)[:candidate_count].tolist()
    else:
        selected_indices = torch.argsort(ranking_logits, descending=True)[:candidate_count].tolist()

    ranked_labels = [class_naming.names[index] for index in selected_indices]
    return ranked_labels, scores


@torch.no_grad()
def _sequence_log_prob(
    llm: LLMWrapper,
    text: str,
    ctx_kv,
    first_logits: Optional[torch.Tensor] = None,
) -> float:
    """
    Compute the generated-label log probability conditioned on the query context.
    """
    token_ids = _class_token_ids(llm, text)
    if first_logits is None:
        raise ValueError("first_logits are required to score a label continuation")
    log_p = 0.0
    cur_kv = copy.deepcopy(ctx_kv)
    current_logits = first_logits
    for index, token_id in enumerate(token_ids):
        log_p += torch.log_softmax(current_logits, dim=-1)[token_id].item()
        if index + 1 < len(token_ids):
            ids = torch.tensor([[token_id]], device=llm.device)
            tok_emb = llm.model.get_input_embeddings()(ids).to(llm.model_dtype)
            logits, cur_kv, _ = llm.prefill(
                tok_emb, past_kv=cur_kv, use_cache=True
            )
            current_logits = logits[0, -1]
    return log_p


@torch.no_grad()
def autoregressive_decode(
    llm: LLMWrapper,
    past_kv,
    next_logits: torch.Tensor,
    max_new_tokens: int = 24,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Tuple[str, object]:
    """Decode an assistant continuation with a strict generated-token limit."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if temperature < 0:
        raise ValueError("reasoning temperature must be non-negative")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")

    generated: List[int] = []
    current_logits = next_logits.float().clone()
    current_kv = past_kv
    eos_id = llm.tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        step_logits = current_logits.clone()
        if temperature == 0.0:
            next_id = int(torch.argmax(step_logits).item())
        else:
            scaled = step_logits / temperature
            probabilities = torch.softmax(scaled, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            remove = cumulative - sorted_probs >= top_p
            sorted_probs[remove] = 0
            filtered = torch.zeros_like(probabilities).scatter(
                0, sorted_indices, sorted_probs
            )
            filtered = filtered / filtered.sum()
            next_id = int(torch.multinomial(filtered, num_samples=1).item())

        if eos_id is not None and next_id == eos_id:
            break

        generated.append(next_id)
        token_embeds = llm.model.get_input_embeddings()(
            torch.tensor([[next_id]], device=llm.device)
        ).to(llm.model_dtype)
        logits, current_kv, _ = llm.prefill(
            token_embeds, past_kv=current_kv, use_cache=True
        )
        current_logits = logits[0, -1].float()

    decoded_text = llm.tokenizer.decode(generated, skip_special_tokens=True)
    return decoded_text, current_kv


@torch.no_grad()
def evaluate_dataset(
    llm: LLMWrapper,
    builder: EmbedBuilder,
    img_tok: ImageTokenizer,
    data_by_class: Dict[int, List[torch.Tensor]],
    class_naming: ClassNaming,
    past_kv,
    cfg: PromptConfig = DEFAULT_PROMPT_CFG,
    answer_mode: str = "first_token",
    temperature: float = 0.0,
    rank_k: int = 3,
    sample_ranking: bool = False,
    verbose: bool = False,
    desc: str = "Evaluating",
) -> Tuple[float, List[str], List[str]]:
    """
    Evaluate classification accuracy over a dataset dictionary mapping class_id -> list of images.

    Parameters
    ----------
    llm           : LLMWrapper instance
    builder       : EmbedBuilder instance
    img_tok       : ImageTokenizer instance
    data_by_class : Dict[class_id, List[torch.Tensor]]
    class_naming  : ClassNaming instance
    past_kv       : Accumulated KV cache
    cfg           : PromptConfig
    answer_mode   : 'first_token' or 'full_prob'
    temperature   : Class-sampling temperature; 0 is greedy argmax
    verbose       : If True, log every single image classification
    desc          : Progress bar description

    Returns
    -------
    (accuracy_percentage, all_true_names, all_pred_names)
    """
    total_imgs = sum(len(imgs) for imgs in data_by_class.values())
    if total_imgs == 0:
        return 0.0, [], []

    all_true: List[str] = []
    all_pred: List[str] = []
    correct = 0
    topk_correct = 0
    count = 0

    pbar = tqdm(total=total_imgs, desc=desc, leave=False, dynamic_ncols=True)
    for cls_id in range(len(class_naming.classes)):
        cls_name = class_naming.name(cls_id)
        images = data_by_class.get(cls_id, [])
        for img_idx, img in enumerate(images):
            try:
                pred_name, scores = classify_image(
                    llm, builder, img_tok, img,
                    class_naming=class_naming,
                    past_kv=past_kv,
                    cfg=cfg,
                    answer_mode=answer_mode,
                    temperature=temperature,
                    rank_k=rank_k,
                    sample_ranking=sample_ranking,
                )
                ranked_names = pred_name
                pred_name = ranked_names[0]
            except Exception as e:
                logging.error(f"Error classifying {cls_name} img {img_idx}: {e}")
                ranked_names = [class_naming.names[0]]
                pred_name = ranked_names[0]
                scores = {}

            is_ok = (pred_name == cls_name)
            correct += int(is_ok)
            topk_correct += int(cls_name in ranked_names)
            count += 1
            all_true.append(cls_name)
            all_pred.append(pred_name)

            pbar.update(1)
            pbar.set_postfix({
                "top1": f"{100.0 * correct / count:.1f}%",
                f"top{rank_k}": f"{100.0 * topk_correct / count:.1f}%",
                "corr": f"{correct}/{count}",
            })

            if verbose:
                true_disp = class_naming.display_name(cls_id)
                pred_disp = class_naming.display_name(pred_name)
                logging.info(
                    f"  [{count:4d}] true={true_disp:20s} pred={pred_disp:20s} "
                    f"ranked={', '.join(ranked_names)} {'✓' if is_ok else '✗'}"
                )
                if scores:
                    top = sorted(scores.items(), key=lambda x: -x[1])[:5]
                    logging.debug(f"    top-5: {top}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    pbar.close()
    accuracy = 100.0 * correct / total_imgs if total_imgs > 0 else 0.0
    topk_accuracy = 100.0 * topk_correct / total_imgs if total_imgs > 0 else 0.0
    logging.info(
        "%s accuracy: top-1=%.2f%%, top-%d=%.2f%%",
        desc, accuracy, rank_k, topk_accuracy,
    )
    return accuracy, all_true, all_pred


# ---------------------------------------------------------------------------
# 11. Dataset Utilities
# ---------------------------------------------------------------------------

def load_cifar10(data_dir: str, train: bool = True) -> torchvision.datasets.CIFAR10:
    transform = transforms.ToTensor()   # (3, 32, 32) in [0, 1]
    return torchvision.datasets.CIFAR10(
        root=data_dir, train=train, download=True, transform=transform
    )


def subset_dataset(
    dataset,
    subset_size: Union[int, float],
    balanced: bool = True,
    seed: Optional[int] = 0,
) -> Dict[int, List[torch.Tensor]]:
    """Select a total-size subset from a dataset, optionally balanced by class.

    Integer sizes count examples across the whole dataset. A float in [0, 1]
    selects that fraction of the whole dataset. Balanced selection divides the
    requested total as evenly as possible among classes. Unbalanced selection
    takes the first requested examples in dataset order.
    """
    _validate_subset_spec(subset_size, "subset size")
    total_size = _resolve_subset_count(subset_size, len(dataset))
    targets = getattr(dataset, "targets", None)
    if targets is None:
        records = [dataset[index] for index in range(len(dataset))]
        targets = [int(label) for _, label in records]
    else:
        records = None
        targets = [int(label) for label in targets]

    class_ids = sorted(set(targets))
    selected_indices: List[int]
    if not balanced:
        selected_indices = list(range(total_size))
    else:
        indices_by_class: Dict[int, List[int]] = {class_id: [] for class_id in class_ids}
        for index, class_id in enumerate(targets):
            indices_by_class[class_id].append(index)

        selection_rng = random.Random(seed) if seed is not None else None
        if selection_rng is not None:
            for class_indices in indices_by_class.values():
                selection_rng.shuffle(class_indices)
            remainder_order = list(class_ids)
            selection_rng.shuffle(remainder_order)
        else:
            remainder_order = list(class_ids)

        per_class = {class_id: 0 for class_id in class_ids}
        remaining = total_size
        while remaining > 0:
            progressed = False
            for class_id in remainder_order:
                if per_class[class_id] < len(indices_by_class[class_id]):
                    per_class[class_id] += 1
                    remaining -= 1
                    progressed = True
                    if remaining == 0:
                        break
            if not progressed:
                break
        selected_indices = [
            index
            for class_id in class_ids
            for index in indices_by_class[class_id][:per_class[class_id]]
        ]
        selected_indices.sort()

    class_buckets: Dict[int, List[torch.Tensor]] = {class_id: [] for class_id in class_ids}
    for index in selected_indices:
        if records is None:
            image, label = dataset[index]
        else:
            image, label = records[index]
        class_buckets[int(label)].append(image)
    return class_buckets


def _validate_subset_spec(value: Union[int, float], name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be an integer count or a fraction in [0, 1]")
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"{name} count must be non-negative")
    elif not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} fraction must be in [0, 1]")


def _parse_subset_spec(value: str) -> Union[int, float]:
    """Parse integer counts or decimal proportions for CLI sample sizes."""
    try:
        parsed: Union[int, float]
        if "." not in value and "e" not in value.lower():
            parsed = int(value)
        else:
            parsed = float(value)
        _validate_subset_spec(parsed, "subset size")
        return parsed
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parse_seed(value: str) -> Optional[int]:
    if value.strip().lower() in ("none", "null"):
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seed must be an integer or None") from exc


def _resolve_subset_count(value: Union[int, float], available: int) -> int:
    _validate_subset_spec(value, "subset size")
    requested = math.floor(available * value) if isinstance(value, float) else value
    return min(requested, available)


def _subset_buckets_by_total(
    data_by_class: Dict[int, List[torch.Tensor]],
    total_size: int,
    balanced: bool = True,
) -> Dict[int, List[torch.Tensor]]:
    """Take a total-size subset from existing class buckets."""
    class_ids = sorted(data_by_class)
    result = {class_id: [] for class_id in class_ids}
    remaining = min(total_size, sum(len(data_by_class[cid]) for cid in class_ids))
    if not balanced:
        for class_id in class_ids:
            take = min(remaining, len(data_by_class[class_id]))
            result[class_id] = data_by_class[class_id][:take]
            remaining -= take
            if remaining == 0:
                break
        return result

    per_class = {class_id: 0 for class_id in class_ids}
    while remaining > 0:
        progressed = False
        for class_id in class_ids:
            if per_class[class_id] < len(data_by_class[class_id]):
                per_class[class_id] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break
    return {
        class_id: data_by_class[class_id][:per_class[class_id]]
        for class_id in class_ids
    }


def build_training_pairs(
    data_by_class: Dict[int, List[torch.Tensor]],
    class_naming: ClassNaming,
    balanced: bool = True,
    seed: Optional[int] = 0,
    epoch: int = 1,
) -> List[Tuple[torch.Tensor, str]]:
    """Build per-image training examples, optionally equalizing class counts."""
    counts = [len(data_by_class.get(cid, [])) for cid in range(len(class_naming.classes))]
    per_class_count = min(counts, default=0) if balanced else None
    pairs = [
        (image, class_naming.name(class_id))
        for class_id in range(len(class_naming.classes))
        for image in data_by_class.get(class_id, [])[
            :per_class_count if per_class_count is not None else None
        ]
    ]
    if seed is not None:
        random.Random(seed + epoch - 1).shuffle(pairs)
    return pairs


def plot_confusion_matrix(
    true_labels: List[str],
    pred_labels: List[str],
    class_names: List[str],
    out_path: Path,
):
    if not HAS_MATPLOTLIB:
        return
    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    idx = {c: i for i, c in enumerate(class_names)}
    for t, p in zip(true_labels, pred_labels):
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += 1
    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n)))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(class_names)
    thresh = cm.max() / 2
    for i in range(n):
        for j in range(n):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    logging.info(f"Confusion matrix saved to {out_path}")


# ---------------------------------------------------------------------------
# 12. Argument Parsing & Config Loading
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="LLM Learner")

    p.add_argument("--model-name", default="HuggingFaceTB/SmolLM3-3B",
                   help="HuggingFace model name")
    p.add_argument("--device", default=None,
                   help="cuda / cpu / None (auto)")
    p.add_argument("--attn-impl", default=None,
                   choices=["eager", "sdpa", "flash_attention_2"],
                   help="Attention implementation override")
    p.add_argument("--dtype", default="auto",
                   choices=["auto", "float32", "float16", "bfloat16"],
                   help="Model dtype. Default: auto (model's native)")

    # Image tokenizer
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--img-mode", choices=["floats", "byte_embeds"], default="byte_embeds")
    p.add_argument("--img-reduction", choices=["random_proj", "sum", "avg"],
                   default="random_proj",
                   help="Patch reduction method")
    # KV cache compression
    p.add_argument("--kv-policy", choices=["noop", "streaming", "attn_query"],
                   default="attn_query")
    p.add_argument("--kv-sink", type=int, default=4,
                   help="StreamingLLM: number of sink tokens")
    p.add_argument("--kv-topk", type=int, default=128,
                   help="AttentionQuery: top-k positions to keep")
    p.add_argument("--attractor-range", type=int, nargs=2, default=[-10, -1],
                   metavar=("START", "END"),
                   help="AttentionQuery: attractor query range (supports negatives)")
    p.add_argument("--neighbor-window", type=int, nargs=2, default=[0, 0],
                   metavar=("BEFORE", "AFTER"),
                   help="AttentionQuery: neighbor window around selected positions")
    p.add_argument("--attractor-round", choices=["assistant", "user", "turn_tail"],
                   default="assistant",
                   help="Attractor query type: 'assistant'=user Q + forced asst A, 'user'=user Q only, 'turn_tail'=use in-turn tail tokens")
    p.add_argument("--attractor-temporary", action="store_true", default=False,
                   help="If set, attractor query prefill is temporary/transient (not appended to KV cache)")
    p.add_argument("--attractor-user-prompt", type=str, default=None,
                   help="Override user prompt for attractor query")
    p.add_argument("--attractor-asst-prompt", type=str, default=None,
                   help="Override assistant prompt for attractor query")

    p.add_argument("--train-samples", type=_parse_subset_spec, default=100,
                   help="Total training subset size: integer count or fraction in [0, 1] (default: 100)")
    p.add_argument("--test-samples", type=_parse_subset_spec, default=100,
                   help="Total test subset size: integer count or fraction in [0, 1] (default: 100)")
    p.add_argument("--seed", type=_parse_seed, default=0,
                   help="Training shuffle/random seed (default: 0); use None for ordered training")
    p.add_argument("--balanced-sampling", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Balance the total train/test subset across classes (default: enabled)")
    p.add_argument("--teaching-epochs", type=int, default=2,
                   help="Number of teaching passes over training data")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Class-ranking sampling temperature (used only with --sample-ranking; default: 1.0)")
    p.add_argument("--rank-k", type=int, default=3,
                   help="Number of distinct ranked class candidates to return")
    p.add_argument("--sample-ranking", action=argparse.BooleanOptionalAction, default=False,
                   help="Sample distinct ranked classes without replacement instead of deterministic top-k (default: disabled)")
    p.add_argument("--reasoning", action=argparse.BooleanOptionalAction, default=False,
                   help="After each teaching feedback prefill, autoregressively generate a short visual explanation (default: disabled)")
    p.add_argument("--reasoning-max-new-tokens", type=int, default=DEFAULT_REASONING_MAX_NEW_TOKENS,
                   help="Maximum number of autoregressively generated reasoning tokens")
    p.add_argument("--reasoning-temperature", type=float, default=1.0,
                   help="Reasoning generation temperature (default: 0.7)")
    p.add_argument("--reasoning-top-p", type=float, default=0.7,
                   help="Nucleus sampling top-p for reasoning generation (default: 0.9)")
    p.add_argument("--query-prompt", type=str, default=None,
                   help="Override the per-image query prompt (supports {class_names_list} and {rank_k})")
    p.add_argument("--reasoning-prompt", type=str, default=None,
                   help="Override the user-facing reasoning instruction")
    p.add_argument("--feedback-user-prompt", type=str, default=None,
                   help="Override the feedback USER-turn template (supports {class_name}, {ranked_labels}, {rank_feedback}, {reasoning_instruction}, {true_rank}, {rank_k})")
    p.add_argument("--feedback-acknowledgement", type=str, default=None,
                   help="Override the fixed assistant acknowledgement when reasoning is disabled")
    p.add_argument("--ranking-instruction", type=str, default=None,
                   help="Override deterministic ranking instruction (supports {rank_k})")
    p.add_argument("--sampled-ranking-instruction", type=str, default=None,
                   help="Override sampled-ranking instruction (supports {rank_k})")
    p.add_argument("--single-ranking-instruction", type=str, default=None,
                   help="Override the single-label ranking instruction")
    p.add_argument("--rank-feedback-top1", type=str, default=None,
                   help="Override feedback text when the true class is top-ranked")
    p.add_argument("--rank-feedback-in-topk", type=str, default=None,
                   help="Override feedback text when the true class is in top-k but not first (supports {true_rank}, {rank_k})")
    p.add_argument("--rank-feedback-missed", type=str, default=None,
                   help="Override feedback text when the true class is outside top-k (supports {rank_k})")
    p.add_argument("--answer-mode", choices=["first_token", "full_prob"],
                   default="first_token",
                   help="Score candidate labels by first token or complete token sequence")
    p.add_argument("--guess-prob-gt", type=float, default=0.0,
                   help="Legacy experiment: probability of replacing the sampled teaching guess with ground truth")
    p.add_argument("--guess-prob-random", type=float, default=0.0,
                   help="Legacy experiment: probability of replacing the sampled teaching guess with a random label")
    p.add_argument("--eval-every", type=int, default=1,
                   help="Evaluate validation/test accuracy every N epochs (and at the final epoch)")
    p.add_argument("--train-eval-samples", type=_parse_subset_spec, default=None,
                   help="Total training examples for post-teaching evaluation; integer or fraction (default: min selected train/test total)")
    p.add_argument("--skip-val", action="store_true", default=False,
                   help="Skip validation/test evaluation passes")
    p.add_argument("--sanity-check", action="store_true", default=False,
                   help="Sanity check mode: evaluate on a subset of training images to test recall")
    p.add_argument("--verbose", action="store_true", default=False,
                   help="Enable verbose per-image logging during evaluation and teaching")

    # Class labels
    p.add_argument("--generic-class-names", action="store_true", default=False,
                   help="Deprecated alias for --class-label-mode generic")
    p.add_argument("--class-label-mode", "--label-style",
                   dest="class_label_mode",
                   choices=["direct", "letters", "mc_letters", "digits", "mc_digits", "generic"],
                   default="direct",
                   help="Labels used for teaching and prediction (default: direct)")
    p.add_argument("--include-class-options", action="store_true", default=False,
                   help="Make guess-correct query prompts list each label and its CIFAR-10 class")
    p.add_argument("--disable-instruction", action="store_true", default=False,
                   help="Disable the first-round instruction prompt turn")

    # Data / logging
    p.add_argument("--data-dir", default="./datasets")
    p.add_argument("--log-dir", default="./logs")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--no-confusion", action="store_true", default=False)

    # Config file
    p.add_argument("--config", default=None,
                   help="Python config file to override defaults")

    args, _ = p.parse_known_args()
    if args.config and Path(args.config).exists():
        _apply_config(args.config, p)
    args = p.parse_args()

    # Post-process
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.dtype_obj = {
        "auto": None, "float32": torch.float32,
        "float16": torch.float16, "bfloat16": torch.bfloat16,
    }[args.dtype]
    _validate_subset_spec(args.train_samples, "--train-samples")
    _validate_subset_spec(args.test_samples, "--test-samples")
    if args.train_eval_samples is not None:
        _validate_subset_spec(args.train_eval_samples, "--train-eval-samples")
    if args.temperature < 0:
        p.error("--temperature must be non-negative")
    if args.rank_k < 1:
        p.error("--rank-k must be at least 1")
    if args.reasoning_max_new_tokens < 0:
        p.error("--reasoning-max-new-tokens must be non-negative")
    if args.reasoning and args.reasoning_max_new_tokens < 1:
        p.error("--reasoning-max-new-tokens must be at least 1 when --reasoning is enabled")
    if args.reasoning_temperature < 0:
        p.error("--reasoning-temperature must be non-negative")
    if not 0.0 < args.reasoning_top_p <= 1.0:
        p.error("--reasoning-top-p must be in (0, 1]")
    if not 0.0 <= args.guess_prob_gt <= 1.0:
        p.error("--guess-prob-gt must be between 0 and 1")
    if not 0.0 <= args.guess_prob_random <= 1.0:
        p.error("--guess-prob-random must be between 0 and 1")

    return args


def _apply_config(config_path: str, parser: argparse.ArgumentParser):
    """Load a Python config file and update parser defaults."""
    with open(config_path) as f:
        src = f.read()
    tree = ast.parse(src, filename=config_path)
    cfg: Dict = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    try:
                        val = eval(
                            compile(ast.Expression(node.value), config_path, "eval"),
                            {"__builtins__": {}}, {}
                        )
                        cfg[t.id] = val
                    except Exception:
                        pass
    for key, val in cfg.items():
        dest = key.replace("-", "_")
        for action in parser._actions:
            if action.dest == dest:
                action.default = val
                parser._defaults[dest] = val
                break


# ---------------------------------------------------------------------------
# 13. Logging Setup
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str, name: str = "llm_learner", level: str = "INFO"):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / f"{name}.log"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(asctime)s] %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return log_file


# ---------------------------------------------------------------------------
# 14. Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    setup_logging(args.log_dir, level=args.log_level)
    log_dir = Path(args.log_dir)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    logging.info("=" * 70)
    logging.info("LLM Learner")
    logging.info("=" * 70)
    for k, v in sorted(vars(args).items()):
        if not k.startswith("_") and k != "dtype_obj":
            logging.info(f"  {k}: {v}")

    # -------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------
    logging.info("\nLoading CIFAR-10 ...")
    train_ds = load_cifar10(args.data_dir, train=True)
    train_by_class = subset_dataset(
        train_ds,
        args.train_samples,
        balanced=args.balanced_sampling,
        seed=args.seed,
    )

    # Validation/Test dataset
    if args.skip_val:
        logging.info("Validation evaluation is skipped (--skip-val)")
        test_by_class = {}
    elif args.sanity_check:
        logging.info("SANITY CHECK MODE: Testing on the selected training subset")
        test_by_class = subset_dataset(
            train_ds,
            args.test_samples,
            balanced=args.balanced_sampling,
            seed=args.seed,
        )
    else:
        test_ds  = load_cifar10(args.data_dir, train=False)
        test_by_class = subset_dataset(
            test_ds,
            args.test_samples,
            balanced=args.balanced_sampling,
            seed=args.seed,
        )

    # Post-teaching train evaluation uses the smaller selected dataset subset
    # by default; an explicit integer or fraction can override that size.
    train_selected_total = sum(len(images) for images in train_by_class.values())
    test_selected_total = sum(len(images) for images in test_by_class.values())
    if args.train_eval_samples is None:
        train_eval_total = min(
            train_selected_total,
            test_selected_total if test_by_class else train_selected_total,
        )
    else:
        train_eval_total = _resolve_subset_count(
            args.train_eval_samples,
            len(train_ds),
        )
        train_eval_total = min(train_eval_total, train_selected_total)
    train_eval_by_class = _subset_buckets_by_total(
        train_by_class,
        train_eval_total,
        balanced=args.balanced_sampling,
    )

    n_train_total = sum(len(v) for v in train_by_class.values())
    n_train_eval_total = sum(len(v) for v in train_eval_by_class.values())
    n_test_total = sum(len(v) for v in test_by_class.values())

    logging.info(f"  Train:      {n_train_total} images")
    logging.info(f"  Train-Eval: {n_train_eval_total} images")
    logging.info(f"  Val/Test:   {n_test_total} images" + (" (sanity check: from train)" if args.sanity_check else (" (skipped)" if args.skip_val else "")))

    # -------------------------------------------------------------------
    # Class naming
    # -------------------------------------------------------------------
    class_naming = ClassNaming(
        CIFAR10_CLASSES,
        use_generic=args.generic_class_names,
        label_mode=args.class_label_mode,
    )
    if class_naming.use_generic:
        mapping_str = ", ".join(f"{class_naming.name(i)}: {class_naming.original(i)}" for i in range(len(CIFAR10_CLASSES)))
        logging.info(f"Class names (generic mode): {mapping_str}")
    else:
        logging.info(f"Class names: {class_naming.names}")

    # -------------------------------------------------------------------
    # LLM
    # -------------------------------------------------------------------
    logging.info("\nInitializing LLM ...")
    llm = LLMWrapper(
        args.model_name,
        device=args.device,
        attn_implementation=args.attn_impl,
        torch_dtype=args.dtype_obj,
    )

    # -------------------------------------------------------------------
    # Image Tokenizer
    # -------------------------------------------------------------------
    logging.info("\nInitializing Image Tokenizer ...")
    vocab_embs = llm.get_vocab_embeds() if args.img_mode == "byte_embeds" else None
    img_tok = ImageTokenizer(
        patch_size=args.patch_size,
        img_size=32,
        mode=args.img_mode,
        reduction=args.img_reduction,
        llm_dim=llm.hidden_dim,
        vocab_embeds=vocab_embs,
    ).to(llm.device)
    logging.info(f"  {img_tok}")

    # -------------------------------------------------------------------
    # Chat template + EmbedBuilder
    # -------------------------------------------------------------------
    logging.info("\nExtracting chat template ...")
    extractor = ChatTemplateExtractor(llm.tokenizer)
    chat_segs = extractor.segments
    builder   = EmbedBuilder(llm, chat_segs)

    # -------------------------------------------------------------------
    # KV Cache Compressor
    # -------------------------------------------------------------------
    if args.kv_policy == "noop":
        compressor = NoopCompressor()
    elif args.kv_policy == "streaming":
        compressor = StreamingLLMCompressor(
            num_sink=args.kv_sink, keep_last=args.kv_topk
        )
    elif args.kv_policy == "attn_query":
        compressor = AttentionQueryCompressor(
            top_k=args.kv_topk,
            attractor_range=tuple(args.attractor_range),
            neighbor_window=tuple(args.neighbor_window),
            attractor_round=args.attractor_round,
            attractor_temporary=args.attractor_temporary,
        )
    logging.info(f"KV compressor: {compressor}")

    # -------------------------------------------------------------------
    # Teaching Phase
    # -------------------------------------------------------------------
    logging.info("\n" + "=" * 70)
    logging.info("TEACHING PHASE")
    logging.info("=" * 70)

    past_kv    = None
    prompt_cfg = DEFAULT_PROMPT_CFG
    prompt_cfg.include_class_options = (
        args.include_class_options or args.answer_mode == "full_prob"
    )
    prompt_cfg.verbose = args.verbose
    if args.query_prompt is not None:
        prompt_cfg.guess_user_tmpl = args.query_prompt
    if args.reasoning_prompt is not None:
        prompt_cfg.reasoning_prompt = args.reasoning_prompt
    if args.feedback_user_prompt is not None:
        prompt_cfg.feedback_user_prompt = args.feedback_user_prompt
    if args.feedback_acknowledgement is not None:
        prompt_cfg.feedback_acknowledgement = args.feedback_acknowledgement
    if args.ranking_instruction is not None:
        prompt_cfg.ranking_instruction = args.ranking_instruction
    if args.sampled_ranking_instruction is not None:
        prompt_cfg.sampled_ranking_instruction = args.sampled_ranking_instruction
    if args.single_ranking_instruction is not None:
        prompt_cfg.single_ranking_instruction = args.single_ranking_instruction
    if args.rank_feedback_top1 is not None:
        prompt_cfg.rank_feedback_top1 = args.rank_feedback_top1
    if args.rank_feedback_in_topk is not None:
        prompt_cfg.rank_feedback_in_topk = args.rank_feedback_in_topk
    if args.rank_feedback_missed is not None:
        prompt_cfg.rank_feedback_missed = args.rank_feedback_missed
    if args.disable_instruction:
        prompt_cfg.use_instruction_turn = False
    prompt_cfg.attractor_round = args.attractor_round
    prompt_cfg.attractor_temporary = args.attractor_temporary
    if args.attractor_user_prompt is not None:
        prompt_cfg.attractor_user_prompt = args.attractor_user_prompt
        prompt_cfg.attractor_query_user_prompt = args.attractor_user_prompt
    if args.attractor_asst_prompt is not None:
        prompt_cfg.attractor_asst_prompt = args.attractor_asst_prompt

    _log_prompt_catalog(
        prompt_cfg,
        class_naming,
        chat_segs,
        answer_mode=args.answer_mode,
        temperature=args.temperature,
        attractor_round=args.attractor_round,
        patch_count=img_tok.num_patches,
        rank_k=min(args.rank_k, len(class_naming.names)),
        sample_ranking=args.sample_ranking,
        reasoning_enabled=args.reasoning,
        reasoning_max_new_tokens=args.reasoning_max_new_tokens,
    )

    if prompt_cfg.use_instruction_turn and prompt_cfg.instruction_user_prompt:
        logging.info("Prefilling first-round instruction prompt...")
        past_kv, info = prefill_instruction(
            llm, builder, past_kv, cfg=prompt_cfg,
            class_naming=class_naming,
            rank_k=min(args.rank_k, len(class_naming.names)),
            reasoning_enabled=args.reasoning,
            sample_ranking=args.sample_ranking,
        )
        logging.info(f"  -> Instruction prefilled (+{info['tokens_added']} tokens), "
                     f"cache now: {kv_seq_len(past_kv)} tokens")

    history: List[Dict] = []
    last_val_true: List[str] = []
    last_val_pred: List[str] = []

    epoch_pbar = tqdm(range(1, args.teaching_epochs + 1), desc="Epochs", dynamic_ncols=True)
    for epoch in epoch_pbar:
        logging.info(f"\n-- Epoch {epoch}/{args.teaching_epochs} --")

        training_pairs = build_training_pairs(
            train_by_class,
            class_naming,
            balanced=args.balanced_sampling,
            seed=args.seed,
            epoch=epoch,
        )
        correct_count = 0
        train_pbar = tqdm(
            enumerate(training_pairs),
            total=len(training_pairs),
            desc=f"Ep {epoch} Online Teaching",
            leave=False,
            dynamic_ncols=True,
        )
        for index, (image, true_name) in train_pbar:
            past_kv, was_correct, predicted = teach_guess_correct(
                llm, builder, img_tok, image, true_name, class_naming,
                past_kv, compressor=compressor, cfg=prompt_cfg,
                answer_mode=args.answer_mode,
                temperature=args.temperature,
                rank_k=min(args.rank_k, len(class_naming.names)),
                sample_ranking=args.sample_ranking,
                reasoning_enabled=args.reasoning,
                reasoning_max_new_tokens=args.reasoning_max_new_tokens,
                reasoning_temperature=args.reasoning_temperature,
                reasoning_top_p=args.reasoning_top_p,
                guess_prob_gt=args.guess_prob_gt,
                guess_prob_random=args.guess_prob_random,
            )
            correct_count += int(was_correct)
            acc = correct_count / (index + 1) * 100.0
            train_pbar.set_postfix({
                "online": f"{acc:.1f}%",
                "cache": kv_seq_len(past_kv),
            })
            if args.verbose and ((index + 1) % 10 == 0 or index == 0):
                logging.info(
                    "  img %d/%d: true=%s, pred=%s %s "
                    "(pre-feedback online accuracy=%.1f%%, cache=%d)",
                    index + 1,
                    len(training_pairs),
                    class_naming.display_name(true_name),
                    ", ".join(class_naming.display_name(name) for name in predicted),
                    "✓" if was_correct else "✗",
                    acc,
                    kv_seq_len(past_kv),
                )
        logging.info(
            "Epoch %d online pre-feedback teaching accuracy: %.2f%%",
            epoch,
            acc if training_pairs else 0.0,
        )

        # Periodic evaluation (every eval_every epochs, plus final epoch)
        is_last_epoch = (epoch == args.teaching_epochs)
        should_eval = (epoch % args.eval_every == 0) or is_last_epoch

        train_acc = None
        val_acc = None

        if should_eval:
            cur_cache_len = kv_seq_len(past_kv)
            logging.info(f"Evaluating at epoch {epoch} (cache: {cur_cache_len} tokens)...")

            if n_train_eval_total > 0:
                train_acc, _, _ = evaluate_dataset(
                    llm, builder, img_tok, train_eval_by_class,
                    class_naming=class_naming, past_kv=past_kv,
                    cfg=prompt_cfg, answer_mode=args.answer_mode,
                    temperature=args.temperature,
                    rank_k=min(args.rank_k, len(class_naming.names)),
                    sample_ranking=args.sample_ranking,
                    verbose=args.verbose, desc=f"Ep {epoch} Eval (Train)",
                )

            if not args.skip_val and n_test_total > 0:
                val_acc, last_val_true, last_val_pred = evaluate_dataset(
                    llm, builder, img_tok, test_by_class,
                    class_naming=class_naming, past_kv=past_kv,
                    cfg=prompt_cfg, answer_mode=args.answer_mode,
                    temperature=args.temperature,
                    rank_k=min(args.rank_k, len(class_naming.names)),
                    sample_ranking=args.sample_ranking,
                    verbose=args.verbose, desc=f"Ep {epoch} Eval (Val)",
                )

            parts = [f"Epoch {epoch:2d}/{args.teaching_epochs}", f"KV Cache: {cur_cache_len} tokens"]
            if train_acc is not None:
                parts.append(f"Train Acc: {train_acc:.2f}%")
            if val_acc is not None:
                parts.append(f"Val Acc: {val_acc:.2f}%")
            logging.info(" | ".join(parts))

            history.append({
                "epoch": epoch,
                "kv_len": cur_cache_len,
                "train_acc": train_acc,
                "val_acc": val_acc,
            })

            post = {}
            if train_acc is not None:
                post["train"] = f"{train_acc:.1f}%"
            if val_acc is not None:
                post["val"] = f"{val_acc:.1f}%"
            epoch_pbar.set_postfix(post)

    if past_kv is None or kv_seq_len(past_kv) == 0:
        logging.error("Teaching produced empty KV cache. Exiting.")
        return

    logging.info(f"\nTeaching complete. Final KV cache length: {kv_seq_len(past_kv)}")

    # -------------------------------------------------------------------
    # Results & Summary Table
    # -------------------------------------------------------------------
    logging.info("\n" + "=" * 70)
    logging.info("TRAINING SUMMARY")
    logging.info("=" * 70)

    header = f"| {'Epoch':^7} | {'KV Len':^8} | {'Train Acc':^13} | {'Val Acc':^13} |"
    sep = f"|{'-'*9}+{'-'*10}+{'-'*15}+{'-'*15}|"
    logging.info(sep)
    logging.info(header)
    logging.info(sep)
    for row in history:
        tr_s = f"{row['train_acc']:.2f}%" if row['train_acc'] is not None else "N/A"
        va_s = f"{row['val_acc']:.2f}%" if row['val_acc'] is not None else "N/A"
        logging.info(f"| {row['epoch']:^7d} | {row['kv_len']:^8d} | {tr_s:^13s} | {va_s:^13s} |")
    logging.info(sep)

    # Save to CSV
    csv_file = log_dir / "metrics.csv"
    try:
        with open(csv_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "kv_len", "train_acc", "val_acc"])
            for row in history:
                writer.writerow([
                    row["epoch"],
                    row["kv_len"],
                    f"{row['train_acc']:.4f}" if row['train_acc'] is not None else "",
                    f"{row['val_acc']:.4f}" if row['val_acc'] is not None else "",
                ])
        logging.info(f"Saved metrics CSV to: {csv_file}")
    except Exception as e:
        logging.warning(f"Failed to write metrics CSV: {e}")

    # Per-class breakdown & Confusion Matrix (from final val run, if available)
    if last_val_true and last_val_pred:
        logging.info("\nPer-class accuracy (Final Val Pass):")
        class_correct: Dict[str, int] = {}
        class_total:   Dict[str, int] = {}
        for t, p in zip(last_val_true, last_val_pred):
            class_total[t]   = class_total.get(t, 0) + 1
            class_correct[t] = class_correct.get(t, 0) + int(t == p)
        for cls_id in range(len(class_naming.classes)):
            name = class_naming.name(cls_id)
            disp = class_naming.display_name(cls_id)
            n = class_total.get(name, 0)
            c = class_correct.get(name, 0)
            bar = "█" * c + "░" * (n - c)
            logging.info(f"  {disp:20s}: {c}/{n}  {bar}")

        if not args.no_confusion and HAS_MATPLOTLIB:
            try:
                disp_true = [class_naming.display_name(t) for t in last_val_true]
                disp_pred = [class_naming.display_name(p) for p in last_val_pred]
                disp_classes = [class_naming.display_name(i) for i in range(len(class_naming.classes))]
                plot_confusion_matrix(
                    disp_true, disp_pred, disp_classes,
                    log_dir / "confusion_matrix.png",
                )
            except Exception as e:
                logging.warning(f"Confusion matrix failed: {e}")


if __name__ == "__main__":
    main()

