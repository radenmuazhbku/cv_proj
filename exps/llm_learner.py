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
5. Teaching input modes:
   - 'one-class': one batch per class ("class1 <e1> <e2> ...")
   - 'mixed': interleaved classes ("<e1> class1 <e2> class2 ...")
   - 'guess-correct': model guesses; feedback given per example

Usage
-----
    python exps/llm_learner.py
    python exps/llm_learner.py --model-name Qwen/Qwen3-0.6B
    python exps/llm_learner.py --img-mode byte_embeds --img-reduction sum
    python exps/llm_learner.py --kv-policy streaming --kv-sink 4 --kv-last 64
    python exps/llm_learner.py --kv-policy attn_query --kv-topk 128 \\
        --attractor-range -10 -1 --neighbor-window 1 1
    python exps/llm_learner.py --input-mode guess-correct --train-per-class 5
    python exps/llm_learner.py --generic-class-names
    python exps/llm_learner.py --config exps/configs/llm_learner_config.py
"""

import argparse
import ast
import gc
import logging
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

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

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
            full = tok.apply_chat_template(
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
            gen_prompt = tok.apply_chat_template(
                msgs2, tokenize=False, add_generation_prompt=True
            )
            u2s_gp = gen_prompt.index(_SENT_U2)
            u2e_gp = u2s_gp + len(_SENT_U2)
            asst_open = gen_prompt[u2e_gp:]
            if user_to_asst.endswith(asst_open) and asst_open:
                user_close = user_to_asst[: len(user_to_asst) - len(asst_open)]
                return user_close, asst_open
        except Exception:
            pass
        # Can't split, treat everything as the separator
        return "", user_to_asst

    def _extract_single_turn_fallback(self) -> ChatSegments:
        """Minimal fallback: single-turn extraction."""
        tok = self.tokenizer
        msgs = [{"role": "user", "content": _SENT_U1}]
        try:
            full = tok.apply_chat_template(
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
    def embed_text(self, text: str) -> torch.Tensor:
        """
        Tokenize text and embed it.

        Returns:
            (1, seq_len, hidden_dim) float tensor on model's device.
            Returns empty tensor (1, 0, hidden_dim) for empty string.
        """
        if not text:
            return torch.zeros(1, 0, self.hidden_dim,
                               device=self.device, dtype=self.model_dtype)
        ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
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
    ):
        self.top_k           = top_k
        self.attractor_range = attractor_range
        self.neighbor_window = neighbor_window
        self.agg             = agg
        self.num_sink        = num_sink

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
        # attention_maps: tuple of (1, num_heads, seq_q, seq_k) per layer
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
                f"attractor={self.attractor_range}, neighbor={self.neighbor_window})")


# ---------------------------------------------------------------------------
# 6. Embed Builder
# ---------------------------------------------------------------------------

class EmbedBuilder:
    """
    Build mixed text + visual embedding sequences using extracted chat tags.

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
        _add("asst_tail",    self.llm.embed_text(self.chat.asst_tail))

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

    def __init__(self, classes: List[str], use_generic: bool = False):
        self.classes      = classes
        self.use_generic  = use_generic
        self._id_to_name  = {
            i: (f"class{i+1}" if use_generic else cls)
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
        """Return display name; in generic mode, also prints the original class name if known."""
        if isinstance(name_or_id, int):
            cid = name_or_id
            cname = self.name(cid)
        else:
            cname = name_or_id
            cid = self._name_to_id.get(cname, None)

        if self.use_generic and cid is not None and 0 <= cid < len(self.classes):
            return f"{cname} ({self.classes[cid]})"
        return cname

    @property
    def names(self) -> List[str]:
        return [self._id_to_name[i] for i in range(len(self.classes))]


# ---------------------------------------------------------------------------
# 8. Prompt Config
# ---------------------------------------------------------------------------

@dataclass
class PromptConfig:
    """
    Text templates for each part of the conversation.

    Template variables: {class_name}, {class_names_list}
    """
    # First-round instruction / system setup turn
    instruction_user_prompt: str = (
        "You are a large language model. You were never trained on images directly, "
        "but in this session you will perform in-context learning analogous to deep neural network "
        "SGD classification training, assisted by a KV cache compression algorithm.\n"
        "Because earlier KV cache entries are dynamically removed, your self-attention at recent "
        "timesteps may not attend back to all earlier inputs.\n\n"
        "In each turn, you will be given a batch of image patch vectors and label pairs. "
        "Respond in a way that consolidates memory.\n\n"
        "At the end of training, there will be test images unseen during in-context learning."
    )
    instruction_asst_prompt: str = "Understood."
    use_instruction_turn: bool = True

    # Teaching (one-class mode)
    teach_user_tmpl: str  = "Here are examples of {class_name}."
    teach_asst_tmpl: str  = "Understood. I have memorized {class_name}."

    # Teaching (mixed mode)
    mixed_intro_tmpl: str = "Match each visual to its class."
    mixed_label_sep: str  = " "    # text placed between embed and next-label
    mixed_asst_tmpl: str  = "I have memorized all examples."

    # Guess-correct mode
    guess_user_tmpl: str      = ""    # user turn is just the visual embed
    guess_correct_user: str   = "{class_name}"
    guess_correct_asst: str   = "correct, {class_name}"
    guess_wrong_asst: str     = "wrong, it is {class_name}"

    # Test / classification query
    test_user_tmpl: str = "What class is this image?"

    # Sep between images in a batch
    image_sep: str = " "


# Default config
DEFAULT_PROMPT_CFG = PromptConfig()


# ---------------------------------------------------------------------------
# 9. Teaching Functions
# ---------------------------------------------------------------------------

@torch.no_grad()
def prefill_instruction(
    llm: LLMWrapper,
    builder: EmbedBuilder,
    past_kv,
    cfg: PromptConfig = DEFAULT_PROMPT_CFG,
) -> Tuple[object, Dict]:
    """
    Prefill the first-round instruction turn (user instruction + assistant response).
    This establishes task context before any image training data is ingested.
    """
    turn_emb, positions = builder.build_teaching_turn(
        user_text=cfg.instruction_user_prompt,
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
def teach_one_class(
    llm: LLMWrapper,
    builder: EmbedBuilder,
    img_tok: ImageTokenizer,
    class_name: str,
    images: torch.Tensor,          # (N, C, H, W)
    past_kv,
    compressor: KVCacheCompressor,
    cfg: PromptConfig = DEFAULT_PROMPT_CFG,
    is_first_turn: bool = False,
) -> Tuple[object, Dict]:
    """
    'one-class' input mode: present all images of one class together.

    Layout:
        user:      <user_tmpl> <embed1> <sep> <embed2> ... <embedN>
        assistant: <asst_tmpl>

    Returns:
        (past_kv, info_dict)
    """
    device = llm.device
    dtype  = llm.model_dtype

    # Image embeddings: (N, num_patches, D)
    imgs = images.to(device)
    patches = img_tok(imgs).to(dtype)   # (N, num_patches, D)

    # Flatten all patches into a single sequence with optional sep embeds
    sep_emb = llm.embed_text(cfg.image_sep) if cfg.image_sep else None
    vis_parts = []
    for i in range(patches.shape[0]):
        vis_parts.append(patches[i:i+1])    # (1, num_patches, D)
        if sep_emb is not None and i < patches.shape[0] - 1:
            vis_parts.append(sep_emb)
    visual_embeds = torch.cat(vis_parts, dim=1)   # (1, total_tokens, D)

    user_text = cfg.teach_user_tmpl.format(class_name=class_name)
    asst_text = cfg.teach_asst_tmpl.format(class_name=class_name)

    turn_emb, positions = builder.build_teaching_turn(
        user_text=user_text,
        visual_embeds=visual_embeds,
        assistant_text=asst_text,
        is_first_turn=is_first_turn,
    )

    prev_len = kv_seq_len(past_kv)
    need_attn = isinstance(compressor, AttentionQueryCompressor)
    logits, past_kv, attn_maps = llm.prefill(
        turn_emb, past_kv=past_kv, use_cache=True, output_attentions=need_attn
    )

    # Absolute positions of visual tokens in the full KV sequence
    abs_vis = None
    if "visual" in positions:
        vs, ve = positions["visual"]
        abs_vis = (prev_len + vs, prev_len + ve)

    full_len = kv_seq_len(past_kv)
    past_kv = compressor.compress(past_kv, attention_maps=attn_maps, data_range=abs_vis)
    post_len = kv_seq_len(past_kv)

    return past_kv, {
        "tokens_added": full_len - prev_len,
        "tokens_after_compress": post_len,
        "visual_range": abs_vis,
    }


@torch.no_grad()
def teach_mixed(
    llm: LLMWrapper,
    builder: EmbedBuilder,
    img_tok: ImageTokenizer,
    batch_images: List[torch.Tensor],  # list of (C, H, W)
    batch_labels: List[str],           # corresponding class names
    past_kv,
    compressor: KVCacheCompressor,
    cfg: PromptConfig = DEFAULT_PROMPT_CFG,
    is_first_turn: bool = False,
) -> Tuple[object, Dict]:
    """
    'mixed' input mode: interleave visual embeds and class labels in one turn.

    Layout:
        user:      <intro> <embed1> label1 <sep> <embed2> label2 ...
        assistant: <mixed_asst_tmpl>
    """
    device = llm.device
    dtype  = llm.model_dtype

    stacked = torch.stack(batch_images).to(device)
    patches = img_tok(stacked).to(dtype)  # (N, num_patches, D)

    prefix = builder.chat.global_prefix if is_first_turn else builder.chat.turn_separator
    sep_emb = llm.embed_text(cfg.mixed_label_sep) if cfg.mixed_label_sep else None

    parts: List[torch.Tensor] = []

    def _add(x: torch.Tensor):
        x = x.to(device=device, dtype=dtype)
        if x.shape[1] > 0:
            parts.append(x)

    _add(llm.embed_text(prefix))
    if cfg.mixed_intro_tmpl:
        _add(llm.embed_text(cfg.mixed_intro_tmpl))

    vis_start = sum(p.shape[1] for p in parts)
    for i, (label, patch) in enumerate(zip(batch_labels, patches)):
        _add(patch.unsqueeze(0))                   # (1, num_patches, D)
        _add(llm.embed_text(label))
        if i < len(batch_labels) - 1 and sep_emb is not None:
            _add(sep_emb)
    vis_end = sum(p.shape[1] for p in parts)

    _add(llm.embed_text(builder.chat.user_to_asst))
    _add(llm.embed_text(cfg.mixed_asst_tmpl))
    _add(llm.embed_text(builder.chat.asst_tail))

    turn_emb = torch.cat(parts, dim=1)
    prev_len = kv_seq_len(past_kv)

    need_attn = isinstance(compressor, AttentionQueryCompressor)
    logits, past_kv, attn_maps = llm.prefill(
        turn_emb, past_kv=past_kv, use_cache=True, output_attentions=need_attn
    )

    abs_vis = (prev_len + vis_start, prev_len + vis_end)
    past_kv = compressor.compress(past_kv, attention_maps=attn_maps, data_range=abs_vis)
    post_len = kv_seq_len(past_kv)

    return past_kv, {
        "tokens_added": kv_seq_len(past_kv) - prev_len,
        "tokens_after_compress": post_len,
        "visual_range": abs_vis,
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
    cfg: PromptConfig = DEFAULT_PROMPT_CFG,
    is_first_turn: bool = False,
    guess_prob_gt: float = 0.0,
    guess_prob_random: float = 0.0,
) -> Tuple[object, bool, str]:
    """
    'guess-correct' mode for a single image.

    Dialogue (all within one accumulated KV):
        [turn 1] user: <visual embed>
                 assistant: <model's guess>
        [turn 2] user: <true class name>
                 assistant: "correct, <class>" or "wrong, it is <class>"

    Guess mode flags:
        guess_prob_gt: probability of assistant guessing true ground truth instead of argmax (0=argmax, 1=always true)
        guess_prob_random: probability of assistant guessing random label instead of argmax (0=argmax, 1=always random)

    Returns:
        (past_kv, was_correct, guessed_class_name)
    """
    device = llm.device
    dtype  = llm.model_dtype

    patches = img_tok(image.unsqueeze(0).to(device)).to(dtype)  # (1, num_patches, D)

    # -- Turn 1: show visual, ask model to predict --
    q_emb, _ = builder.build_query_turn(
        user_text=cfg.guess_user_tmpl,
        visual_embeds=patches,
        is_first_turn=is_first_turn,
    )
    logits, past_kv_q, _ = llm.prefill(q_emb, past_kv=past_kv, use_cache=True)
    next_logits = logits[0, -1, :]      # (vocab_size,)
    argmax_name = _argmax_class(llm, next_logits, class_naming)

    # Decide assistant guess:
    # 1. guess_prob_gt: interpolate between argmax and always true GT
    # 2. guess_prob_random: interpolate between argmax and always random label
    if guess_prob_gt > 0.0 and random.random() < guess_prob_gt:
        guessed_name = true_class_name
    elif guess_prob_random > 0.0 and random.random() < guess_prob_random:
        guessed_name = random.choice(class_naming.names)
    else:
        guessed_name = argmax_name

    # Append guessed_name as assistant response
    pred_emb = llm.embed_text(guessed_name + builder.chat.asst_tail)
    pred_emb = pred_emb.to(device=device, dtype=dtype)
    _, past_kv_q, _ = llm.prefill(pred_emb, past_kv=past_kv_q, use_cache=True)

    # -- Turn 2: user reveals true label, assistant acknowledges --
    was_correct = (guessed_name == true_class_name)
    if was_correct:
        asst_feedback = cfg.guess_correct_asst.format(class_name=true_class_name)
    else:
        asst_feedback = cfg.guess_wrong_asst.format(class_name=true_class_name)

    fb_emb, _ = builder.build_teaching_turn(
        user_text=cfg.guess_correct_user.format(class_name=true_class_name),
        visual_embeds=None,
        assistant_text=asst_feedback,
        is_first_turn=False,
    )
    _, past_kv_q, _ = llm.prefill(fb_emb, past_kv=past_kv_q, use_cache=True)

    return past_kv_q, was_correct, guessed_name


# ---------------------------------------------------------------------------
# 10. Classification
# ---------------------------------------------------------------------------

@torch.no_grad()
def classify_image(
    llm: LLMWrapper,
    builder: EmbedBuilder,
    img_tok: ImageTokenizer,
    image: torch.Tensor,       # (C, H, W)
    class_naming: ClassNaming,
    past_kv,
    cfg: PromptConfig = DEFAULT_PROMPT_CFG,
    answer_mode: str = "first_token",
) -> Tuple[str, Dict[str, float]]:
    """
    Classify a single test image given the accumulated KV cache.

    Parameters
    ----------
    answer_mode : 'first_token' | 'full_prob'
        first_token - compare softmax probability of first token of each class name.
        full_prob   - compare summed log-prob of full class name token sequence.

    Returns
    -------
    (predicted_class_name, {name: score})
    """
    device = llm.device
    dtype  = llm.model_dtype

    patches = img_tok(image.unsqueeze(0).to(device)).to(dtype)  # (1, N, D)

    q_emb, positions = builder.build_query_turn(
        user_text=cfg.test_user_tmpl,
        visual_embeds=patches,
        is_first_turn=(past_kv is None or kv_seq_len(past_kv) == 0),
    )

    if answer_mode == "first_token":
        logits, _, _ = llm.prefill(q_emb, past_kv=past_kv, use_cache=False)
        next_logits = logits[0, -1, :]     # (vocab_size,)
        probs = torch.softmax(next_logits, dim=-1)
        scores = {}
        for name in class_naming.names:
            tid = llm.tokenizer(name, return_tensors="pt").input_ids[0, 0].item()
            scores[name] = probs[tid].item()

    elif answer_mode == "full_prob":
        # Prefill context first, then score each class name sequence
        _, ctx_kv, _ = llm.prefill(q_emb, past_kv=past_kv, use_cache=True)
        scores = {}
        for name in class_naming.names:
            log_p = _sequence_log_prob(llm, name, ctx_kv)
            scores[name] = log_p
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    else:
        raise ValueError(f"Unknown answer_mode: {answer_mode}")

    predicted = max(scores, key=scores.get)
    return predicted, scores


def _argmax_class(llm: LLMWrapper, logits: torch.Tensor, class_naming: ClassNaming) -> str:
    """Given (vocab_size,) logits, pick the class whose first token has highest prob."""
    probs = torch.softmax(logits, dim=-1)
    best, best_p = class_naming.names[0], -1.0
    for name in class_naming.names:
        tid = llm.tokenizer(name, return_tensors="pt").input_ids[0, 0].item()
        p = probs[tid].item()
        if p > best_p:
            best_p = p
            best = name
    return best


@torch.no_grad()
def _sequence_log_prob(llm: LLMWrapper, text: str, ctx_kv) -> float:
    """
    Compute sum of log-probs of tokens in `text` conditioned on `ctx_kv`.
    """
    ids = llm.tokenizer(text, return_tensors="pt").input_ids.to(llm.device)
    emb = llm.model.get_input_embeddings()(ids).to(llm.model_dtype)

    log_p = 0.0
    cur_kv = ctx_kv
    for t in range(emb.shape[1]):
        tok_emb = emb[:, t:t+1]
        logits, cur_kv, _ = llm.prefill(tok_emb, past_kv=cur_kv, use_cache=True)
        log_p += torch.log_softmax(logits[0, 0], dim=-1)[ids[0, t]].item()
    return log_p


# ---------------------------------------------------------------------------
# 11. Dataset Utilities
# ---------------------------------------------------------------------------

def load_cifar10(data_dir: str, train: bool = True) -> torchvision.datasets.CIFAR10:
    transform = transforms.ToTensor()   # (3, 32, 32) in [0, 1]
    return torchvision.datasets.CIFAR10(
        root=data_dir, train=train, download=True, transform=transform
    )


def subset_per_class(
    dataset,
    n_per_class: int,
) -> Dict[int, List[torch.Tensor]]:
    """
    Return a dict mapping class_id -> list of image tensors (up to n_per_class each).
    Images are taken in dataset order (first N per class).
    """
    class_buckets: Dict[int, List] = {i: [] for i in range(10)}
    for img, label in dataset:
        if len(class_buckets[label]) < n_per_class:
            class_buckets[label].append(img)
        if all(len(v) >= n_per_class for v in class_buckets.values()):
            break
    return class_buckets


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

    # Model
    # Qwen/Qwen3-0.6B
    # Qwen/Qwen3-1.7B
    # Qwen/Qwen3-4B
    # Qwen/Qwen3-8B
    # HuggingFaceTB/SmolLM-135M
    # HuggingFaceTB/SmolLM3-3B
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
    p.add_argument("--kv-last", type=int, default=512,
                   help="StreamingLLM: number of recent tokens to keep")
    p.add_argument("--kv-topk", type=int, default=128,
                   help="AttentionQuery: top-k positions to keep")
    p.add_argument("--attractor-range", type=int, nargs=2, default=[-10, -1],
                   metavar=("START", "END"),
                   help="AttentionQuery: attractor query range (supports negatives)")
    p.add_argument("--neighbor-window", type=int, nargs=2, default=[0, 0],
                   metavar=("BEFORE", "AFTER"),
                   help="AttentionQuery: neighbor window around selected positions")

    # Input / teaching mode
    p.add_argument("--input-mode",
                   choices=["one-class", "mixed", "guess-correct"],
                   default="mixed")
    p.add_argument("--train-per-class", type=int, default=100,
                   help="Training images per class")
    p.add_argument("--test-per-class", type=int, default=10,
                   help="Test images per class")
    p.add_argument("--teaching-epochs", type=int, default=2,
                   help="Number of teaching passes over training data")
    p.add_argument("--batch-size", type=int, default=10,
                   help="Images per teaching batch (for mixed mode)")
    p.add_argument("--guess-prob-gt", type=float, default=0.9,
                   help="Probability of assistant guessing true ground truth instead of argmax (0=argmax, 1=always true GT)")
    p.add_argument("--guess-prob-random", type=float, default=0.1,
                   help="Probability of assistant guessing random label instead of argmax (0=argmax, 1=always random)")

    # Class names
    p.add_argument("--generic-class-names", action="store_true", default=False,
    # p.add_argument("--generic-class-names", action="store_true", default=False,
                   help="Use class1, class2, ... instead of real names")
    p.add_argument("--disable-instruction", action="store_true", default=False,
                   help="Disable the first-round instruction prompt turn")

    # Classification
    p.add_argument("--answer-mode", choices=["first_token", "full_prob"],
                   default="full_prob")

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
    test_ds  = load_cifar10(args.data_dir, train=False)

    train_by_class = subset_per_class(train_ds, args.train_per_class)
    test_by_class  = subset_per_class(test_ds,  args.test_per_class)

    logging.info(f"  Train: {sum(len(v) for v in train_by_class.values())} images")
    logging.info(f"  Test:  {sum(len(v) for v in test_by_class.values())} images")

    # -------------------------------------------------------------------
    # Class naming
    # -------------------------------------------------------------------
    class_naming = ClassNaming(CIFAR10_CLASSES, use_generic=args.generic_class_names)
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
            num_sink=args.kv_sink, keep_last=args.kv_last
        )
    elif args.kv_policy == "attn_query":
        compressor = AttentionQueryCompressor(
            top_k=args.kv_topk,
            attractor_range=tuple(args.attractor_range),
            neighbor_window=tuple(args.neighbor_window),
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
    if args.disable_instruction:
        prompt_cfg.use_instruction_turn = False

    if prompt_cfg.use_instruction_turn and prompt_cfg.instruction_user_prompt:
        logging.info("Prefilling first-round instruction prompt...")
        past_kv, info = prefill_instruction(llm, builder, past_kv, cfg=prompt_cfg)
        logging.info(f"  -> Instruction prefilled (+{info['tokens_added']} tokens), "
                     f"cache now: {kv_seq_len(past_kv)} tokens")

    for epoch in range(args.teaching_epochs):
        logging.info(f"\n-- Epoch {epoch + 1}/{args.teaching_epochs} --")

        if args.input_mode == "one-class":
            # Teach one class at a time (all images of that class together)
            for cls_id, cls_name_orig in enumerate(CIFAR10_CLASSES):
                cls_name = class_naming.name(cls_id)
                cls_disp = class_naming.display_name(cls_id)
                images   = train_by_class[cls_id]
                if not images:
                    continue

                images_tensor = torch.stack(images)   # (N, C, H, W)
                prev_len = kv_seq_len(past_kv)
                logging.info(f"  Teaching '{cls_disp}' ({len(images)} imgs), "
                             f"cache={prev_len}")

                past_kv, info = teach_one_class(
                    llm, builder, img_tok,
                    class_name=cls_name,
                    images=images_tensor,
                    past_kv=past_kv,
                    compressor=compressor,
                    cfg=prompt_cfg,
                    is_first_turn=(past_kv is None or kv_seq_len(past_kv) == 0),
                )
                logging.info(f"    -> +{info['tokens_added']} tokens, "
                             f"cache after compress: {info['tokens_after_compress']}")

        elif args.input_mode == "mixed":
            # Interleave images + labels in mini-batches
            import random
            all_imgs: List[torch.Tensor] = []
            all_lbls: List[str] = []
            for cls_id in range(len(CIFAR10_CLASSES)):
                for img in train_by_class[cls_id]:
                    all_imgs.append(img)
                    all_lbls.append(class_naming.name(cls_id))

            pairs = list(zip(all_imgs, all_lbls))
            random.shuffle(pairs)
            all_imgs_s, all_lbls_s = zip(*pairs) if pairs else ([], [])

            bs = args.batch_size
            is_first = (past_kv is None or kv_seq_len(past_kv) == 0)
            for i in range(0, len(all_imgs_s), bs):
                b_imgs = list(all_imgs_s[i:i+bs])
                b_lbls = list(all_lbls_s[i:i+bs])
                prev_len = kv_seq_len(past_kv)
                logging.info(f"  Mixed batch {i//bs + 1}: {len(b_imgs)} items, "
                             f"cache={prev_len}")
                past_kv, info = teach_mixed(
                    llm, builder, img_tok, b_imgs, b_lbls, past_kv, compressor,
                    cfg=prompt_cfg, is_first_turn=(is_first and i == 0),
                )
                logging.info(f"    -> cache after compress: {info['tokens_after_compress']}")

        elif args.input_mode == "guess-correct":
            # Per-image guess + correction dialogue
            all_imgs  = []
            all_cls_ids = []
            for cls_id in range(len(CIFAR10_CLASSES)):
                for img in train_by_class[cls_id]:
                    all_imgs.append(img)
                    all_cls_ids.append(cls_id)

            correct_count = 0
            is_first = (past_kv is None or kv_seq_len(past_kv) == 0)
            for i, (img, cls_id) in enumerate(zip(all_imgs, all_cls_ids)):
                cls_name = class_naming.name(cls_id)
                past_kv, was_correct, predicted = teach_guess_correct(
                    llm, builder, img_tok, img, cls_name, class_naming,
                    past_kv, cfg=prompt_cfg,
                    is_first_turn=(is_first and i == 0),
                    guess_prob_gt=args.guess_prob_gt,
                    guess_prob_random=args.guess_prob_random,
                )
                correct_count += int(was_correct)
                if (i + 1) % 10 == 0 or i == 0:
                    acc = correct_count / (i + 1) * 100
                    true_disp = class_naming.display_name(cls_id)
                    pred_disp = class_naming.display_name(predicted)
                    logging.info(
                        f"  img {i+1}/{len(all_imgs)}: "
                        f"true={true_disp}, pred={pred_disp} "
                        f"{'✓' if was_correct else '✗'} "
                        f"(running acc={acc:.1f}%, cache={kv_seq_len(past_kv)})"
                    )

    if past_kv is None or kv_seq_len(past_kv) == 0:
        logging.error("Teaching produced empty KV cache. Exiting.")
        return

    logging.info(f"\nTeaching complete. Final KV cache length: {kv_seq_len(past_kv)}")

    # -------------------------------------------------------------------
    # Testing Phase
    # -------------------------------------------------------------------
    logging.info("\n" + "=" * 70)
    logging.info("TESTING PHASE")
    logging.info("=" * 70)
    logging.info(f"Answer mode: {args.answer_mode}")

    all_true: List[str] = []
    all_pred: List[str] = []
    correct = 0
    total   = 0

    for cls_id in range(len(CIFAR10_CLASSES)):
        cls_name      = class_naming.name(cls_id)
        images        = test_by_class[cls_id]

        for img_idx, img in enumerate(images):
            try:
                pred_name, scores = classify_image(
                    llm, builder, img_tok, img,
                    class_naming=class_naming,
                    past_kv=past_kv,
                    cfg=prompt_cfg,
                    answer_mode=args.answer_mode,
                )
            except Exception as e:
                logging.error(f"Error classifying {cls_name} img {img_idx}: {e}")
                pred_name = class_naming.names[0]

            is_ok = pred_name == cls_name
            correct += int(is_ok)
            total   += 1
            all_true.append(cls_name)
            all_pred.append(pred_name)

            true_disp = class_naming.display_name(cls_id)
            pred_disp = class_naming.display_name(pred_name)
            logging.info(
                f"  [{total:4d}] true={true_disp:20s} pred={pred_disp:20s} "
                f"{'✓' if is_ok else '✗'}"
            )
            if args.log_level == "DEBUG":
                top = sorted(scores.items(), key=lambda x: -x[1])[:5]
                logging.debug(f"    top-5: {top}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    # -------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    logging.info("\n" + "=" * 70)
    logging.info("RESULTS")
    logging.info("=" * 70)
    logging.info(f"  Total:    {total}")
    logging.info(f"  Correct:  {correct}")
    logging.info(f"  Accuracy: {accuracy:.2f}%")
    logging.info(f"  KV policy:   {compressor}")
    logging.info(f"  Input mode:  {args.input_mode}")
    logging.info(f"  Answer mode: {args.answer_mode}")
    logging.info(f"  Image mode:  {args.img_mode}/{args.img_reduction}")

    # Per-class breakdown
    logging.info("\nPer-class accuracy:")
    class_correct: Dict[str, int] = {}
    class_total:   Dict[str, int] = {}
    for t, p in zip(all_true, all_pred):
        class_total[t]   = class_total.get(t, 0) + 1
        class_correct[t] = class_correct.get(t, 0) + int(t == p)
    for cls_id in range(len(class_naming.classes)):
        name = class_naming.name(cls_id)
        disp = class_naming.display_name(cls_id)
        n = class_total.get(name, 0)
        c = class_correct.get(name, 0)
        bar = "█" * c + "░" * (n - c)
        logging.info(f"  {disp:20s}: {c}/{n}  {bar}")

    # Confusion matrix
    if not args.no_confusion and HAS_MATPLOTLIB:
        try:
            disp_true = [class_naming.display_name(t) for t in all_true]
            disp_pred = [class_naming.display_name(p) for p in all_pred]
            disp_classes = [class_naming.display_name(i) for i in range(len(class_naming.classes))]
            plot_confusion_matrix(
                disp_true, disp_pred, disp_classes,
                log_dir / "confusion_matrix.png",
            )
        except Exception as e:
            logging.warning(f"Confusion matrix failed: {e}")


if __name__ == "__main__":
    main()

