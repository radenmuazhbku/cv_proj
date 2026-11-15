"""
Sequential in-context learning (ICL) with a persistent LLM KV cache.

Phase 1: explain the embedding/label format, provide initial labeled examples,
         provide one unlabeled embedding, and receive "Understood."
Phase 2: teach additional labeled examples in sequential batches.
Phase 3: evaluate each example using an isolated copy of the teaching KV cache.

The LLM is frozen. There is no optimizer, backpropagation, feedback, or reasoning.
Edit the uppercase constants and prompt strings below.
"""

import copy
import csv
import gc
import random
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================================
# USER-EDITABLE CONSTANTS
# =============================================================================

VERBOSE = True
MODEL_NAME = "Qwen/Qwen3-8B"
# MODEL_NAME = "Qwen/Qwen3-4B"
# MODEL_NAME = "Qwen/Qwen3-1.7B"
# MODEL_NAME = "Qwen/Qwen3-0.6B"
# MODEL_NAME = "HuggingFaceTB/SmolLM3-3B"
# MODEL_NAME = "HuggingFaceTB/SmolLM-135M"

DATA_DIR = "./datasets"
OUTPUT_DIR = "./logs/sequential_icl"

IMAGE_SIZE = 32
PATCH_SIZE = 16
IMAGE_MODE = "byte_embeds"       # "byte_embeds" or "floats"
IMAGE_REDUCTION = "sum"          # "sum", "avg", or "random_proj"

# CIFAR-10 uses one-token labels A through J.
CLASS_NAMES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

TRAIN_SUBSET_SIZE = 100         # Total number of CIFAR-10 train examples to use.
TEST_SAMPLES_PER_CLASS = 1      # Retained for optional test-split evaluation.
INITIAL_EXAMPLES_PER_CLASS = 1  # Examples per class included in Phase 1.
BATCH_SIZE = 5                  # Labeled examples per Phase-2 user message.
NUM_SUBSETS = 1                 # Split remaining training data into this many subsets.
TEACHING_EPOCHS = 3             # Total passes over all teaching subsets.
EPOCHS_PER_EVAL = 1             # Evaluate after this many teaching epochs.
SHUFFLE_SEED = 0                # None preserves the original CIFAR-10 train order.

# Original CIFAR-10 *training split* index to use for the final hash query.
# None selects one example randomly from the selected TRAIN_SUBSET_SIZE items.
EVAL_DATASET_INDEX = None

EVALUATE_TRAIN = True
EVALUATE_TEST = False
TRAIN_EVAL_SAMPLES_PER_CLASS = 0  # 0 means use all available training examples.
# TRAIN_EVAL_SAMPLES_PER_CLASS = 10  # 0 means use all available training examples.

# "first_token": one forward pass for the query; softmax over label token IDs.
# "full_prob": one forward pass per candidate label (e.g. 10 passes for A-J).
# "generate": free-run autoregressive generation; the generated text is the class label.
# PREDICTION_MODE = "first_token"
# PREDICTION_MODE = "full_prob"
PREDICTION_MODE = "generate"

# KV sink is deliberately large, so it should not truncate normal experiments.
KV_SINK = 1_000_000

# Editable natural-language instruction and message text.
INITIAL_INSTRUCTION = (
    "You are learning an image classification task through in-context examples. "
    "Each example is presented as the text embeds: followed by image "
    "embedding vectors, then a colon and its class label. There may be any "
    "number of labeled examples. The final item is an image embedding followed "
    "by a colon with no label. Infer its class from the preceding "
    "examples. The available class labels are: {labels}. "
    # "For now, do not classify the final item. Reply with exactly: Understood. "
    "For every later batch of labeled examples, do not acknowledge; simply "
    "receive them as additional context. Do not provide feedback, reasoning, "
    "or explanations. Reply immediately only with one of labels, {labels}"
)
INITIAL_ACK = "Understood."

# Empty thinking block used before assistant label generation.
# These are referenced by build_labeled_examples() for each teaching batch.
THINK_OPEN = "<think>\n"
THINK_CLOSE = "\n</think>\n\n"

EMBED_PREFIX = ""
LABEL_SEPARATOR = " : "
EXAMPLE_NEWLINE = "\n"

EVAL_PROMPT = ""

# SWAP_EMBED_WITH_HASH: when True, replace image embedding vectors with
# deterministic hash tokens derived from the ORIGINAL CIFAR-10 split index.
# The same index is used for teaching and evaluation, independent of batch order.
# This tests
# basic id:label recall — the LLM should attend to the hash token id and
# output the exact class label, without needing vision embeddings at all.
# SWAP_EMBED_WITH_HASH = False
SWAP_EMBED_WITH_HASH = True
HASH_SEPARATOR = "|"
PATCH_COUNT = (IMAGE_SIZE // PATCH_SIZE) ** 2

# Assistant answer prefix prefilled before every answer (first_token /
# full_prob / generate). Steers the model toward a concise class-label
# continuation instead of chatter or parroting "Understood.".
# ANSWER_PREFIX = "The class of the final item is: "
ANSWER_PREFIX = ""

# Temporary sanity prefill: before teaching starts, prefill one unlabeled
# query ("embeds: <image> :") against a temporary copy of the teaching cache
# and let the model generate freely. The assistant should reply with the
# class label only. Does not modify the persistent teaching cache.
# Set True to enable.
SANITY_PREFILL_ROUND = True
# SANITY_PREFILL_ROUND = False


def vprint(*args, **kwargs):
    """Print diagnostics only when VERBOSE is enabled."""
    if VERBOSE:
        print(*args, **kwargs)

def vprint(*args, **kwargs):
    """Print only when VERBOSE is enabled."""
    if VERBOSE:
        print(*args, **kwargs)


def kv_seq_len(cache):
    if cache is None:
        return 0
    if hasattr(cache, "get_seq_length"):
        return int(cache.get_seq_length())
    if isinstance(cache, (tuple, list)) and cache:
        return int(cache[0][0].shape[2])
    return 0


def apply_chat_template(tokenizer, messages, **kwargs):
    # Qwen3 supports enable_thinking=False. Other tokenizers may not.
    try:
        return tokenizer.apply_chat_template(
            messages, enable_thinking=False, **kwargs
        )
    except (TypeError, ValueError):
        return tokenizer.apply_chat_template(messages, **kwargs)
    
# =============================================================================
# CHAT-TAG EXTRACTION
# =============================================================================

def extract_chat_tags(tokenizer):
    """
    Extract raw chat-template strings using sentinel messages.

    Returned dictionary:
      global_prefix: first BOS/system/user-open segment
      user_to_asst: user-close + assistant-open segment
      asst_tail: assistant-close/EOS after assistant content
      turn_separator: assistant tail + next user-open between turns
      user_close, asst_open: split generation boundary where available
    """
    sentinels = {
        "u1": "\\x01USR1\\x01",
        "a1": "\\x01AST1\\x01",
        "u2": "\\x01USR2\\x01",
        "a2": "\\x01AST2\\x01",
    }
    messages = [
        {"role": "user", "content": sentinels["u1"]},
        {"role": "assistant", "content": sentinels["a1"]},
        {"role": "user", "content": sentinels["u2"]},
        {"role": "assistant", "content": sentinels["a2"]},
    ]

    full = apply_chat_template(
        tokenizer, messages, tokenize=False, add_generation_prompt=False
    )
    u1s = full.index(sentinels["u1"])
    a1s = full.index(sentinels["a1"])
    u2s = full.index(sentinels["u2"])
    a2s = full.index(sentinels["a2"])
    u1e = u1s + len(sentinels["u1"])
    a1e = a1s + len(sentinels["a1"])
    u2e = u2s + len(sentinels["u2"])
    a2e = a2s + len(sentinels["a2"])

    global_prefix = full[:u1s]
    user_to_asst = full[u2e:a2s]
    asst_tail = full[a2e:]
    turn_separator = full[a1e:u2s]

    one_turn = apply_chat_template(
        tokenizer,
        [{"role": "user", "content": sentinels["u2"]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    marker = one_turn.index(sentinels["u2"]) + len(sentinels["u2"])
    asst_open = one_turn[marker:]
    if asst_open and asst_open in user_to_asst:
        split_at = user_to_asst.rfind(asst_open)
        user_close = user_to_asst[:split_at]
    else:
        user_close, asst_open = "", user_to_asst

    tags = {
        "global_prefix": global_prefix,
        "user_to_asst": user_to_asst,
        "asst_tail": asst_tail,
        "turn_separator": turn_separator,
        "user_close": user_close,
        "asst_open": asst_open,
    }

    vprint("Extracted chat-template strings:")
    for name, value in tags.items():
        vprint(f"  {name:16s} = {value!r}")
    return tags




# =============================================================================
# MODEL AND EMBEDDING FUNCTIONS
# =============================================================================

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    embedding = model.get_input_embeddings()
    model_device = embedding.weight.device
    model_dtype = embedding.weight.dtype
    hidden_dim = model.config.hidden_size if MODEL_NAME != "google/gemma-4-E2B" else model.config.text_config.hidden_size

    vprint(f"Model: {MODEL_NAME}")
    vprint(f"Device: {model_device}; dtype: {model_dtype}; hidden size: {hidden_dim}")
    return model, tokenizer, model_device, model_dtype, hidden_dim
# from transformers import AutoProcessor, AutoModelForMultimodalLM

# processor = AutoProcessor.from_pretrained("google/gemma-4-E2B")
# model = AutoModelForMultimodalLM.from_pretrained("google/gemma-4-E2B", device_map="auto")

@torch.no_grad()
def embed_text(model, tokenizer, text, device, dtype, add_special_tokens=False):
    if not text:
        return torch.empty((1, 0, model.config.hidden_size), device=device, dtype=dtype)
    ids = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=add_special_tokens,
    ).input_ids.to(device)
    return model.get_input_embeddings()(ids).to(dtype=dtype)


class ImageTokenizer(nn.Module):
    """Torch module: image patches -> LLM-width continuous embedding vectors."""

    def __init__(self, patch_size, img_size, mode, reduction, llm_dim, vocab_embeds):
        super().__init__()
        assert img_size % patch_size == 0
        assert mode in ("floats", "byte_embeds")
        assert reduction in ("sum", "avg", "random_proj")
        self.patch_size = patch_size
        self.img_size = img_size
        self.mode = mode
        self.reduction = reduction
        self.llm_dim = llm_dim
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid
        self.patch_channels = patch_size * patch_size * 3

        if mode == "byte_embeds":
            self.register_buffer("byte_table", vocab_embeds[:256].detach().clone().float())
        else:
            self.register_buffer("byte_table", None)

        if reduction == "random_proj":
            self.proj = nn.Linear(
                self.patch_channels, llm_dim,
                bias=(mode == "floats"),
            )
        elif mode == "floats":
            self.register_buffer(
                "fixed_weight",
                torch.randn(self.patch_channels, llm_dim) * (llm_dim ** -0.5),
            )

    def forward(self, images):
        batch = images.shape[0]
        patches = (
            images.unfold(2, self.patch_size, self.patch_size)
            .unfold(3, self.patch_size, self.patch_size)
            .contiguous()
            .view(batch, 3, self.grid, self.grid, -1)
            .permute(0, 2, 3, 1, 4)
            .contiguous()
            .view(batch, self.num_patches, self.patch_channels)
        )

        if self.mode == "floats":
            if self.reduction == "random_proj":
                return self.proj(patches)
            output = patches @ self.fixed_weight
            return output / self.patch_channels if self.reduction == "avg" else output

        byte_values = (patches * 255.0).long().clamp(0, 255)
        vectors = self.byte_table[byte_values]
        if self.reduction == "sum":
            return vectors.sum(dim=2)
        if self.reduction == "avg":
            return vectors.mean(dim=2)
        return torch.einsum("cd,bncd->bnd", self.proj.weight.T, vectors)



# =============================================================================
# SEQUENTIAL MESSAGE BUILDING
# =============================================================================

def image_to_embeds(image_tokenizer, image, device, dtype):
    return image_tokenizer(image.unsqueeze(0).to(device)).to(dtype=dtype)


def index_to_hash_embeds(model, tokenizer, image_index, patch_count, device, dtype):
    """Generate deterministic hash token embeddings from image dataset index.

    The hash is derived from image_index, truncated to patch_count characters,
    with HASH_SEPARATOR between chars to prevent tokenizer BPE merging.
    Returns token embeddings that replace image vectors in the prefill.
    """
    import hashlib
    hash_str = hashlib.sha256(str(image_index).encode()).hexdigest()
    hash_str = hash_str[:patch_count]
    separated = HASH_SEPARATOR.join(hash_str)
    return embed_text(model, tokenizer, separated, device, dtype)


def append_text(parts, model, tokenizer, text, device, dtype):
    if text:
        parts.append(embed_text(model, tokenizer, text, device, dtype))


def build_labeled_examples(model, tokenizer, tags, image_tokenizer, examples,
                           device, dtype, first_turn=False):
    """Build assistant-turn prefill for labeled teaching examples:

        embeds: <vectors1> : label1
        embeds: <vectors2> : label2

    Everything is assistant-turn prefill; no "Understood." between batches.
    For Phase 2 batches, prefix includes turn_separator + user_to_asst to
    switch from the previous turn into the assistant turn.
    """
    prefix = tags["global_prefix"] if first_turn else tags["turn_separator"] + tags["user_to_asst"]
    parts = []
    append_text(parts, model, tokenizer, prefix, device, dtype)
    # Empty think tags: skip thinking, go straight to labels
    append_text(parts, model, tokenizer, THINK_OPEN, device, dtype)
    append_text(parts, model, tokenizer, THINK_CLOSE, device, dtype)

    for image, label, dataset_index in examples:
        append_text(parts, model, tokenizer, "\n" + EMBED_PREFIX, device, dtype)
        if SWAP_EMBED_WITH_HASH:
            parts.append(index_to_hash_embeds(
                model, tokenizer, dataset_index, PATCH_COUNT, device, dtype,
            ))
        else:
            parts.append(image_to_embeds(image_tokenizer, image, device, dtype))
        append_text(
            parts, model, tokenizer,
            LABEL_SEPARATOR + str(label) + EXAMPLE_NEWLINE,
            device, dtype,
        )

    return torch.cat([part for part in parts if part.shape[1]], dim=1)

def build_initial_turn(model, tokenizer, tags, image_tokenizer, instruction,
                       initial_examples, device, dtype):
    """
    Phase 1: instruction + N labeled examples.
    Force-prefill INITIAL_ACK ("Understood.") exactly once after the initial
    user round (instruction + format). No unlabeled query is included here,
    so the model never learns "embeds: <embed> : -> Understood." and is free
    to generate the class label during evaluation/sanity rounds.
    """
    parts = []
    append_text(parts, model, tokenizer, tags["global_prefix"], device, dtype)
    append_text(parts, model, tokenizer, instruction, device, dtype)

    for image, label, dataset_index in initial_examples:
        append_text(parts, model, tokenizer, "\n" + EMBED_PREFIX, device, dtype)
        if SWAP_EMBED_WITH_HASH:
            parts.append(index_to_hash_embeds(
                model, tokenizer, dataset_index, PATCH_COUNT, device, dtype,
            ))
        else:
            parts.append(image_to_embeds(image_tokenizer, image, device, dtype))
        append_text(
            parts, model, tokenizer,
            LABEL_SEPARATOR + str(label) + EXAMPLE_NEWLINE,
            device, dtype,
        )

    # Force-prefill the single acknowledgment after the initial user round.
    append_text(parts, model, tokenizer, tags["user_to_asst"], device, dtype)
    append_text(parts, model, tokenizer, INITIAL_ACK, device, dtype)
    append_text(parts, model, tokenizer, tags["asst_tail"], device, dtype)
    return torch.cat([part for part in parts if part.shape[1]], dim=1)


def build_eval_query(model, tokenizer, tags, image_tokenizer, image,
                     dataset_index, past_kv, device, dtype):
    """Build a query using the same original dataset index as its teaching example."""
    parts = []
    append_text(parts, model, tokenizer, tags["turn_separator"], device, dtype)
    append_text(parts, model, tokenizer, EVAL_PROMPT, device, dtype)
    if SWAP_EMBED_WITH_HASH:
        parts.append(index_to_hash_embeds(
            model, tokenizer, dataset_index, PATCH_COUNT, device, dtype,
        ))
    else:
        parts.append(image_to_embeds(image_tokenizer, image, device, dtype))
    append_text(parts, model, tokenizer, LABEL_SEPARATOR, device, dtype)
    append_text(parts, model, tokenizer, tags["user_close"], device, dtype)
    append_text(parts, model, tokenizer, tags["asst_open"], device, dtype)
    return torch.cat([part for part in parts if part.shape[1]], dim=1)


# =============================================================================
# KV CACHE AND LABEL SCORING
# =============================================================================

def kv_seq_len(cache):
    if cache is None:
        return 0
    if hasattr(cache, "get_seq_length"):
        return int(cache.get_seq_length())
    if isinstance(cache, (tuple, list)) and cache:
        return int(cache[0][0].shape[2])
    return 0


def apply_kv_sink(cache):
    if cache is None or kv_seq_len(cache) <= KV_SINK:
        return cache
    indices = torch.arange(KV_SINK, device="cpu")
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        from transformers import DynamicCache
        kept = DynamicCache()
        for key, value in zip(cache.key_cache, cache.value_cache):
            kept.key_cache.append(key[:, :, indices.to(key.device), :])
            kept.value_cache.append(value[:, :, indices.to(value.device), :])
        return kept
    return tuple(
        (key[:, :, indices.to(key.device), :], value[:, :, indices.to(value.device), :])
        for key, value in cache
    )


@torch.no_grad()
def model_prefill(model, embeds, past_kv=None):
    output = model(
        inputs_embeds=embeds,
        past_key_values=past_kv,
        use_cache=True,
        return_dict=True,
    )
    return output.logits, output.past_key_values


def get_label_token_id(tokenizer, label):
    ids = tokenizer(" " + label, add_special_tokens=False).input_ids
    if len(ids) != 1:
        raise ValueError(
            f"Label {label!r} must be exactly one token for this scoring mode; got {ids}"
        )
    return int(ids[0])


@torch.no_grad()
def predict_first_token(model, tokenizer, query_embeds, main_cache, class_names):
    """Single query forward pass; normalize only across the class-token logits."""
    output = model(
        inputs_embeds=query_embeds,
        past_key_values=copy.deepcopy(main_cache),
        use_cache=True,
        return_dict=True,
    )
    logits = output.logits[0, -1].float()
    ids = torch.tensor(
        [get_label_token_id(tokenizer, name) for name in class_names],
        device=logits.device,
    )
    probabilities = torch.softmax(logits[ids], dim=0)
    return {
        name: float(probability)
        for name, probability in zip(class_names, probabilities.tolist())
    }


@torch.no_grad()
def predict_full_prob(model, tokenizer, query_embeds, main_cache, class_names):
    """
    Score each candidate independently. For CIFAR-10 this performs 10 model
    forward passes, one with each candidate label token appended to the query.
    The probability is the candidate token's next-token probability.
    """
    scores = {}
    for label in class_names:
        token_id = get_label_token_id(tokenizer, label)
        token = torch.tensor([[token_id]], device=query_embeds.device)
        token_embed = model.get_input_embeddings()(token).to(query_embeds.dtype)

        # Every candidate starts from a fresh copy of the same teaching cache.
        candidate_input = torch.cat([query_embeds, token_embed], dim=1)
        output = model(
            inputs_embeds=candidate_input,
            past_key_values=copy.deepcopy(main_cache),
            use_cache=False,
            return_dict=True,
        )
        query_last_position = query_embeds.shape[1] - 1
        candidate_log_probability = torch.log_softmax(
            output.logits[0, query_last_position].float(), dim=-1
        )[token_id]
        scores[label] = float(candidate_log_probability.exp().item())

    total = sum(scores.values())
    if total > 0:
        scores = {label: value / total for label, value in scores.items()}
    return scores


@torch.no_grad()
def predict_one(model, tokenizer, tags, image_tokenizer, image, dataset_index,
                main_cache, class_names, device, dtype):
    query = build_eval_query(
        model, tokenizer, tags, image_tokenizer, image, dataset_index,
        main_cache, device, dtype,
    )
    # Prepend the answer prefix as assistant text so the model continues
    # from it with a concise class label instead of chatter or parroting.
    answer_prefix_embed = embed_text(
        model, tokenizer, ANSWER_PREFIX, device, dtype
    )
    answer_query = torch.cat([query, answer_prefix_embed], dim=1)
    scores = None
    if PREDICTION_MODE == "first_token":
        scores = predict_first_token(
            model, tokenizer, answer_query, main_cache, class_names
        )
        prediction = max(scores, key=scores.get)
    elif PREDICTION_MODE == "full_prob":
        scores = predict_full_prob(
            model, tokenizer, answer_query, main_cache, class_names
        )
        prediction = max(scores, key=scores.get)
    elif PREDICTION_MODE == "generate":
        # Free-run autoregressive generation: the assistant's response is the
        # generated class label directly, continuing from the answer prefix.
        logits, gen_cache = model_prefill(
            model, answer_query, past_kv=copy.deepcopy(main_cache),
        )
        prediction, _ = decode_reasoning(
            model, tokenizer, gen_cache, logits[0, -1].float(),
            device, dtype,
        )
        prediction = prediction.strip()
    else:
        raise ValueError(f"Unknown PREDICTION_MODE: {PREDICTION_MODE}")

    if VERBOSE:
        vprint("Temporary evaluation query:")
        vprint(repr(
            tags["turn_separator"] + EVAL_PROMPT
            + "<embed>" + LABEL_SEPARATOR
            + tags["user_close"] + tags["asst_open"]
            + ANSWER_PREFIX
        ))
        if scores is not None:
            vprint("Candidate probabilities:")
            for label, probability in sorted(
                scores.items(), key=lambda item: item[1], reverse=True
            ):
                vprint(f"  {label}: {probability:.8f}")
        vprint("Predicted:", prediction)
    return prediction, scores


# =============================================================================
# DATA AND TEACHING
# =============================================================================

def load_data():
    """Load a shuffled-or-ordered subset from CIFAR-10's official train split.

    Every returned pair is (transformed_image, A-J_label, original_dataset_index).
    Labels are read directly from train_set.targets, never inferred from ordering.
    """
    transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
    ])
    train_set = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=transform
    )

    train_indices = list(range(len(train_set)))
    if SHUFFLE_SEED is not None:
        random.Random(SHUFFLE_SEED).shuffle(train_indices)
    # With SHUFFLE_SEED=None, this is simply indices 0, 1, 2, ... .
    train_indices = train_indices[:min(TRAIN_SUBSET_SIZE, len(train_indices))]

    train_pairs = []
    for dataset_index in tqdm(
        train_indices, desc="Loading CIFAR-10 train subset", disable=not VERBOSE
    ):
        image, _ = train_set[dataset_index]
        class_id = int(train_set.targets[dataset_index])
        train_pairs.append((image, CLASS_NAMES[class_id], dataset_index))

    # Preserve the original script's small optional test set: first N examples
    # per class, with disjoint global IDs so hashes cannot overlap the train split.
    test_counts = [0] * len(CLASS_NAMES)
    test_pairs = []
    for test_index, target in enumerate(test_set.targets):
        class_id = int(target)
        if test_counts[class_id] >= TEST_SAMPLES_PER_CLASS:
            continue
        image, _ = test_set[test_index]
        test_pairs.append((image, CLASS_NAMES[class_id], len(train_set) + test_index))
        test_counts[class_id] += 1
        if all(count >= TEST_SAMPLES_PER_CLASS for count in test_counts):
            break

    print(f"CIFAR-10 train subset: {len(train_pairs)} / {len(train_set)} examples")
    print(f"Train subset order: {'shuffled' if SHUFFLE_SEED is not None else 'default dataset order'}")
    vprint("CIFAR-10 class-to-label mapping:")
    for class_id, class_name in enumerate(train_set.classes):
        vprint(f"  {class_id}: {class_name} -> {CLASS_NAMES[class_id]}")
    return train_pairs, test_pairs


def split_subsets(pairs, number, seed):
    """Split pairs; seed=None preserves the incoming order."""
    ordered = list(pairs)
    if seed is not None:
        random.Random(seed).shuffle(ordered)
    number = max(1, min(number, len(ordered)))
    subsets = [[] for _ in range(number)]
    for index, pair in enumerate(ordered):
        subsets[index % number].append(pair)
    return subsets


def teach_batch(model, tokenizer, tags, image_tokenizer, batch,
                main_cache, device, dtype):
    """Append one batch to the persistent teaching conversation."""
    turn = build_labeled_examples(
        model, tokenizer, tags, image_tokenizer, batch,
        device, dtype, first_turn=False,
    )
    _, updated_cache = model_prefill(model, turn, main_cache)
    return apply_kv_sink(updated_cache)


@torch.no_grad()
def evaluate(model, tokenizer, tags, image_tokenizer, pairs,
             main_cache, device, dtype, description):
    correct = 0
    total = 0
    per_class = {label: [0, 0] for label in CLASS_NAMES}

    progress = tqdm(pairs, desc=description, disable=not VERBOSE)
    for image, true_label, dataset_index in progress:
        # predict_one only reads a copy of the persistent main cache.
        try:
            prediction, _ = predict_one(
                model, tokenizer, tags, image_tokenizer, image, dataset_index,
                main_cache, CLASS_NAMES, device, dtype,
            )
        except Exception as exc:
            vprint(f"Prediction failed: {exc}")
            prediction = ""

        total += 1
        is_correct = prediction == true_label
        correct += int(is_correct)
        per_class[true_label][1] += 1
        per_class[true_label][0] += int(is_correct)
        progress.set_postfix(accuracy=f"{100 * correct / total:.2f}%")

    accuracy = 100.0 * correct / max(1, total)
    vprint(f"{description}: accuracy = {accuracy:.2f}% ({correct}/{total})")
    for label, (class_correct, class_total) in per_class.items():
        if class_total:
            vprint(
                f"  {label}: {class_correct}/{class_total} "
                f"({100 * class_correct / class_total:.1f}%)"
            )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return accuracy


MAX_NEW_TOKENS = 100
# TEMPERATURE = 1.0
TEMPERATURE = 0.0
TOP_P = 0.9
def decode_reasoning(model, tokenizer, past_kv, next_logits, device, dtype):
    generated = []
    current_logits = next_logits.float()
    current_cache = past_kv
    eos_id = tokenizer.eos_token_id
    nl_id = tokenizer("\n", add_special_tokens=False).input_ids
    nl_id = nl_id[0] if nl_id else None

    if VERBOSE:
        topk = torch.topk(current_logits, 10)
        vprint("Top-10 logits for next token:")
        for tok_id, logit_val in zip(topk.indices.tolist(), topk.values.tolist()):
            tok_str = tokenizer.decode([tok_id])
            vprint(f"  {tok_str!r} ({tok_id}): {logit_val:.4f}")

    for _ in range(MAX_NEW_TOKENS):
        if TEMPERATURE <= 0:
            next_id = int(torch.argmax(current_logits).item())
        else:
            probs = torch.softmax(current_logits / TEMPERATURE, dim=-1)
            sorted_probs, sorted_ids = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=0)
            remove = cumulative - sorted_probs >= TOP_P
            sorted_probs[remove] = 0
            filtered = torch.zeros_like(probs).scatter(0, sorted_ids, sorted_probs)
            filtered /= filtered.sum()
            next_id = int(torch.multinomial(filtered, 1).item())

        if eos_id is not None and next_id == eos_id:
            break
        generated.append(next_id)
        token = torch.tensor([[next_id]], device=device)
        token_embeds = model.get_input_embeddings()(token).to(dtype)
        logits, current_cache = model_prefill(
            model, token_embeds, current_cache
        )
        current_logits = logits[0, -1].float()

    # vprint(f" GEN\n{generated}\n--")

    return tokenizer.decode(generated, skip_special_tokens=True), current_cache
def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, device, dtype, hidden_dim = load_model()
    tags = extract_chat_tags(tokenizer)

    # Ensure class labels are single-token candidates for the two scoring modes.
    for label in CLASS_NAMES:
        get_label_token_id(tokenizer, label)

    image_tokenizer = ImageTokenizer(
        PATCH_SIZE, IMAGE_SIZE, IMAGE_MODE, IMAGE_REDUCTION,
        hidden_dim, model.get_input_embeddings().weight,
    ).to(device=device)

    train_pairs, test_pairs = load_data()

    # Select the first configured number of examples per class for Phase 1.
    # Keep the original dataset index attached to every image throughout.
    initial_examples = []
    initial_indices = set()
    initial_counts = {label: 0 for label in CLASS_NAMES}
    # One pass over the selected list preserves its current order (shuffled
    # when SHUFFLE_SEED is set, original CIFAR order when it is None).
    for pair in train_pairs:
        image, label, dataset_index = pair
        if initial_counts[label] < INITIAL_EXAMPLES_PER_CLASS:
            initial_examples.append(pair)
            initial_indices.add(dataset_index)
            initial_counts[label] += 1

    remaining_pairs = [pair for pair in train_pairs if pair[2] not in initial_indices]
    subsets = split_subsets(remaining_pairs, NUM_SUBSETS, SHUFFLE_SEED)

    # Pick the final hash-query example from the same 100-item training subset,
    # so its label/hash mapping has already appeared in the teaching context.
    if EVAL_DATASET_INDEX is None:
        # Independent random choice on each run; SHUFFLE_SEED only controls
        # the ordering/subset selection, not this final query choice.
        eval_pair = random.choice(train_pairs)
    else:
        matches = [pair for pair in train_pairs if pair[2] == EVAL_DATASET_INDEX]
        if not matches:
            raise ValueError(
                f"EVAL_DATASET_INDEX={EVAL_DATASET_INDEX} is not in the selected "
                f"CIFAR-10 train subset. Set it to one of the selected indices "
                "or use None for random selection."
            )
        eval_pair = matches[0]

    eval_image, eval_label, eval_dataset_index = eval_pair
    print(
        "Final hash-evaluation CIFAR-10 train index: "
        f"{eval_dataset_index} (label={eval_label})"
    )

    # The final unlabeled query is evaluated after all selected training
    # examples have been added to the teaching context.
    instruction = INITIAL_INSTRUCTION.format(labels=", ".join(CLASS_NAMES))

    if VERBOSE:
        vprint("\nPHASE 1 — INITIAL PREFILL")
        vprint("  > USER instruction:")
        vprint(instruction)
        vprint("  > USER examples:")
        for _, label, dataset_index in initial_examples:
            vprint(f"{EMBED_PREFIX}<hash:index={dataset_index}>{LABEL_SEPARATOR}{label}")
        vprint("  > ASSISTANT:", INITIAL_ACK)

    initial_turn = build_initial_turn(
        model, tokenizer, tags, image_tokenizer, instruction,
        initial_examples, device, dtype,
    )
    _, main_cache = model_prefill(model, initial_turn)
    main_cache = apply_kv_sink(main_cache)
    vprint(f"Phase-1 persistent KV length: {kv_seq_len(main_cache)}")


    history = []
    teaching_epoch = 0
    while teaching_epoch < TEACHING_EPOCHS:
        teaching_epoch += 1
        vprint(
            f"\n{'=' * 20} TEACHING EPOCH "
            f"{teaching_epoch}/{TEACHING_EPOCHS} {'=' * 20}"
        )

        for subset_index, subset in enumerate(subsets, start=1):
            subset_pairs = list(subset)
            if SHUFFLE_SEED is not None:
                random.Random(
                    SHUFFLE_SEED + teaching_epoch * 1000 + subset_index
                ).shuffle(subset_pairs)

            vprint(
                f"Subset {subset_index}/{len(subsets)}: "
                f"{len(subset_pairs)} examples"
            )
            for start in tqdm(
                range(0, len(subset_pairs), BATCH_SIZE),
                desc=f"Teaching subset {subset_index}", disable=not VERBOSE,
            ):
                batch = subset_pairs[start:start + BATCH_SIZE]
                if VERBOSE:
                    vprint(f"\nPHASE 2 — TEACHING BATCH ({len(batch)} examples)")
                    for _, label, dataset_index in batch:
                        vprint(f"{EMBED_PREFIX}<hash:index={dataset_index}>{LABEL_SEPARATOR}{label}")
                    # No assistant acknowledgment after each teaching batch.

                main_cache = teach_batch(
                    model, tokenizer, tags, image_tokenizer, batch,
                    main_cache, device, dtype,
                )

        vprint(f"Persistent KV length after teaching: {kv_seq_len(main_cache)}")

        should_evaluate = (
            teaching_epoch % max(1, EPOCHS_PER_EVAL) == 0
            or teaching_epoch == TEACHING_EPOCHS
        )
        if not should_evaluate:
            continue

        train_accuracy = None
        test_accuracy = None

        if EVALUATE_TRAIN:
            train_eval_pairs = list(train_pairs)
            if TRAIN_EVAL_SAMPLES_PER_CLASS > 0:
                train_eval_pairs = []
                for label in CLASS_NAMES:
                    class_examples = [pair for pair in train_pairs if pair[1] == label]
                    train_eval_pairs.extend(class_examples[:TRAIN_EVAL_SAMPLES_PER_CLASS])
            vprint("\nPHASE 3 — TRAIN ACCURACY (temporary cache per example)")
            train_accuracy = evaluate(
                model, tokenizer, tags, image_tokenizer, train_eval_pairs,
                main_cache, device, dtype,
                f"Epoch {teaching_epoch} train",
            )

        if EVALUATE_TEST:
            vprint("\nPHASE 3 — TEST ACCURACY (temporary cache per example)")
            test_accuracy = evaluate(
                model, tokenizer, tags, image_tokenizer, test_pairs,
                main_cache, device, dtype,
                f"Epoch {teaching_epoch} test",
            )

        history.append({
            "epoch": teaching_epoch,
            "kv_len": kv_seq_len(main_cache),
            "train_acc": train_accuracy,
            "test_acc": test_accuracy,
        })

        vprint(
            f"Epoch {teaching_epoch}: train={train_accuracy}, "
            f"test={test_accuracy}, persistent_kv={kv_seq_len(main_cache)}"
        )

    # Final hash-recall sanity query. This is deliberately after all 100
    # selected training examples have been prefetched into the persistent cache.
    if SANITY_PREFILL_ROUND:
        vprint("\nFINAL HASH SANITY EVALUATION — temporary cache copy")
        vprint(f"  CIFAR-10 train index: {eval_dataset_index}")
        vprint(f"  reference label     : {eval_label!r}")
        sanity_query = build_eval_query(
            model, tokenizer, tags, image_tokenizer, eval_image,
            eval_dataset_index, None, device, dtype,
        )
        sanity_answer_prefix = embed_text(
            model, tokenizer, ANSWER_PREFIX, device, dtype
        )
        sanity_answer_query = torch.cat(
            [sanity_query, sanity_answer_prefix], dim=1
        )
        sanity_logits, sanity_cache = model_prefill(
            model, sanity_answer_query, past_kv=copy.deepcopy(main_cache),
        )
        sanity_generated, _ = decode_reasoning(
            model, tokenizer, sanity_cache,
            sanity_logits[0, -1].float(), device, dtype,
        )
        vprint(f"  generated label     : {sanity_generated.strip()!r}")
        vprint(f"  exact match         : {sanity_generated.strip() == eval_label}")

    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["epoch", "kv_len", "train_acc", "test_acc"]
        )
        writer.writeheader()
        writer.writerows(history)
    vprint(f"\nSaved metrics: {metrics_path}")
    vprint("Done.")


if __name__ == "__main__":
    main()
