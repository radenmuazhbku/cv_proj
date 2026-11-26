"""CIFAR-10 ICL classification using K-means patch IDs as existing LLM tokens.

The vocabulary is never extended: selected existing token IDs stand for visual
patch centroids. The shared codebook is fitted across all patch positions.

Run from the repository root (adjust MODEL_NAME and other constants below):

    .venv/bin/python exps/cifar10_patch/cifar_classify_icl.py

Example configurations (edit the constants in CONFIG before running):

    # Qwen3, 2x2 patches, 128 centroids, full configured demonstration set
    MODEL_NAME = "Qwen/Qwen3-8B"
    PATCH_SIZE = 16
    KMEANS_CENTROIDS = 128
    TRAIN_SUBSET_SIZE = 100
    EVAL_SUBSET_SIZE = None

    # SmolLM3, 4x4 patches, 64 centroids, quick evaluation on 20 queries
    MODEL_NAME = "HuggingFaceTB/SmolLM3-3B"
    PATCH_SIZE = 8
    KMEANS_CENTROIDS = 64
    TRAIN_SUBSET_SIZE = 100
    EVAL_SUBSET_SIZE = 20
    ENABLE_REASONING = False

    # Coarser patches, larger codebook, smaller demonstration set
    MODEL_NAME = "HuggingFaceTB/SmolLM3-3B"
    PATCH_SIZE = 32
    KMEANS_CENTROIDS = 256
    TRAIN_SUBSET_SIZE = 50
    EVAL_SUBSET_SIZE = 10
    ENABLE_REASONING = False

PATCH_SIZE must divide IMAGE_SIZE. The default shared codebook is used for all
spatial positions. K-means fits on the configured training demonstrations, so
increase TRAIN_SUBSET_SIZE to expose it to more CIFAR patches. For a fixed
contiguous vocabulary interval, set CENTROID_TOKEN_START to its first token ID;
otherwise the selector searches for a printable interval automatically.
"""

import copy
import random
import time

import torch
import torchvision
import torchvision.transforms as T
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# =============================================================================
# CONFIG
# =============================================================================

# Supported choices: "Qwen/Qwen3-8B" or "HuggingFaceTB/SmolLM3-3B".
# Qwen3 is the original sanity-check model; SmolLM3-3B is a smaller alternative.
MODEL_NAME = "Qwen/Qwen3-8B"
DATA_DIR = "./datasets"

TRAIN_SUBSET_SIZE = 100
SHUFFLE_SEED = 0          # None = original CIFAR-10 train dataset order
EVAL_DATASET_INDEX = 50   # Index into the selected subset for single-sample diagnostics

# K-means tokenizer. PATCH_SIZE must divide IMAGE_SIZE. All positions share
# this codebook by default. CENTROID_TOKEN_START selects a contiguous usable
# token range; set None to use the vocabulary filter's first eligible range.
PATCH_SIZE = 16
KMEANS_CENTROIDS = 128
KMEANS_SAMPLES = 100000
KMEANS_ITERS = 50
CENTROID_TOKEN_START = None
VOCAB_FILTER = "printable"  # "printable" or stub selector "unicode_script"

# Eval sanity train:
# True  = evaluate accuracy across all examples in the trainset and report final accuracy.
# False = evaluate only a single query sample.
EVAL_SANITY_TRAIN = True

# Subset of trainset for accelerated evaluation:
# None or 0 = evaluate on all examples in TRAIN_SUBSET_SIZE.
# Else integer > 0 = random subset of size EVAL_SUBSET_SIZE (must be <= TRAIN_SUBSET_SIZE).
EVAL_SUBSET_SIZE = None

IMAGE_SIZE = 32
PATCH_SIZE = 16
PATCH_COUNT = (IMAGE_SIZE // PATCH_SIZE) ** 2

CLASS_NAMES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
VERBOSE = True
TOP_K = 20

# Reasoning configuration:
# ENABLE_REASONING = False
ENABLE_REASONING = True
MAX_REASONING_TOKENS = 512
TEMPERATURE = 0.6          # 0.6 - 0.7 recommended for Qwen reasoning models
TOP_P = 0.95               # Recommended top-p for reasoning

EMBED_PREFIX = ""
LABEL_SEPARATOR = " : "
INITIAL_ACK = "Understood."
THINK_OPEN = "<think>\n"
THINK_CLOSE = "\n</think>\n\n"

INITIAL_INSTRUCTION = (
    "You will see examples of image patch-token sequences followed by their class label. "
    "The final sequence is a query image. Find its class from the examples and answer "
    "with the class label only. "
    "Answer immediately with the class label and nothing else."
    "Available labels: "
    + ", ".join(CLASS_NAMES)
    + "."
)


# =============================================================================
# PATCH K-MEANS AND EXISTING-VOCAB SELECTION
# =============================================================================
def patchify(images, patch_size):
    n, c, h, w = images.shape
    if h % patch_size or w % patch_size:
        raise ValueError(f"Image shape {h}x{w} must divide patch size {patch_size}")
    grid = h // patch_size
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    return patches.permute(0, 2, 3, 1, 4, 5).contiguous().reshape(
        n, grid * grid, c * patch_size * patch_size
    )


def squared_distances(x, centers):
    return (x.square().sum(1, keepdim=True) + centers.square().sum(1)[None] - 2 * x @ centers.T).clamp_min(0)


@torch.no_grad()
def fit_patch_codebook(images, patch_size, k, sample_limit, n_iters, device, seed):
    patches = patchify(images, patch_size).reshape(-1, 3 * patch_size * patch_size).to(device)
    if len(patches) > sample_limit:
        gen = torch.Generator(device=device).manual_seed(seed)
        patches = patches[torch.randperm(len(patches), generator=gen, device=device)[:sample_limit]]
    k = min(k, len(patches))
    gen = torch.Generator(device=device).manual_seed(seed)
    centers = patches[torch.randperm(len(patches), generator=gen, device=device)[:k]].clone()
    for _ in tqdm(range(n_iters), desc="K-means shared codebook", leave=False):
        sums = torch.zeros_like(centers)
        counts = torch.zeros(k, device=device)
        min_dist = torch.full((len(patches),), float("inf"), device=device)
        assignments = torch.empty(len(patches), dtype=torch.long, device=device)
        for start in range(0, len(patches), 8192):
            x = patches[start:start + 8192]
            distances = squared_distances(x, centers)
            values, ids = distances.min(1)
            assignments[start:start + len(x)] = ids
            min_dist[start:start + len(x)] = values
        sums.index_add_(0, assignments, patches)
        counts.index_add_(0, assignments, torch.ones(len(patches), device=device))
        active = counts > 0
        centers[active] = sums[active] / counts[active, None]
        if (~active).any():
            empty = (~active).nonzero().flatten()
            centers[empty] = patches[min_dist.topk(len(empty)).indices]
    return centers


def patchify_encode(images, centers, patch_size, device):
    patches = patchify(images, patch_size)
    codes = torch.empty(patches.shape[:2], dtype=torch.long)
    for start in range(0, len(images), 512):
        batch = patches[start:start + 512].to(device)
        for pos in range(batch.shape[1]):  # shared codebook at every position
            codes[start:start + len(batch), pos] = squared_distances(batch[:, pos], centers).argmin(1).cpu()
    return codes


def contiguous_vocab_range(tokenizer, count, start=None):
    """Return existing token IDs [start, end); choose a readable contiguous range."""
    vocab_size = len(tokenizer)
    candidates = []
    for token_id in range(vocab_size):
        piece = tokenizer.convert_ids_to_tokens(token_id)
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        # Keep ordinary printable pieces, avoid special/control markers and whitespace-only pieces.
        if piece and not piece.startswith(("<|", "[", "\\")) and decoded and decoded.isprintable() and not decoded.isspace():
            candidates.append(token_id)
    if start is not None:
        if start < 0 or start + count > vocab_size:
            raise ValueError(f"Requested vocab interval [{start}, {start + count}) exceeds vocabulary size {vocab_size}")
        return list(range(start, start + count))
    candidate_set = set(candidates)
    for first in candidates:
        if all(token_id in candidate_set for token_id in range(first, first + count)):
            return list(range(first, first + count))
    raise ValueError(f"Could not find {count} contiguous printable vocabulary tokens")


def unicode_script_vocab_stub(tokenizer, count):
    """TODO: select a script-specific token range (e.g. Arabic or Han); placeholder."""
    raise NotImplementedError("Unicode-script vocabulary selection is a planned stub")


@torch.no_grad()
def embed_text(model, tokenizer, text, device, dtype, add_special_tokens=False):
    if not text:
        return torch.empty((1, 0, model.config.hidden_size), device=device, dtype=dtype)
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=add_special_tokens).input_ids.to(device)
    return model.get_input_embeddings()(ids).to(dtype=dtype)


@torch.no_grad()
def image_codes_to_embeds(model, codes, centroid_token_ids, device, dtype):
    token_ids = torch.tensor([centroid_token_ids[i] for i in codes.tolist()], device=device).unsqueeze(0)
    return model.get_input_embeddings()(token_ids).to(dtype=dtype)


def append_text(parts, model, tokenizer, text, device, dtype):
    if text:
        parts.append(embed_text(model, tokenizer, text, device, dtype))


# =============================================================================
# CIFAR-10
# =============================================================================

def load_cifar10_examples():
    transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
    ])
    train_set = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=transform
    )

    indices = list(range(len(train_set)))
    if SHUFFLE_SEED is not None:
        random.Random(SHUFFLE_SEED).shuffle(indices)
    indices = indices[:min(TRAIN_SUBSET_SIZE, len(indices))]

    # Keep original CIFAR-10 training-split index, image tensor, and label.
    examples = []
    for dataset_index in indices:
        image, target = train_set[dataset_index]
        label = CLASS_NAMES[int(target)]
        examples.append((dataset_index, image, label))

    return train_set, examples


# =============================================================================
# CHAT-TEMPLATE EXTRACTION
# =============================================================================

def apply_chat_template(tokenizer, messages, enable_thinking=None, **kwargs):
    if enable_thinking is not None:
        try:
            return tokenizer.apply_chat_template(
                messages, enable_thinking=enable_thinking, **kwargs
            )
        except (TypeError, ValueError):
            pass
    return tokenizer.apply_chat_template(messages, **kwargs)


def extract_chat_tags(tokenizer):
    sentinels = {
        "u1": "\x01USR1\x01",
        "a1": "\x01AST1\x01",
        "u2": "\x01USR2\x01",
        "a2": "\x01AST2\x01",
    }
    messages = [
        {"role": "user", "content": sentinels["u1"]},
        {"role": "assistant", "content": sentinels["a1"]},
        {"role": "user", "content": sentinels["u2"]},
        {"role": "assistant", "content": sentinels["a2"]},
    ]
    full = apply_chat_template(
        tokenizer, messages, tokenize=False, add_generation_prompt=False, enable_thinking=True
    )

    u1s = full.index(sentinels["u1"])
    a1s = full.index(sentinels["a1"])
    u2s = full.index(sentinels["u2"])
    a2s = full.index(sentinels["a2"])
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
        enable_thinking=True,
    )
    marker = one_turn.index(sentinels["u2"]) + len(sentinels["u2"])
    asst_open = one_turn[marker:]

    if asst_open and asst_open in user_to_asst:
        split_at = user_to_asst.rfind(asst_open)
        user_close = user_to_asst[:split_at]
    else:
        user_close, asst_open = "", user_to_asst

    # Strip any hard-coded think tags from asst_open so build functions have explicit control
    for tag in [THINK_CLOSE, THINK_OPEN, "<think>\n\n</think>\n\n", "<think>", "</think>"]:
        asst_open = asst_open.replace(tag, "")

    return {
        "global_prefix": global_prefix,
        "user_to_asst": user_to_asst,
        "asst_tail": asst_tail,
        "turn_separator": turn_separator,
        "user_close": user_close,
        "asst_open": asst_open,
    }


# =============================================================================
# CONVERSATION & EMBEDDING BUILDERS
# =============================================================================

def build_display_conversation(tags, examples, query_example, code_by_index, centroid_token_ids,
                               tokenizer, enable_reasoning=ENABLE_REASONING):
    """Build formatted raw conversation string for logging.

    Patch token sequences are displayed as their selected vocabulary pieces.
    """
    parts = [
        tags["global_prefix"],
        INITIAL_INSTRUCTION,
        tags["user_to_asst"],
        INITIAL_ACK,
        tags["turn_separator"],
    ]

    for dataset_index, _, label in examples:
        placeholder = " ".join(
            tokenizer.decode([centroid_token_ids[int(code)]]) for code in code_by_index[dataset_index]
        )
        parts.extend([
            "\n" + EMBED_PREFIX,
            placeholder,
            LABEL_SEPARATOR + label,
        ])

    query_index = query_example[0]
    q_placeholder = " ".join(
        tokenizer.decode([centroid_token_ids[int(code)]]) for code in code_by_index[query_index]
    )
    parts.extend([
        "\n" + EMBED_PREFIX,
        q_placeholder,
        LABEL_SEPARATOR,
        tags["user_close"],
        tags["asst_open"],
    ])

    if enable_reasoning:
        parts.append(THINK_OPEN)
    else:
        parts.append(THINK_OPEN)
        parts.append(THINK_CLOSE)

    return "".join(parts)


@torch.no_grad()
def build_demonstration_embeds(model, tokenizer, tags, examples, code_by_index, centroid_token_ids, device, dtype):
    """Build shared demonstration prefix embeddings (instruction + all labeled examples)."""
    parts = []
    prefix = tags["global_prefix"] + INITIAL_INSTRUCTION + tags["user_to_asst"] + INITIAL_ACK + tags["turn_separator"]
    append_text(parts, model, tokenizer, prefix, device, dtype)

    for i, (dataset_index, image, label) in enumerate(examples):
        append_text(parts, model, tokenizer, "\n" + EMBED_PREFIX, device, dtype)
        parts.append(image_codes_to_embeds(model, code_by_index[dataset_index], centroid_token_ids, device, dtype))
        append_text(parts, model, tokenizer, LABEL_SEPARATOR + str(label), device, dtype)

    return torch.cat([p for p in parts if p.shape[1] > 0], dim=1)


@torch.no_grad()
def build_query_embeds(model, tokenizer, tags, query_example, code_by_index, centroid_token_ids, device, dtype, enable_reasoning=ENABLE_REASONING):
    """Build query suffix embeddings (query item + colon + assistant turn tags)."""
    q_idx, _, _ = query_example
    parts = []
    append_text(parts, model, tokenizer, "\n" + EMBED_PREFIX, device, dtype)
    parts.append(image_codes_to_embeds(model, code_by_index[q_idx], centroid_token_ids, device, dtype))

    suffix = LABEL_SEPARATOR + tags["user_close"] + tags["asst_open"]
    append_text(parts, model, tokenizer, suffix, device, dtype)

    if enable_reasoning:
        append_text(parts, model, tokenizer, THINK_OPEN, device, dtype)
    else:
        append_text(parts, model, tokenizer, THINK_OPEN + THINK_CLOSE, device, dtype)

    return torch.cat([p for p in parts if p.shape[1] > 0], dim=1)


# =============================================================================
# PREDICTION & EVALUATION ROUTINES
# =============================================================================

@torch.no_grad()
def decode_reasoning(
    model,
    tokenizer,
    past_kv,
    initial_logits,
    device,
    max_tokens=MAX_REASONING_TOKENS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    query_desc="",
):
    """Autoregressively generate reasoning tokens in <think>...</think> (non-streaming by default).

    Tracks token count and time taken.
    Transitions past </think> to place the KV-cache at the assistant answer position.
    """
    think_close_id = tokenizer.convert_tokens_to_ids("</think>")
    eos_id = tokenizer.eos_token_id

    curr_cache = past_kv
    current_logits = initial_logits
    reasoning_tokens = []
    closed_naturally = False

    start_time = time.time()

    with torch.inference_mode():
        for step in range(max_tokens):
            if temperature <= 0:
                next_id = int(torch.argmax(current_logits).item())
            else:
                probs = torch.softmax(current_logits / temperature, dim=-1)
                sorted_probs, sorted_ids = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=0)
                remove = cumulative_probs - sorted_probs > top_p
                sorted_probs[remove] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum()
                next_id = int(sorted_ids[torch.multinomial(sorted_probs, 1)].item())

            if next_id == think_close_id:
                closed_naturally = True
                break
            if eos_id is not None and next_id == eos_id:
                closed_naturally = True
                break

            reasoning_tokens.append(next_id)

            # --- Streaming print option (commented out) ---
            # if VERBOSE:
            #     tok_str = tokenizer.decode([next_id], skip_special_tokens=False)
            #     print(tok_str, end="", flush=True)

            token_tensor = torch.tensor([[next_id]], device=device)
            step_out = model(
                input_ids=token_tensor,
                past_key_values=curr_cache,
                use_cache=True,
                return_dict=True,
            )
            curr_cache = step_out.past_key_values
            current_logits = step_out.logits[0, -1, :].float()

    elapsed_sec = time.time() - start_time
    tok_count = len(reasoning_tokens)
    tok_per_sec = tok_count / elapsed_sec if elapsed_sec > 0 else 0.0

    # Non-streaming decode: display the complete reasoning trace block at once
    if VERBOSE:
        reasoning_text = tokenizer.decode(reasoning_tokens, skip_special_tokens=False)
        header_desc = f" ({query_desc})" if query_desc else ""
        print("\n" + "=" * 88)
        print(f"MODEL REASONING / THINKING TRACE{header_desc}")
        print("=" * 88)
        print(reasoning_text.strip() if reasoning_text.strip() else "[No reasoning tokens generated]")
        print("=" * 88)
        print(
            f"Reasoning tokens: {tok_count} / {max_tokens} "
            f"| Time: {elapsed_sec:.2f}s ({tok_per_sec:.1f} tok/s) "
            f"| Closed naturally: {closed_naturally}"
        )
        print("=" * 88)

    # Step transition past </think> to place KV-cache and logits at assistant answer position
    transition_str = "</think>\n\n" if closed_naturally else "\n</think>\n\n"
    transition_ids = tokenizer(
        transition_str, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)

    with torch.inference_mode():
        trans_out = model(
            input_ids=transition_ids,
            past_key_values=curr_cache,
            use_cache=True,
            return_dict=True,
        )
    final_logits = trans_out.logits[0, -1, :].float()

    return curr_cache, final_logits, reasoning_tokens, closed_naturally, elapsed_sec


@torch.no_grad()
def predict_query(
    model,
    tokenizer,
    past_kv,
    query_embeds,
    device,
    label_token_ids,
    enable_reasoning=ENABLE_REASONING,
    query_desc="",
):
    """Evaluate one query starting from past_kv.

    Optionally runs streaming free-run reasoning in <think>...</think>,
    transitions through </think>, and reads next-token class probabilities.
    """
    with torch.inference_mode():
        query_out = model(
            inputs_embeds=query_embeds,
            past_key_values=past_kv,
            use_cache=True,
            return_dict=True,
        )
    curr_cache = query_out.past_key_values
    current_logits = query_out.logits[0, -1, :].float()

    reasoning_tokens = []
    closed_naturally = False
    reasoning_time = 0.0

    if enable_reasoning:
        curr_cache, logits, reasoning_tokens, closed_naturally, reasoning_time = decode_reasoning(
            model=model,
            tokenizer=tokenizer,
            past_kv=curr_cache,
            initial_logits=current_logits,
            device=device,
            max_tokens=MAX_REASONING_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            query_desc=query_desc,
        )
    else:
        logits = current_logits

    class_logits = torch.tensor(
        [logits[label_token_ids[label]].item() for label in CLASS_NAMES],
        dtype=torch.float64,
    )
    class_probs = torch.softmax(class_logits, dim=0)
    pred_class = CLASS_NAMES[int(torch.argmax(class_probs).item())]

    return pred_class, logits, class_probs, reasoning_tokens, closed_naturally, reasoning_time


@torch.no_grad()
def evaluate_sanity_train(
    model,
    tokenizer,
    tags,
    code_by_index,
    centroid_token_ids,
    eval_examples,
    base_cache,
    device,
    dtype,
    label_token_ids,
    enable_reasoning=ENABLE_REASONING,
):
    """Evaluate accuracy across the evaluation examples and report accuracy."""
    correct = 0
    total = 0
    total_reasoning_tokens = 0
    total_reasoning_time = 0.0
    per_class = {label: [0, 0] for label in CLASS_NAMES}

    print(f"\nRunning eval sanity train across {len(eval_examples)} examples...")
    progress = tqdm(eval_examples, desc="Eval Sanity Train", disable=not VERBOSE)
    for q_idx, q_image, true_label in progress:
        query_embeds = build_query_embeds(
            model, tokenizer, tags,
            (q_idx, q_image, true_label),
            code_by_index, centroid_token_ids,
            device, dtype,
            enable_reasoning=enable_reasoning,
        )
        curr_cache = copy.deepcopy(base_cache)
        query_desc = f"eval {total + 1}/{len(eval_examples)}: index={q_idx}, true={true_label}"
        pred_label, logits, class_probs, reasoning_tokens, closed_naturally, reasoning_time = predict_query(
            model, tokenizer, curr_cache, query_embeds,
            device, label_token_ids, enable_reasoning=enable_reasoning,
            query_desc=query_desc,
        )

        total_reasoning_tokens += len(reasoning_tokens)
        total_reasoning_time += reasoning_time

        is_correct = (pred_label == true_label)
        correct += int(is_correct)
        total += 1
        per_class[true_label][1] += 1
        per_class[true_label][0] += int(is_correct)
        progress.set_postfix(accuracy=f"{100.0 * correct / total:.2f}%")

    accuracy = 100.0 * correct / max(1, total)
    print("\n" + "=" * 88)
    print(f"EVAL SANITY TRAIN FINAL ACCURACY: {accuracy:.2f}% ({correct}/{total})")
    print("=" * 88)
    if enable_reasoning and total > 0:
        avg_tokens = total_reasoning_tokens / total
        avg_time = total_reasoning_time / total
        print(f"Reasoning summary: {total_reasoning_tokens} total tokens | {total_reasoning_time:.2f}s total (avg {avg_tokens:.1f} tok, {avg_time:.2f}s / sample)")
        print("=" * 88)
    print("Per-class accuracy breakdown:")
    for label, (class_correct, class_total) in per_class.items():
        pct = 100.0 * class_correct / class_total if class_total else 0.0
        print(f"  Class {label}: {class_correct:3d} / {class_total:3d} ({pct:5.1f}%)")
    print("=" * 88)
    return accuracy


# =============================================================================
# MAIN
# =============================================================================

@torch.no_grad()
def main():
    print(f"Loading CIFAR-10 training split (subset size = {TRAIN_SUBSET_SIZE})...")
    train_set, examples = load_cifar10_examples()

    # Determine subset of trainset to use for evaluation
    if EVAL_SUBSET_SIZE and EVAL_SUBSET_SIZE > 0:
        if EVAL_SUBSET_SIZE > len(examples):
            raise ValueError(
                f"EVAL_SUBSET_SIZE={EVAL_SUBSET_SIZE} must be <= trainset size ({len(examples)})."
            )
        eval_examples = random.Random(SHUFFLE_SEED).sample(examples, EVAL_SUBSET_SIZE)
        print(f"Using random evaluation subset: {len(eval_examples)} of {len(examples)} examples.")
    else:
        eval_examples = examples
        print(f"Using full training set for evaluation: {len(eval_examples)} examples.")

    # Select single query item for primary diagnostics
    if EVAL_DATASET_INDEX is None:
        eval_example = random.choice(examples)
    else:
        eval_example = examples[EVAL_DATASET_INDEX]

    eval_index, eval_image, eval_label = eval_example

    print("\nCIFAR-10 class mapping:")
    for class_id, class_name in enumerate(train_set.classes):
        print(f"  CIFAR class {class_id:2d} {class_name:12s} -> {CLASS_NAMES[class_id]}")

    print("\nLoading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    device = model.get_input_embeddings().weight.device
    dtype = model.get_input_embeddings().weight.dtype

    # Validate class labels map to single tokens
    label_token_ids = {}
    for label in CLASS_NAMES:
        ids = tokenizer(label, add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise ValueError(f"Class label {label!r} is not a single token: {ids}.")
        label_token_ids[label] = ids[0]

    if IMAGE_SIZE % PATCH_SIZE:
        raise ValueError(f"PATCH_SIZE={PATCH_SIZE} must divide IMAGE_SIZE={IMAGE_SIZE}")
    print(f"\nFitting shared K-means patch tokenizer (patch={PATCH_SIZE}, k={KMEANS_CENTROIDS})...")
    centers = fit_patch_codebook(
        torch.stack([image for _, image, _ in examples]), PATCH_SIZE, KMEANS_CENTROIDS,
        KMEANS_SAMPLES, KMEANS_ITERS, device, SHUFFLE_SEED or 0,
    )
    code_by_index = {}
    for start in range(0, len(examples), 128):
        batch_examples = examples[start:start + 128]
        batch_codes = patchify_encode(torch.stack([image for _, image, _ in batch_examples]), centers, PATCH_SIZE, device)
        code_by_index.update({ex[0]: code for ex, code in zip(batch_examples, batch_codes)})
    if VOCAB_FILTER == "unicode_script":
        centroid_token_ids = unicode_script_vocab_stub(tokenizer, len(centers))
    else:
        centroid_token_ids = contiguous_vocab_range(tokenizer, len(centers), CENTROID_TOKEN_START)
    print(f"Mapped {len(centers)} centroids to existing vocabulary IDs "
          f"[{centroid_token_ids[0]}, {centroid_token_ids[-1] + 1}); vocabulary unchanged.")

    print("\nSelected training examples:")
    for i, (dataset_index, _, label) in enumerate(examples):
        patch_tokens = " ".join(
            tokenizer.decode([centroid_token_ids[int(code)]]) for code in code_by_index[dataset_index]
        )
        print(f"  {i:3d} index={dataset_index:5d}  patches={patch_tokens!r}  label={label}")
    print(f"\nDiagnostic query index={eval_index}; class={eval_label}; "
          f"patch codes={code_by_index[eval_index].tolist()}")

    tags = extract_chat_tags(tokenizer)
    display_conversation = build_display_conversation(
        tags, examples, eval_example, code_by_index, centroid_token_ids, tokenizer,
        enable_reasoning=ENABLE_REASONING
    )

    if VERBOSE:
        print("\n" + "=" * 88)
        print("FULL RAW PREFILL CONVERSATION (INCLUDING CHAT TAGS)")
        print("=" * 88)
        print(display_conversation, end="")
        print("\n" + "=" * 88)
        print("END OF RAW PREFILL CONVERSATION")
        print("=" * 88)

    # Build shared demonstration prefix
    print("\nPrefilling demonstration context into base KV cache...")
    demo_embeds = build_demonstration_embeds(
        model, tokenizer, tags, examples, code_by_index, centroid_token_ids, device, dtype
    )
    print(f"Demonstration context token count: {demo_embeds.shape[1]}")

    with torch.inference_mode():
        demo_out = model(inputs_embeds=demo_embeds, use_cache=True, return_dict=True)
    base_cache = demo_out.past_key_values

    # Step diagnostic query sample
    print("\nEvaluating diagnostic query sample...")
    diag_query_embeds = build_query_embeds(
        model, tokenizer, tags, eval_example, code_by_index, centroid_token_ids,
        device, dtype, enable_reasoning=ENABLE_REASONING,
    )
    diag_cache = copy.deepcopy(base_cache)
    pred_class, logits, class_probs, reasoning_tokens, closed_naturally, reasoning_time = predict_query(
        model, tokenizer, diag_cache, diag_query_embeds, device, label_token_ids,
        enable_reasoning=ENABLE_REASONING, query_desc="DIAGNOSTIC QUERY",
    )

    if VERBOSE:
        full_vocab_probs = torch.softmax(logits, dim=-1)
        print("\nTop next-token logits and full-vocabulary softmax:")
        top_values, top_ids = torch.topk(logits, min(TOP_K, logits.numel()))
        for token_id, logit in zip(top_ids.tolist(), top_values.tolist()):
            probability = float(full_vocab_probs[token_id].item())
            token_text = tokenizer.decode([token_id])
            print(
                f"  token={token_id:7d}  text={token_text!r:16s}  "
                f"logit={logit:10.4f}  softmax={probability:.8e}"
            )

        print("\nClass-token scores:")
        for label, token_id, class_prob in zip(
            CLASS_NAMES,
            [label_token_ids[label] for label in CLASS_NAMES],
            class_probs.tolist(),
        ):
            class_logit = float(logits[token_id].item())
            full_prob = float(full_vocab_probs[token_id].item())
            print(
                f"  {label}: token_id={token_id:6d}  logit={class_logit:10.4f}  "
                f"class_softmax={class_prob:.8f}  full_vocab_softmax={full_prob:.8e}"
            )

    greedy_token_id = int(torch.argmax(logits).item())
    greedy_text = tokenizer.decode([greedy_token_id], skip_special_tokens=True).strip()

    print("\nDiagnostic query result:")
    print(f"  greedy next token = {greedy_text!r} (token_id={greedy_token_id})")
    print(f"  best class pred   = {pred_class}")
    print(f"  reference label   = {eval_label}")
    print(f"  correct           = {pred_class == eval_label}")

    # Run full eval sanity train if enabled
    if EVAL_SANITY_TRAIN:
        evaluate_sanity_train(
            model, tokenizer, tags, code_by_index, centroid_token_ids, eval_examples, base_cache,
            device, dtype, label_token_ids,
            enable_reasoning=ENABLE_REASONING,
        )


if __name__ == "__main__":
    main()
