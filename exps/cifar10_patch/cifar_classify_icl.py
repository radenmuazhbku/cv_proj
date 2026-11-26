"""CIFAR-10 ICL classification using K-means patch IDs as existing LLM tokens.

The vocabulary is never extended: selected existing token IDs stand for visual
patch centroids. The shared codebook is fitted across all patch positions.

Examples from the repository root:

    .venv/bin/python exps/cifar10_patch/cifar_classify_icl.py
    .venv/bin/python exps/cifar10_patch/cifar_classify_icl.py --model Qwen/Qwen3-8B --patch-size 8 --k 128
    .venv/bin/python exps/cifar10_patch/cifar_classify_icl.py --eval-train --train-eval-subset 100
    # Reproduce v2's shuffled 100-example train recall sanity evaluation with Qwen:
    .venv/bin/python exps/cifar10_patch/cifar_classify_icl.py --model Qwen/Qwen3-8B --train-subset 100 --subset-sampling shuffled --eval-teach --seed 0 --patch-size 16 --k 256 --kmeans-samples 100000 --kmeans-iters 50 --prediction-mode generate --answer-prefix ''
    .venv/bin/python exps/cifar10_patch/cifar_classify_icl.py --prediction-mode full_prob --answer-prefix "Class: "
    .venv/bin/python exps/cifar10_patch/cifar_classify_icl.py --use-reasoning --max-reasoning-tokens 512 --answer-prefill-after-reasoning

The defaults teach on 1,000 balanced train examples and evaluate on 100
balanced test examples. --eval-train evaluates a disjoint train sanity subset;
--eval-teach instead repeats each teaching image as a query, matching v2's
train recall sanity protocol. Balanced subset sizes must be multiples of 10.
Use --subset-sampling shuffled for v2's seeded shuffle-and-slice sampling. The
shared codebook is fitted on teaching images only; all query splits reuse one
unchanged teaching KV cache.
Patch size must divide 32. Vocabulary entries are existing model tokens.
"""

import argparse
import copy
import random
import time
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as T
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

# =============================================================================
# CLI
# =============================================================================

CLASS_NAMES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
IMAGE_SIZE = 32
EMBED_PREFIX = ""
LABEL_SEPARATOR = " : "
INITIAL_ACK = "Understood."
EXAMPLE_NEWLINE = "\n"
EVAL_PROMPT = ""
THINK_OPEN = "<think>\n"

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3-8B",
    # parser.add_argument("--model", default="HuggingFaceTB/SmolLM3-3B",
                        help="HF model ID, e.g. Qwen/Qwen3-8B or HuggingFaceTB/SmolLM3-3B")
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    parser.add_argument("--train-subset", type=int, default=1000,
                        help="balanced labeled teaching examples; multiple of 10")
    parser.add_argument("--subset-sampling", choices=("balanced", "shuffled"), default="balanced",
                        help="balanced class counts, or v2-compatible random shuffled ordering")
    parser.add_argument("--test-subset", type=int, default=100,
                        help="balanced test examples; multiple of 10")
    parser.add_argument("--eval-train", action="store_true",
                        help="also evaluate a disjoint balanced subset of the train split")
    parser.add_argument("--eval-teach", action="store_true",
                        help="evaluate the teaching examples themselves (v2 train recall sanity mode)")
    parser.add_argument("--train-eval-subset", type=int, default=100,
                        help="balanced train sanity examples; multiple of 10")
    parser.add_argument("--patch-size", type=int, choices=(1, 2, 4, 8, 16, 32), default=16)
    parser.add_argument("--k", type=int, default=256, help="shared K-means centroid count")
    parser.add_argument("--kmeans-samples", type=int, default=100000)
    parser.add_argument("--kmeans-iters", type=int, default=50)
    parser.add_argument("--centroid-token-start", type=int, default=None)
    parser.add_argument("--vocab-filter", choices=("printable", "unicode_script"), default="printable")
    parser.add_argument("--prediction-mode", choices=("generate", "first_token", "full_prob"), default="generate")
    parser.add_argument("--answer-prefix", default="")
    parser.add_argument("--use-reasoning", action="store_true",
                        help="generate a bounded <think>...</think> section before the answer")
    parser.add_argument("--max-reasoning-tokens", type=int, default=512)
    parser.add_argument("--reasoning-temperature", type=float, default=0.6)
    parser.add_argument("--reasoning-top-p", type=float, default=0.95)
    parser.add_argument("--answer-prefill-after-reasoning", action="store_true",
                        help="insert --reasoning-answer-prefix after </think>")
    parser.add_argument("--reasoning-answer-prefix", default="The class of the final item is: ")
    parser.add_argument("--max-label-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("logs/cifar_patch/icl_scores.json"))
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def select_indices(targets, count, seed, sampling="balanced", excluded=()):
    if count <= 0:
        raise ValueError(f"Subset size must be positive, got {count}")
    excluded = set(excluded)
    rng = random.Random(seed)
    if sampling == "shuffled":
        pool = [i for i in range(len(targets)) if i not in excluded]
        rng.shuffle(pool)
        if count > len(pool):
            raise ValueError(f"Requested {count} examples from a pool of {len(pool)}")
        return pool[:count]
    if count % 10:
        raise ValueError(f"Balanced subset size must be a multiple of 10, got {count}")
    per_class = count // 10
    pools = []
    for class_id in range(10):
        pool = [i for i, target in enumerate(targets) if int(target) == class_id and i not in excluded]
        rng.shuffle(pool)
        if len(pool) < per_class:
            raise ValueError(f"Class {class_id} has {len(pool)} eligible examples; {per_class} required")
        pools.append(pool[:per_class])
    return [pools[c][j] for j in range(per_class) for c in range(10)]


def make_instruction():
    labels = ", ".join(CLASS_NAMES)
    return (
        "You are learning an image classification task through in-context examples. "
        "Each example is presented as text embeds: followed by image patch tokens, "
        "then a colon and its class label. The final item is an image followed by "
        "a colon with no label. Infer its class from the preceding examples. "
        f"The available class labels are: {labels}. Reply immediately only with one of these labels, {labels}."
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

def load_cifar10_examples(args):
    transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
    ])
    train_set = torchvision.datasets.CIFAR10(
        root=str(args.data_root), train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root=str(args.data_root), train=False, download=True, transform=transform
    )
    teach_indices = select_indices(
        train_set.targets, args.train_subset, args.seed, args.subset_sampling
    )
    train_eval_indices = []
    if args.eval_train and not args.eval_teach:
        train_eval_indices = select_indices(
            train_set.targets, args.train_eval_subset, args.seed + 1,
            args.subset_sampling, excluded=teach_indices,
        )
    test_indices = select_indices(
        test_set.targets, args.test_subset, args.seed + 2, args.subset_sampling
    )

    def examples_from(dataset, indices, offset=0):
        result = []
        for index in indices:
            image, target = dataset[index]
            result.append((offset + index, image, CLASS_NAMES[int(target)]))
        return result

    return (
        train_set,
        examples_from(train_set, teach_indices),
        examples_from(train_set, train_eval_indices),
        examples_from(test_set, test_indices, offset=len(train_set)),
    )


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
        tokenizer, messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
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
        enable_thinking=False,
    )
    marker = one_turn.index(sentinels["u2"]) + len(sentinels["u2"])
    asst_open = one_turn[marker:]

    if asst_open and asst_open in user_to_asst:
        split_at = user_to_asst.rfind(asst_open)
        user_close = user_to_asst[:split_at]
    else:
        user_close, asst_open = "", user_to_asst

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
                               tokenizer):
    """Build formatted raw conversation string for logging.

    Patch token sequences are displayed as their selected vocabulary pieces.
    """
    parts = [
        tags["global_prefix"],
        INITIAL_INSTRUCTION,
    ]

    for dataset_index, _, label in examples:
        placeholder = " ".join(
            tokenizer.decode([centroid_token_ids[int(code)]]) for code in code_by_index[dataset_index]
        )
        parts.extend([
            "\n" + EMBED_PREFIX,
            placeholder,
            LABEL_SEPARATOR + label + EXAMPLE_NEWLINE,
        ])

    parts.extend([tags["user_to_asst"], INITIAL_ACK, tags["asst_tail"]])

    query_index = query_example[0]
    q_placeholder = " ".join(
        tokenizer.decode([centroid_token_ids[int(code)]]) for code in code_by_index[query_index]
    )
    parts.extend([
        tags["turn_separator"] + EVAL_PROMPT + "\n" + EMBED_PREFIX,
        q_placeholder,
        LABEL_SEPARATOR,
        tags["user_close"],
        tags["asst_open"],
        ANSWER_PREFIX,
    ])

    return "".join(parts)


@torch.no_grad()
def build_demonstration_embeds(model, tokenizer, tags, examples, code_by_index, centroid_token_ids, device, dtype):
    """Build shared demonstration prefix embeddings (instruction + all labeled examples)."""
    parts = []
    prefix = tags["global_prefix"] + INITIAL_INSTRUCTION
    append_text(parts, model, tokenizer, prefix, device, dtype)

    for i, (dataset_index, image, label) in enumerate(examples):
        append_text(parts, model, tokenizer, "\n" + EMBED_PREFIX, device, dtype)
        parts.append(image_codes_to_embeds(model, code_by_index[dataset_index], centroid_token_ids, device, dtype))
        append_text(parts, model, tokenizer, LABEL_SEPARATOR + str(label) + EXAMPLE_NEWLINE, device, dtype)

    append_text(parts, model, tokenizer, tags["user_to_asst"] + INITIAL_ACK + tags["asst_tail"], device, dtype)

    return torch.cat([p for p in parts if p.shape[1] > 0], dim=1)


@torch.no_grad()
def build_query_embeds(model, tokenizer, tags, query_example, code_by_index, centroid_token_ids,
                       device, dtype, use_reasoning=False):
    """Build query suffix embeddings (query item + colon + assistant turn tags)."""
    q_idx, _, _ = query_example
    parts = []
    append_text(parts, model, tokenizer, tags["turn_separator"] + EVAL_PROMPT + "\n" + EMBED_PREFIX, device, dtype)
    parts.append(image_codes_to_embeds(model, code_by_index[q_idx], centroid_token_ids, device, dtype))

    suffix = LABEL_SEPARATOR + tags["user_close"] + tags["asst_open"]
    suffix += THINK_OPEN if use_reasoning else ANSWER_PREFIX
    append_text(parts, model, tokenizer, suffix, device, dtype)

    return torch.cat([p for p in parts if p.shape[1] > 0], dim=1)


# =============================================================================
# PREDICTION & EVALUATION ROUTINES
# =============================================================================

@torch.no_grad()
def decode_answer(
    model,
    tokenizer,
    past_kv,
    initial_logits,
    device,
    max_tokens=8,
    query_desc="",
):
    """Greedily generate a short answer continuation, following the teaching script."""
    eos_id = tokenizer.eos_token_id

    curr_cache = past_kv
    current_logits = initial_logits
    generated_tokens = []
    stopped = False

    start_time = time.time()

    with torch.inference_mode():
        for step in range(max_tokens):
            next_id = int(torch.argmax(current_logits).item())
            if eos_id is not None and next_id == eos_id:
                stopped = True
                break

            generated_tokens.append(next_id)

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
    generated = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    if VERBOSE:
        print(f"Generated answer{f' ({query_desc})' if query_desc else ''}: {generated!r}")
    return curr_cache, current_logits, generated_tokens, stopped, elapsed_sec


@torch.no_grad()
def decode_reasoning(model, tokenizer, past_kv, initial_logits, device,
                     max_tokens, temperature, top_p, query_desc=""):
    """Generate a bounded reasoning section and advance past its closing tag."""
    close_id = tokenizer.convert_tokens_to_ids("</think>")
    eos_id = tokenizer.eos_token_id
    current_cache = past_kv
    logits = initial_logits
    generated = []
    closed = False
    started = time.time()
    with torch.inference_mode():
        for _ in range(max_tokens):
            if temperature <= 0:
                next_id = int(logits.argmax().item())
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                sorted_probs, sorted_ids = probs.sort(descending=True)
                cumulative = sorted_probs.cumsum(0)
                remove = cumulative - sorted_probs > top_p
                sorted_probs[remove] = 0
                sorted_probs /= sorted_probs.sum().clamp_min(1e-12)
                next_id = int(sorted_ids[torch.multinomial(sorted_probs, 1)].item())
            if next_id == close_id:
                closed = True
                break
            if eos_id is not None and next_id == eos_id:
                break
            generated.append(next_id)
            output = model(
                input_ids=torch.tensor([[next_id]], device=device),
                past_key_values=current_cache,
                use_cache=True,
                return_dict=True,
            )
            current_cache = output.past_key_values
            logits = output.logits[0, -1].float()
    elapsed = time.time() - started
    if VERBOSE:
        trace = tokenizer.decode(generated, skip_special_tokens=True).strip()
        print(f"Reasoning{f' ({query_desc})' if query_desc else ''}: {trace!r}; "
              f"tokens={len(generated)}/{max_tokens}, closed={closed}, time={elapsed:.2f}s")
    closing = tokenizer("</think>" + "\n\n", add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    with torch.inference_mode():
        output = model(
            input_ids=closing,
            past_key_values=current_cache,
            use_cache=True,
            return_dict=True,
        )
    return output.past_key_values, output.logits[0, -1].float(), generated, closed, elapsed


@torch.no_grad()
def predict_query(
    model,
    tokenizer,
    past_kv,
    query_embeds,
    device,
    label_token_ids,
    settings,
    query_desc="",
):
    """Evaluate a query with the teaching script's short answer-prefix convention."""
    with torch.inference_mode():
        query_out = model(
            inputs_embeds=query_embeds,
            past_key_values=past_kv,
            use_cache=True,
            return_dict=True,
        )
    curr_cache = query_out.past_key_values
    current_logits = query_out.logits[0, -1, :].float()

    reasoning_time = 0.0
    if settings.use_reasoning:
        curr_cache, current_logits, _, _, reasoning_time = decode_reasoning(
            model, tokenizer, curr_cache, current_logits, device,
            settings.max_reasoning_tokens, settings.reasoning_temperature,
            settings.reasoning_top_p, query_desc,
        )
        if settings.answer_prefill_after_reasoning and settings.reasoning_answer_prefix:
            answer_prefill = embed_text(
                model, tokenizer, settings.reasoning_answer_prefix, device, query_embeds.dtype
            )
            with torch.inference_mode():
                prefill_out = model(
                    inputs_embeds=answer_prefill,
                    past_key_values=curr_cache,
                    use_cache=True,
                    return_dict=True,
                )
            curr_cache = prefill_out.past_key_values
            current_logits = prefill_out.logits[0, -1, :].float()

    class_logits = current_logits[torch.tensor([label_token_ids[label] for label in CLASS_NAMES], device=current_logits.device)]
    class_probs = torch.softmax(class_logits, dim=0).cpu()
    if PREDICTION_MODE == "first_token":
        pred_class = CLASS_NAMES[int(class_probs.argmax().item())]
        return pred_class, current_logits, class_probs, [], True, reasoning_time
    if PREDICTION_MODE == "full_prob":
        log_scores = []
        for label in CLASS_NAMES:
            label_id = label_token_ids[label]
            label_input = torch.tensor([[label_id]], device=device)
            label_embed = model.get_input_embeddings()(label_input).to(query_embeds.dtype)
            joint_input = torch.cat([query_embeds, label_embed], dim=1)
            candidate_out = model(
                inputs_embeds=joint_input,
                past_key_values=copy.deepcopy(past_kv),
                use_cache=False,
                return_dict=True,
            )
            log_scores.append(torch.log_softmax(candidate_out.logits[0, -2].float(), dim=-1)[label_id])
        class_probs = torch.softmax(torch.stack(log_scores), dim=0).cpu()
        pred_class = CLASS_NAMES[int(class_probs.argmax().item())]
        return pred_class, current_logits, class_probs, [], True, reasoning_time
    if PREDICTION_MODE != "generate":
        raise ValueError(f"Unknown PREDICTION_MODE: {PREDICTION_MODE}")
    _, _, generated_ids, stopped, elapsed = decode_answer(
        model, tokenizer, curr_cache, current_logits, device, MAX_LABEL_TOKENS, query_desc
    )
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    pred_class = next((label for label in CLASS_NAMES if answer.startswith(label)), answer)
    return pred_class, current_logits, class_probs, generated_ids, stopped, elapsed + reasoning_time


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
    settings,
    split_name,
):
    """Evaluate queries from isolated copies of the shared teaching KV cache."""
    correct = 0
    total = 0
    per_class = {label: [0, 0] for label in CLASS_NAMES}
    records = []

    print(f"\nEvaluating {split_name} on {len(eval_examples)} examples...")
    progress = tqdm(eval_examples, desc=f"Eval {split_name}", disable=not VERBOSE)
    for q_idx, q_image, true_label in progress:
        query_embeds = build_query_embeds(
            model, tokenizer, tags,
            (q_idx, q_image, true_label),
            code_by_index, centroid_token_ids,
            device, dtype, use_reasoning=settings.use_reasoning,
        )
        curr_cache = copy.deepcopy(base_cache)
        query_desc = f"eval {total + 1}/{len(eval_examples)}: index={q_idx}, true={true_label}"
        pred_label, logits, class_probs, reasoning_tokens, closed_naturally, reasoning_time = predict_query(
            model, tokenizer, curr_cache, query_embeds,
            device, label_token_ids, settings,
            query_desc=query_desc,
        )

        is_correct = (pred_label == true_label)
        correct += int(is_correct)
        total += 1
        per_class[true_label][1] += 1
        per_class[true_label][0] += int(is_correct)
        records.append({
            "dataset_index": q_idx,
            "true_label": true_label,
            "predicted_label": pred_label,
            "correct": is_correct,
            "scores": {label: float(score) for label, score in zip(CLASS_NAMES, class_probs.tolist())},
        })
        progress.set_postfix(accuracy=f"{100.0 * correct / total:.2f}%")

    accuracy = 100.0 * correct / max(1, total)
    print("\n" + "=" * 88)
    print(f"{split_name.upper()} ACCURACY: {accuracy:.2f}% ({correct}/{total})")
    print("=" * 88)
    print("Per-class accuracy breakdown:")
    for label, (class_correct, class_total) in per_class.items():
        pct = 100.0 * class_correct / class_total if class_total else 0.0
        print(f"  Class {label}: {class_correct:3d} / {class_total:3d} ({pct:5.1f}%)")
    print("=" * 88)
    return {"accuracy": accuracy, "correct": correct, "total": total,
            "per_class": per_class, "examples": records}


# =============================================================================
# MAIN
# =============================================================================

@torch.no_grad()
def main():
    global INITIAL_INSTRUCTION, ANSWER_PREFIX, MAX_LABEL_TOKENS, PREDICTION_MODE, VERBOSE
    args = parse_args()
    required_sizes = [("train-subset", args.train_subset), ("test-subset", args.test_subset)]
    if args.eval_train and not args.eval_teach:
        required_sizes.append(("train-eval-subset", args.train_eval_subset))
    for name, value in required_sizes:
        if value <= 0 or (args.subset_sampling == "balanced" and value % 10):
            suffix = " and a multiple of 10" if args.subset_sampling == "balanced" else ""
            raise ValueError(f"--{name} must be positive{suffix}, got {value}")
    if args.k < 1 or args.kmeans_samples < 1 or args.kmeans_iters < 1 or args.max_label_tokens < 1:
        raise ValueError("--k, --kmeans-samples, --kmeans-iters, and --max-label-tokens must be positive")
    if args.max_reasoning_tokens < 1:
        raise ValueError("--max-reasoning-tokens must be positive")
    if args.reasoning_temperature < 0:
        raise ValueError("--reasoning-temperature must be non-negative")
    if not 0 < args.reasoning_top_p <= 1:
        raise ValueError("--reasoning-top-p must be in (0, 1]")
    if args.use_reasoning and args.prediction_mode != "generate":
        raise ValueError("--use-reasoning currently requires --prediction-mode generate")
    if args.answer_prefill_after_reasoning and not args.use_reasoning:
        raise ValueError("--answer-prefill-after-reasoning requires --use-reasoning")
    if IMAGE_SIZE % args.patch_size:
        raise ValueError(f"--patch-size={args.patch_size} must divide image size {IMAGE_SIZE}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    VERBOSE = not args.quiet
    PREDICTION_MODE = args.prediction_mode
    ANSWER_PREFIX = args.answer_prefix
    MAX_LABEL_TOKENS = args.max_label_tokens
    INITIAL_INSTRUCTION = make_instruction()

    train_eval_count = args.train_subset if args.eval_teach else (args.train_eval_subset if args.eval_train else 0)
    print(f"Loading CIFAR subsets: teach={args.train_subset}, test={args.test_subset}, "
          f"train_eval={train_eval_count}, sampling={args.subset_sampling}")
    train_set, teach_examples, train_eval_examples, test_examples = load_cifar10_examples(args)
    if args.eval_teach:
        train_eval_examples = teach_examples
    print(f"Train teach: {len(teach_examples)}; test evaluation: {len(test_examples)}")
    if args.eval_train or args.eval_teach:
        print(f"Train sanity evaluation: {len(train_eval_examples)} "
              f"({'teaching examples' if args.eval_teach else 'disjoint examples'})")

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    device = model.get_input_embeddings().weight.device
    dtype = model.get_input_embeddings().weight.dtype

    label_token_ids = {}
    for label in CLASS_NAMES:
        ids = tokenizer(" " + label, add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise ValueError(f"Class label {label!r} is not a single token: {ids}")
        label_token_ids[label] = int(ids[0])

    print(f"Fitting shared patch K-means: patch={args.patch_size}, k={args.k}")
    centers = fit_patch_codebook(
        torch.stack([image for _, image, _ in teach_examples]),
        args.patch_size, args.k, args.kmeans_samples, args.kmeans_iters,
        device, args.seed,
    )
    all_examples = teach_examples + train_eval_examples + test_examples
    code_by_index = {}
    for start in range(0, len(all_examples), 128):
        batch_examples = all_examples[start:start + 128]
        batch_codes = patchify_encode(
            torch.stack([image for _, image, _ in batch_examples]),
            centers, args.patch_size, device,
        )
        code_by_index.update({
            example[0]: code for example, code in zip(batch_examples, batch_codes)
        })

    if args.vocab_filter == "unicode_script":
        centroid_token_ids = unicode_script_vocab_stub(tokenizer, len(centers))
    else:
        centroid_token_ids = contiguous_vocab_range(
            tokenizer, len(centers), args.centroid_token_start
        )
    print(
        f"Centroids map to existing vocabulary IDs "
        f"[{centroid_token_ids[0]}, {centroid_token_ids[-1] + 1}); vocabulary unchanged."
    )

    tags = extract_chat_tags(tokenizer)
    display_query = test_examples[0]
    if VERBOSE:
        display = build_display_conversation(
            tags, teach_examples, display_query, code_by_index, centroid_token_ids, tokenizer
        )
        print("\n" + "=" * 80)
        print(display, end="")
        print("\n" + "=" * 80)

    print("Prefilling teaching examples once and preserving the base KV cache...")
    demo_embeds = build_demonstration_embeds(
        model, tokenizer, tags, teach_examples, code_by_index,
        centroid_token_ids, device, dtype,
    )
    with torch.inference_mode():
        demo_out = model(inputs_embeds=demo_embeds, use_cache=True, return_dict=True)
    base_cache = demo_out.past_key_values

    # Each split evaluation starts every query from a fresh copy of this same
    # teaching cache; query tokens never accumulate in the shared cache.
    results = {
        "config": {
            "model": args.model,
            "train_subset": args.train_subset,
            "test_subset": args.test_subset,
            "eval_train": args.eval_train or args.eval_teach,
            "eval_teach": args.eval_teach,
            "subset_sampling": args.subset_sampling,
            "train_eval_subset": len(train_eval_examples),
            "patch_size": args.patch_size,
            "k": len(centers),
            "prediction_mode": args.prediction_mode,
            "answer_prefix": args.answer_prefix,
            "use_reasoning": args.use_reasoning,
            "max_reasoning_tokens": args.max_reasoning_tokens,
            "reasoning_temperature": args.reasoning_temperature,
            "reasoning_top_p": args.reasoning_top_p,
            "answer_prefill_after_reasoning": args.answer_prefill_after_reasoning,
            "reasoning_answer_prefix": args.reasoning_answer_prefix,
            "seed": args.seed,
        },
        "test": evaluate_sanity_train(
            model, tokenizer, tags, code_by_index, centroid_token_ids,
            test_examples, base_cache, device, dtype, label_token_ids, args, "test",
        ),
    }
    if args.eval_train or args.eval_teach:
        results["train_sanity"] = evaluate_sanity_train(
            model, tokenizer, tags, code_by_index, centroid_token_ids,
            train_eval_examples, base_cache, device, dtype, label_token_ids, args, "train_sanity",
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"Saved per-example class scores and metrics to {args.output.resolve()}")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
