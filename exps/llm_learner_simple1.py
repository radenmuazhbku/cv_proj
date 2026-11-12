"""
LLM learner: online in-context image teaching through a persistent KV cache.

The model is frozen. Each training image is queried, the predicted label is
appended as an assistant response, and the ground-truth feedback is appended
as the next user/assistant turn. No optimizer or backpropagation is used.

Edit the constants and prompt strings below; there is no argparse/config layer.
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

# Main model. Other choices:
MODEL_NAME = "Qwen/Qwen3-8B"
# MODEL_NAME = "Qwen/Qwen3-4B"
# MODEL_NAME = "Qwen/Qwen3-1.7B"
# MODEL_NAME = "Qwen/Qwen3-0.6B"
# MODEL_NAME = "HuggingFaceTB/SmolLM3-3B"
# MODEL_NAME = "HuggingFaceTB/SmolLM-135M"

DATA_DIR = "./data"
OUTPUT_DIR = "./runs/llm_learner"

IMAGE_SIZE = 32
PATCH_SIZE = 16
IMAGE_MODE = "byte_embeds"       # "byte_embeds" or "floats"
IMAGE_REDUCTION = "sum"          # "sum", "avg", or "random_proj"

# CIFAR-10 is used as the example dataset. Change names to arbitrary labels
# when using a different dataset with the same class-index ordering.
CLASS_NAMES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
# CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
#                "dog", "frog", "horse", "ship", "truck"]

TRAIN_SAMPLES_PER_CLASS = 100
TEST_SAMPLES_PER_CLASS = 100
SHUFFLE_SEED = 0
TEACHING_EPOCHS = 2
EVAL_EVERY = 1
EVALUATE_TRAIN_SUBSET = 100
SKIP_VALIDATION = False

# KV sink: retain the first N cache positions. Set much larger than the
# expected context to make this effectively a no-op while keeping one policy.
KV_SINK = 1_000_000

RANK_K = 3
ANSWER_TEMPERATURE = 0.0
SAMPLE_RANKING = False
REASONING = True
REASONING_ONLY_ON_ERROR = False
REASONING_MAX_NEW_TOKENS = 24
REASONING_TEMPERATURE = 0.7
REASONING_TOP_P = 0.9

# Prompt strings are deliberately kept here, beside the experiment settings.
INSTRUCTION_USER_PROMPT = (
    "Rank the {rank_k} most likely classes for each image. "
    "The allowed labels are: {class_names_list}. Return labels only, "
    "comma-separated. After each prediction, I will provide the true label. "
    "Then state whether your prediction was correct and briefly explain. "
    "Remember these rules. At the end, classify unseen test images."
)
INSTRUCTION_ASSISTANT_PROMPT = "Understood."

IMAGE_QUERY_PROMPT = (
    "Classify this image. Choose from: {class_names_list}. "
    "Return the top {rank_k} labels, ranked by likelihood, comma-separated. "
    "Use only the listed labels."
)

FEEDBACK_USER_PROMPT = (
    "The true label is {true_label}. Your ranked prediction was "
    "{predicted_labels}. {rank_feedback} {reasoning_instruction}"
)
FEEDBACK_ACKNOWLEDGEMENT = "Feedback received."
REASONING_PROMPT = (
    "Briefly explain why the prediction was correct or incorrect. "
    "Update your understanding of the relevant class features."
)
RANK_FEEDBACK_TOP1 = "The true label was ranked first."
RANK_FEEDBACK_IN_TOPK = "The true label was ranked {rank} of {rank_k}."
RANK_FEEDBACK_MISSED = "The true label was outside the top {rank_k}."


# =============================================================================
# SMALL UTILITIES
# =============================================================================

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

    try:
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
    except Exception as exc:
        vprint(f"WARNING: chat-template extraction failed: {exc}")
        try:
            one_turn = apply_chat_template(
                tokenizer,
                [{"role": "user", "content": sentinels["u1"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            marker = one_turn.index(sentinels["u1"])
            end = marker + len(sentinels["u1"])
            prefix, assistant_open = one_turn[:marker], one_turn[end:]
        except Exception:
            prefix, assistant_open = "", "\n"
        tags = {
            "global_prefix": prefix,
            "user_to_asst": assistant_open,
            "asst_tail": "",
            "turn_separator": assistant_open,
            "user_close": "",
            "asst_open": assistant_open,
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
    hidden_dim = model.config.hidden_size

    vprint(f"Model: {MODEL_NAME}")
    vprint(f"Device: {model_device}; dtype: {model_dtype}; hidden size: {hidden_dim}")
    return model, tokenizer, model_device, model_dtype, hidden_dim


@torch.no_grad()
def embed_text(model, tokenizer, text, device, dtype, add_special_tokens=True):
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
        return torch.einsum("dc,bncd->bnd", self.proj.weight, vectors)


def build_turn(model, tokenizer, tags, user_text, visual_embeds, assistant_text,
               device, dtype, first_turn=False, include_tail=True):
    """Build one complete user/assistant turn, inserting visual vectors in user content."""
    prefix = tags["global_prefix"] if first_turn else tags["turn_separator"]
    parts = [
        embed_text(model, tokenizer, prefix, device, dtype),
        embed_text(model, tokenizer, user_text, device, dtype),
    ]
    if visual_embeds is not None and visual_embeds.shape[1]:
        parts.append(visual_embeds.to(device=device, dtype=dtype))
    parts.append(embed_text(model, tokenizer, tags["user_to_asst"], device, dtype))
    parts.append(embed_text(model, tokenizer, assistant_text, device, dtype))
    if include_tail:
        parts.append(embed_text(model, tokenizer, tags["asst_tail"], device, dtype))
    parts = [part for part in parts if part.shape[1]]
    return torch.cat(parts, dim=1)


def build_query_turn(model, tokenizer, tags, user_text, visual_embeds,
                     device, dtype, first_turn=False):
    """Build a user turn ending at the assistant generation boundary."""
    prefix = tags["global_prefix"] if first_turn else tags["turn_separator"]
    assistant_open = tags["asst_open"] or tags["user_to_asst"]
    parts = [
        embed_text(model, tokenizer, prefix, device, dtype),
        embed_text(model, tokenizer, user_text, device, dtype),
    ]
    if visual_embeds is not None and visual_embeds.shape[1]:
        parts.append(visual_embeds.to(device=device, dtype=dtype))
    parts.append(embed_text(model, tokenizer, tags["user_close"], device, dtype))
    parts.append(embed_text(model, tokenizer, assistant_open, device, dtype))
    parts = [part for part in parts if part.shape[1]]
    return torch.cat(parts, dim=1)


@torch.no_grad()
def prefill(model, embeds, past_kv=None, use_cache=True):
    output = model(
        inputs_embeds=embeds,
        past_key_values=past_kv,
        use_cache=use_cache,
        return_dict=True,
    )
    return output.logits, output.past_key_values if use_cache else None


def apply_kv_sink(cache):
    """Streaming-style sink policy; KV_SINK is intentionally very large."""
    if cache is None:
        return None
    seq_len = kv_seq_len(cache)
    if seq_len <= KV_SINK:
        return cache
    # Retain the first KV_SINK positions. This branch should not be reached
    # with the default setting, but makes the policy explicit and deterministic.
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


# =============================================================================
# PREDICTION, TEACHING, AND EVALUATION
# =============================================================================

def class_token_ids(tokenizer, label):
    ids = tokenizer(" " + label, add_special_tokens=False).input_ids
    if not ids:
        raise ValueError(f"Label {label!r} produced no tokens")
    return [int(token_id) for token_id in ids]


@torch.no_grad()
def predict_image(model, tokenizer, tags, image_tokenizer, image, class_names,
                  past_kv, device, dtype, rank_k=RANK_K):
    image_embeds = image_tokenizer(image.unsqueeze(0).to(device)).to(dtype)
    prompt = IMAGE_QUERY_PROMPT.format(
        class_names_list=", ".join(class_names),
        rank_k=min(rank_k, len(class_names)),
    )
    query = build_query_turn(
        model, tokenizer, tags, prompt, image_embeds,
        device, dtype, first_turn=(kv_seq_len(past_kv) == 0),
    )
    logits, query_cache = prefill(model, query, past_kv, use_cache=True)
    next_logits = logits[0, -1].float()

    # Score first token for each class. For multi-token/prefix-colliding labels,
    # use full sequence probability conditioned on the same query context.
    token_ids = [class_token_ids(tokenizer, name) for name in class_names]
    first_ids = [ids[0] for ids in token_ids]
    if len(set(first_ids)) == len(first_ids):
        label_logits = torch.log_softmax(next_logits, dim=-1)[
            torch.tensor(first_ids, device=next_logits.device)
        ]
    else:
        label_scores = []
        for ids in token_ids:
            score = 0.0
            current_logits = next_logits
            current_cache = copy.deepcopy(query_cache)
            for pos, token_id in enumerate(ids):
                score += torch.log_softmax(current_logits, dim=-1)[token_id].item()
                if pos + 1 < len(ids):
                    token = torch.tensor([[token_id]], device=device)
                    token_embed = model.get_input_embeddings()(token).to(dtype)
                    out_logits, current_cache = prefill(
                        model, token_embed, current_cache, use_cache=True
                    )
                    current_logits = out_logits[0, -1].float()
            label_scores.append(score)
        label_logits = torch.tensor(label_scores, device=next_logits.device)

    probs = torch.softmax(label_logits, dim=0)
    scores = {name: float(prob) for name, prob in zip(class_names, probs.tolist())}
    count = min(rank_k, len(class_names))
    if SAMPLE_RANKING and ANSWER_TEMPERATURE > 0:
        weights = torch.softmax(label_logits / ANSWER_TEMPERATURE, dim=0)
        indices = torch.multinomial(weights, count, replacement=False).tolist()
    else:
        indices = torch.argsort(label_logits, descending=True)[:count].tolist()
    ranked = [class_names[index] for index in indices]

    if VERBOSE:
        print("Image query:")
        print(prompt)
        print("<image embeddings: {} patch vectors>".format(image_embeds.shape[1]))
        prefix = tags["global_prefix"] if kv_seq_len(past_kv) == 0 else tags["turn_separator"]
        assistant_open = tags["asst_open"] or tags["user_to_asst"]
        print("FULL QUERY FLOW:")
        print(repr(prefix + prompt + "\n<embed>" + tags["user_close"] + assistant_open))
        print("Chat boundary:", repr(tags["user_close"]), repr(assistant_open))
        print("Candidate scores:")
        for name, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
            print(f"  {name:12s} {score:.8g}")
        print("Selected ranking:", ", ".join(ranked))
    return ranked, scores, query_cache


@torch.no_grad()
def decode_reasoning(model, tokenizer, past_kv, next_logits, device, dtype):
    generated = []
    current_logits = next_logits.float()
    current_cache = past_kv
    eos_id = tokenizer.eos_token_id

    for _ in range(REASONING_MAX_NEW_TOKENS):
        if REASONING_TEMPERATURE <= 0:
            next_id = int(torch.argmax(current_logits).item())
        else:
            probs = torch.softmax(current_logits / REASONING_TEMPERATURE, dim=-1)
            sorted_probs, sorted_ids = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=0)
            remove = cumulative - sorted_probs >= REASONING_TOP_P
            sorted_probs[remove] = 0
            filtered = torch.zeros_like(probs).scatter(0, sorted_ids, sorted_probs)
            filtered /= filtered.sum()
            next_id = int(torch.multinomial(filtered, 1).item())

        if eos_id is not None and next_id == eos_id:
            break
        generated.append(next_id)
        token = torch.tensor([[next_id]], device=device)
        token_embeds = model.get_input_embeddings()(token).to(dtype)
        logits, current_cache = prefill(
            model, token_embeds, current_cache, use_cache=True
        )
        current_logits = logits[0, -1].float()

    return tokenizer.decode(generated, skip_special_tokens=True), current_cache


@torch.no_grad()
def teach_one(model, tokenizer, tags, image_tokenizer, image, true_label,
              class_names, past_kv, device, dtype):
    ranked, _, query_cache = predict_image(
        model, tokenizer, tags, image_tokenizer, image,
        class_names, past_kv, device, dtype,
    )
    predicted = ", ".join(ranked)
    answer = embed_text(
        model, tokenizer, " " + predicted + tags["asst_tail"],
        device, dtype, add_special_tokens=False,
    )
    _, query_cache = prefill(model, answer, query_cache, use_cache=True)

    true_rank = ranked.index(true_label) + 1 if true_label in ranked else None
    correct = ranked[0] == true_label
    if true_rank == 1:
        rank_feedback = RANK_FEEDBACK_TOP1
    elif true_rank is not None:
        rank_feedback = RANK_FEEDBACK_IN_TOPK.format(rank=true_rank, rank_k=len(ranked))
    else:
        rank_feedback = RANK_FEEDBACK_MISSED.format(rank_k=len(ranked))

    reasoning_instruction = REASONING_PROMPT if REASONING else ""
    feedback = FEEDBACK_USER_PROMPT.format(
        true_label=true_label,
        predicted_labels=predicted,
        rank_feedback=rank_feedback,
        reasoning_instruction=reasoning_instruction,
    ).strip()

    if REASONING and (not REASONING_ONLY_ON_ERROR or not correct):
        feedback_embeds = build_query_turn(
            model, tokenizer, tags, feedback, None, device, dtype, first_turn=False
        )
        feedback_logits, query_cache = prefill(
            model, feedback_embeds, query_cache, use_cache=True
        )
        vprint("Feedback user turn:", feedback)
        reasoning_text, query_cache = decode_reasoning(
            model, tokenizer, query_cache, feedback_logits[0, -1],
            device, dtype,
        )
        tail = embed_text(
            model, tokenizer, tags["asst_tail"], device, dtype,
            add_special_tokens=False,
        )
        if tail.shape[1]:
            _, query_cache = prefill(model, tail, query_cache, use_cache=True)
        vprint("Generated assistant feedback:", reasoning_text)
    else:
        feedback_turn = build_turn(
            model, tokenizer, tags, feedback, None,
            FEEDBACK_ACKNOWLEDGEMENT, device, dtype,
            first_turn=False, include_tail=True,
        )
        _, query_cache = prefill(model, feedback_turn, query_cache, use_cache=True)

    return apply_kv_sink(query_cache), correct, ranked


@torch.no_grad()
def evaluate_dataset(model, tokenizer, tags, image_tokenizer, data_by_class,
                     class_names, past_kv, device, dtype, description):
    total = sum(len(images) for images in data_by_class.values())
    if total == 0:
        return 0.0, [], []

    true_labels, predicted_labels = [], []
    correct = 0
    progress = tqdm(total=total, desc=description, leave=False)

    for class_id, true_label in enumerate(class_names):
        for image in data_by_class.get(class_id, []):
            try:
                # Some Transformers cache implementations mutate the supplied
                # cache in place. Evaluate on a copy so test queries never teach.
                eval_context = copy.deepcopy(past_kv)
                ranked, _, _ = predict_image(
                    model, tokenizer, tags, image_tokenizer, image,
                    class_names, eval_context, device, dtype,
                )
                prediction = ranked[0]
            except Exception as exc:
                vprint(f"Evaluation prediction failed for {true_label}: {exc}")
                prediction = class_names[0]

            correct += int(prediction == true_label)
            true_labels.append(true_label)
            predicted_labels.append(prediction)
            progress.update(1)
            progress.set_postfix(accuracy=f"{100 * correct / len(true_labels):.1f}%")

    progress.close()
    accuracy = 100.0 * correct / total
    print(f"{description}: top-1 accuracy = {accuracy:.2f}%")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return accuracy, true_labels, predicted_labels


# =============================================================================
# DATA
# =============================================================================

def load_data():
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

    def collect(dataset, per_class):
        buckets = {i: [] for i in range(len(CLASS_NAMES))}
        for image, label in tqdm(dataset, desc="Loading dataset"):
            if label < len(CLASS_NAMES) and len(buckets[label]) < per_class:
                buckets[label].append(image)
            if all(len(items) >= per_class for items in buckets.values()):
                break
        return buckets

    train_by_class = collect(train_set, TRAIN_SAMPLES_PER_CLASS)
    test_by_class = collect(test_set, TEST_SAMPLES_PER_CLASS)
    return train_by_class, test_by_class


def make_training_pairs(data_by_class, epoch):
    pairs = []
    for class_id, images in data_by_class.items():
        pairs.extend((image, CLASS_NAMES[class_id]) for image in images)
    rng = random.Random(None if SHUFFLE_SEED is None else SHUFFLE_SEED + epoch)
    rng.shuffle(pairs)
    return pairs


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, device, dtype, hidden_dim = load_model()
    tags = extract_chat_tags(tokenizer)

    image_tokenizer = ImageTokenizer(
        PATCH_SIZE, IMAGE_SIZE, IMAGE_MODE, IMAGE_REDUCTION,
        hidden_dim, model.get_input_embeddings().weight,
    ).to(device=device)

    vprint("\nInitial instruction turn:")
    instruction = INSTRUCTION_USER_PROMPT.format(
        rank_k=min(RANK_K, len(CLASS_NAMES)),
        class_names_list=", ".join(CLASS_NAMES),
    )
    vprint("USER:", instruction)
    vprint("ASSISTANT:", INSTRUCTION_ASSISTANT_PROMPT)
    instruction_turn = build_turn(
        model, tokenizer, tags, instruction, None,
        INSTRUCTION_ASSISTANT_PROMPT, device, dtype,
        first_turn=True, include_tail=True,
    )
    _, past_kv = prefill(model, instruction_turn, use_cache=True)
    past_kv = apply_kv_sink(past_kv)
    print(f"Instruction context: {kv_seq_len(past_kv)} KV positions")

    train_by_class, test_by_class = load_data()
    train_eval_by_class = {
        class_id: images[:EVALUATE_TRAIN_SUBSET]
        for class_id, images in train_by_class.items()
    }
    history = []
    final_true, final_pred = [], []

    for epoch in range(1, TEACHING_EPOCHS + 1):
        print(f"\n{'=' * 24} EPOCH {epoch}/{TEACHING_EPOCHS} {'=' * 24}")
        pairs = make_training_pairs(train_by_class, epoch)
        online_correct = 0

        for index, (image, true_label) in enumerate(
            tqdm(pairs, desc=f"Epoch {epoch}: online teaching")
        ):
            past_kv, was_correct, ranked = teach_one(
                model, tokenizer, tags, image_tokenizer, image,
                true_label, CLASS_NAMES, past_kv, device, dtype,
            )
            online_correct += int(was_correct)
            if VERBOSE and (index == 0 or (index + 1) % 10 == 0):
                print(
                    f"[{index + 1}/{len(pairs)}] true={true_label}, "
                    f"pred={', '.join(ranked)}, "
                    f"correct={was_correct}, "
                    f"online_acc={100 * online_correct / (index + 1):.1f}%, "
                    f"kv={kv_seq_len(past_kv)}"
                )

        online_accuracy = 100 * online_correct / max(1, len(pairs))
        print(f"Epoch {epoch} online pre-feedback accuracy: {online_accuracy:.2f}%")

        should_evaluate = (epoch % EVAL_EVERY == 0) or (epoch == TEACHING_EPOCHS)
        train_accuracy = None
        validation_accuracy = None

        if should_evaluate:
            print(f"Evaluating epoch {epoch}; KV length={kv_seq_len(past_kv)}")
            if EVALUATE_TRAIN_SUBSET > 0:
                train_accuracy, _, _ = evaluate_dataset(
                    model, tokenizer, tags, image_tokenizer,
                    train_eval_by_class, CLASS_NAMES, past_kv, device, dtype,
                    f"Epoch {epoch} train evaluation",
                )
            if not SKIP_VALIDATION:
                validation_accuracy, final_true, final_pred = evaluate_dataset(
                    model, tokenizer, tags, image_tokenizer,
                    test_by_class, CLASS_NAMES, past_kv, device, dtype,
                    f"Epoch {epoch} validation",
                )

            history.append({
                "epoch": epoch,
                "kv_len": kv_seq_len(past_kv),
                "online_acc": online_accuracy,
                "train_acc": train_accuracy,
                "val_acc": validation_accuracy,
            })
            print(
                f"Epoch {epoch}: online={online_accuracy:.2f}%, "
                f"train={train_accuracy}, validation={validation_accuracy}, "
                f"KV={kv_seq_len(past_kv)}"
            )

    print("\nTraining complete.")
    print(f"Final KV cache length: {kv_seq_len(past_kv)}")

    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["epoch", "kv_len", "online_acc", "train_acc", "val_acc"],
        )
        writer.writeheader()
        writer.writerows(history)
    print(f"Saved metrics: {metrics_path}")

    if final_true:
        print("\nFinal validation per-class accuracy:")
        for name in CLASS_NAMES:
            indices = [i for i, label in enumerate(final_true) if label == name]
            count = len(indices)
            class_correct = sum(final_pred[i] == name for i in indices)
            print(f"  {name:12s} {class_correct}/{count}")

    print("Done.")


if __name__ == "__main__":
    main()
