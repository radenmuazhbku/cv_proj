"""Single-pass CIFAR-10 hash-ICL sanity check for Qwen3.

Builds one complete conversation:
  user: task instruction
  assistant: Understood.
  user: all labeled CIFAR-10 hash examples + one unlabeled hash query
  assistant: <think>...</think> (generation position)

The entire prompt is prefetched in ONE model forward pass. The next token is
selected directly from the returned logits; no teaching/KV-cache loop and no
second model forward pass are used.
"""

import hashlib
import random

import torch
import torchvision
import torchvision.transforms as T
from transformers import AutoTokenizer, AutoModelForCausalLM


# =============================================================================
# CONFIG
# =============================================================================

MODEL_NAME = "Qwen/Qwen3-8B"
DATA_DIR = "./datasets"

TRAIN_SUBSET_SIZE = 100
SHUFFLE_SEED = 0          # None = original CIFAR-10 train dataset order
EVAL_DATASET_INDEX = None # None = random item from selected subset; else original CIFAR index

IMAGE_SIZE = 32
PATCH_SIZE = 16
PATCH_COUNT = (IMAGE_SIZE // PATCH_SIZE) ** 2  # Four hash characters
HASH_SEPARATOR = "|"

CLASS_NAMES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
VERBOSE = True
TOP_K = 20

EMBED_PREFIX = "embeds: "
LABEL_SEPARATOR = " : "
INITIAL_ACK = "Understood."
THINK_OPEN = "<think>\n"
THINK_CLOSE = "\n</think>\n\n"

INITIAL_INSTRUCTION = (
    "You are learning an image classification task through in-context examples. "
    "You will see text of the form embeds: followed by a hash identifier for an "
    "image, then a colon and its class label. Treat these sequences as labeled "
    "examples. There may be any number of labeled examples. The final item is "
    "an image hash followed by a colon with no label. Infer its class from the "
    "preceding examples. Your task is to continue immediately after the final "
    "hanging colon with exactly one class label and nothing else. Do not provide "
    "reasoning, explanations, or extra text. Available labels: "
    + ", ".join(CLASS_NAMES)
    + "."
)


# =============================================================================
# HASH + CIFAR-10
# =============================================================================

def index_to_hash(index, length=PATCH_COUNT, separator=HASH_SEPARATOR):
    """Same deterministic SHA-256 prefix used by the full ICL script."""
    digest = hashlib.sha256(str(index).encode("utf-8")).hexdigest()
    return separator.join(digest[:length])


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

    # Keep the original CIFAR-10 training-split index with each sample.
    examples = []
    for dataset_index in indices:
        class_id = int(train_set.targets[dataset_index])
        label = CLASS_NAMES[class_id]
        examples.append((dataset_index, label))

    return train_set, examples


# =============================================================================
# CHAT-TEMPLATE EXTRACTION (same sentinel-based approach as short script)
# =============================================================================

def apply_chat_template(tokenizer, messages, **kwargs):
    try:
        return tokenizer.apply_chat_template(
            messages, enable_thinking=False, **kwargs
        )
    except (TypeError, ValueError):
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
        tokenizer, messages, tokenize=False, add_generation_prompt=False
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


def build_conversation(tags, examples, query_hash):
    parts = [
        tags["global_prefix"],
        INITIAL_INSTRUCTION,
        tags["user_to_asst"],
        INITIAL_ACK,
        tags["turn_separator"],
    ]

    # All demonstrations are in one user message, one example per line.
    for dataset_index, label in examples:
        hash_text = index_to_hash(dataset_index)
        parts.extend([
            "\n" + EMBED_PREFIX,
            hash_text,
            LABEL_SEPARATOR + label,
        ])

    # Final query: same hash format, deliberately no label after the colon.
    parts.extend([
        "\n" + EMBED_PREFIX,
        query_hash[1],
        LABEL_SEPARATOR,
        tags["user_close"],
        tags["asst_open"],
        THINK_OPEN,
        THINK_CLOSE,
    ])
    return "".join(parts)


# =============================================================================
# MAIN: ONE PREFILL FORWARD PASS, THEN READ NEXT-TOKEN DISTRIBUTION
# =============================================================================

def main():
    print("Loading CIFAR-10 training split...")
    train_set, examples = load_cifar10_examples()

    # Choose a query from the selected subset. It may also appear among the
    # demonstrations: this is an explicit hash-to-label recall sanity check.
    selected_indices = {dataset_index for dataset_index, _ in examples}
    if EVAL_DATASET_INDEX is None:
        eval_index, eval_label = random.choice(examples)
    else:
        if EVAL_DATASET_INDEX not in selected_indices:
            raise ValueError(
                f"EVAL_DATASET_INDEX={EVAL_DATASET_INDEX} is not in the selected "
                f"subset of {len(examples)} CIFAR-10 training examples."
            )
        eval_index, eval_label = next(
            (idx, label) for idx, label in examples if idx == EVAL_DATASET_INDEX
        )

    query_hash = (eval_index, index_to_hash(eval_index))

    print("\nCIFAR-10 class mapping:")
    for class_id, class_name in enumerate(train_set.classes):
        print(f"  CIFAR class {class_id:2d} {class_name:12s} -> {CLASS_NAMES[class_id]}")

    print("\nSelected training examples (original CIFAR-10 index, hash, label):")
    for dataset_index, label in examples:
        print(
            f"  index={dataset_index:5d}  hash={index_to_hash(dataset_index):11s}  "
            f"label={label} ({train_set.classes[int(train_set.targets[dataset_index])]})"
        )

    print("\nFinal query:")
    print(f"  original train index = {eval_index}")
    print(f"  hash                 = {query_hash[1]}")
    print(f"  CIFAR-10 class       = {train_set.classes[int(train_set.targets[eval_index])]}")
    print(f"  reference label      = {eval_label}")

    print("\nLoading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    device = model.get_input_embeddings().weight.device

    tags = extract_chat_tags(tokenizer)
    conversation = build_conversation(tags, examples, query_hash)

    if VERBOSE:
        print("\n" + "=" * 88)
        print("FULL RAW PREFILL CONVERSATION (INCLUDING CHAT TAGS)")
        print("=" * 88)
        print(conversation, end="")
        print("\n" + "=" * 88)
        print("END OF RAW PREFILL CONVERSATION")
        print("=" * 88)

    input_ids = tokenizer(
        conversation, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)

    print(f"\nInput token count: {input_ids.shape[1]}")
    print("Last input tokens:", repr(tokenizer.decode(input_ids[0, -40:])))

    # Exactly ONE model forward pass. No cache continuation or teaching loop.
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )

    logits = outputs.logits[0, -1, :].float()
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

    # Report the model's probability for each valid class label token.
    label_token_ids = {}
    for label in CLASS_NAMES:
        ids = tokenizer(label, add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise ValueError(
                f"Class label {label!r} is not a single token: {ids}. "
                "Use labels that each map to one token for this next-token test."
            )
        label_token_ids[label] = ids[0]

    class_logits = torch.tensor(
        [logits[label_token_ids[label]].item() for label in CLASS_NAMES],
        dtype=torch.float64,
    )
    class_probs = torch.softmax(class_logits, dim=0)

    print("\nClass-token scores:")
    for label, token_id, class_logit, class_prob in zip(
        CLASS_NAMES,
        [label_token_ids[label] for label in CLASS_NAMES],
        class_logits.tolist(),
        class_probs.tolist(),
    ):
        full_prob = float(full_vocab_probs[token_id].item())
        print(
            f"  {label}: token_id={token_id:6d}  logit={class_logit:10.4f}  "
            f"class_softmax={class_prob:.8f}  full_vocab_softmax={full_prob:.8e}"
        )

    # Greedy decode from the already-computed logits; no extra forward pass.
    predicted_token_id = int(torch.argmax(logits).item())
    predicted_text = tokenizer.decode(
        [predicted_token_id], skip_special_tokens=True
    ).strip()
    best_class_position = int(torch.argmax(class_probs).item())
    best_class = CLASS_NAMES[best_class_position]

    print("\nGreedy next token:")
    print(f"  token_id = {predicted_token_id}")
    print(f"  decoded  = {predicted_text!r}")
    print("\nBest among the ten class labels (class-only softmax):")
    print(f"  prediction = {best_class}")
    print(f"  reference  = {eval_label}")
    print(f"  correct    = {best_class == eval_label}")


if __name__ == "__main__":
    main()
