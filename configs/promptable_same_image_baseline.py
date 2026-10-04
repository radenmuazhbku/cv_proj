"""Historical same-image box-prompt training settings."""

from pathlib import Path

cfg = {
    "image_dir": Path("datasets/mscoco/train2017"),
    "annotation_file": Path("datasets/mscoco_prompt_split/train.json"),
    "validation_image_dir": Path("datasets/mscoco/val2017"),
    "validation_annotation_file": Path("datasets/mscoco_prompt_split/seen_val.json"),
    "test_image_dir": Path("datasets/mscoco/val2017"),
    "test_annotation_file": Path("datasets/mscoco_prompt_split/unseen_test.json"),
    "eval_every": 1_000,
    "eval_batch_size": 4,
    "max_validation_examples": None,
    "max_test_examples": None,
    "steps": 10_000,
    "log_every": 1_000,
    "batch_size": 1,
    "negative_ratio": 1.0,
    "negative_iou_threshold": 0.1,
    "max_examples": 10_000,
    "seed": 7,
    "lr": 1e-4,
    "objectness_weight": 1.0,
    "bbox_weight": 5.0,
    "giou_weight": 2.0,
    "resolution": 560,
    "output": Path("logs/promptable_same_image_baseline.pt"),
    "device": "cuda",
}
