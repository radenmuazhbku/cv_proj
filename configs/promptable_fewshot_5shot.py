"""Cross-image visual few-shot configuration (one to five support shots)."""

from pathlib import Path

cfg = {
    "seed": 7,
    "train_image_dir": Path("datasets/mscoco/train2017"),
    "train_annotation_file": Path("datasets/mscoco_prompt_split/train.json"),
    "validation_image_dir": Path("datasets/mscoco/val2017"),
    "validation_support_image_dir": Path("datasets/mscoco/val2017"),
    "validation_annotation_file": Path("datasets/mscoco_prompt_split/seen_val.json"),
    "validation_support_annotation_file": Path("datasets/mscoco_prompt_split/seen_support.json"),
    "test_image_dir": Path("datasets/mscoco/val2017"),
    "test_support_image_dir": Path("datasets/mscoco/val2017"),
    "test_annotation_file": Path("datasets/mscoco_prompt_split/unseen_test.json"),
    "test_support_annotation_file": Path("datasets/mscoco_prompt_split/unseen_support.json"),
    "episodes": 10_000,
    "max_shots": 5,
    "negative_ratio": 0.5,
    "negative_eval_per_class": 100,
    "steps": 10_000,
    "eval_every": 1_000,
    "selection_metric": "coco_map",
    "coco_selection_images": 500,
    "batch_size": 2,
    "coco_eval_batch_size": 8,
    "num_queries": 100,
    "resolution": 560,
    "lr": 1e-4,
    "output": Path("logs/promptable_fewshot_5shot.pt"),
    "figure_dir": Path("logs/promptable_fewshot_figures/5shot"),
}
