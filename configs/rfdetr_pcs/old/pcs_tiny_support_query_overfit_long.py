"""Longer four-episode run to test whether separate-support box fit keeps improving."""

from pathlib import Path

cfg = {
    "seed": 73,
    "query_image_dir": Path("datasets/mscoco/train2017"),
    "query_annotation_file": Path("datasets/mscoco_prompt_split_tiny/train.json"),
    "support_image_dir": Path("datasets/mscoco/val2017"),
    "support_annotation_file": Path("datasets/mscoco_prompt_split_tiny/seen_support.json"),
    "resolution": 280,
    "positional_encoding_size": 37,
    "num_queries": 20,
    "bbox_reparam": False,
    "segmentation": True,
    "shots": 2,
    "negative_prompt_ratio": 0.0,
    "overfit_episodes": 4,
    "steps": 5000,
    "batch_size": 2,
    "lr": 0.0001,
    "support_loss_weight": 0.5,
    "clip_grad_norm": 0.1,
    "eval_every": 250,
    "max_checkpoints": 3,
    "device": "cuda",
    "output_dir": Path("logs/rfdetr_pcs_tiny_support_query_overfit_long"),
}
