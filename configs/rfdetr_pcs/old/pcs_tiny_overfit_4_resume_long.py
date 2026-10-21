"""Continue the original four-episode checkpoint for 5,000 more steps."""

from pathlib import Path

cfg = {
    "seed": 73,
    "train_image_dir": Path("datasets/mscoco/train2017"),
    "train_annotation_file": Path("datasets/mscoco_prompt_split_tiny/train.json"),
    "resolution": 280,
    "positional_encoding_size": 37,
    "num_queries": 20,
    "bbox_reparam": False,
    "segmentation": True,
    "include_prompt_box": True,
    "negative_ratio": 0.0,
    "negative_prompt_ratio": 0.0,
    "stage_sizes": [4],
    "stage_steps": [5000],
    "batch_size": 4,
    "num_workers": 0,
    "lr": 0.0001,
    "clip_grad_norm": 0.1,
    "eval_every": 100,
    "sanity_samples": 4,
    "max_checkpoints": 3,
    "keep_checkpoint_steps": [6500],
    "resume_checkpoint": Path("logs/rfdetr_pcs_tiny_overfit/checkpoint_step_001500.pt"),
    "device": "cuda",
    "output_dir": Path("logs/rfdetr_pcs_tiny_overfit_4_resume_long"),
}
