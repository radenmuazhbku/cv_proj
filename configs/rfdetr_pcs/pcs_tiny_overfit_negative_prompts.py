"""Same-image diagnostic with one annotation-sampled negative prompt per positive."""

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
    # Empty-remainder episodes are disabled so this isolates negative boxes.
    "negative_ratio": 0.0,
    # 1.0 means an expected one negative-class box per positive prompt.
    "negative_prompt_ratio": 1.0,
    "max_prompt_boxes": 1,
    "stage_sizes": [4],
    "stage_steps": [500],
    "batch_size": 4,
    "num_workers": 0,
    "lr": 0.0001,
    "clip_grad_norm": 0.1,
    "eval_every": 100,
    "sanity_samples": 4,
    "max_checkpoints": 3,
    "keep_checkpoint_steps": [500],
    "device": "cuda",
    "output_dir": Path("logs/rfdetr_pcs_tiny_overfit_negative_prompts"),
}
