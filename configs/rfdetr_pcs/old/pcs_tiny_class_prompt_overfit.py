"""Same-image PCS overfit ladder: 4 -> 100 -> 1000 fixed training episodes."""

from pathlib import Path

cfg = {
    "conditioning_mode": "class_embedding",
    "seed": 73,
    "pretrain_weights": "rf-detr-base.pth",
    "train_image_dir": Path("datasets/mscoco/train2017"),
    "train_annotation_file": Path("datasets/mscoco_prompt_split_tiny/train.json"),
    "resolution": 280,
    # Keep the pretrained DINOv2 positional grid; the input resolution is resized
    # independently and RF-DETR interpolates the feature position encodings.
    "positional_encoding_size": 37,
    "num_queries": 20,
    # Random learned reference points start at zero; bbox_reparam=True interprets
    # their zero width/height directly and collapses initial boxes.
    "bbox_reparam": False,
    "segmentation": True,
    "include_prompt_box": True,
    "negative_ratio": 0.0,
    "negative_prompt_ratio": 0.0,
    "negative_class_ratio": 0.0,
    "pretrained_class_count": 90,
    "num_prompt_classes": 91,
    "init_class_embeddings_from_detector": True,
    "stage_sizes": [4, 100, 1000],
    "stage_steps": [1500, 2000, 3000],
    "batch_size": 4,
    "num_workers": 0,
    "lr": 0.0001,
    "clip_grad_norm": 0.1,
    "eval_every": 100,
    "sanity_samples": 8,
    "max_checkpoints": 3,
    "keep_checkpoint_steps": [1500, 3500, 6500],
    "device": "cuda",
    "output_dir": Path("logs/rfdetr_pcs_tiny_class_prompt_overfit"),
}
