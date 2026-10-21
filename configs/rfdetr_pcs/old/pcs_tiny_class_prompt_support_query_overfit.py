"""Four separate-image episodes with auxiliary support-image detection loss."""

from pathlib import Path

cfg = {
    "conditioning_mode": "class_embedding",
    "seed": 73,
    "pretrain_weights": "rf-detr-base.pth",
    "pretrained_class_count": 90,
    "num_prompt_classes": 91,
    "init_class_embeddings_from_detector": True,
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
    "steps": 1000,
    "batch_size": 2,
    "lr": 0.0001,
    "support_loss_weight": 0.5,
    "clip_grad_norm": 0.1,
    "eval_every": 100,
    "device": "cuda",
    "output_dir": Path("logs/rfdetr_pcs_tiny_class_prompt_support_query_overfit"),
}
