"""Full FSOD Training (MS COCO, 5-shot): standard 200k-step compute budget."""

from pathlib import Path
AUG_AGGRESSIVE = {
    "HorizontalFlip": {"p": 0.5},
    "VerticalFlip": {"p": 0.5},
    "Rotate": {"limit": 45, "p": 0.5},
    "Affine": {
        "scale": (0.8, 1.2),
        "translate_percent": (-0.1, 0.1),
        "rotate": (-15, 15),
        "shear": (-5, 5),
        "p": 0.5,
    },
    "ColorJitter": {
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.2,
        "hue": 0.1,
        "p": 0.5,
    },
}

AUG_CONSERVATIVE = {
    "HorizontalFlip": {"p": 0.5},
    "RandomBrightnessContrast": {
        "brightness_limit": 0.1,
        "contrast_limit": 0.1,
        "p": 0.3,
    },
}

cfg = {
    "aug_config": AUG_CONSERVATIVE,
    "mode": "support_query",
        # "conditioning_schedule": [
    #     {"mode": "class_embedding", "steps": 100_000},
    #     {"mode": "visual_prompt", "steps": 800_000},
    # ],
    "conditioning_mode": "visual_prompt",

        "conditioning_sampling_schedule": [
        {"class_embedding": 0.5, "visual_prompt": 0.5, "steps": 250_000},
        {"class_embedding": 0.2, "visual_prompt": 0.8, "steps": 250_000}
    ],
    "steps": 500_000,
    "eval_every": 50_000,
    "log_every": 10_000,
    "seed": 7,
    "use_pretrained_model": False,
    # "init_checkpoint": Path("rf-detr-base.pth"),
    # Base training dataset (COCO 60 Base classes, novel classes removed)
    "train_image_dir": Path("datasets/mscoco/train2017"),
    "train_annotation_file": Path("datasets/mscoco_fsod_splits/coco_base_train.json"),
    "train_support_image_dir": Path("datasets/mscoco/train2017"),
    "train_support_annotation_file": Path("datasets/mscoco_fsod_splits/coco_base_train.json"),
    # Validation & Test (5-shot support conditioning, full 5,000 val2017 test queries)
    "validation_image_dir": Path("datasets/mscoco/val2017"),
    "validation_annotation_file": Path("datasets/mscoco_fsod_splits/coco_val_full.json"),
    "validation_support_image_dir": Path("datasets/mscoco/train2017"),
    "validation_support_annotation_file": Path("datasets/mscoco_fsod_splits/coco_fsod_novel_5shot_seed0.json"),
    "test_image_dir": Path("datasets/mscoco/val2017"),
    "test_annotation_file": Path("datasets/mscoco_fsod_splits/coco_val_full.json"),
    "test_support_image_dir": Path("datasets/mscoco/train2017"),
    "test_support_annotation_file": Path("datasets/mscoco_fsod_splits/coco_fsod_novel_5shot_seed0.json"),
    "episodes": 200_000,
    "shots": 5,
    "negative_prompt_ratio": 0.0,
    "negative_class_ratio": 0.1,
    "validation_negative_class_ratio": 0.0,
    "test_negative_class_ratio": 0.0,
    "support_loss_weight": 0.5,
    "include_prompt_box_in_loss": False,
    "batch_repeat": 1,
    "batch_size": 2,
    "eval_batch_size": 2,
    "num_workers": 4,
    "resolution": 560,
    "num_queries": 100,
    "positional_encoding_size": 37,
    "bbox_reparam": False,
    "segmentation": True,
    "lr": 1e-4,
    "weight_decay": 0.05,
    "lr_scheduler": "cosine",
    "warmup_steps": 0.05,
    "warmup_start_factor": 0.1,
    "min_lr_ratio": 0.01,
    "clip_grad_norm": 0.1,
    "validation_episodes": 500,
    "test_episodes": 2_000,
    "compute_coco_eval": True,
    # Standard 20 COCO Novel category IDs:
    "novel_category_ids": [1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72],
    "selection_metric": "matched_box_iou",
    "train_sanity": True,
    "train_sanity_samples": 8,
    "validation_sanity": True,
    "validation_sanity_samples": 8,
    "test_sanity": True,
    "test_sanity_samples": 16,
    "inference_plot_mode": "compact",
    "inference_extra_predictions": 0,
    "evaluate_test_at_end": True,
    "max_checkpoints": 3,
    "device": "cuda",
    "output_dir": None
    # "figure_dir": Path("logs/rfdetr_pcs_coco_fs5/figures"),
}

