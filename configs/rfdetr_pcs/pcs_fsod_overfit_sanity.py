"""Fast end-to-end FSOD sanity overfit run with bounding box and segmentation.

Exercises:
- Support-query multi-shot conditioning (strictly disjoint support and query images)
- Mask instance segmentation training (loss_mask_ce, loss_mask_dice)
- Bounding box detection (loss_bbox, loss_giou)
- Presence classification (loss_presence)
- Standard COCOeval evaluation for both bbox and segm
"""

from pathlib import Path

cfg = {
    "mode": "support_query",
    "conditioning_mode": "visual_prompt",
    "seed": 42,
    "use_pretrained_model": True,
    "init_checkpoint": Path("rf-detr-base.pth"),
    "train_image_dir": Path("datasets/VOC2012/JPEGImages"),
    "train_annotation_file": Path("datasets/VOC2012/annotations/voc_2012_val.json"),
    "train_support_image_dir": Path("datasets/VOC2012/JPEGImages"),
    "train_support_annotation_file": Path("datasets/voc_fsod_splits/split_1/voc_split1_novel_2shot_seed0.json"),
    "validation_image_dir": Path("datasets/VOC2012/JPEGImages"),
    "validation_annotation_file": Path("datasets/VOC2012/annotations/voc_2012_val.json"),
    "validation_support_image_dir": Path("datasets/VOC2012/JPEGImages"),
    "validation_support_annotation_file": Path("datasets/voc_fsod_splits/split_1/voc_split1_novel_2shot_seed0.json"),
    "test_image_dir": Path("datasets/VOC2012/JPEGImages"),
    "test_annotation_file": Path("datasets/VOC2012/annotations/voc_2012_val.json"),
    "test_support_image_dir": Path("datasets/VOC2012/JPEGImages"),
    "test_support_annotation_file": Path("datasets/voc_fsod_splits/split_1/voc_split1_novel_2shot_seed0.json"),
    "episodes": 200,
    "shots": 2,
    "negative_prompt_ratio": 0.0,
    "negative_class_ratio": 0.0,
    "support_loss_weight": 0.5,
    "include_prompt_box_in_loss": False,
    "batch_repeat": 1,
    "batch_size": 2,
    "eval_batch_size": 2,
    "num_workers": 0,
    "resolution": 280,
    "num_queries": 20,
    "positional_encoding_size": 37,
    "bbox_reparam": False,
    "segmentation": True,
    "lr": 1e-4,
    "weight_decay": 0.01,
    "lr_scheduler": "cosine",
    "warmup_steps": 10,
    "warmup_start_factor": 0.1,
    "min_lr_ratio": 0.01,
    "clip_grad_norm": 0.1,
    "steps": 150,
    "log_every": 10,
    "eval_every": 50,
    "validation_episodes": 8,
    "test_episodes": 8,
    "compute_coco_eval": True,
    "novel_category_ids": [3, 6, 10, 14, 18],  # bird, bus, cow, motorbike, sofa
    "selection_metric": "matched_box_iou",
    "train_sanity": False,
    "validation_sanity": True,
    "validation_sanity_samples": 2,
    "test_sanity": True,
    "test_sanity_samples": 2,
    "inference_plot_mode": "compact",
    "inference_extra_predictions": 0,
    "evaluate_test_at_end": True,
    "max_checkpoints": 2,
    "device": "cuda",
    "output_dir": Path("logs/pcs_fsod_overfit_sanity"),
    "figure_dir": Path("logs/pcs_fsod_overfit_sanity/figures"),
}
