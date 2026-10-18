"""COCO class-split annotation generation settings."""

from pathlib import Path

cfg = {
    "train_source_annotation": Path(
        "datasets/mscoco/annotations_trainval2017/annotations/instances_train2017.json"
    ),
    "validation_source_annotation": Path(
        "datasets/mscoco/annotations_trainval2017/annotations/instances_val2017.json"
    ),
    "output_dir": Path("datasets/mscoco_prompt_split"),
    "train_classes": [1, 3, 8, 9, 10, 11, 13, 14, 15, 16],
    "test_classes": [17, 18, 19, 20, 21, 22, 23, 24, 25, 27],
    "train_images": 10_000,
    "support_images_per_class": 5,
    "seed": 7,
}
