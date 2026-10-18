"""COCO class-split annotation generation settings."""

from pathlib import Path

train_list = list(range(1,71))
test_list = list(range(71,81))
cfg = {
    "train_source_annotation": Path(
        "datasets/mscoco/annotations_trainval2017/annotations/instances_train2017.json"
    ),
    "validation_source_annotation": Path(
        "datasets/mscoco/annotations_trainval2017/annotations/instances_val2017.json"
    ),
    "output_dir": Path("datasets/mscoco_prompt_split"),
    "train_classes": train_list,
    "test_classes": test_list,
    "train_images": 100_000,
    "support_images_per_class": 5,
    "seed": 0,
}
