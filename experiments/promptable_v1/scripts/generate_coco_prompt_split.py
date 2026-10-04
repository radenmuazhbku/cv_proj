"""Generate deterministic train/held-out COCO prompt subsets."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def filter_coco(
    source: dict[str, Any],
    category_ids: set[int],
    image_count: int | None,
    seed: int,
) -> dict[str, Any]:
    annotations = [ann for ann in source["annotations"] if ann["category_id"] in category_ids]
    image_ids_with_targets = sorted({ann["image_id"] for ann in annotations})
    rng = random.Random(seed)
    rng.shuffle(image_ids_with_targets)
    selected_image_ids = set(image_ids_with_targets[:image_count] if image_count else image_ids_with_targets)
    annotations = [ann for ann in annotations if ann["image_id"] in selected_image_ids]
    images = [image for image in source["images"] if image["id"] in selected_image_ids]
    categories = [category for category in source["categories"] if category["id"] in category_ids]
    return {"info": source.get("info", {}), "licenses": source.get("licenses", []), "images": images, "annotations": annotations, "categories": categories}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-source-annotation", type=Path, required=True)
    parser.add_argument("--validation-source-annotation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-classes", nargs="+", type=int, required=True)
    parser.add_argument("--test-classes", nargs="+", type=int, required=True)
    parser.add_argument("--train-images", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    train_classes, test_classes = set(args.train_classes), set(args.test_classes)
    if train_classes & test_classes:
        raise ValueError("train and test class IDs must be disjoint")
    with args.train_source_annotation.open() as file:
        train_source = json.load(file)
    with args.validation_source_annotation.open() as file:
        validation_source = json.load(file)
    train = filter_coco(train_source, train_classes, args.train_images, args.seed)
    seen_validation = filter_coco(validation_source, train_classes, None, args.seed + 1)
    unseen_test = filter_coco(validation_source, test_classes, None, args.seed + 2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in (("train.json", train), ("seen_val.json", seen_validation), ("unseen_test.json", unseen_test)):
        with (args.output_dir / name).open("w") as file:
            json.dump(payload, file)
    print(f"train images={len(train['images'])} annotations={len(train['annotations'])}")
    print(f"seen validation images={len(seen_validation['images'])} annotations={len(seen_validation['annotations'])}")
    print(f"unseen test images={len(unseen_test['images'])} annotations={len(unseen_test['annotations'])}")
    print(f"wrote train.json, seen_val.json, and unseen_test.json in {args.output_dir}")


if __name__ == "__main__":
    main()
