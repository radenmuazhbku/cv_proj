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
    parser.add_argument("--source-annotation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-classes", nargs="+", type=int, required=True)
    parser.add_argument("--test-classes", nargs="+", type=int, required=True)
    parser.add_argument("--train-images", type=int, default=100)
    parser.add_argument("--test-images", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    train_classes, test_classes = set(args.train_classes), set(args.test_classes)
    if train_classes & test_classes:
        raise ValueError("train and test class IDs must be disjoint")
    with args.source_annotation.open() as file:
        source = json.load(file)
    train = filter_coco(source, train_classes, args.train_images, args.seed)
    test = filter_coco(source, test_classes, args.test_images, args.seed + 1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in (("train.json", train), ("test.json", test)):
        with (args.output_dir / name).open("w") as file:
            json.dump(payload, file)
    print(f"train images={len(train['images'])} annotations={len(train['annotations'])}")
    print(f"test images={len(test['images'])} annotations={len(test['annotations'])}")
    print(f"wrote {args.output_dir / 'train.json'} and {args.output_dir / 'test.json'}")


if __name__ == "__main__":
    main()
