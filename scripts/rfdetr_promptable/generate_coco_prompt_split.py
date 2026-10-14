"""Generate deterministic train/held-out COCO prompt subsets."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from cv_proj.configuration import parse_args_with_python_config


def filter_coco(
    source: dict[str, Any],
    category_ids: set[int],
    image_count: int | None,
    seed: int,
) -> dict[str, Any]:
    if image_count is not None and image_count <= 0:
        raise ValueError("image_count must be positive when specified")
    annotations = [ann for ann in source["annotations"] if ann["category_id"] in category_ids]
    image_ids_with_targets = sorted({ann["image_id"] for ann in annotations})
    rng = random.Random(seed)
    rng.shuffle(image_ids_with_targets)
    selected_image_ids = set(
        image_ids_with_targets if image_count is None else image_ids_with_targets[:image_count]
    )
    annotations = [ann for ann in annotations if ann["image_id"] in selected_image_ids]
    images = [image for image in source["images"] if image["id"] in selected_image_ids]
    categories = [category for category in source["categories"] if category["id"] in category_ids]
    return {"info": source.get("info", {}), "licenses": source.get("licenses", []), "images": images, "annotations": annotations, "categories": categories}


def split_support_query_coco(
    source: dict[str, Any], category_ids: set[int], support_images_per_class: int, seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reserve disjoint, deterministic support images and retain all other queries."""
    if support_images_per_class <= 0:
        raise ValueError("support_images_per_class must be positive")
    class_annotations = [
        ann for ann in source["annotations"] if ann["category_id"] in category_ids
    ]
    support_ids: set[int] = set()
    for category_id in sorted(category_ids):
        positive_ids = sorted(
            {ann["image_id"] for ann in class_annotations if ann["category_id"] == category_id}
        )
        if len(positive_ids) < 2:
            continue
        rng = random.Random(seed + category_id)
        rng.shuffle(positive_ids)
        support_ids.update(positive_ids[: min(support_images_per_class, len(positive_ids) - 1)])

    query_ids = {image["id"] for image in source["images"]} - support_ids
    support_images = [image for image in source["images"] if image["id"] in support_ids]
    query_images = [image for image in source["images"] if image["id"] in query_ids]
    categories = [category for category in source["categories"] if category["id"] in category_ids]
    support = {
        "info": source.get("info", {}),
        "licenses": source.get("licenses", []),
        "images": support_images,
        "annotations": [ann for ann in class_annotations if ann["image_id"] in support_ids],
        "categories": categories,
    }
    query = {
        "info": source.get("info", {}),
        "licenses": source.get("licenses", []),
        "images": query_images,
        "annotations": [ann for ann in class_annotations if ann["image_id"] in query_ids],
        "categories": categories,
    }
    return support, query


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--train-source-annotation", type=Path)
    parser.add_argument("--validation-source-annotation", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--train-classes", nargs="+", type=int)
    parser.add_argument("--test-classes", nargs="+", type=int)
    parser.add_argument("--train-images", type=int, default=10000)
    parser.add_argument("--support-images-per-class", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    args = parse_args_with_python_config(parser)

    required = (
        "train_source_annotation", "validation_source_annotation", "output_dir",
        "train_classes", "test_classes",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error(f"missing required config/arguments: {', '.join('--' + name.replace('_', '-') for name in missing)}")

    train_classes, test_classes = set(args.train_classes), set(args.test_classes)
    if train_classes & test_classes:
        raise ValueError("train and test class IDs must be disjoint")
    with args.train_source_annotation.open() as file:
        train_source = json.load(file)
    with args.validation_source_annotation.open() as file:
        validation_source = json.load(file)
    train = filter_coco(train_source, train_classes, args.train_images, args.seed)
    seen_support, seen_validation = split_support_query_coco(
        validation_source, train_classes, args.support_images_per_class, args.seed + 1
    )
    unseen_support, unseen_test = split_support_query_coco(
        validation_source, test_classes, args.support_images_per_class, args.seed + 2
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("train.json", train),
        ("seen_support.json", seen_support),
        ("seen_val.json", seen_validation),
        ("unseen_support.json", unseen_support),
        ("unseen_test.json", unseen_test),
    ):
        with (args.output_dir / name).open("w") as file:
            json.dump(payload, file)
    print(f"train images={len(train['images'])} annotations={len(train['annotations'])}")
    print(f"seen support images={len(seen_support['images'])} annotations={len(seen_support['annotations'])}")
    print(f"seen query images={len(seen_validation['images'])} annotations={len(seen_validation['annotations'])}")
    print(f"unseen support images={len(unseen_support['images'])} annotations={len(unseen_support['annotations'])}")
    print(f"unseen query images={len(unseen_test['images'])} annotations={len(unseen_test['annotations'])}")
    print(f"wrote train.json and support/query annotation files in {args.output_dir}")


if __name__ == "__main__":
    main()
