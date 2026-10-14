#!/usr/bin/env python3
"""Build an RF-DETR COCO dataset from three generated DFFF COCO folders.

The required inputs map directly to train, valid, and test. Each folder may be
randomly sampled by fraction or exact image count before its RF-DETR split is
written. The generated folder layout is compatible with RF-DETR's COCO format
auto-detection: ``train/_annotations.coco.json`` (and valid/test equivalents).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


OUTPUT_SPLITS = ("train", "valid", "test")


@dataclass(frozen=True)
class ImageSample:
    image: dict
    annotations: list[dict]
    source_path: Path


def load_coco_folder(folder: Path) -> tuple[list[ImageSample], list[dict]]:
    """Load a single generated DFFF COCO folder such as ``coco_dfff/set1``."""
    annotation_path = folder / "annotations" / "instances.json"
    if not annotation_path.is_file():
        raise FileNotFoundError(f"COCO annotations not found: {annotation_path}")
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not {"images", "annotations", "categories"}.issubset(data):
        raise ValueError(f"{annotation_path} is not a COCO detection annotation file")

    annotations_by_image: defaultdict[int, list[dict]] = defaultdict(list)
    for annotation in data["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)

    samples = []
    for image in data["images"]:
        image_path = folder / image["file_name"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Image referenced by {annotation_path} does not exist: {image_path}")
        samples.append(ImageSample(image, annotations_by_image[image["id"]], image_path))
    if not samples:
        raise ValueError(f"No images found in {folder}")
    return samples, data["categories"]


def requested_count(total: int, fraction: float | None, count: int | None) -> int:
    if count is not None:
        if count < 0 or count > total:
            raise ValueError(f"Requested count {count} must be between 0 and {total}")
        return count
    assert fraction is not None
    if not 0 <= fraction <= 1:
        raise ValueError(f"Fraction {fraction} must be between 0 and 1")
    return round(total * fraction)


def sample_images(samples: list[ImageSample], count: int, rng: random.Random) -> list[ImageSample]:
    """Randomly sample images without replacement, preserving a random order."""
    return rng.sample(samples, count)


def write_rfdetr_split(target_dir: Path, split: str, samples: list[ImageSample], categories: list[dict]) -> None:
    split_dir = target_dir / split
    split_dir.mkdir(parents=True)
    images = []
    annotations = []
    for image_id, sample in enumerate(samples, start=1):
        filename = f"{image_id:06d}_{sample.source_path.name}"
        shutil.copy2(sample.source_path, split_dir / filename)
        images.append({
            "id": image_id,
            "file_name": filename,
            "width": sample.image["width"],
            "height": sample.image["height"],
        })
        for annotation in sample.annotations:
            annotations.append({**annotation, "id": len(annotations) + 1, "image_id": image_id})

    (split_dir / "_annotations.coco.json").write_text(
        json.dumps(
            {
                "info": {"description": f"RF-DETR-ready DFFF {split} split"},
                "licenses": [],
                "images": images,
                "annotations": annotations,
                "categories": categories,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"{split}: {len(images)} images, {len(annotations)} annotations")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_source", type=Path, help="Generated COCO folder to use as train")
    parser.add_argument("valid_source", type=Path, help="Generated COCO folder to use as valid")
    parser.add_argument("test_source", type=Path, help="Generated COCO folder to use as test")
    parser.add_argument("target_dir", nargs="?", type=Path, default=Path("datasets/rfdetr_dfff"),
                        help="RF-DETR output folder (default: datasets/rfdetr_dfff)")
    allocation = parser.add_mutually_exclusive_group()
    allocation.add_argument("--fractions", nargs=3, type=float, metavar=("TRAIN", "VALID", "TEST"),
                            help="Random sample fractions per source folder (default: 1.0 1.0 1.0)")
    allocation.add_argument("--counts", nargs=3, type=int, metavar=("TRAIN", "VALID", "TEST"),
                            help="Random sample image counts per source folder")
    parser.add_argument("--seed", type=int, help="Seed for reproducible random sampling; omit for a new draw")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing target directory")
    args = parser.parse_args()

    source_dirs = tuple(path.resolve() for path in (args.train_source, args.valid_source, args.test_source))
    target_dir = args.target_dir.resolve()
    if len(set(source_dirs)) != len(source_dirs):
        parser.error("Train, valid, and test source folders must be different")
    if any(not path.is_dir() for path in source_dirs):
        parser.error("Every source folder must exist")
    if any(target_dir == source or target_dir in source.parents or source in target_dir.parents for source in source_dirs):
        parser.error("Target directory must not overlap a source folder")
    if target_dir.exists() and not args.overwrite:
        parser.error(f"Target directory already exists: {target_dir} (use --overwrite to replace it)")

    fractions = tuple(args.fractions) if args.fractions else (1.0, 1.0, 1.0)
    counts = tuple(args.counts) if args.counts else (None, None, None)
    rng = random.Random(args.seed)
    selected = {}
    categories: list[dict] | None = None
    try:
        for output_split, source_dir, fraction, count in zip(OUTPUT_SPLITS, source_dirs, fractions, counts, strict=True):
            samples, source_categories = load_coco_folder(source_dir)
            if categories is None:
                categories = source_categories
            elif categories != source_categories:
                raise ValueError(f"Category definitions in {source_dir} differ from the other source folders")
            selected[output_split] = sample_images(samples, requested_count(len(samples), fraction, count), rng)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True)
    for output_split in OUTPUT_SPLITS:
        write_rfdetr_split(target_dir, output_split, selected[output_split], categories or [])


if __name__ == "__main__":
    main()
