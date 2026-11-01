#!/usr/bin/env python3
"""Build an RF-DETR COCO dataset split from a single images folder and COCO annotation file.

The script randomly partitions the dataset into splits:
- No fractions given or 1 fraction (e.g. 1.0): 1 split only (train).
- 2 fractions/counts (e.g. 0.8 0.2): 2 splits only (train, valid).
- 3 fractions/counts (e.g. 0.8 0.1 0.1): 3 splits (train, valid, test).

It preserves all bounding box, segmentation mask, and keypoint annotations and category definitions.
The generated folder layout is compatible with RF-DETR's Roboflow / COCO format:
``target_dir/train/_annotations.coco.json`` (and valid/test equivalents).

python scripts/rfdetr/split_coco_for_rfdetr.py --images-dir datasets/FetSelect_Dataset/images/ \
      --annotations datasets/FetSelect_Dataset/split_H1H2.json \
        --output-dir datasets/split_H1H2 \
            --fractions 0.8 0.1 0.1

python scripts/rfdetr/split_coco_for_rfdetr.py --images-dir datasets/FetSelect_Dataset/images/ \
--annotations datasets/FetSelect_Dataset/scalebar_H1H2.json \
--output-dir datasets/scalebar_H1H2 \
--fractions 0.8 0.1 0.1
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SPLIT_NAMES_BY_COUNT = {
    1: ("train",),
    2: ("train", "valid"),
    3: ("train", "valid", "test"),
}


@dataclass(frozen=True)
class ImageSample:
    image: dict[str, Any]
    annotations: list[dict[str, Any]]
    source_path: Path


def load_coco_dataset(
    images_dir: Path,
    annotation_path: Path,
) -> tuple[list[ImageSample], list[dict[str, Any]], dict[str, Any]]:
    if not annotation_path.is_file():
        raise FileNotFoundError(f"COCO annotation file not found: {annotation_path}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")

    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not {"images", "annotations", "categories"}.issubset(data):
        raise ValueError(
            f"{annotation_path} is not a valid COCO annotation file (missing images, annotations, or categories)"
        )

    annotations_by_image: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in data["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)

    samples: list[ImageSample] = []
    missing_images: list[str] = []

    for image in data["images"]:
        filename = image["file_name"]
        image_path = images_dir / filename
        if not image_path.is_file():
            # Check basename fallback in images_dir
            fallback_path = images_dir / Path(filename).name
            if fallback_path.is_file():
                image_path = fallback_path
            else:
                missing_images.append(filename)
                continue

        samples.append(ImageSample(image, annotations_by_image[image["id"]], image_path))

    if missing_images:
        if not samples:
            raise FileNotFoundError(
                f"None of the {len(data['images'])} images in {annotation_path} could be found in {images_dir}. "
                f"Sample missing file: {missing_images[0]}"
            )
        print(
            f"Warning: {len(missing_images)}/{len(data['images'])} images referenced in annotations "
            f"were not found in {images_dir} and were skipped."
        )

    if not samples:
        raise ValueError(f"No valid images found for dataset at {annotation_path}")

    coco_meta = {
        "info": data.get("info", {}),
        "licenses": data.get("licenses", []),
    }
    return samples, data["categories"], coco_meta


def compute_split_counts(
    total: int,
    fractions: list[float] | None,
    counts: list[int] | None,
) -> dict[str, int]:
    if counts is not None:
        if len(counts) not in SPLIT_NAMES_BY_COUNT:
            raise ValueError(f"--counts accepts 1, 2, or 3 values, got {len(counts)}")
        splits = SPLIT_NAMES_BY_COUNT[len(counts)]
        if any(c < 0 for c in counts):
            raise ValueError(f"Counts cannot be negative: {counts}")
        if sum(counts) > total:
            raise ValueError(
                f"Sum of requested counts ({sum(counts)}) exceeds total available images ({total})"
            )
        return {split: count for split, count in zip(splits, counts)}

    fracs = fractions if fractions is not None else [1.0]
    if len(fracs) not in SPLIT_NAMES_BY_COUNT:
        raise ValueError(f"--fractions accepts 1, 2, or 3 values, got {len(fracs)}")
    splits = SPLIT_NAMES_BY_COUNT[len(fracs)]
    if any(f < 0 for f in fracs):
        raise ValueError(f"Fractions cannot be negative: {fracs}")

    total_frac = sum(fracs)
    if total_frac <= 0:
        raise ValueError("Sum of fractions must be greater than 0")

    if len(fracs) == 1:
        f = fracs[0]
        n = total if f >= 1.0 else round(total * f)
        return {"train": n}

    norm_fracs = [f / total_frac for f in fracs]
    split_counts: dict[str, int] = {}
    allocated = 0
    for split, norm_f in zip(splits[:-1], norm_fracs[:-1]):
        n = round(total * norm_f)
        split_counts[split] = n
        allocated += n

    last_split = splits[-1]
    split_counts[last_split] = max(0, total - allocated)
    return split_counts


def write_rfdetr_split(
    target_dir: Path,
    split: str,
    samples: list[ImageSample],
    categories: list[dict[str, Any]],
    coco_meta: dict[str, Any],
    *,
    preserve_filenames: bool = False,
    symlink: bool = False,
) -> None:
    split_dir = target_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []

    num_boxes = 0
    num_masks = 0
    num_keypoints = 0

    for new_image_id, sample in enumerate(samples, start=1):
        if preserve_filenames:
            filename = sample.source_path.name
        else:
            filename = f"{new_image_id:06d}_{sample.source_path.name}"

        dest_image_path = split_dir / filename
        if symlink:
            if dest_image_path.is_symlink() or dest_image_path.exists():
                dest_image_path.unlink()
            dest_image_path.symlink_to(sample.source_path.resolve())
        else:
            shutil.copy2(sample.source_path, dest_image_path)

        image_entry = dict(sample.image)
        image_entry["id"] = new_image_id
        image_entry["file_name"] = filename
        images.append(image_entry)

        for annotation in sample.annotations:
            ann = dict(annotation)
            ann["id"] = len(annotations) + 1
            ann["image_id"] = new_image_id

            if "bbox" in ann:
                num_boxes += 1
            if ann.get("segmentation"):
                num_masks += 1
            if ann.get("keypoints"):
                num_keypoints += 1
                if "num_keypoints" not in ann:
                    kp = ann["keypoints"]
                    ann["num_keypoints"] = sum(1 for v in kp[2::3] if v > 0)

            annotations.append(ann)

    info = dict(coco_meta.get("info", {}))
    info["description"] = info.get("description", "") or f"RF-DETR-ready {split} split"

    split_coco = {
        "info": info,
        "licenses": coco_meta.get("licenses", []),
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    (split_dir / "_annotations.coco.json").write_text(
        json.dumps(split_coco, indent=2) + "\n",
        encoding="utf-8",
    )

    stats = [f"{len(images)} images", f"{len(annotations)} annotations"]
    details = []
    if num_boxes > 0:
        details.append(f"{num_boxes} boxes")
    if num_masks > 0:
        details.append(f"{num_masks} masks")
    if num_keypoints > 0:
        details.append(f"{num_keypoints} keypoints")
    if details:
        stats.append(f"({', '.join(details)})")

    print(f"{split:5s}: " + ", ".join(stats))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Positional or named arguments for flexibility
    parser.add_argument("images_dir", nargs="?", type=Path, default=None, help="Folder containing images")
    parser.add_argument("annotations", nargs="?", type=Path, default=None, help="COCO annotation JSON file")
    parser.add_argument(
        "target_dir",
        nargs="?",
        type=Path,
        default=None,
        help="RF-DETR output folder (default: datasets/rfdetr_dfff)",
    )

    parser.add_argument("--images-dir", "-i", type=Path, dest="opt_images_dir", help="Folder containing images")
    parser.add_argument("--annotations", "-a", type=Path, dest="opt_annotations", help="COCO annotation JSON file")
    parser.add_argument("--output-dir", "-o", type=Path, dest="opt_target_dir", help="RF-DETR output folder")

    allocation = parser.add_mutually_exclusive_group()
    allocation.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        metavar="FRAC",
        help=(
            "Split fractions or ratios: omit or '1.0' for train only; "
            "'0.8 0.2' for train and valid only; '0.8 0.1 0.1' for train, valid, test."
        ),
    )
    allocation.add_argument(
        "--counts",
        nargs="+",
        type=int,
        metavar="COUNT",
        help=(
            "Exact image counts per split: 1 count for train only; "
            "2 counts for train and valid only; 3 counts for train, valid, test."
        ),
    )

    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducible random sampling (default: 42)")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing target directory")
    parser.add_argument("--symlink", action="store_true", help="Symlink images instead of copying")
    parser.add_argument(
        "--preserve-filenames",
        action="store_true",
        help="Preserve original filenames instead of prefixing with 6-digit image ID",
    )

    args = parser.parse_args()

    images_dir = args.opt_images_dir or args.images_dir
    annotation_file = args.opt_annotations or args.annotations
    target_dir = args.opt_target_dir or args.target_dir or Path("datasets/rfdetr_dfff")

    if images_dir is None or annotation_file is None:
        parser.error("Both images directory and annotation file must be specified.")

    images_dir = images_dir.resolve()
    annotation_file = annotation_file.resolve()
    target_dir = target_dir.resolve()

    if target_dir == images_dir or target_dir in images_dir.parents or images_dir in target_dir.parents:
        parser.error("Target directory must not overlap the images folder")

    if target_dir.exists() and not args.overwrite:
        parser.error(f"Target directory already exists: {target_dir} (use --overwrite to replace it)")

    try:
        samples, categories, coco_meta = load_coco_dataset(images_dir, annotation_file)
        counts = compute_split_counts(len(samples), args.fractions, args.counts)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    rng = random.Random(args.seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)

    split_samples: dict[str, list[ImageSample]] = {}
    offset = 0
    for split_name, count in counts.items():
        split_samples[split_name] = [samples[i] for i in indices[offset : offset + count]]
        offset += count

    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True)

    splits_summary = ", ".join(f"{split}={len(s_list)}" for split, s_list in split_samples.items())
    print(f"Splitting {len(samples)} images into: {splits_summary} -> {target_dir}")

    for output_split, s_list in split_samples.items():
        write_rfdetr_split(
            target_dir=target_dir,
            split=output_split,
            samples=s_list,
            categories=categories,
            coco_meta=coco_meta,
            preserve_filenames=args.preserve_filenames,
            symlink=args.symlink,
        )


if __name__ == "__main__":
    main()
