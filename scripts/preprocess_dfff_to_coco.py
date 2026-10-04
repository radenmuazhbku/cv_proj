#!/usr/bin/env python3
"""Convert the DFFF Excel annotations into four COCO dataset splits.

The source directory may contain a nested ``dfff`` directory. Images are
assigned from their path to Set1, Set2, Internal, or External; they are never
matched globally by filename alone. The Excel columns must be: fname,
structure, h_min, w_min, h_max, and w_max. Coordinates are (y_min, x_min,
y_max, x_max).
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
REQUIRED_COLUMNS = {"fname", "structure", "h_min", "w_min", "h_max", "w_max"}
SOURCE_SPLITS = {
    "Set1-Training&ValidationSetsCNN": "set1",
    "Set2-Training&Validation SetsANNScoringsystem": "set2",
    "Internal_Test_Set": "internal",
    "External_Test_Set": "external",
}


def split_and_relative_path(source_dir: Path, image_path: Path) -> tuple[str, Path] | None:
    """Get the DFFF split and path below its split directory for one image."""
    parts = image_path.relative_to(source_dir).parts
    for index, part in enumerate(parts):
        split = SOURCE_SPLITS.get(part)
        if split:
            return split, Path(*parts[index + 1 :])
    return None


def index_images(source_dir: Path) -> dict[str, dict[str, tuple[Path, Path]]]:
    """Index images by split and filename, rejecting duplicates within a split."""
    images: dict[str, dict[str, tuple[Path, Path]]] = {split: {} for split in SOURCE_SPLITS.values()}
    for image_path in source_dir.rglob("*"):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        resolved = split_and_relative_path(source_dir, image_path)
        if resolved is None:
            continue
        split, relative_path = resolved
        if image_path.name in images[split]:
            previous, _ = images[split][image_path.name]
            raise ValueError(
                f"Duplicate filename in {split}: {image_path.name} ({previous} and {image_path})"
            )
        images[split][image_path.name] = (image_path, relative_path)
    empty_splits = [split for split, files in images.items() if not files]
    if empty_splits:
        raise FileNotFoundError(f"No images found for DFFF split(s): {', '.join(empty_splits)}")
    return images


def read_annotations(excel_path: Path) -> pd.DataFrame:
    annotations = pd.read_excel(excel_path)
    missing_columns = REQUIRED_COLUMNS - set(annotations.columns)
    if missing_columns:
        raise ValueError(f"Missing required Excel columns: {', '.join(sorted(missing_columns))}")
    annotations = annotations.loc[:, ["fname", "structure", "h_min", "w_min", "h_max", "w_max"]].copy()
    if annotations.isna().any().any():
        raise ValueError("The Excel annotations contain missing values.")
    annotations["fname"] = annotations["fname"].astype(str)
    annotations["structure"] = annotations["structure"].astype(str)
    for column in ("h_min", "w_min", "h_max", "w_max"):
        annotations[column] = pd.to_numeric(annotations[column], errors="raise")
    return annotations


def assign_annotations_to_splits(
    annotations: pd.DataFrame, images: dict[str, dict[str, tuple[Path, Path]]]
) -> dict[str, pd.DataFrame]:
    filename_splits: defaultdict[str, list[str]] = defaultdict(list)
    for split, split_images in images.items():
        for filename in split_images:
            filename_splits[filename].append(split)

    assignments: dict[str, str] = {}
    for filename in annotations["fname"].unique():
        candidates = filename_splits.get(filename, [])
        if not candidates:
            raise FileNotFoundError(f"No image found for annotated filename: {filename}")
        if len(candidates) > 1:
            raise ValueError(
                f"Annotated filename exists in multiple splits and cannot be assigned safely: {filename} "
                f"({', '.join(candidates)})"
            )
        assignments[filename] = candidates[0]

    annotations["split"] = annotations["fname"].map(assignments)
    return {split: annotations.loc[annotations["split"] == split].copy() for split in images}


def build_split_coco(
    split: str,
    split_images: dict[str, tuple[Path, Path]],
    split_annotations: pd.DataFrame,
    categories: list[dict],
    category_ids: dict[str, int],
) -> tuple[dict, list[tuple[Path, Path]]]:
    image_records = []
    files_to_copy = []
    image_ids = {}
    for image_id, filename in enumerate(sorted(split_images), start=1):
        source_path, relative_path = split_images[filename]
        with Image.open(source_path) as image:
            width, height = image.size
        image_ids[filename] = image_id
        image_records.append(
            {"id": image_id, "file_name": (Path("images") / relative_path).as_posix(),
             "width": width, "height": height}
        )
        files_to_copy.append((source_path, relative_path))

    annotations = []
    for annotation_id, row in enumerate(split_annotations.itertuples(index=False), start=1):
        x_min, y_min = float(row.w_min), float(row.h_min)
        x_max, y_max = float(row.w_max), float(row.h_max)
        box_width, box_height = x_max - x_min, y_max - y_min
        image = image_records[image_ids[row.fname] - 1]
        if box_width <= 0 or box_height <= 0:
            raise ValueError(f"Invalid box in {split} for {row.fname}: {row}")
        if x_min < 0 or y_min < 0 or x_max > image["width"] or y_max > image["height"]:
            raise ValueError(f"Box is outside {row.fname} ({image['width']}x{image['height']}): {row}")
        annotations.append(
            {"id": annotation_id, "image_id": image_ids[row.fname],
             "category_id": category_ids[row.structure],
             "bbox": [x_min, y_min, box_width, box_height],
             "area": box_width * box_height, "iscrowd": 0}
        )

    return (
        {"info": {"description": f"DFFF {split} split converted from ObjectDetection.xlsx"},
         "licenses": [], "images": image_records, "annotations": annotations,
         "categories": categories},
        files_to_copy,
    )


def write_split(target_dir: Path, split: str, coco: dict, files_to_copy: list[tuple[Path, Path]]) -> None:
    split_dir = target_dir / split
    (split_dir / "images").mkdir(parents=True)
    (split_dir / "annotations").mkdir()
    for source_path, relative_path in files_to_copy:
        destination = split_dir / "images" / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
    (split_dir / "annotations" / "instances.json").write_text(
        json.dumps(coco, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path, help="Directory containing ObjectDetection.xlsx and DFFF images")
    parser.add_argument(
        "target_dir",
        nargs="?",
        type=Path,
        default=Path("datasets/coco_dfff"),
        help="Output directory for the four COCO splits (default: datasets/coco_dfff)",
    )
    parser.add_argument("--excel", type=Path, help="Excel file path, relative to source_dir unless absolute")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing target directory")
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    excel_path = args.excel or Path("ObjectDetection.xlsx")
    excel_path = excel_path if excel_path.is_absolute() else source_dir / excel_path
    target_dir = args.target_dir.resolve()
    if not source_dir.is_dir():
        parser.error(f"Source directory does not exist: {source_dir}")
    if not excel_path.is_file():
        parser.error(f"Excel file does not exist: {excel_path}")
    if target_dir == source_dir or source_dir in target_dir.parents or target_dir in source_dir.parents:
        parser.error("Target directory must not be the source directory or one of its ancestors or descendants")
    if target_dir.exists() and not args.overwrite:
        parser.error(f"Target directory already exists: {target_dir} (use --overwrite to replace it)")

    images = index_images(source_dir)
    all_annotations = read_annotations(excel_path)
    annotations_by_split = assign_annotations_to_splits(all_annotations, images)
    category_names = sorted(all_annotations["structure"].unique())
    categories = [
        {"id": category_id, "name": name, "supercategory": "object"}
        for category_id, name in enumerate(category_names, start=1)
    ]
    category_ids = {category["name"]: category["id"] for category in categories}

    results = {
        split: build_split_coco(split, images[split], annotations_by_split[split], categories, category_ids)
        for split in images
    }
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True)
    for split, (coco, files_to_copy) in results.items():
        write_split(target_dir, split, coco, files_to_copy)
        print(f"{split}: {len(coco['images'])} images, {len(coco['annotations'])} annotations")


if __name__ == "__main__":
    main()
