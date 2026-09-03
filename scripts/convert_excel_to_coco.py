#!/usr/bin/env python3
"""Convert Excel bounding-box annotations and images to COCO detection format.

The Excel file must contain: fname, structure, h_min, w_min, h_max, w_max.
Coordinates are interpreted as (y_min, x_min, y_max, x_max).
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


def image_index(source_dir: Path) -> dict[str, Path]:
    """Return image paths keyed by filename, rejecting ambiguous filenames."""
    matches: defaultdict[str, list[Path]] = defaultdict(list)
    for path in source_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            matches[path.name].append(path)

    ambiguous = {name: paths for name, paths in matches.items() if len(paths) > 1}
    if ambiguous:
        details = "; ".join(
            f"{name}: {', '.join(str(path) for path in paths)}"
            for name, paths in sorted(ambiguous.items())
        )
        raise ValueError(f"Image filenames must be unique under the source directory. {details}")
    return {name: paths[0] for name, paths in matches.items()}


def build_coco(source_dir: Path, excel_path: Path) -> tuple[dict, list[tuple[Path, str]]]:
    annotations = pd.read_excel(excel_path)
    missing_columns = REQUIRED_COLUMNS - set(annotations.columns)
    if missing_columns:
        raise ValueError(f"Missing required Excel columns: {', '.join(sorted(missing_columns))}")

    annotations = annotations.loc[:, sorted(REQUIRED_COLUMNS)].copy()
    if annotations.isna().any().any():
        raise ValueError("The Excel annotations contain missing values.")

    annotations["fname"] = annotations["fname"].astype(str)
    annotations["structure"] = annotations["structure"].astype(str)
    for column in ("h_min", "w_min", "h_max", "w_max"):
        annotations[column] = pd.to_numeric(annotations[column], errors="raise")

    images_by_name = image_index(source_dir)
    referenced_names = set(annotations["fname"])
    missing_images = sorted(referenced_names - set(images_by_name))
    if missing_images:
        preview = ", ".join(missing_images[:10])
        suffix = "..." if len(missing_images) > 10 else ""
        raise FileNotFoundError(f"No image was found for {len(missing_images)} annotation filename(s): {preview}{suffix}")

    category_names = sorted(annotations["structure"].unique())
    categories = [
        {"id": category_id, "name": name, "supercategory": "object"}
        for category_id, name in enumerate(category_names, start=1)
    ]
    category_ids = {category["name"]: category["id"] for category in categories}

    coco_images = []
    files_to_copy = []
    image_ids = {}
    for image_id, filename in enumerate(sorted(referenced_names), start=1):
        image_path = images_by_name[filename]
        with Image.open(image_path) as image:
            width, height = image.size
        image_ids[filename] = image_id
        coco_images.append(
            {"id": image_id, "file_name": f"images/{filename}", "width": width, "height": height}
        )
        files_to_copy.append((image_path, filename))

    coco_annotations = []
    for annotation_id, row in enumerate(annotations.itertuples(index=False), start=1):
        # Excel convention: h is y, w is x. COCO uses [x, y, width, height].
        x_min, y_min = float(row.w_min), float(row.h_min)
        x_max, y_max = float(row.w_max), float(row.h_max)
        width, height = x_max - x_min, y_max - y_min
        image = coco_images[image_ids[row.fname] - 1]
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid box at Excel row {annotation_id + 1}: {row}")
        if x_min < 0 or y_min < 0 or x_max > image["width"] or y_max > image["height"]:
            raise ValueError(
                f"Box at Excel row {annotation_id + 1} is outside {row.fname} "
                f"({image['width']}x{image['height']}): {row}"
            )
        coco_annotations.append(
            {
                "id": annotation_id,
                "image_id": image_ids[row.fname],
                "category_id": category_ids[row.structure],
                "bbox": [x_min, y_min, width, height],
                "area": width * height,
                "iscrowd": 0,
            }
        )

    return (
        {"info": {"description": "Converted from ObjectDetection.xlsx"}, "licenses": [],
         "images": coco_images, "annotations": coco_annotations, "categories": categories},
        files_to_copy,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path, help="Directory containing the Excel file and images")
    parser.add_argument("target_dir", type=Path, help="New output directory for the COCO dataset")
    parser.add_argument("--excel", type=Path, help="Excel file path, relative to source_dir unless absolute")
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    excel_path = args.excel or Path("ObjectDetection.xlsx")
    excel_path = excel_path if excel_path.is_absolute() else source_dir / excel_path
    target_dir = args.target_dir.resolve()
    if not source_dir.is_dir():
        parser.error(f"Source directory does not exist: {source_dir}")
    if not excel_path.is_file():
        parser.error(f"Excel file does not exist: {excel_path}")
    if target_dir.exists():
        parser.error(f"Target directory already exists: {target_dir}")

    coco, files_to_copy = build_coco(source_dir, excel_path)
    (target_dir / "images").mkdir(parents=True)
    (target_dir / "annotations").mkdir()
    for source_path, filename in files_to_copy:
        shutil.copy2(source_path, target_dir / "images" / filename)
    annotation_path = target_dir / "annotations" / "instances.json"
    annotation_path.write_text(json.dumps(coco, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {len(coco['images'])} images, {len(coco['annotations'])} annotations, and "
          f"{len(coco['categories'])} categories to {target_dir}")


if __name__ == "__main__":
    main()
