"""Convert PASCAL VOC dataset (XML annotations) to standard MS COCO format JSON.

Supports VOC 2012 (and VOC 2007) directory structures:
  VOC_ROOT/
    Annotations/
      *.xml
    JPEGImages/
      *.jpg
    ImageSets/Main/
      train.txt, val.txt, trainval.txt, etc.

Usage:
    python scripts/convert_voc_to_coco.py \
        --voc-dir datasets/VOC2012 \
        --output-dir datasets/VOC2012/annotations \
        --splits train val trainval
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

# Standard 20 PASCAL VOC categories (in canonical order)
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

# Mapping from VOC category names to standard MS COCO 80 category IDs (useful when evaluating COCO-pretrained detectors)
VOC_TO_COCO_CATEGORY_IDS = {
    "aeroplane": 5,      # COCO: airplane
    "bicycle": 2,        # COCO: bicycle
    "bird": 16,          # COCO: bird
    "boat": 9,           # COCO: boat
    "bottle": 44,        # COCO: bottle
    "bus": 6,            # COCO: bus
    "car": 3,            # COCO: car
    "cat": 17,           # COCO: cat
    "chair": 62,         # COCO: chair
    "cow": 21,           # COCO: cow
    "diningtable": 67,   # COCO: dining table
    "dog": 18,           # COCO: dog
    "horse": 19,         # COCO: horse
    "motorbike": 4,      # COCO: motorcycle
    "person": 1,         # COCO: person
    "pottedplant": 64,   # COCO: potted plant
    "sheep": 20,         # COCO: sheep
    "sofa": 63,          # COCO: couch
    "train": 7,          # COCO: train
    "tvmonitor": 72,     # COCO: tv
}


def build_categories(use_coco_ids: bool = False) -> list[dict[str, Any]]:
    categories = []
    for i, name in enumerate(VOC_CLASSES):
        cat_id = VOC_TO_COCO_CATEGORY_IDS[name] if use_coco_ids else (i + 1)
        categories.append({
            "id": cat_id,
            "name": name,
            "supercategory": "none",
        })
    return categories


def parse_voc_xml(xml_path: Path) -> dict[str, Any]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.findtext("filename")
    if not filename:
        filename = f"{xml_path.stem}.jpg"

    size_elem = root.find("size")
    if size_elem is not None:
        width = int(size_elem.findtext("width", "0"))
        height = int(size_elem.findtext("height", "0"))
        depth = int(size_elem.findtext("depth", "3"))
    else:
        width, height, depth = 0, 0, 3

    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip().lower()
        if name not in VOC_CLASSES:
            continue

        difficult_text = obj.findtext("difficult", "0")
        difficult = int(difficult_text) if difficult_text.isdigit() else 0

        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        xmin = float(bndbox.findtext("xmin", "0"))
        ymin = float(bndbox.findtext("ymin", "0"))
        xmax = float(bndbox.findtext("xmax", "0"))
        ymax = float(bndbox.findtext("ymax", "0"))

        objects.append({
            "name": name,
            "difficult": difficult,
            "bbox": [xmin, ymin, xmax, ymax],
        })

    return {
        "filename": filename,
        "width": width,
        "height": height,
        "depth": depth,
        "objects": objects,
    }


def image_stem_to_id(stem: str, fallback_index: int) -> int:
    """Deterministically convert a VOC stem like '2008_000008' to integer 2008000008."""
    digits = stem.replace("_", "")
    if digits.isdigit():
        return int(digits)
    return fallback_index


def convert_voc_split(
    voc_dir: Path,
    image_set_name: str,
    categories: list[dict[str, Any]],
    cat_name_to_id: dict[str, int],
    include_difficult: bool = True,
) -> dict[str, Any]:
    annotations_dir = voc_dir / "Annotations"
    images_dir = voc_dir / "JPEGImages"
    split_file = voc_dir / "ImageSets" / "Main" / f"{image_set_name}.txt"

    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    with split_file.open("r", encoding="utf-8") as f:
        image_stems = [line.strip().split()[0] for line in f if line.strip()]

    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    annotation_id = 1

    for idx, stem in enumerate(image_stems, start=1):
        xml_path = annotations_dir / f"{stem}.xml"
        if not xml_path.exists():
            continue

        parsed = parse_voc_xml(xml_path)
        img_id = image_stem_to_id(stem, idx)
        file_name = parsed["filename"]
        width = parsed["width"]
        height = parsed["height"]

        # If width/height missing in XML, fallback to checking actual image
        if width <= 0 or height <= 0:
            img_path = images_dir / file_name
            if img_path.exists():
                from PIL import Image
                with Image.open(img_path) as im:
                    width, height = im.size

        coco_images.append({
            "id": img_id,
            "file_name": file_name,
            "width": width,
            "height": height,
        })

        for obj in parsed["objects"]:
            is_difficult = obj["difficult"]
            if not include_difficult and is_difficult:
                continue

            xmin, ymin, xmax, ymax = obj["bbox"]
            # Convert 1-based [xmin, ymin, xmax, ymax] to 0-based [x, y, w, h]
            x = max(0.0, xmin - 1.0)
            y = max(0.0, ymin - 1.0)
            w = max(0.0, xmax - xmin)
            h = max(0.0, ymax - ymin)
            area = round(w * h, 2)

            cat_id = cat_name_to_id[obj["name"]]

            box_poly = [round(x, 2), round(y, 2), round(x + w, 2), round(y, 2), round(x + w, 2), round(y + h, 2), round(x, 2), round(y + h, 2)]
            coco_annotations.append({
                "id": annotation_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                "segmentation": [box_poly],
                "area": area,
                "iscrowd": 0,
                "ignore": int(is_difficult),
            })
            annotation_id += 1

    return {
        "info": {
            "description": f"PASCAL VOC 2012 ({image_set_name}) converted to COCO format",
            "year": 2012,
        },
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--voc-dir",
        type=Path,
        default=Path("datasets/VOC2012"),
        help="Path to VOC root directory containing Annotations/, JPEGImages/, ImageSets/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/VOC2012/annotations"),
        help="Output directory to save converted COCO json files",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "trainval"],
        help="ImageSets/Main split names to convert (e.g. train val trainval)",
    )
    parser.add_argument(
        "--use-coco-category-ids",
        action="store_true",
        help="Map VOC classes to standard COCO category IDs (1..80) instead of 1..20",
    )
    parser.add_argument(
        "--exclude-difficult",
        action="store_true",
        help="Exclude difficult objects from annotations",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    categories = build_categories(use_coco_ids=args.use_coco_category_ids)
    cat_name_to_id = {cat["name"]: cat["id"] for cat in categories}

    print(f"Categories ({len(categories)}):")
    for cat in categories:
        print(f"  {cat['id']:2d}: {cat['name']}")

    for split in args.splits:
        print(f"\nConverting split '{split}' from {args.voc_dir}...")
        coco_dict = convert_voc_split(
            voc_dir=args.voc_dir,
            image_set_name=split,
            categories=categories,
            cat_name_to_id=cat_name_to_id,
            include_difficult=not args.exclude_difficult,
        )
        suffix = "_coco_ids" if args.use_coco_category_ids else ""
        out_file = args.output_dir / f"voc_2012_{split}{suffix}.json"
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(coco_dict, f)

        print(
            f"Wrote {out_file}: "
            f"{len(coco_dict['images'])} images, {len(coco_dict['annotations'])} annotations"
        )


if __name__ == "__main__":
    main()

