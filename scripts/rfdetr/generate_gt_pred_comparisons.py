"""Create GT-only, prediction-only, and overlay comparison panels from COCO labels.

Example:
    .venv/bin/python scripts/generate_gt_pred_comparisons.py \
        --images-dir datasets/rfdetr_dfff/test \
        --annotations datasets/rfdetr_dfff/test/_annotations.coco.json \
        --output-dir logs/comparisons/test --count 10000
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rfdetr import from_checkpoint

GT_COLOR = (0, 230, 118)
PRED_COLOR = (255, 82, 82)
TEXT_COLOR = (255, 255, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("logs/checkpoint_best_ema.pth"))
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True, help="COCO _annotations.coco.json file.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--device", default=None, help="Override checkpoint device, e.g. cuda or cpu.")
    return parser.parse_args()


def draw_boxes(image: np.ndarray, boxes: list[tuple[list[float], str]], color: tuple[int, int, int]) -> np.ndarray:
    """Draw xyxy boxes and labels onto an RGB image."""
    canvas = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for (x1, y1, x2, y2), label in boxes:
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        left, top, right, bottom = draw.textbbox((x1, y1), label, font=font)
        label_y = max(0, y1 - (bottom - top) - 4)
        draw.rectangle((x1, label_y, x1 + (right - left) + 6, label_y + (bottom - top) + 4), fill=color)
        draw.text((x1 + 3, label_y + 2), label, fill=TEXT_COLOR, font=font)
    return np.asarray(canvas)


def add_title(image: np.ndarray, title: str) -> np.ndarray:
    """Add a compact panel title without changing the source image dimensions."""
    canvas = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    left, top, right, bottom = draw.textbbox((0, 0), title, font=font)
    draw.rectangle((0, 0, right - left + 10, bottom - top + 8), fill=(0, 0, 0))
    draw.text((5, 4), title, fill=TEXT_COLOR, font=font)
    return np.asarray(canvas)


def load_coco(path: Path) -> tuple[dict[str, list[tuple[list[float], str]]], dict[str, int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    categories = {category["id"]: category["name"] for category in data["categories"]}
    file_names = {image["id"]: image["file_name"] for image in data["images"]}
    boxes_by_file: dict[str, list[tuple[list[float], str]]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    for annotation in data["annotations"]:
        x, y, width, height = annotation["bbox"]
        filename = file_names[annotation["image_id"]]
        boxes_by_file[filename].append(([x, y, x + width, y + height], categories[annotation["category_id"]]))
        counts[filename] += 1
    return boxes_by_file, counts


def main() -> None:
    args = parse_args()
    if args.count < 1 or args.batch_size < 1:
        raise ValueError("--count and --batch-size must be at least 1")
    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must be between 0 and 1")

    image_paths = sorted(path for path in args.images_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
    image_paths = image_paths[: args.count]
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.images_dir}")

    gt_by_file, _ = load_coco(args.annotations)
    model_kwargs = {"device": args.device} if args.device else {}
    model = from_checkpoint(args.checkpoint, **model_kwargs)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for start in range(0, len(image_paths), args.batch_size):
        batch_paths = image_paths[start : start + args.batch_size]
        images = [np.asarray(Image.open(path).convert("RGB")).copy() for path in batch_paths]
        batch_detections = model.predict(images, threshold=args.threshold, include_source_image=False)
        for path, image, detections in zip(batch_paths, images, batch_detections):
            gt_boxes = [(box, f"GT {name}") for box, name in gt_by_file.get(path.name, [])]
            class_names = detections.data.get("class_name", np.full(len(detections), "unknown", dtype=object))
            pred_boxes = [
                (box.tolist(), f"P {name} {confidence:.2f}")
                for box, name, confidence in zip(detections.xyxy, class_names, detections.confidence)
            ]

            gt_panel = add_title(draw_boxes(image, gt_boxes, GT_COLOR), f"Ground truth ({len(gt_boxes)})")
            pred_panel = add_title(draw_boxes(image, pred_boxes, PRED_COLOR), f"Predictions ({len(pred_boxes)})")
            overlay = draw_boxes(draw_boxes(image, gt_boxes, GT_COLOR), pred_boxes, PRED_COLOR)
            overlay_panel = add_title(overlay, "Overlay: GT green / prediction red")
            Image.fromarray(np.concatenate((gt_panel, pred_panel, overlay_panel), axis=1)).save(args.output_dir / path.name)
            print(f"{path.name}: {len(gt_boxes)} GT, {len(pred_boxes)} predictions")


if __name__ == "__main__":
    main()
