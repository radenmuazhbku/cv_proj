"""Run a trained RF-DETR checkpoint on representative images and save annotated PNGs.

By default the script uses the best EMA checkpoint and the first eight test images.

Example:
    .venv/bin/python scripts/generate_sample_detections.py --count 12 --threshold 0.35
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from rfdetr import from_checkpoint
from supervision import BoxAnnotator, ColorLookup, LabelAnnotator, Position


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("logs/checkpoint_best_ema.pth"))
    parser.add_argument("--images-dir", type=Path, default=Path("datasets/rfdetr_dfff/test"))
    parser.add_argument("--output-dir", type=Path, default=Path("logs/visualizations/detections"))
    parser.add_argument("--count", type=int, default=8, help="Number of sorted input images to render.")
    parser.add_argument("--batch-size", type=int, default=4, help="Inference batch size; use 1 to minimize VRAM use.")
    parser.add_argument("--threshold", type=float, default=0.35, help="Minimum prediction confidence.")
    parser.add_argument("--device", default=None, help="Override checkpoint device, e.g. cuda or cpu.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count < 1:
        raise ValueError("--count must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must be between 0 and 1")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    image_paths = sorted(
        path for path in args.images_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )[: args.count]
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.images_dir}")

    model_kwargs = {"device": args.device} if args.device else {}
    model = from_checkpoint(args.checkpoint, **model_kwargs)
    box_annotator = BoxAnnotator(thickness=2, color_lookup=ColorLookup.CLASS)
    label_annotator = LabelAnnotator(
        color_lookup=ColorLookup.CLASS,
        text_position=Position.TOP_LEFT,
        text_scale=0.45,
        text_padding=2,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for start in range(0, len(image_paths), args.batch_size):
        batch_paths = image_paths[start : start + args.batch_size]
        images = [np.asarray(Image.open(image_path).convert("RGB")).copy() for image_path in batch_paths]
        batch_detections = model.predict(images, threshold=args.threshold, include_source_image=False)
        for image_path, image, detections in zip(batch_paths, images, batch_detections):
            class_names = detections.data.get("class_name", np.full(len(detections), "unknown", dtype=object))
            labels = [f"{name} {confidence:.2f}" for name, confidence in zip(class_names, detections.confidence)]
            annotated = box_annotator.annotate(scene=image, detections=detections)
            annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
            output_path = args.output_dir / image_path.name
            Image.fromarray(annotated).save(output_path)
            print(f"{image_path.name}: {len(detections)} detections -> {output_path}")


if __name__ == "__main__":
    main()
