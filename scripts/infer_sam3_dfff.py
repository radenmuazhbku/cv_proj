#!/usr/bin/env python3
"""Run SAM3 visual-example segmentation on DFFF support/query pairs.

Query annotations are never read for prompts or model inputs. The query image is
concatenated beside one or more labeled support images, and only support boxes
are supplied as positive visual prompts. Query labels are loaded after all
inference finishes, solely for evaluation and result visualization.
"""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from PIL import Image

from cv_proj.sam3_dfff import average_precision_at_iou, bbox_iou, box_proxy_mask_metrics


TEXT_PROMPTS = {"NT": "nuchal translucency (NT)", "nasal bone": "nasal bone"}


def _compose_square(
    support_images: list[Image.Image], query_image: Image.Image, boxes: list[list[float]], resolution: int
) -> tuple[Image.Image, list[list[float]], tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Pack image panels side by side without changing each source aspect ratio."""
    images = support_images + [query_image]
    count = len(images)
    edges = [round(index * resolution / count) for index in range(count + 1)]
    canvas = Image.new("RGB", (resolution, resolution), color=(16, 18, 24))
    rectangles: list[tuple[int, int, int, int]] = []
    scales: list[tuple[float, float, int, int]] = []
    for index, image in enumerate(images):
        left, right = edges[index], edges[index + 1]
        tile_width = right - left
        tile_height = tile_width
        top = (resolution - tile_height) // 2
        scale = min(tile_width / image.width, tile_height / image.height)
        resized_width = max(1, round(image.width * scale))
        resized_height = max(1, round(image.height * scale))
        resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
        image_left = left + (tile_width - resized_width) // 2
        image_top = top + (tile_height - resized_height) // 2
        canvas.paste(resized, (image_left, image_top))
        rectangles.append((image_left, image_top, image_left + resized_width, image_top + resized_height))
        scales.append((resized_width / image.width, resized_height / image.height, image_left, image_top))

    prompt_boxes: list[list[float]] = []
    for box, (scale_x, scale_y, left, top) in zip(boxes, scales[: len(boxes)], strict=True):
        prompt_boxes.append(
            [left + box[0] * scale_x, top + box[1] * scale_y,
             left + box[2] * scale_x, top + box[3] * scale_y]
        )
    return canvas, prompt_boxes, rectangles[-1], rectangles[0]


def _mask_box(mask: np.ndarray) -> list[float]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return [0.0, 0.0, 0.0, 0.0]
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def _predict_one(
    processor: Any,
    class_name: str,
    support_records: list[dict[str, Any]],
    query_record: dict[str, Any],
    resolution: int,
) -> tuple[dict[str, Any], np.ndarray, Image.Image, list[Image.Image]]:
    support_images = [Image.open(record["image_path"]).convert("RGB") for record in support_records]
    query_image = Image.open(query_record["image_path"]).convert("RGB")
    composite, prompt_boxes, query_content_rect, _ = _compose_square(
        support_images,
        query_image,
        [record["bbox_xyxy"] for record in support_records],
        resolution,
    )
    state = processor.set_image(composite)
    state = processor.set_text_prompt(TEXT_PROMPTS.get(class_name, class_name), state)
    for box in prompt_boxes:
        x1, y1, x2, y2 = box
        normalized_box = [
            (x1 + x2) / (2 * resolution),
            (y1 + y2) / (2 * resolution),
            (x2 - x1) / resolution,
            (y2 - y1) / resolution,
        ]
        state = processor.add_geometric_prompt(normalized_box, True, state)

    query_left, query_top, query_right, query_bottom = query_content_rect
    detections: list[tuple[float, np.ndarray]] = []
    masks = state.get("masks")
    scores = state.get("scores")
    if masks is not None and scores is not None:
        mask_array = masks.detach().to("cpu").numpy()
        score_array = scores.detach().to("cpu").numpy().reshape(-1)
        for mask, score in zip(mask_array, score_array, strict=True):
            mask = np.asarray(mask).squeeze().astype(bool)
            query_crop = mask[query_top:query_bottom, query_left:query_right]
            if not query_crop.any():
                continue
            source_mask = np.asarray(
                Image.fromarray(query_crop.astype(np.uint8) * 255).resize(
                    query_image.size, Image.Resampling.NEAREST
                )
            ) > 0
            if source_mask.any():
                detections.append((float(score), source_mask))

    if detections:
        score, prediction_mask = max(detections, key=lambda item: item[0])
        bbox = _mask_box(prediction_mask)
    else:
        score = 0.0
        prediction_mask = np.zeros((query_image.height, query_image.width), dtype=bool)
        bbox = [0.0, 0.0, 0.0, 0.0]
    result = {
        "query_id": query_record["query_id"],
        "class_name": class_name,
        "image_id": query_record["image_id"],
        "image_path": query_record["image_path"],
        "score": score,
        "bbox_xyxy": bbox,
        "mask_found": bool(prediction_mask.any()),
    }
    return result, prediction_mask, composite, support_images


def _evaluate(
    predictions: dict[str, list[dict[str, Any]]],
    masks: dict[str, np.ndarray],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "metric_note": "GT contains boxes only; mask metrics below compare against box-shaped proxies, not true segmentation masks.",
        "classes": {},
    }
    aggregate_fields = (
        "bbox_AP50", "bbox_AP75", "mean_bbox_IoU", "bbox_recall50", "bbox_recall75",
        "box_proxy_mask_IoU", "box_proxy_mask_Dice", "box_coverage_recall", "outside_box_fraction",
    )
    for class_name, records in predictions.items():
        targets = evaluation["classes"].get(class_name, {})
        if not targets:
            summary["classes"][class_name] = {
                "query_images": len(records),
                "evaluated_queries": 0,
                "note": "No query ground-truth boxes; qualitative-only results.",
            }
            continue
        gt_boxes = {query_id: target["bbox_xyxy"] for query_id, target in targets.items()}
        evaluated = [record for record in records if record["query_id"] in targets]
        ious = [bbox_iou(record["bbox_xyxy"], gt_boxes[record["query_id"]]) for record in evaluated]
        mask_values = [box_proxy_mask_metrics(masks[record["query_id"]], gt_boxes[record["query_id"]]) for record in evaluated]
        class_metrics: dict[str, Any] = {
            "query_images": len(records),
            "evaluated_queries": len(targets),
            "negative_query_images": len(records) - len(targets),
            "bbox_AP50": average_precision_at_iou(
                [record for record in records if record["mask_found"]], gt_boxes, 0.50
            ),
            "bbox_AP75": average_precision_at_iou(
                [record for record in records if record["mask_found"]], gt_boxes, 0.75
            ),
            "mean_bbox_IoU": float(np.mean(ious)) if ious else 0.0,
            "bbox_recall50": float(np.mean([value >= 0.50 for value in ious])) if ious else 0.0,
            "bbox_recall75": float(np.mean([value >= 0.75 for value in ious])) if ious else 0.0,
        }
        for field in ("box_proxy_mask_iou", "box_proxy_dice", "box_coverage_recall", "outside_box_fraction"):
            class_metrics[field] = float(np.mean([value[field] for value in mask_values])) if mask_values else 0.0
        # Keep concise metric names in the summary and plot.
        class_metrics["box_proxy_mask_IoU"] = class_metrics.pop("box_proxy_mask_iou")
        class_metrics["box_proxy_mask_Dice"] = class_metrics.pop("box_proxy_dice")
        summary["classes"][class_name] = class_metrics

    measured = [metrics for metrics in summary["classes"].values() if metrics.get("evaluated_queries", 0)]
    summary["macro_average"] = {
        key: float(np.mean([metrics[key] for metrics in measured])) if measured else None
        for key in aggregate_fields
    }
    summary["macro_average"]["evaluated_queries"] = sum(
        metrics.get("evaluated_queries", 0) for metrics in measured
    )
    return summary


def _plot_pair(
    destination: Path,
    class_name: str,
    support_records: list[dict[str, Any]],
    support_images: list[Image.Image],
    query_image: Image.Image,
    prediction_mask: np.ndarray,
    prediction: dict[str, Any],
    ground_truth: dict[str, Any] | None,
) -> None:
    columns = len(support_images) + 1
    figure, axes = plt.subplots(1, columns, figsize=(5 * columns, 5), squeeze=False)
    for index, (record, image) in enumerate(zip(support_records, support_images, strict=True)):
        axis = axes[0, index]
        axis.imshow(image)
        x1, y1, x2, y2 = record["bbox_xyxy"]
        axis.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="cyan", linewidth=2))
        axis.set_title(f"Support {index + 1}: {class_name} (positive box prompt)")
        axis.axis("off")

    query_axis = axes[0, -1]
    query_axis.imshow(query_image)
    overlay = np.zeros((*prediction_mask.shape, 4), dtype=np.float32)
    overlay[..., 1] = 1.0
    overlay[..., 3] = prediction_mask.astype(np.float32) * 0.48
    query_axis.imshow(overlay)
    if ground_truth is not None:
        x1, y1, x2, y2 = ground_truth["bbox_xyxy"]
        query_axis.add_patch(
            Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="yellow", linestyle="--", linewidth=2)
        )
    query_axis.set_title(f"Unprompted query; score={prediction['score']:.3f}\nGT box yellow, predicted mask green")
    query_axis.axis("off")
    figure.suptitle(f"SAM3 visual-example segmentation — {class_name}")
    figure.tight_layout()
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _plot_metrics(destination: Path, summary: dict[str, Any]) -> None:
    class_names = [
        name for name, metrics in summary["classes"].items() if metrics.get("evaluated_queries", 0)
    ]
    if not class_names:
        return
    measures = (
        ("bbox_AP50", "Box AP50"),
        ("bbox_AP75", "Box AP75"),
        ("mean_bbox_IoU", "Mean box IoU"),
        ("box_proxy_mask_IoU", "Proxy mask IoU*"),
    )
    x = np.arange(len(class_names))
    width = 0.8 / len(measures)
    figure, axis = plt.subplots(figsize=(max(8, len(class_names) * 3), 5))
    for index, (key, label) in enumerate(measures):
        values = [summary["classes"][name][key] for name in class_names]
        axis.bar(x - 0.4 + width / 2 + index * width, values, width, label=label)
    axis.set_xticks(x, class_names)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("SAM3 query quality (box annotations only)")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.text(0.01, 0.01, "* Proxy mask IoU compares predicted masks to filled GT boxes; it is not mask-ground-truth IoU.", fontsize=9)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("outputs/sam3_dfff/manifest.json"))
    parser.add_argument("--evaluation", type=Path, help="Separate query-label file; read only after inference")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sam3_dfff/results"))
    parser.add_argument("--checkpoint", type=Path, help="Optional local SAM3 checkpoint; otherwise uses Hugging Face")
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--max-visualizations", type=int, default=40)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    bpe_path = Path(__file__).resolve().parents[1] / "src/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    model = build_sam3_image_model(
        bpe_path=str(bpe_path),
        device=device,
        checkpoint_path=str(args.checkpoint) if args.checkpoint else None,
        load_from_HF=args.checkpoint is None,
    )
    processor = Sam3Processor(
        model, resolution=args.resolution, device=device, confidence_threshold=args.score_threshold
    )

    predictions: dict[str, list[dict[str, Any]]] = {}
    saved_masks: dict[str, np.ndarray] = {}
    plot_jobs: list[tuple[str, list[dict[str, Any]], list[Image.Image], Image.Image, np.ndarray, dict[str, Any]]] = []
    query_number = 0
    precision_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device == "cuda"
        else nullcontext()
    )
    with torch.inference_mode(), precision_context:
        for class_name, class_records in manifest["classes"].items():
            predictions[class_name] = []
            for query in class_records["queries"]:
                prediction, mask, _composite, _support_images = _predict_one(
                    processor,
                    class_name,
                    class_records["supports"],
                    query,
                    args.resolution,
                )
                mask_name = f"mask_{query_number:05d}.png"
                Image.fromarray(mask.astype(np.uint8) * 255).save(args.output_dir / mask_name)
                prediction["mask_path"] = mask_name
                predictions[class_name].append(prediction)
                saved_masks[query["query_id"]] = mask
                if len(plot_jobs) < args.max_visualizations:
                    support_images = [
                        Image.open(record["image_path"]).convert("RGB")
                        for record in class_records["supports"]
                    ]
                    query_image = Image.open(query["image_path"]).convert("RGB")
                    plot_jobs.append(
                        (class_name, class_records["supports"], support_images, query_image, mask, prediction)
                    )
                query_number += 1
                print(
                    f"[{query_number}] {class_name}: {query['query_id']} "
                    f"score={prediction['score']:.3f} mask={prediction['mask_found']}"
                )

    # Deliberately read target boxes only when every model call has completed.
    evaluation_path = args.evaluation or args.manifest.with_name("evaluation.json")
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    metrics = _evaluate(predictions, saved_masks, evaluation)
    for class_name, records in predictions.items():
        gt = evaluation["classes"].get(class_name, {})
        supports = manifest["classes"][class_name]["supports"]
        for index, job in enumerate((job for job in plot_jobs if job[0] == class_name)):
            _, _support_records, support_images, query_image, mask, prediction = job
            _plot_pair(
                args.output_dir / f"visual_{class_name.replace(' ', '_')}_{index:04d}.png",
                class_name,
                supports,
                support_images,
                query_image,
                mask,
                prediction,
                gt.get(prediction["query_id"]),
            )
    _plot_metrics(args.output_dir / "metrics.png", metrics)
    (args.output_dir / "predictions.json").write_text(json.dumps(predictions, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"\nMetrics: {args.output_dir / 'metrics.json'}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()