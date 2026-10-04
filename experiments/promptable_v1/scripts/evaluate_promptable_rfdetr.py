"""Evaluate box-prompted detection on a filtered COCO annotation file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pycocotools.coco import COCO
from torchvision.ops import box_iou

from rfdetr_promptable.config import RFDETRBaseConfig
from rfdetr_promptable.promptable import PromptableDetector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--annotation-file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=560)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--figure-dir", type=Path)
    parser.add_argument("--max-figures-per-class", type=int, default=10)
    parser.add_argument("--max-images", type=int)
    args = parser.parse_args()

    coco = COCO(str(args.annotation_file))
    model = PromptableDetector.from_config(
        RFDETRBaseConfig(
            pretrain_weights=None,
            resolution=args.resolution,
            num_queries=1,
            num_select=1,
            group_detr=1,
            two_stage=False,
            device=args.device,
        )
    ).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model.eval()

    categories = {category["id"]: category["name"] for category in coco.loadCats(coco.getCatIds())}
    metrics: dict[str, dict[str, float | int]] = {}
    figure_counts: dict[int, int] = {}
    image_ids = coco.getImgIds()
    if args.max_images is not None:
        image_ids = image_ids[: args.max_images]
    for image_id in image_ids:
        metadata = coco.loadImgs([image_id])[0]
        ann_ids = coco.getAnnIds(imgIds=[image_id], iscrowd=False)
        annotations = [ann for ann in coco.loadAnns(ann_ids) if ann["bbox"][2] > 1 and ann["bbox"][3] > 1]
        if not annotations:
            continue
        annotation = annotations[0]
        category_id = annotation["category_id"]
        image = Image.open(args.image_dir / metadata["file_name"]).convert("RGB")
        x, y, width, height = annotation["bbox"]
        prediction = model.predict(image, [[x, y, x + width, y + height]], threshold=args.threshold)
        if len(prediction["boxes"]) == 0:
            continue
        target = torch.tensor([[x, y, x + width, y + height]])
        iou = float(box_iou(prediction["boxes"][:1], target)[0, 0])
        result = metrics.setdefault(categories[category_id], {"examples": 0, "mean_score": 0.0, "mean_iou": 0.0, "recall50": 0})
        result["examples"] = int(result["examples"]) + 1
        result["mean_score"] = float(result["mean_score"]) + float(prediction["scores"][0])
        result["mean_iou"] = float(result["mean_iou"]) + iou
        result["recall50"] = int(result["recall50"]) + int(iou >= 0.5)
        if args.figure_dir is not None and figure_counts.get(category_id, 0) < args.max_figures_per_class:
            _save_figures(args.figure_dir, categories[category_id], metadata["file_name"], image, [x, y, x + width, y + height], prediction)
            figure_counts[category_id] = figure_counts.get(category_id, 0) + 1

    if not metrics:
        raise RuntimeError("No prompted test examples produced predictions")
    total = sum(int(result["examples"]) for result in metrics.values())
    for result in metrics.values():
        count = int(result["examples"])
        result["mean_score"] = float(result["mean_score"]) / count
        result["mean_iou"] = float(result["mean_iou"]) / count
        result["recall50"] = int(result["recall50"]) / count
    print(f"examples={total} mean_iou={sum(float(r['mean_iou']) * int(r['examples']) for r in metrics.values()) / total:.4f}")
    if args.figure_dir is not None:
        args.figure_dir.mkdir(parents=True, exist_ok=True)
        (args.figure_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))


def _save_figures(figure_dir: Path, class_name: str, filename: str, image: Image.Image, target_box: list[float], prediction: dict[str, torch.Tensor]) -> None:
    safe_name = class_name.replace(" ", "_")
    stem = Path(filename).stem
    original_dir = figure_dir / "original" / safe_name
    prediction_dir = figure_dir / "prediction" / safe_name
    combined_dir = figure_dir / "prediction_gt" / safe_name
    for directory in (original_dir, prediction_dir, combined_dir):
        directory.mkdir(parents=True, exist_ok=True)

    def draw(path: Path, include_prompt: bool, include_target: bool) -> None:
        figure, axis = plt.subplots(figsize=(8, 6))
        axis.imshow(image)
        if include_prompt or include_target:
            axis.add_patch(patches.Rectangle((target_box[0], target_box[1]), target_box[2] - target_box[0], target_box[3] - target_box[1], fill=False, edgecolor="green" if include_target else "blue", linewidth=2, label="ground truth" if include_target else "prompt"))
        if len(prediction["boxes"]):
            box = prediction["boxes"][0].tolist()
            axis.add_patch(patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, edgecolor="red", linewidth=2, label="prediction"))
        axis.axis("off")
        axis.legend(loc="upper right")
        figure.savefig(path, bbox_inches="tight", dpi=120)
        plt.close(figure)

    draw(original_dir / f"{stem}.png", include_prompt=False, include_target=False)
    draw(prediction_dir / f"{stem}.png", include_prompt=True, include_target=False)
    draw(combined_dir / f"{stem}.png", include_prompt=False, include_target=True)


if __name__ == "__main__":
    main()
