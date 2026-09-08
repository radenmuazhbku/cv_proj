"""Evaluate box-prompted detection on a filtered COCO annotation file."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from pycocotools.coco import COCO
from torchvision.ops import box_iou

from rfdetr_promptable.config import RFDETRNanoConfig
from rfdetr_promptable.promptable import PromptableDetector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--annotation-file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=0.0)
    args = parser.parse_args()

    coco = COCO(str(args.annotation_file))
    model = PromptableDetector.from_config(
        RFDETRNanoConfig(
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

    scores, ious = [], []
    for image_id in coco.getImgIds():
        metadata = coco.loadImgs([image_id])[0]
        ann_ids = coco.getAnnIds(imgIds=[image_id], iscrowd=False)
        annotations = [ann for ann in coco.loadAnns(ann_ids) if ann["bbox"][2] > 1 and ann["bbox"][3] > 1]
        if not annotations:
            continue
        annotation = annotations[0]
        image = Image.open(args.image_dir / metadata["file_name"]).convert("RGB")
        x, y, width, height = annotation["bbox"]
        prediction = model.predict(image, [[x, y, x + width, y + height]], threshold=args.threshold)
        if len(prediction["boxes"]) == 0:
            continue
        target = torch.tensor([[x, y, x + width, y + height]])
        iou = float(box_iou(prediction["boxes"][:1], target)[0, 0])
        scores.append(float(prediction["scores"][0]))
        ious.append(iou)

    if not ious:
        raise RuntimeError("No prompted test examples produced predictions")
    print(f"examples={len(ious)} mean_score={sum(scores) / len(scores):.4f} mean_iou={sum(ious) / len(ious):.4f}")


if __name__ == "__main__":
    main()
