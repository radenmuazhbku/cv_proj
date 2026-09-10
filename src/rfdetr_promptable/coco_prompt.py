"""Minimal COCO episodic dataset for box-prompted RF-DETR training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class CocoBoxPromptDataset(Dataset[dict[str, Any]]):
    """Return positive object prompts and negative boxes with no selected object."""

    def __init__(
        self,
        image_dir: str | Path,
        annotation_file: str | Path,
        resolution: int = 384,
        negative_ratio: float = 1.0,
        negative_iou_threshold: float = 0.1,
        seed: int = 7,
        max_examples: int | None = None,
    ) -> None:
        if negative_ratio < 0:
            raise ValueError("negative_ratio must be non-negative")
        if not 0 <= negative_iou_threshold < 1:
            raise ValueError("negative_iou_threshold must be in [0, 1)")
        self.image_dir = Path(image_dir)
        self.coco = COCO(str(annotation_file))
        self.resolution = resolution
        self.negative_iou_threshold = negative_iou_threshold
        self.seed = seed
        self.records = []
        negative_count = int(negative_ratio)
        negative_fraction = negative_ratio - negative_count
        for image_id in sorted(self.coco.getImgIds()):
            annotation_ids = self.coco.getAnnIds(imgIds=[image_id], iscrowd=False)
            annotations = self.coco.loadAnns(annotation_ids)
            annotations = [annotation for annotation in annotations if annotation["bbox"][2] > 1 and annotation["bbox"][3] > 1]
            if annotations:
                for annotation in annotations:
                    self.records.append((image_id, annotation, True))
                count = negative_count + (1 if negative_fraction and (image_id + seed) % 1000 < negative_fraction * 1000 else 0)
                self.records.extend((image_id, annotations, False) for _ in range(count))
        if max_examples is not None:
            if max_examples <= 0:
                raise ValueError("max_examples must be positive")
            generator = torch.Generator().manual_seed(seed)
            permutation = torch.randperm(len(self.records), generator=generator).tolist()
            self.records = [self.records[index] for index in permutation[:max_examples]]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_id, record, is_positive = self.records[index]
        metadata = self.coco.loadImgs([image_id])[0]
        image = Image.open(self.image_dir / metadata["file_name"]).convert("RGB")
        width, height = image.size
        image_tensor = TF.resize(TF.to_tensor(image), [self.resolution, self.resolution])
        if is_positive:
            x, y, box_width, box_height = record["bbox"]
        else:
            boxes = [annotation["bbox"] for annotation in record]
            x, y, box_width, box_height = _sample_negative_box(
                width, height, boxes, seed=self.seed + index, iou_threshold=self.negative_iou_threshold
            )
        prompt_box = torch.tensor(
            [(x + box_width / 2) / width, (y + box_height / 2) / height, box_width / width, box_height / height],
            dtype=torch.float32,
        )
        return {"image": image_tensor, "prompt_box": prompt_box, "present": float(is_positive), "image_id": image_id}


def collate_prompts(batch: list[dict[str, Any]]) -> tuple[Tensor, Tensor, Tensor]:
    images = torch.stack([item["image"] for item in batch])
    prompt_boxes = torch.stack([item["prompt_box"] for item in batch]).unsqueeze(1)
    present = torch.tensor([[item["present"]] for item in batch], dtype=torch.float32)
    return images, prompt_boxes, present


def _sample_negative_box(
    image_width: int,
    image_height: int,
    boxes: list[list[float]],
    seed: int,
    iou_threshold: float,
) -> tuple[float, float, float, float]:
    """Sample a valid image box whose IoU with selected targets stays below the threshold."""
    generator = torch.Generator().manual_seed(seed)
    target_boxes = torch.tensor(boxes, dtype=torch.float32)
    target_xyxy = torch.stack(
        [target_boxes[:, 0], target_boxes[:, 1], target_boxes[:, 0] + target_boxes[:, 2], target_boxes[:, 1] + target_boxes[:, 3]],
        dim=1,
    )
    for _ in range(50):
        box_width = max(4.0, image_width * 0.05) + float(torch.rand((), generator=generator)) * (image_width * 0.45)
        box_height = max(4.0, image_height * 0.05) + float(torch.rand((), generator=generator)) * (image_height * 0.45)
        x = float(torch.rand((), generator=generator)) * max(1.0, image_width - box_width)
        y = float(torch.rand((), generator=generator)) * max(1.0, image_height - box_height)
        candidate = torch.tensor([[x, y, x + box_width, y + box_height]])
        intersection_lt = torch.maximum(candidate[:, None, :2], target_xyxy[None, :, :2])
        intersection_rb = torch.minimum(candidate[:, None, 2:], target_xyxy[None, :, 2:])
        intersection = (intersection_rb - intersection_lt).clamp_min(0).prod(-1)
        candidate_area = box_width * box_height
        target_area = (target_xyxy[:, 2:] - target_xyxy[:, :2]).clamp_min(0).prod(-1)
        iou = intersection / (candidate_area + target_area - intersection).clamp_min(1e-6)
        if float(iou.max()) <= iou_threshold:
            return x, y, box_width, box_height
    return 0.0, 0.0, max(4.0, image_width * 0.05), max(4.0, image_height * 0.05)
