"""COCO support/query episodes for visual few-shot object prompting.

This is a patched version that properly handles Albumentations augmentation
with bounding box and mask transformation.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from rfdetr_pcs.coco_prompt import COCO


class CocoFewShotEpisodeDataset(Dataset[dict[str, Any]]):
    """Create same-image or separate support/query episodes from COCO data."""

    def __init__(
        self,
        image_dir: str | Path,
        annotation_file: str | Path,
        support_image_dir: str | Path | None = None,
        resolution: int = 560,
        episodes: int = 10_000,
        max_shots: int = 5,
        negative_ratio: float = 0.5,
        negative_prompt_ratio: float = 0.0,
        negative_class_ratio: float = 0.0,
        negative_eval_per_class: int = 100,
        support_annotation_file: str | Path | None = None,
        seed: int = 7,
        training: bool = True,
        include_masks: bool = False,
        aug_config: dict[str, Any] | None = None,
        augmentation_backend: str = "cpu",
    ) -> None:
        if (episodes <= 0 or max_shots <= 0 or negative_ratio < 0
                or negative_prompt_ratio < 0 or negative_class_ratio < 0
                or negative_eval_per_class < 0):
            raise ValueError("episodes and max_shots must be positive; ratios/counts must be non-negative")
        self.image_dir = Path(image_dir)
        self.annotation_file = Path(annotation_file)
        self.coco = COCO(str(annotation_file))
        self.has_separate_support = support_image_dir is not None

        if support_annotation_file is not None:
            self.support_coco = COCO(str(support_annotation_file))
            self.support_image_dir = Path(support_annotation_file).parent if support_annotation_file is not None else Path(support_image_dir)
            self.support_annotation_file = Path(support_annotation_file)
        else:
            self.support_coco = None
            self.support_image_dir = Path(support_image_dir) if support_image_dir is not None else self.image_dir
            self.support_annotation_file = None

        self.resolution = resolution
        self.episodes = episodes
        self.max_shots = max_shots
        self.negative_ratio = negative_ratio
        self.negative_prompt_ratio = negative_prompt_ratio
        self.negative_class_ratio = negative_class_ratio
        self.negative_eval_per_class = negative_eval_per_class
        self.seed = seed
        self.training = training
        self.include_masks = include_masks

        self.category_ids = sorted(set(ann["category_id"] for ann in self.coco.anns.values()))
        self.category_image_instances: dict[int, dict[int, list[dict[str, Any]]]] = {}
        self.support_image_annotations: dict[int, list[dict[str, Any]]] = {}
        self.category_image_instances_all: dict[int, dict[int, list[dict[str, Any]]]] = {}
        for ann in self.coco.anns.values():
            category_id = int(ann["category_id"])
            image_id = int(ann["image_id"])
            self.category_image_instances.setdefault(category_id, {}).setdefault(image_id, []).append(ann)
            self.category_image_instances_all.setdefault(category_id, {}).setdefault(image_id, []).append(ann)

        if self.has_separate_support:
            self.support_category_image_instances: dict[int, dict[int, list[dict[str, Any]]]] = {}
            for ann in self.support_coco.anns.values():
                category_id = int(ann["category_id"])
                image_id = int(ann["image_id"])
                self.support_category_image_instances.setdefault(category_id, {}).setdefault(image_id, []).append(ann)
                self.support_image_annotations[image_id] = self.support_coco.imgToAnns.get(image_id, [])
        else:
            self.support_category_image_instances = self.category_image_instances
            self.support_image_annotations = {img["id"]: self.coco.imgToAnns.get(img["id"], []) for img in self.coco.imgs.values()}

        self.eval_records: list[tuple[int, int, bool]] = []
        if not self.training:
            for category_id in self.category_ids:
                self.eval_records.extend(
                    (category_id, image_id, True) for image_id in sorted(self.category_image_instances[category_id])
                )

        self.aug_config = aug_config
        self.augmentation_backend = augmentation_backend
        self._augmentation_transform = None
        if aug_config and training and augmentation_backend in ("cpu", "albumentations", "albu"):
            self._setup_albumentations_transform()

        # TODO: Restore remaining methods from original file
