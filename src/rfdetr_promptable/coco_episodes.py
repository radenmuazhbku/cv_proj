"""COCO support/query episodes for visual few-shot object prompting."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from rfdetr_promptable.coco_prompt import COCO


class CocoFewShotEpisodeDataset(Dataset[dict[str, Any]]):
    """Create cross-image support/query episodes from COCO annotations.

    A positive episode uses 1..max_shots support image/box examples of one category
    and a different query image containing that category. A negative episode uses
    support examples of the category and a query image with no annotation of it.
    """

    def __init__(
        self,
        image_dir: str | Path,
        annotation_file: str | Path,
        support_image_dir: str | Path | None = None,
        resolution: int = 560,
        episodes: int = 10_000,
        max_shots: int = 5,
        negative_ratio: float = 0.5,
        negative_eval_per_class: int = 100,
        support_annotation_file: str | Path | None = None,
        seed: int = 7,
        training: bool = True,
    ) -> None:
        if episodes <= 0 or max_shots <= 0 or negative_ratio < 0 or negative_eval_per_class < 0:
            raise ValueError("episodes and max_shots must be positive; ratios/counts must be non-negative")
        self.image_dir = Path(image_dir)
        self.support_image_dir = Path(support_image_dir) if support_image_dir is not None else self.image_dir
        self.coco = COCO(str(annotation_file))
        self.support_coco = (
            COCO(str(support_annotation_file))
            if support_annotation_file is not None
            else self.coco
        )
        self.has_separate_support = support_annotation_file is not None
        self.resolution = resolution
        self.episodes = episodes
        self.max_shots = max_shots
        self.negative_ratio = negative_ratio
        self.negative_eval_per_class = negative_eval_per_class
        self.seed = seed
        self.training = training
        categories = self.coco.loadCats(self.coco.getCatIds())
        self.category_ids = sorted(category["id"] for category in categories)
        self.category_images: dict[int, list[int]] = {}
        self.image_categories: dict[int, set[int]] = {}
        self.image_annotations: dict[int, list[dict[str, Any]]] = {}
        self.support_category_images: dict[int, list[int]] = {}
        self.support_image_annotations: dict[int, list[dict[str, Any]]] = {}
        for image_id in sorted(self.coco.getImgIds()):
            anns = [ann for ann in self.coco.loadAnns(self.coco.getAnnIds(imgIds=[image_id], iscrowd=False)) if ann["bbox"][2] > 1 and ann["bbox"][3] > 1]
            self.image_annotations[image_id] = anns
            category_set = {ann["category_id"] for ann in anns}
            self.image_categories[image_id] = category_set
            for category_id in category_set:
                self.category_images.setdefault(category_id, []).append(image_id)
        for image_id in sorted(self.support_coco.getImgIds()):
            anns = [
                ann for ann in self.support_coco.loadAnns(
                    self.support_coco.getAnnIds(imgIds=[image_id], iscrowd=False)
                )
                if ann["bbox"][2] > 1 and ann["bbox"][3] > 1
            ]
            self.support_image_annotations[image_id] = anns
            for category_id in {ann["category_id"] for ann in anns}:
                self.support_category_images.setdefault(category_id, []).append(image_id)
        if self.has_separate_support:
            overlap = set(self.coco.getImgIds()) & set(self.support_coco.getImgIds())
            if overlap:
                raise ValueError(
                    f"support and query annotation files overlap on {len(overlap)} image IDs"
                )
            self.category_ids = [
                category_id for category_id in self.category_ids
                if self.category_images.get(category_id) and self.support_category_images.get(category_id)
            ]
        else:
            self.category_ids = [
                category_id for category_id in self.category_ids
                if len(self.category_images.get(category_id, [])) >= 2
            ]
        if not self.category_ids:
            raise ValueError("COCO annotation subset needs at least one category in two or more images")
        self.eval_records: list[tuple[int, int, bool]] = []
        if not training:
            for category_id in self.category_ids:
                self.eval_records.extend(
                    (category_id, image_id, True) for image_id in self.category_images[category_id]
                )
                absent_image_ids = [
                    image_id
                    for image_id in sorted(self.image_categories)
                    if category_id not in self.image_categories[image_id]
                ]
                absent_image_ids.sort(key=lambda image_id: (self.seed * 1_000_003 + image_id) % 2**32)
                self.eval_records.extend(
                    (category_id, image_id, False)
                    for image_id in absent_image_ids[: self.negative_eval_per_class]
                )

    def __len__(self) -> int:
        return self.episodes if self.training else len(self.eval_records)

    def _load_tensor(
        self, image_id: int, support: bool = False
    ) -> tuple[Tensor, int, int]:
        coco = self.support_coco if support else self.coco
        metadata = coco.loadImgs([image_id])[0]
        image_root = self.support_image_dir if support else self.image_dir
        image = Image.open(image_root / metadata["file_name"]).convert("RGB")
        width, height = image.size
        tensor = TF.resize(TF.to_tensor(image), [self.resolution, self.resolution])
        return tensor, width, height

    @staticmethod
    def _normalize_box(box: list[float], width: int, height: int) -> Tensor:
        x, y, box_width, box_height = box
        return torch.tensor(
            [(x + box_width / 2) / width, (y + box_height / 2) / height, box_width / width, box_height / height],
            dtype=torch.float32,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = random.Random(self.seed + index)
        if self.training:
            category_id, fixed_query_id = self._sample_training_category(rng)
            positive_episode = False
        else:
            category_id, fixed_query_id, positive_episode = self.eval_records[index]
        category_image_ids = self.category_images[category_id]
        if self.training:
            shot_count = rng.randint(1, min(self.max_shots, len(category_image_ids) - 1))
            positive_episode = rng.random() >= self.negative_ratio / (1.0 + self.negative_ratio)
            if positive_episode:
                query_id = rng.choice(category_image_ids)
                support_pool = [image_id for image_id in category_image_ids if image_id != query_id]
                shot_count = min(shot_count, len(support_pool))
                support_ids = rng.sample(support_pool, shot_count)
            else:
                support_ids = rng.sample(category_image_ids, shot_count)
                support_set = set(support_ids)
                query_candidates = [
                    image_id
                    for image_id, present_categories in self.image_categories.items()
                    if image_id not in support_set and category_id not in present_categories
                ]
                if not query_candidates:
                    positive_episode = True
                    query_id = rng.choice(category_image_ids)
                    support_pool = [image_id for image_id in category_image_ids if image_id != query_id]
                    support_ids = rng.sample(support_pool, min(shot_count, len(support_pool)))
                else:
                    query_id = rng.choice(query_candidates)
        else:
            query_id = fixed_query_id
            if self.has_separate_support:
                support_pool = self.support_category_images[category_id]
                support_ids = sorted(
                    support_pool,
                    key=lambda image_id: (self.seed * 1_000_003 + image_id) % 2**32,
                )[: self.max_shots]
            else:
                support_pool = [image_id for image_id in category_image_ids if image_id != query_id]
                support_ids = sorted(
                    support_pool,
                    key=lambda image_id: (self.seed * 1_000_003 + image_id) % 2**32,
                )[: self.max_shots]
            shot_count = len(support_ids)

        support_images, support_boxes = [], []
        for support_id in support_ids:
            support_annotations = (
                self.support_image_annotations if self.has_separate_support else self.image_annotations
            )
            annotations = [ann for ann in support_annotations[support_id] if ann["category_id"] == category_id]
            annotation = rng.choice(annotations)
            tensor, width, height = self._load_tensor(support_id, support=self.has_separate_support)
            support_images.append(tensor)
            support_boxes.append(self._normalize_box(annotation["bbox"], width, height))

        query_tensor, query_width, query_height = self._load_tensor(query_id)
        if positive_episode:
            targets = [ann for ann in self.image_annotations[query_id] if ann["category_id"] == category_id]
            target_boxes = torch.stack(
                [self._normalize_box(ann["bbox"], query_width, query_height) for ann in targets]
            )
            target_box = target_boxes[rng.randrange(len(target_boxes))]
            present = 1.0
        else:
            target_box = torch.tensor([0.5, 0.5, 0.0, 0.0], dtype=torch.float32)
            target_boxes = target_box[None, :]
            present = 0.0
        return {
            "support_images": torch.stack(support_images),
            "support_boxes": torch.stack(support_boxes),
            "query_image": query_tensor,
            "target_box": target_box,
            "target_boxes": target_boxes,
            "present": present,
            "category_id": category_id,
            "query_image_id": query_id,
            "support_image_ids": support_ids,
        }

    def _sample_training_category(self, rng: random.Random) -> tuple[int, int]:
        return rng.choice(self.category_ids), -1


def collate_few_shot(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Flatten variable-shot support sets and retain query-to-concept assignments."""
    support_images = torch.cat([item["support_images"] for item in batch], dim=0)
    support_boxes = torch.cat([item["support_boxes"] for item in batch], dim=0)
    support_group_ids = torch.cat(
        [torch.full((item["support_images"].shape[0],), index, dtype=torch.long) for index, item in enumerate(batch)]
    )
    max_targets = max(item["target_boxes"].shape[0] for item in batch)
    query_eval_boxes = torch.zeros((len(batch), max_targets, 4), dtype=torch.float32)
    query_eval_counts = torch.zeros(len(batch), dtype=torch.long)
    for index, item in enumerate(batch):
        target_boxes = item["target_boxes"]
        query_eval_boxes[index, : target_boxes.shape[0]] = target_boxes
        query_eval_counts[index] = target_boxes.shape[0] if item["present"] else 0
    return {
        "support_images": support_images,
        "support_boxes": support_boxes,
        "support_group_ids": support_group_ids,
        "query_images": torch.stack([item["query_image"] for item in batch]),
        "query_boxes": torch.stack([item["target_box"] for item in batch])[:, None, :],
        "query_eval_boxes": query_eval_boxes,
        "query_eval_counts": query_eval_counts,
        "present": torch.tensor([[item["present"]] for item in batch], dtype=torch.float32),
        "query_to_support": torch.arange(len(batch), dtype=torch.long),
        "category_ids": [item["category_id"] for item in batch],
        "query_image_ids": [item["query_image_id"] for item in batch],
        "support_image_ids": [item["support_image_ids"] for item in batch],
    }
