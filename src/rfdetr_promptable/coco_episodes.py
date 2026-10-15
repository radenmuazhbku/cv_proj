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
    """Create same-image visual-prompt episodes from COCO annotations.

    One instance box in an image is the visual prompt; the other instances of
    that category in the same image are detection targets. Images must contain
    at least two instances of the chosen category.
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
        self.category_image_instances = {
            category_id: {
                image_id: [ann for ann in self.image_annotations[image_id] if ann["category_id"] == category_id]
                for image_id in image_ids
                if sum(ann["category_id"] == category_id for ann in self.image_annotations[image_id]) >= 2
            }
            for category_id, image_ids in self.category_images.items()
        }
        self.category_image_instances = {
            category_id: records for category_id, records in self.category_image_instances.items() if records
        }
        self.category_ids = sorted(self.category_image_instances)
        if not self.category_ids:
            raise ValueError("COCO annotation subset needs an image with at least two instances of one category")
        self.eval_records: list[tuple[int, int, bool]] = []
        if not training:
            for category_id in self.category_ids:
                self.eval_records.extend(
                    (category_id, image_id, True) for image_id in sorted(self.category_image_instances[category_id])
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
            positive_episode = True
        else:
            category_id, fixed_query_id, positive_episode = self.eval_records[index]
        if self.training:
            query_id = rng.choice(sorted(self.category_image_instances[category_id]))
        else:
            query_id = fixed_query_id
        annotations = self.category_image_instances[category_id][query_id]
        prompt_index = rng.randrange(len(annotations)) if self.training else (
            (self.seed + query_id + category_id) % len(annotations)
        )
        prompt_annotation = annotations[prompt_index]
        target_annotations = [ann for ann in annotations if ann["id"] != prompt_annotation["id"]]
        query_tensor, query_width, query_height = self._load_tensor(query_id)
        prompt_box = self._normalize_box(prompt_annotation["bbox"], query_width, query_height)
        target_boxes = torch.stack(
            [self._normalize_box(ann["bbox"], query_width, query_height) for ann in target_annotations]
        )
        present = 1.0
        return {
            "support_images": query_tensor.unsqueeze(0),
            "support_boxes": prompt_box.unsqueeze(0),
            "query_image": query_tensor,
            "target_box": target_boxes[0],
            "target_boxes": target_boxes,
            "present": present,
            "category_id": category_id,
            "query_image_id": query_id,
            "support_image_ids": [query_id],
            "prompt_annotation_id": prompt_annotation["id"],
        }

    def _sample_training_category(self, rng: random.Random) -> tuple[int, int]:
        category_id = rng.choice(self.category_ids)
        return category_id, -1


def collate_few_shot(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate query targets and the current single prompt per episode.

    TODO: Preserve every support image/box and its full annotation set when the
    dataset is extended to multi-shot episodes; this collator currently takes only
    ``support_boxes[0]`` and builds targets from the remaining same-image instances.
    """
    max_targets = max(item["target_boxes"].shape[0] for item in batch)
    query_eval_boxes = torch.zeros((len(batch), max_targets, 4), dtype=torch.float32)
    query_eval_counts = torch.zeros(len(batch), dtype=torch.long)
    for index, item in enumerate(batch):
        target_boxes = item["target_boxes"]
        query_eval_boxes[index, : target_boxes.shape[0]] = target_boxes
        query_eval_counts[index] = target_boxes.shape[0] if item["present"] else 0
    return {
        "support_boxes": torch.stack([item["support_boxes"][0] for item in batch]),
        "query_images": torch.stack([item["query_image"] for item in batch]),
        "query_boxes": query_eval_boxes,
        "query_target_mask": query_eval_counts.gt(0).unsqueeze(1) & (
            torch.arange(max_targets).unsqueeze(0) < query_eval_counts.unsqueeze(1)
        ),
        "query_eval_boxes": query_eval_boxes,
        "query_eval_counts": query_eval_counts,
        "present": torch.tensor([[item["present"]] for item in batch], dtype=torch.float32),
        "category_ids": [item["category_id"] for item in batch],
        "query_image_ids": [item["query_image_id"] for item in batch],
        "support_image_ids": [item["support_image_ids"] for item in batch],
        "prompt_annotation_ids": [item["prompt_annotation_id"] for item in batch],
    }
