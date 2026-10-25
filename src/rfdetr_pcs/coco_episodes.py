"""COCO support/query episodes for visual few-shot object prompting."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from rfdetr_pcs.coco_prompt import COCO


class CocoFewShotEpisodeDataset(Dataset[dict[str, Any]]):
    """Create same-image or separate support/query episodes from COCO data.

    Same-image episodes require at least two instances of the chosen category
    so at least one remains after selecting a positive prompt. Separate
    support/query episodes accept query images with one or more targets.
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
        self.negative_prompt_ratio = negative_prompt_ratio
        self.negative_class_ratio = negative_class_ratio
        self.negative_eval_per_class = negative_eval_per_class
        self.seed = seed
        self.training = training
        self.include_masks = include_masks
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
            if overlap and not self.training:
                raise ValueError(
                    f"support and query annotation files overlap on {len(overlap)} image IDs during evaluation"
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
                if sum(ann["category_id"] == category_id for ann in self.image_annotations[image_id])
                >= (1 if self.has_separate_support else 2)
            }
            for category_id, image_ids in self.category_images.items()
        }
        self.category_image_instances_all = {
            category_id: {
                image_id: [ann for ann in self.image_annotations[image_id] if ann["category_id"] == category_id]
                for image_id in image_ids
            }
            for category_id, image_ids in self.category_images.items()
        }
        self.category_image_instances = {
            category_id: records for category_id, records in self.category_image_instances.items() if records
        }
        self.category_ids = sorted(
            category_id for category_id in self.category_image_instances
            if not self.has_separate_support
            or len(self.support_category_images.get(category_id, [])) >= self.max_shots
        )
        if not self.category_ids:
            requirement = (
                "a query image with an instance and enough support images for max_shots"
                if self.has_separate_support
                else "an image with at least two instances of one category"
            )
            raise ValueError(f"COCO annotation subset needs {requirement}")
        self.eval_records: list[tuple[int, int, bool]] = []
        if not training:
            for category_id in self.category_ids:
                self.eval_records.extend(
                    (category_id, image_id, True) for image_id in sorted(self.category_image_instances[category_id])
                )

        self.aug_config = aug_config
        self.augmentation_backend = augmentation_backend
        self._augmentation_pipeline = None
        if self.aug_config and self.training:
            self._setup_augmentation_pipeline()

    def __len__(self) -> int:
        return self.episodes if self.training else len(self.eval_records)

    def _setup_augmentation_pipeline(self) -> None:
        backend = self.augmentation_backend
        if backend in ("kornia", "gpu"):
            try:
                import kornia.augmentation as K
                self._augmentation_pipeline = K.AugmentationSequential(
                    *self._build_kornia_transforms(),
                    data_keys=["input"],
                    random_apply=True,
                )
            except ImportError:
                import warnings
                warnings.warn(
                    "Kornia not installed, falling back to CPU augmentation. "
                    "Install with: pip install 'rfdetr[augment]'"
                )
                backend = "albumentations"
        if backend in ("albumentations", "albu", "cpu"):
            try:
                import albumentations as A
                self._augmentation_pipeline = A.Compose(self._build_albumentations_transform())
            except ImportError:
                import warnings
                warnings.warn(
                    "Albumentations not installed, falling back to torchvision. "
                    "Install with: pip install 'rfdetr[augment]'"
                )
                self._augmentation_pipeline = None

    def _build_albumentations_transform(self) -> list:
        from rfdetr_pcs.datasets.transforms import AlbumentationsWrapper
        transforms = AlbumentationsWrapper.from_config(self.aug_config or {})
        return [t.transform for t in transforms]

    def _build_kornia_transforms(self) -> list:
        import kornia.augmentation as K
        from rfdetr_pcs.datasets.kornia_transforms import build_kornia_pipeline
        if self.aug_config:
            return build_kornia_pipeline(self.aug_config, self.resolution, with_masks=self.include_masks)
        return []

    def _apply_augmentation(self, image_tensor: Tensor) -> Tensor:
        if self._augmentation_pipeline is None or not self.training:
            return image_tensor
        if isinstance(self._augmentation_pipeline, type(None)):
            return image_tensor
        if "kornia" in str(type(self._augmentation_pipeline)):
            import kornia.augmentation as K
            if isinstance(image_tensor, Tensor):
                image_tensor = image_tensor.unsqueeze(0)
                augmented = self._augmentation_pipeline(image_tensor)
                return augmented.squeeze(0)
        else:
            import numpy as np
            image_np = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
            augmented = self._augmentation_pipeline(image=image_np)
            return torch.from_numpy(augmented["image"].transpose(2, 0, 1)).float().to(image_tensor.device)

    def _load_tensor(
        self, image_id: int, support: bool = False
    ) -> tuple[Tensor, int, int]:
        # coco = self.support_coco if support else self.coco
        metadata = coco.loadImgs([image_id])[0]
        image_root = self.support_image_dir if support else self.image_dir
        image = Image.open(image_root / metadata["file_name"]).convert("RGB")
        width, height = image.size
        tensor = TF.resize(TF.to_tensor(image), [self.resolution, self.resolution])
        # if not support and self.training and self.aug_config:
        # if self.training and self.aug_config:
        tensor = self._apply_augmentation(tensor)
        return tensor, width, height

    @staticmethod
    def _normalize_box(box: list[float], width: int, height: int) -> Tensor:
        x, y, box_width, box_height = box
        return torch.tensor(
            [(x + box_width / 2) / width, (y + box_height / 2) / height, box_width / width, box_height / height],
            dtype=torch.float32,
        )

    @staticmethod
    def _ann_to_mask(coco: COCO, ann: dict[str, Any], width: int, height: int) -> Tensor:
        if "segmentation" in ann and ann["segmentation"]:
            return torch.from_numpy(coco.annToMask(ann).copy()).unsqueeze(0)
        x, y, w, h = [int(round(v)) for v in ann["bbox"]]
        mask = torch.zeros((1, height, width), dtype=torch.uint8)
        mask[0, max(0, y) : min(height, y + h), max(0, x) : min(width, x + w)] = 1
        return mask

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = random.Random(self.seed + index)
        if self.has_separate_support:
            negative_class_episode = False
            # In separate support/query mode a positive support set defines the
            # concept even when the query is a negative example. Evaluation can
            # therefore include deterministic absent-query cases too.
            if self.negative_class_ratio > 0 and rng.random() < self.negative_class_ratio / (1.0 + self.negative_class_ratio):
                query_candidates = [image_id for image_id, anns in self.image_annotations.items() if anns]
                query_id = rng.choice(query_candidates)
                absent_categories = [
                    category_id for category_id in self.category_ids
                    if category_id not in self.image_categories[query_id]
                    and self.support_category_images.get(category_id)
                ]
                if absent_categories:
                    category_id = rng.choice(absent_categories)
                    negative_class_episode = True
                else:
                    category_id, _ = self._sample_training_category(rng)
                    query_id = rng.choice(sorted(self.category_image_instances[category_id]))
            else:
                category_id, fixed_query_id = (
                    self._sample_training_category(rng) if self.training else
                    (self.eval_records[index][0], self.eval_records[index][1])
                )
                query_id = (
                    rng.choice(sorted(self.category_image_instances[category_id]))
                    if self.training else fixed_query_id
                )
            query_annotations = (
                [] if negative_class_episode
                else self.category_image_instances[category_id][query_id]
            )
            query_tensor, query_width, query_height = self._load_tensor(query_id)
            target_boxes = torch.stack(
                [self._normalize_box(ann["bbox"], query_width, query_height) for ann in query_annotations]
            ) if query_annotations else torch.empty((0, 4), dtype=torch.float32)
            target_masks = None
            if self.include_masks:
                target_masks = torch.stack(
                    [
                        TF.resize(
                            self._ann_to_mask(self.coco, ann, query_width, query_height),
                            [self.resolution, self.resolution],
                            interpolation=InterpolationMode.NEAREST,
                        ).squeeze(0).bool()
                        for ann in query_annotations
                    ]
                ) if query_annotations else torch.zeros(
                    (0, self.resolution, self.resolution), dtype=torch.bool
                )

            support_ids = sorted(self.support_category_images[category_id])
            if self.training:
                # Exclude query_id so support and query images are strictly disjoint
                candidate_support_ids = [sid for sid in support_ids if sid != query_id]
                if len(candidate_support_ids) >= self.max_shots:
                    support_ids = rng.sample(candidate_support_ids, self.max_shots)
                else:
                    support_ids = rng.sample(support_ids, min(self.max_shots, len(support_ids)))
            else:
                candidate_support_ids = [sid for sid in support_ids if sid != query_id]
                if candidate_support_ids:
                    support_ids = candidate_support_ids
                start = (self.seed + query_id + category_id) % len(support_ids)
                support_ids = [support_ids[(start + shot) % len(support_ids)] for shot in range(self.max_shots)]
            support_images, support_boxes, support_labels, support_prompt_valid = [], [], [], []
            support_target_boxes, support_target_masks, support_target_prompt_masks = [], [], []
            support_prompt_masks = []
            prompt_annotations, support_prompt_annotation_ids = [], []
            for support_id in support_ids:
                all_support_annotations = self.support_image_annotations[support_id]
                support_annotations = [
                    ann for ann in all_support_annotations if ann["category_id"] == category_id
                ]
                prompt_annotation = support_annotations[rng.randrange(len(support_annotations))]
                prompt_annotations.append(prompt_annotation)
                # TODO: Replace annotation-sampled distractors with SAM3-style
                # iterative prompts mined from this model's confident false positives.
                negative_candidates = [
                    ann for ann in all_support_annotations if ann["category_id"] != category_id
                ]
                negative_count = self._negative_prompt_count(
                    1, len(negative_candidates), rng
                )
                negative_annotations = rng.sample(negative_candidates, negative_count)
                shot_prompts = [prompt_annotation, *negative_annotations]
                support_prompt_annotation_ids.append([ann["id"] for ann in shot_prompts])
                support_tensor, support_width, support_height = self._load_tensor(support_id, support=True)
                support_images.append(support_tensor)
                support_boxes.append(torch.stack([
                    self._normalize_box(ann["bbox"], support_width, support_height)
                    for ann in shot_prompts
                ]))
                support_labels.append(torch.tensor(
                    [1, *([0] * negative_count)], dtype=torch.long
                ))
                support_prompt_valid.append(torch.ones(len(shot_prompts), dtype=torch.bool))
                support_target_boxes.append(torch.stack([
                    self._normalize_box(ann["bbox"], support_width, support_height)
                    for ann in support_annotations
                ]))
                support_target_prompt_masks.append(torch.tensor(
                    [ann["id"] == prompt_annotation["id"] for ann in support_annotations],
                    dtype=torch.bool,
                ))
                if self.include_masks:
                    support_prompt_masks.append(
                        TF.resize(
                            self._ann_to_mask(self.support_coco, prompt_annotation, support_width, support_height),
                            [self.resolution, self.resolution],
                            interpolation=InterpolationMode.NEAREST,
                        ).squeeze(0).bool()
                    )
                    support_target_masks.append(torch.stack([
                        TF.resize(
                            self._ann_to_mask(self.support_coco, ann, support_width, support_height),
                            [self.resolution, self.resolution],
                            interpolation=InterpolationMode.NEAREST,
                        ).squeeze(0).bool()
                        for ann in support_annotations
                    ]))
            max_prompts = max(boxes.shape[0] for boxes in support_boxes)
            padded_support_boxes = torch.zeros((len(support_boxes), max_prompts, 4), dtype=torch.float32)
            padded_support_labels = torch.zeros((len(support_boxes), max_prompts), dtype=torch.long)
            padded_support_valid = torch.zeros((len(support_boxes), max_prompts), dtype=torch.bool)
            for shot, (boxes, labels, valid) in enumerate(
                zip(support_boxes, support_labels, support_prompt_valid, strict=True)
            ):
                count = boxes.shape[0]
                padded_support_boxes[shot, :count] = boxes
                padded_support_labels[shot, :count] = labels
                padded_support_valid[shot, :count] = valid
            return {
                "support_images": torch.stack(support_images),
                "support_boxes": padded_support_boxes,
                "support_prompt_labels": padded_support_labels,
                "support_prompt_valid": padded_support_valid,
                "support_target_boxes": support_target_boxes,
                "support_target_prompt_masks": support_target_prompt_masks,
                "support_prompt_masks": torch.stack(support_prompt_masks) if self.include_masks else None,
                "support_target_masks": support_target_masks if self.include_masks else None,
                "query_image": query_tensor,
                "target_boxes": target_boxes,
                "target_masks": target_masks,
                "present": float(not negative_class_episode),
                "is_empty_remainder": False,
                "category_id": category_id,
                "query_image_id": query_id,
                "support_image_ids": support_ids,
                "prompt_annotation_ids": [ann["id"] for ann in prompt_annotations],
                "support_prompt_annotation_ids": support_prompt_annotation_ids,
            }

        negative_class_episode = False
        if self.training and rng.random() < self.negative_class_ratio / (1.0 + self.negative_class_ratio):
            query_candidates = [image_id for image_id, anns in self.image_annotations.items() if anns]
            query_id = rng.choice(query_candidates)
            absent_categories = [
                category_id for category_id in self.category_ids
                if category_id not in self.image_categories[query_id]
            ]
            if absent_categories:
                category_id = rng.choice(absent_categories)
                negative_class_episode = True
            else:
                category_id, _ = self._sample_training_category(rng)
        else:
            category_id = None
        if not negative_class_episode:
            if self.training:
                if category_id is None:
                    category_id, _ = self._sample_training_category(rng)
                is_empty_remainder = rng.random() < self.negative_ratio / (1.0 + self.negative_ratio)
                image_pool = (
                    self.category_image_instances_all[category_id]
                    if is_empty_remainder else self.category_image_instances[category_id]
                )
                query_id = rng.choice(sorted(image_pool))
                annotations = image_pool[query_id]
                if is_empty_remainder:
                    prompt_indices = list(range(len(annotations)))
                else:
                    prompt_count = min(self.max_shots, len(annotations) - 1)
                    prompt_indices = sorted(rng.sample(range(len(annotations)), prompt_count))
            else:
                category_id, fixed_query_id, _ = self.eval_records[index]
                is_empty_remainder = False
                query_id = fixed_query_id
                annotations = self.category_image_instances[category_id][query_id]
                prompt_indices = [(self.seed + query_id + category_id) % len(annotations)]
            prompt_annotations = [annotations[prompt_index] for prompt_index in prompt_indices]
            positive_prompt_count = len(prompt_annotations)
            negative_candidates = [ann for ann in self.image_annotations[query_id]
                                   if ann["category_id"] != category_id]
            # TODO: Match SAM3's interactive training by mining high-confidence
            # false-positive predictions instead of sampling annotated distractors.
            negative_count = self._negative_prompt_count(
                positive_prompt_count, len(negative_candidates), rng
            )
            negative_annotations = rng.sample(negative_candidates, negative_count)
            prompt_annotations.extend(negative_annotations)
            prompt_labels = torch.tensor(
                [1] * positive_prompt_count + [0] * negative_count, dtype=torch.long
            )
            target_annotations = [
                ann for ann in annotations if ann["id"] not in {x["id"] for x in prompt_annotations}
            ]
        else:
            is_empty_remainder = False
            negative_candidates = self.image_annotations[query_id]
            prompt_count = min(self.max_shots, len(negative_candidates))
            prompt_annotations = rng.sample(negative_candidates, prompt_count)
            prompt_labels = torch.zeros(prompt_count, dtype=torch.long)
            positive_prompt_count = 0
            negative_count = prompt_count
            target_annotations = []
        prompt_annotation_ids = {ann["id"] for ann in prompt_annotations}
        query_tensor, query_width, query_height = self._load_tensor(query_id)
        prompt_boxes = torch.stack(
            [self._normalize_box(ann["bbox"], query_width, query_height) for ann in prompt_annotations]
        )
        target_boxes = torch.stack([self._normalize_box(ann["bbox"], query_width, query_height)
                                    for ann in target_annotations]) if target_annotations else torch.empty((0, 4))
        target_masks = None
        prompt_masks = None
        if self.include_masks:
            target_masks = torch.stack([
                    TF.resize(
                        self._ann_to_mask(self.coco, ann, query_width, query_height),
                        [self.resolution, self.resolution],
                        interpolation=InterpolationMode.NEAREST,
                    ).squeeze(0).bool()
                    for ann in target_annotations
                ]) if target_annotations else torch.zeros((0, self.resolution, self.resolution), dtype=torch.bool)
            prompt_masks = torch.stack([
                TF.resize(
                    self._ann_to_mask(self.coco, ann, query_width, query_height),
                    [self.resolution, self.resolution],
                    interpolation=InterpolationMode.NEAREST,
                ).squeeze(0).bool()
                for ann in prompt_annotations
            ])
        present = float(not negative_class_episode)
        return {
            # In same-image mode, prompts belong to query_image itself. We avoid duplicating query_tensor.
            "support_images": query_tensor.unsqueeze(0),
            "support_boxes": prompt_boxes,
            "query_image": query_tensor,
            "target_boxes": target_boxes,
            "target_masks": target_masks,
            "prompt_masks": prompt_masks,
            "prompt_labels": prompt_labels,
            "prompt_target_mask": torch.tensor(
                [not is_empty_remainder] * positive_prompt_count + [False] * negative_count,
                dtype=torch.bool,
            ),
            "present": present,
            "is_empty_remainder": is_empty_remainder,
            "category_id": category_id,
            "query_image_id": query_id,
            "support_image_ids": [query_id] * len(prompt_annotations),
            "prompt_annotation_ids": [ann["id"] for ann in prompt_annotations],
            "prompt_annotation_id": prompt_annotations[0]["id"],
        }

    def _sample_training_category(self, rng: random.Random) -> tuple[int, int]:
        category_id = rng.choice(self.category_ids)
        return category_id, -1

    def _negative_prompt_count(self, positive_count: int, available_count: int, rng: random.Random) -> int:
        """Sample negatives at the configured expected ratio to positive prompts."""
        if self.negative_prompt_ratio == 0 or available_count == 0 or positive_count == 0:
            return 0
        expected = positive_count * self.negative_prompt_ratio
        count = int(expected)
        if rng.random() < expected - count:
            count += 1
        return min(count, available_count)


def collate_few_shot(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate variable prompt counts and remaining same-image query targets."""
    max_targets = max(item["target_boxes"].shape[0] for item in batch)
    max_prompts = max(item["support_boxes"].shape[0] for item in batch)
    query_eval_boxes = torch.zeros((len(batch), max_targets, 4), dtype=torch.float32)
    query_eval_counts = torch.zeros(len(batch), dtype=torch.long)
    support_boxes = torch.zeros((len(batch), max_prompts, 4), dtype=torch.float32)
    prompt_labels = torch.zeros((len(batch), max_prompts), dtype=torch.long)
    prompt_padding_mask = torch.ones((len(batch), max_prompts), dtype=torch.bool)
    prompt_target_mask = torch.zeros((len(batch), max_prompts), dtype=torch.bool)
    prompt_masks = None
    if batch[0].get("prompt_masks") is not None:
        resolution = batch[0]["prompt_masks"].shape[-1]
        prompt_masks = torch.zeros((len(batch), max_prompts, resolution, resolution), dtype=torch.bool)
    for index, item in enumerate(batch):
        target_boxes = item["target_boxes"]
        query_eval_boxes[index, : target_boxes.shape[0]] = target_boxes
        query_eval_counts[index] = target_boxes.shape[0] if item["present"] else 0
        prompt_count = item["support_boxes"].shape[0]
        support_boxes[index, :prompt_count] = item["support_boxes"]
        prompt_labels[index, :prompt_count] = item.get(
            "prompt_labels", torch.ones(prompt_count, dtype=torch.long)
        )
        prompt_padding_mask[index, :prompt_count] = False
        if item.get("prompt_target_mask") is not None:
            prompt_target_mask[index, :prompt_count] = item["prompt_target_mask"]
        if prompt_masks is not None:
            prompt_masks[index, :prompt_count] = item["prompt_masks"]
    result = {
        "support_boxes": support_boxes,
        "prompt_labels": prompt_labels,
        "prompt_padding_mask": prompt_padding_mask,
        "prompt_target_mask": prompt_target_mask,
        "is_empty_remainder": torch.tensor([bool(item.get("is_empty_remainder", False)) for item in batch]),
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
        "prompt_annotation_ids": [item["prompt_annotation_ids"] for item in batch],
    }
    if batch[0].get("target_masks") is not None:
        resolution = batch[0]["target_masks"].shape[-1]
        target_masks = torch.zeros((len(batch), max_targets, resolution, resolution), dtype=torch.bool)
        for index, item in enumerate(batch):
            masks = item["target_masks"]
            target_masks[index, : masks.shape[0]] = masks
        result["query_masks"] = target_masks
        result["prompt_masks"] = prompt_masks
    return result


def collate_support_query(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate separate-image episodes for ``PCSDetector.forward_support_query``.

    Every item has a fixed number of support shots, one query image, and a
    variable number of query instances/prompts. Padding is represented by
    ``query_target_mask`` and ``support_padding_mask``.
    """
    if not batch or batch[0].get("support_images") is None:
        raise ValueError("collate_support_query expects separate support/query episode records")
    support_count = batch[0]["support_images"].shape[0]
    if any(item["support_images"].shape[0] != support_count for item in batch):
        raise ValueError("Each episode in a batch must have the same number of support shots")
    max_targets = max(item["target_boxes"].shape[0] for item in batch)
    max_prompts = max(item["support_boxes"].shape[1] for item in batch)
    target_boxes = torch.zeros((len(batch), max_targets, 4), dtype=torch.float32)
    target_mask = torch.zeros((len(batch), max_targets), dtype=torch.bool)
    support_boxes = torch.zeros((len(batch), support_count, max_prompts, 4), dtype=torch.float32)
    support_labels = torch.zeros((len(batch), support_count, max_prompts), dtype=torch.long)
    support_padding_mask = torch.ones((len(batch), support_count, max_prompts), dtype=torch.bool)
    for index, item in enumerate(batch):
        count = item["target_boxes"].shape[0]
        target_boxes[index, :count] = item["target_boxes"]
        target_mask[index, :count] = True
        prompt_count = item["support_boxes"].shape[1]
        support_boxes[index, :, :prompt_count] = item["support_boxes"]
        support_labels[index, :, :prompt_count] = item["support_prompt_labels"]
        support_padding_mask[index, :, :prompt_count] = ~item["support_prompt_valid"]
    max_support_targets = max(
        boxes.shape[0] for item in batch for boxes in item["support_target_boxes"]
    )
    support_target_boxes = torch.zeros(
        (len(batch), support_count, max_support_targets, 4), dtype=torch.float32
    )
    support_target_mask = torch.zeros(
        (len(batch), support_count, max_support_targets), dtype=torch.bool
    )
    support_target_prompt_mask = torch.zeros(
        (len(batch), support_count, max_support_targets), dtype=torch.bool
    )
    for batch_index, item in enumerate(batch):
        for shot_index, boxes in enumerate(item["support_target_boxes"]):
            count = boxes.shape[0]
            support_target_boxes[batch_index, shot_index, :count] = boxes
            support_target_mask[batch_index, shot_index, :count] = True
            prompt_mask = item["support_target_prompt_masks"][shot_index]
            support_target_prompt_mask[batch_index, shot_index, :count] = prompt_mask
    result: dict[str, Any] = {
        "support_images": torch.stack([item["support_images"] for item in batch]),
        "support_boxes": support_boxes,
        "support_labels": support_labels,
        "support_padding_mask": support_padding_mask,
        "query_images": torch.stack([item["query_image"] for item in batch]).unsqueeze(1),
        "query_boxes": target_boxes,
        "query_target_mask": target_mask,
        "present": torch.tensor([float(item["present"]) for item in batch], dtype=torch.float32),
        "support_target_boxes": support_target_boxes,
        "support_target_mask": support_target_mask,
        "support_target_prompt_mask": support_target_prompt_mask,
        "category_ids": [item["category_id"] for item in batch],
        "query_image_ids": [item["query_image_id"] for item in batch],
        "support_image_ids": [item["support_image_ids"] for item in batch],
        "prompt_annotation_ids": [item["prompt_annotation_ids"] for item in batch],
    }
    if batch[0].get("target_masks") is not None:
        resolution = batch[0]["target_masks"].shape[-1]
        target_masks = torch.zeros((len(batch), max_targets, resolution, resolution), dtype=torch.bool)
        for index, item in enumerate(batch):
            masks = item["target_masks"]
            target_masks[index, :masks.shape[0]] = masks
        result["query_masks"] = target_masks
        support_masks = torch.zeros(
            (len(batch), support_count, max_support_targets, resolution, resolution), dtype=torch.bool
        )
        for batch_index, item in enumerate(batch):
            for shot_index, boxes in enumerate(item["support_target_boxes"]):
                count = boxes.shape[0]
                support_masks[batch_index, shot_index, :count] = item["support_target_masks"][shot_index]
        result["support_target_masks"] = support_masks
        result["support_prompt_masks"] = torch.stack([item["support_prompt_masks"] for item in batch])
    return result
