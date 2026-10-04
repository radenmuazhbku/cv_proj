"""Dataset preparation and box-proxy metrics for SAM3 DFFF experiments."""

from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_CLASSES = ("nasal bone", "NT")
SPLITS = ("set1", "set2", "internal", "external")


def read_coco(path: Path) -> dict[str, Any]:
    """Read a COCO JSON file and validate the fields required by this workflow."""
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("images", "annotations", "categories"):
        if not isinstance(data.get(key), list):
            raise ValueError(f"{path} must contain a list named {key!r}")
    return data


def _xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, width, height = box
    return [float(x), float(y), float(x + width), float(y + height)]


def prepare_manifest(
    dataset_root: Path,
    output_dir: Path,
    support_split: str = "set1",
    query_split: str = "internal",
    class_names: tuple[str, ...] = DEFAULT_CLASSES,
    shots: int = 1,
    max_queries_per_class: int | None = None,
    seed: int = 7,
    copy_images: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Make per-class support/query records and isolate query labels for evaluation.

    The inference manifest intentionally omits query annotations. Ground-truth
    bounding boxes are written to a separate evaluation JSON file and should be
    opened only after predictions have been generated.
    """
    dataset_root = dataset_root.resolve()
    output_dir = output_dir.resolve()
    if support_split not in SPLITS or query_split not in SPLITS:
        raise ValueError(f"Splits must be selected from: {', '.join(SPLITS)}")
    if shots < 1:
        raise ValueError("shots must be at least 1")
    if max_queries_per_class is not None and max_queries_per_class < 1:
        raise ValueError("max_queries_per_class must be positive when specified")

    support_data = read_coco(dataset_root / support_split / "annotations/instances.json")
    query_data = (
        support_data
        if support_split == query_split
        else read_coco(dataset_root / query_split / "annotations/instances.json")
    )
    support_categories = {c["name"].strip().casefold(): c["id"] for c in support_data["categories"]}
    query_categories = {c["name"].strip().casefold(): c["id"] for c in query_data["categories"]}
    support_images = {image["id"]: image for image in support_data["images"]}
    query_images = {image["id"]: image for image in query_data["images"]}

    support_by_class: dict[int, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for annotation in support_data["annotations"]:
        support_by_class[annotation["category_id"]].append(
            (support_images[annotation["image_id"]], annotation)
        )
    query_by_class: dict[int, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for annotation in query_data["annotations"]:
        query_by_class[annotation["category_id"]].append((query_images[annotation["image_id"]], annotation))

    rng = random.Random(seed)
    manifest: dict[str, Any] = {
        "format_version": 1,
        "dataset_root": str(dataset_root),
        "support_split": support_split,
        "query_split": query_split,
        "classes": {},
    }
    evaluation: dict[str, Any] = {"format_version": 1, "classes": {}}
    selected_supports: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for class_name in class_names:
        normalized_name = class_name.strip().casefold()
        if normalized_name not in support_categories:
            raise ValueError(f"Class {class_name!r} is not in {support_split} categories")
        candidates = list(support_by_class[support_categories[normalized_name]])
        if not candidates:
            raise ValueError(f"No support annotations for {class_name!r} in {support_split}")
        rng.shuffle(candidates)
        selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
        seen_ids: set[int] = set()
        for image, annotation in candidates:
            if image["id"] not in seen_ids:
                selected.append((image, annotation))
                seen_ids.add(image["id"])
            if len(selected) == shots:
                break
        if len(selected) < shots:
            raise ValueError(
                f"Requested {shots} support shots for {class_name!r}, but found only "
                f"{len(selected)} distinct images"
            )
        selected_supports[class_name] = selected
    selected_support_image_ids = {
        image["id"] for selected in selected_supports.values() for image, _ in selected
    }

    for class_name in class_names:
        normalized_name = class_name.strip().casefold()
        unique_supports = selected_supports[class_name]

        supports = [
            {
                "image_id": image["id"],
                "image_path": str((dataset_root / support_split / image["file_name"]).resolve()),
                "bbox_xyxy": _xywh_to_xyxy(annotation["bbox"]),
            }
            for image, annotation in unique_supports
        ]

        queries: list[dict[str, Any]] = []
        class_evaluation: dict[str, Any] = {}
        query_category_id = query_categories.get(normalized_name)
        if query_category_id is None:
            raise ValueError(f"Class {class_name!r} is not in {query_split} categories")

        if query_data["annotations"]:
            positive_by_image = {
                image["id"]: annotation for image, annotation in query_by_class[query_category_id]
            }
            query_images_for_class = [
                image
                for image in query_data["images"]
                if support_split != query_split or image["id"] not in selected_support_image_ids
            ]
            positive_images = [image for image in query_images_for_class if image["id"] in positive_by_image]
            negative_images = [image for image in query_images_for_class if image["id"] not in positive_by_image]
            rng.shuffle(positive_images)
            rng.shuffle(negative_images)
            query_candidates = positive_images + negative_images
            if max_queries_per_class is not None:
                query_candidates = query_candidates[:max_queries_per_class]
            for index, image in enumerate(query_candidates):
                query_id = f"{class_name}:{query_split}:{image['id']}:{index}"
                annotation = positive_by_image.get(image["id"])
                queries.append(
                    {
                        "query_id": query_id,
                        "image_id": image["id"],
                        "image_path": str((dataset_root / query_split / image["file_name"]).resolve()),
                        "has_ground_truth": annotation is not None,
                    }
                )
                if annotation is not None:
                    class_evaluation[query_id] = {
                        "image_id": image["id"],
                        "bbox_xyxy": _xywh_to_xyxy(annotation["bbox"]),
                    }
        else:
            # External DFFF images currently have no labels: include them for
            # qualitative prediction, but do not invent evaluation targets.
            for index, image in enumerate(query_data["images"]):
                if support_split == query_split and image["id"] in selected_support_image_ids:
                    continue
                queries.append(
                    {
                        "query_id": f"{class_name}:{query_split}:{image['id']}:{index}",
                        "image_id": image["id"],
                        "image_path": str((dataset_root / query_split / image["file_name"]).resolve()),
                        "has_ground_truth": False,
                    }
                )
            if max_queries_per_class is not None:
                queries = queries[:max_queries_per_class]

        if not queries:
            raise ValueError(f"No eligible query images for {class_name!r} in {query_split}")
        manifest["classes"][class_name] = {"supports": supports, "queries": queries}
        evaluation["classes"][class_name] = class_evaluation

    if copy_images:
        cache_dir = output_dir / "images"
        path_mapping: dict[str, str] = {}
        for class_records in manifest["classes"].values():
            records = class_records["supports"] + class_records["queries"]
            support_keys = {record["image_path"] for record in class_records["supports"]}
            for record in records:
                old_path = Path(record["image_path"])
                split = support_split if str(old_path) in support_keys else query_split
                relative = old_path.relative_to(dataset_root / split)
                new_path = cache_dir / split / relative
                new_path.parent.mkdir(parents=True, exist_ok=True)
                key = str(old_path)
                if key not in path_mapping:
                    shutil.copy2(old_path, new_path)
                    path_mapping[key] = str(new_path.resolve())
                record["image_path"] = path_mapping[key]
        manifest["dataset_root"] = str(cache_dir.resolve())

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2) + "\n", encoding="utf-8")
    return manifest, evaluation


def bbox_iou(first: list[float], second: list[float]) -> float:
    """Compute IoU for two XYXY boxes."""
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union > 0 else 0.0


def box_proxy_mask_metrics(mask: Any, box: list[float]) -> dict[str, float]:
    """Compare a predicted mask to the GT box rectangle (a proxy, not GT mask)."""
    import numpy as np

    prediction = np.asarray(mask, dtype=bool)
    height, width = prediction.shape
    x1 = max(0, min(width, int(box[0])))
    y1 = max(0, min(height, int(box[1])))
    x2 = max(x1, min(width, int(round(box[2]))))
    y2 = max(y1, min(height, int(round(box[3]))))
    target = np.zeros((height, width), dtype=bool)
    target[y1:y2, x1:x2] = True
    intersection = int((prediction & target).sum())
    predicted_count = int(prediction.sum())
    target_count = int(target.sum())
    union = int((prediction | target).sum())
    precision = intersection / predicted_count if predicted_count else 0.0
    recall = intersection / target_count if target_count else 0.0
    dice = 2 * intersection / (predicted_count + target_count) if predicted_count + target_count else 0.0
    return {
        "box_proxy_mask_iou": intersection / union if union else 0.0,
        "box_proxy_dice": dice,
        "box_coverage_recall": recall,
        "outside_box_fraction": 1.0 - precision if predicted_count else 0.0,
    }


def average_precision_at_iou(
    predictions: list[dict[str, Any]], ground_truth: dict[str, list[float]], threshold: float
) -> float:
    """101-point interpolated AP for one prediction per query image."""
    if not ground_truth:
        return 0.0
    ranked = sorted(predictions, key=lambda item: item["score"], reverse=True)
    matched: set[str] = set()
    true_positives: list[int] = []
    false_positives: list[int] = []
    for prediction in ranked:
        query_id = prediction["query_id"]
        target = ground_truth.get(query_id)
        is_match = target is not None and query_id not in matched
        is_match = bool(is_match and bbox_iou(prediction["bbox_xyxy"], target) >= threshold)
        true_positives.append(int(is_match))
        false_positives.append(int(not is_match))
        if is_match:
            matched.add(query_id)
    cumulative_tp: list[int] = []
    cumulative_fp: list[int] = []
    tp = fp = 0
    for true_positive, false_positive in zip(true_positives, false_positives, strict=True):
        tp += true_positive
        fp += false_positive
        cumulative_tp.append(tp)
        cumulative_fp.append(fp)
    recalls = [value / len(ground_truth) for value in cumulative_tp]
    precisions = [tp_value / (tp_value + fp_value) for tp_value, fp_value in zip(cumulative_tp, cumulative_fp, strict=True)]
    return sum(
        max((precision for recall, precision in zip(recalls, precisions, strict=True) if recall >= point), default=0.0)
        for point in (step / 100 for step in range(101))
    ) / 101