"""Train and evaluate same-image box-prompted COCO detection."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from pycocotools.cocoeval import COCOeval

from cv_proj.configuration import parse_args_with_python_config
from rfdetr_promptable.coco_episodes import CocoFewShotEpisodeDataset, collate_few_shot
from rfdetr_promptable.config import RFDETRBaseConfig
from rfdetr_promptable.promptable import PromptableDetector
from rfdetr_promptable.utilities.box_ops import box_cxcywh_to_xyxy


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_model(resolution: int, num_queries: int, device: str) -> PromptableDetector:
    # RF-DETR detector/query/box weights are intentionally random. The backbone
    # constructor loads DINOv2 weights because resolution 560 matches patch size 14.
    config = RFDETRBaseConfig(
        pretrain_weights=None,
        resolution=resolution,
        num_queries=num_queries,
        num_select=num_queries,
        group_detr=1,
        two_stage=False,
        device=device,
    )
    return PromptableDetector.from_config(config).to(device)


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _parse_train_sanity_samples(value: str) -> int | float:
    """Parse a batch count (integer) or a train-loader fraction (decimal/percent)."""
    try:
        if value.endswith("%"):
            fraction = float(value[:-1]) / 100.0
            if not 0.0 < fraction <= 1.0:
                raise ValueError
            return fraction
        if any(marker in value.lower() for marker in (".", "e")):
            fraction = float(value)
            if not 0.0 < fraction <= 1.0:
                raise ValueError
            return fraction
        count = int(value)
        if count <= 0:
            raise ValueError
        return count
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "use a positive integer batch count (e.g. 2), a fraction in (0, 1] (e.g. 0.02), or a percent (e.g. 2%)"
        ) from exc


def _parse_schedule_steps(value: str) -> int | float:
    """Parse warmup as an integer step count or a fractional share of training."""
    try:
        if value.endswith("%"):
            fraction = float(value[:-1]) / 100.0
            if not 0.0 <= fraction < 1.0:
                raise ValueError
            return fraction
        if any(marker in value.lower() for marker in (".", "e")):
            fraction = float(value)
            if not 0.0 <= fraction < 1.0:
                raise ValueError
            return fraction
        count = int(value)
        if count < 0:
            raise ValueError
        return count
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "use a non-negative integer step count or a fraction in [0, 1) (e.g. 500, 0.05, or 5%)"
        ) from exc


def _resolve_warmup_steps(value: int | float, total_steps: int) -> int:
    """Float values below one represent a fraction; integers are absolute steps."""
    if isinstance(value, float):
        return min(total_steps - 1, int(round(value * total_steps)))
    return min(total_steps - 1, value)


def _scheduled_lr(base_lr: float, step: int, total_steps: int, warmup_steps: int,
                  warmup_start_factor: float, min_lr_ratio: float, schedule: str) -> float:
    """Learning rate for the optimizer update at zero-based ``step``."""
    if warmup_steps > 0 and step < warmup_steps:
        progress = step / warmup_steps
        return base_lr * (warmup_start_factor + (1.0 - warmup_start_factor) * progress)
    if schedule == "constant":
        return base_lr
    decay_steps = max(1, total_steps - warmup_steps - 1)
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def _build_fixed_diagnostic_loader(
    dataset: CocoFewShotEpisodeDataset,
    batch_size: int,
    sample_budget: int | float,
    seed: int,
) -> DataLoader:
    """Build one deterministic, reusable diagnostic subset with a positive episode."""
    total_batches = math.ceil(len(dataset) / batch_size)
    batch_count = (
        min(int(sample_budget), total_batches)
        if isinstance(sample_budget, int)
        else min(max(1, math.ceil(total_batches * sample_budget)), total_batches)
    )
    sample_count = min(batch_count * batch_size, len(dataset))
    rng = random.Random(seed + 83_017)
    indices = rng.sample(range(len(dataset)), sample_count)

    # A qualitative localization check is useless if a small random subset happens
    # to contain only negative episodes. Replace its first item with a fixed positive.
    if not any(dataset[index]["present"] for index in indices):
        candidates = [index for index in rng.sample(range(len(dataset)), len(dataset)) if index not in indices]
        for candidate in candidates:
            if dataset[candidate]["present"]:
                indices[0] = candidate
                break
    generator = torch.Generator().manual_seed(seed + 83_018)
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_few_shot,
        num_workers=0,
        pin_memory=True,
        generator=generator,
    )


def run_train_sanity_eval(
    model: PromptableDetector,
    loader: DataLoader,
    device: torch.device,
    figure_dir: Path,
    step: int,
    plot_mode: str = "overlay",
    extra_predictions: int = 0,
    evaluate_prompt_box_iou: bool = True,
) -> dict[str, object]:
    """Evaluate fixed training episodes and save GT/prediction diagnostic panels.

    When enabled, prompt-box IoU is reported separately from IoU against the
    remaining query targets.
    """
    subset = loader.dataset
    dataset = subset.dataset if isinstance(subset, Subset) else subset
    category_names = {
        item["id"]: item["name"] for item in dataset.coco.loadCats(dataset.coco.getCatIds())
    }
    was_training = model.training
    model.eval()
    records: list[dict[str, object]] = []
    with torch.inference_mode():
        for batch_index, raw_batch in enumerate(loader):
            batch = _move_batch(raw_batch, device)
            outputs = model.forward_same_image(batch["query_images"], batch["support_boxes"])
            scores = outputs["pred_logits"].sigmoid()[..., 0]
            predicted_boxes = outputs["pred_boxes"]
            for item_index, category_id in enumerate(batch["category_ids"]):
                query_id = int(batch["query_image_ids"][item_index])
                score, query_index = scores[item_index].max(dim=0)
                prediction = predicted_boxes[item_index, query_index]
                target_count = int(batch["query_eval_counts"][item_index])
                targets = batch["query_eval_boxes"][item_index, :target_count]
                best_iou = None
                if target_count:
                    best_iou = float(torch.stack([_box_iou_cxcywh(prediction, target) for target in targets]).max())
                prompt_iou = None
                if evaluate_prompt_box_iou and target_count:
                    prompt_iou = float(
                        _box_iou_cxcywh(prediction, batch["support_boxes"][item_index])
                    )

                metadata = dataset.coco.loadImgs([query_id])[0]
                image_path = dataset.image_dir / metadata["file_name"]
                image = Image.open(image_path).convert("RGB")
                width, height = image.size
                prompt_target_count = int(target_count > 0)
                red_prediction_count = target_count + prompt_target_count
                prediction_count = min(
                    scores.shape[1], red_prediction_count + extra_predictions
                )
                plot_scores, plot_indices = scores[item_index].topk(prediction_count)
                plot_predictions = [
                    _cxcywh_pixels(predicted_boxes[item_index, query_index], width, height)
                    for query_index in plot_indices
                ]
                target_xyxy = [_cxcywh_pixels(target, width, height) for target in targets]
                support_examples = _load_support_visuals(dataset, batch, item_index, category_names[category_id])
                safe_class = category_names[category_id].replace(" ", "_")
                sample_root = figure_dir / f"batch_{batch_index:03d}"
                sample_root.mkdir(parents=True, exist_ok=True)
                summary = f"{category_names[category_id]} | {'positive' if target_count else 'negative'}"
                if best_iou is not None:
                    summary += f" | best IoU={best_iou:.3f}"
                _save_episode_figures(
                    sample_root,
                    category_names[category_id],
                    metadata["file_name"],
                    image,
                    plot_predictions,
                    plot_scores.detach().cpu().tolist(),
                    target_xyxy,
                    support_examples,
                    plot_mode=plot_mode,
                    red_prediction_count=min(scores.shape[1], red_prediction_count),
                )
                sample_view = "overlay" if plot_mode == "overlay" else "support_query"
                sample_path = sample_root / sample_view / safe_class / f"{Path(metadata['file_name']).stem}.png"
                records.append(
                    {
                        "step": step,
                        "query_image_id": query_id,
                        "category_id": category_id,
                        "category_name": category_names[category_id],
                        "present": bool(target_count),
                        "score": float(score),
                        "best_iou": best_iou,
                        "prompt_iou": prompt_iou,
                        "prediction_cxcywh": prediction.detach().cpu().tolist(),
                        "plotted_prediction_count": prediction_count,
                        "plotted_red_prediction_count": min(red_prediction_count, prediction_count),
                        "plotted_extra_prediction_count": max(0, prediction_count - red_prediction_count),
                        "prompt_prediction_slots_included": bool(prompt_target_count),
                        "target_count": target_count,
                        "support_image_ids": [support_id for support_id, _, _, _ in support_examples],
                        "figure": str(sample_path),
                    }
                )
    model.train(was_training)

    positive_records = [record for record in records if record["present"]]
    negative_records = [record for record in records if not record["present"]]
    prompt_iou_records = [record for record in records if record["prompt_iou"] is not None]
    return {
        "step": step,
        "sample_count": len(records),
        "positive_count": len(positive_records),
        "negative_count": len(negative_records),
        "mean_positive_iou": (
            sum(float(record["best_iou"]) for record in positive_records) / len(positive_records)
            if positive_records else None
        ),
        "mean_prompt_iou": (
            sum(float(record["prompt_iou"]) for record in prompt_iou_records) / len(prompt_iou_records)
            if prompt_iou_records else None
        ),
        "prompt_box_iou_enabled": evaluate_prompt_box_iou,
        "mean_positive_score": (
            sum(float(record["score"]) for record in positive_records) / len(positive_records)
            if positive_records else None
        ),
        "mean_negative_score": (
            sum(float(record["score"]) for record in negative_records) / len(negative_records)
            if negative_records else None
        ),
        "samples": records,
    }


def save_rfdetr_training_plots(
    train_records: list[dict[str, float | int]],
    metrics: dict[str, object],
    metrics_csv: Path,
    output_dir: Path,
) -> None:
    """Adapt few-shot logs to RF-DETR's standard plots and refresh PNGs after eval."""
    import csv

    from rfdetr.visualize.training import plot_loss_metrics, plot_map_metrics, plot_metrics

    rows: list[dict[str, float | int]] = []
    for record in train_records:
        step = int(record["step"])
        row: dict[str, float | int] = {"epoch": step, "step": step}
        for source, target in (
            ("loss", "train/loss"),
            ("objectness", "train/loss_objectness"),
            ("bbox", "train/loss_bbox"),
            ("giou", "train/loss_giou"),
        ):
            row[target] = float(record[source])
        rows.append(row)

    validation_steps = sorted(
        (int(key.removeprefix("validation_step_")), value)
        for key, value in metrics.items()
        if key.startswith("validation_step_") and isinstance(value, dict)
    )
    for step, result in validation_steps:
        selection = result.get("coco_selection", {})
        episodic = result.get("episodic", {})
        if not isinstance(selection, dict) or not isinstance(episodic, dict):
            continue
        row = {"epoch": step, "step": step}
        latest_train = next(
            (record for record in reversed(train_records) if int(record["step"]) <= step),
            None,
        )
        if latest_train is not None:
            for source, target in (
                ("loss", "train/loss"),
                ("objectness", "train/loss_objectness"),
                ("bbox", "train/loss_bbox"),
                ("giou", "train/loss_giou"),
            ):
                row[target] = float(latest_train[source])
        # RF-DETR's visualizer expects its CSVLogger metric names.
        for source, target in (
            ("mAP50", "val/mAP_50"),
            ("mAP", "val/mAP_50_95"),
            ("mAP75", "val/mAP_75"),
            ("mAR100", "val/mAR"),
        ):
            value = selection.get(source)
            if isinstance(value, (int, float)):
                row[target] = float(value)
        macro_iou = episodic.get("macro_positive_iou")
        if isinstance(macro_iou, (int, float)):
            row["val/macro_positive_iou"] = float(macro_iou)
        rows.append(row)

    if not rows:
        return
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(column for row in rows for column in row))
    with metrics_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    figures = [
        plot_metrics(metrics_csv, output_dir / "training_metrics.png"),
        plot_loss_metrics(metrics_csv, output_dir / "loss_metrics.png", loss_log_scale=True),
    ]
    map_columns = ("val/mAP_50", "val/mAP_50_95", "val/mAP_75")
    has_map_values = any(
        isinstance(row.get(column), (int, float))
        for row in rows
        for column in map_columns
    )
    if has_map_values:
        figures.append(plot_map_metrics(metrics_csv, output_dir / "map_metrics.png"))
    else:
        print("skipping RF-DETR mAP plot: this evaluation run has no COCO mAP metrics")
    import matplotlib.pyplot as plt

    for figure in figures:
        plt.close(figure)

    eval_steps = [step for step, _ in validation_steps]
    if eval_steps:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        val_iou = [
            metrics[f"validation_step_{step}"]["episodic"].get("macro_positive_iou")
            for step in eval_steps
        ]
        sanity_iou = [
            metrics.get(f"train_sanity_step_{step}", {}).get("mean_positive_iou")
            for step in eval_steps
        ]
        axes[0].plot(eval_steps, val_iou, marker="o", label="validation macro positive IoU")
        if any(value is not None for value in sanity_iou):
            axes[0].plot(eval_steps, sanity_iou, marker="o", label="train sanity positive IoU")
        axes[0].set_ylabel("IoU")
        axes[0].set_ylim(0, 1)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        positive_scores = [
            metrics.get(f"train_sanity_step_{step}", {}).get("mean_positive_score")
            for step in eval_steps
        ]
        negative_scores = [
            metrics.get(f"train_sanity_step_{step}", {}).get("mean_negative_score")
            for step in eval_steps
        ]
        if any(value is not None for value in positive_scores):
            axes[1].plot(eval_steps, positive_scores, marker="o", label="train sanity positive score")
        if any(value is not None for value in negative_scores):
            axes[1].plot(eval_steps, negative_scores, marker="o", label="train sanity negative score")
        axes[1].set_xlabel("Training step")
        axes[1].set_ylabel("Objectness score")
        axes[1].set_ylim(0, 1)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        fig.suptitle("Promptable Few-Shot Diagnostics")
        fig.tight_layout()
        fig.savefig(output_dir / "promptable_diagnostics.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def run_episode_eval(
    model: PromptableDetector,
    loader: DataLoader,
    device: torch.device,
    figure_dir: Path | None = None,
    max_figures_per_class: int = 10,
    plot_mode: str = "overlay",
    extra_predictions: int = 0,
    evaluate_prompt_box_iou: bool = True,
) -> dict[str, object]:
    model.eval()
    by_category: dict[int, dict[str, object]] = {}
    figure_counts: dict[int, int] = {}
    dataset = loader.dataset
    category_names = {item["id"]: item["name"] for item in dataset.coco.loadCats(dataset.coco.getCatIds())}
    with torch.inference_mode():
        for raw_batch in tqdm(loader, desc="eval", leave=False):
            batch = _move_batch(raw_batch, device)
            outputs = model.forward_same_image(batch["query_images"], batch["support_boxes"])
            scores = outputs["pred_logits"].sigmoid()[..., 0]
            gt_present = batch["present"].flatten() > 0.5
            for index, category_id in enumerate(batch["category_ids"]):
                bucket = by_category.setdefault(category_id, {"examples": 0, "positive": 0, "negative": 0, "positive_score": 0.0, "negative_score": 0.0, "matched_iou": 0.0, "prompt_iou": 0.0, "prompt_iou_count": 0, "ground_truth_objects": 0, "presence_true_positive_at_0_5": 0, "presence_false_positive_at_0_5": 0, "recall50": 0, "recall75": 0})
                bucket["examples"] = int(bucket["examples"]) + 1
                score = float(scores[index].max())
                if bool(gt_present[index]):
                    pred_box = outputs["pred_boxes"][index, scores[index].argmax()]
                    target_count = int(batch["query_eval_counts"][index])
                    gt_boxes = batch["query_eval_boxes"][index, :target_count]
                    ious = torch.stack([_box_iou_cxcywh(pred_box, gt_box) for gt_box in gt_boxes])
                    best_target = int(ious.argmax())
                    gt_box = gt_boxes[best_target]
                    iou = float(ious[best_target])
                    if evaluate_prompt_box_iou:
                        prompt_iou = _box_iou_cxcywh(
                            pred_box, batch["support_boxes"][index]
                        )
                        bucket["prompt_iou"] = float(bucket["prompt_iou"]) + float(prompt_iou)
                        bucket["prompt_iou_count"] = int(bucket["prompt_iou_count"]) + 1
                    bucket["positive"] = int(bucket["positive"]) + 1
                    bucket["positive_score"] = float(bucket["positive_score"]) + score
                    bucket["matched_iou"] = float(bucket["matched_iou"]) + iou
                    bucket["ground_truth_objects"] = int(bucket["ground_truth_objects"]) + target_count
                    bucket["presence_true_positive_at_0_5"] = int(bucket["presence_true_positive_at_0_5"]) + int(score >= 0.5)
                    bucket["recall50"] = int(bucket["recall50"]) + int(iou >= 0.5)
                    bucket["recall75"] = int(bucket["recall75"]) + int(iou >= 0.75)
                    if figure_dir is not None and figure_counts.get(category_id, 0) < max_figures_per_class:
                        metadata = dataset.coco.loadImgs([int(batch["query_image_ids"][index])])[0]
                        image = Image.open(dataset.image_dir / metadata["file_name"]).convert("RGB")
                        width, height = image.size
                        gt_boxes_xyxy = [
                            _cxcywh_pixels(box, width, height) for box in gt_boxes
                        ]
                        prompt_target_count = int(target_count > 0)
                        red_prediction_count = target_count + prompt_target_count
                        prediction_count = min(
                            scores.shape[1], red_prediction_count + extra_predictions
                        )
                        plot_scores, plot_indices = scores[index].topk(prediction_count)
                        plot_predictions = [
                            _cxcywh_pixels(outputs["pred_boxes"][index, query_index], width, height)
                            for query_index in plot_indices
                        ]
                        support_examples = _load_support_visuals(
                            dataset, batch, index, category_names[category_id]
                        )
                        _save_episode_figures(
                            figure_dir, category_names[category_id], metadata["file_name"], image,
                            plot_predictions, plot_scores.detach().cpu().tolist(),
                            gt_boxes_xyxy, support_examples,
                            plot_mode=plot_mode,
                            red_prediction_count=min(scores.shape[1], red_prediction_count),
                        )
                        figure_counts[category_id] = figure_counts.get(category_id, 0) + 1
                else:
                    bucket["negative"] = int(bucket["negative"]) + 1
                    bucket["negative_score"] = float(bucket["negative_score"]) + score
                    bucket["presence_false_positive_at_0_5"] = int(bucket["presence_false_positive_at_0_5"]) + int(score >= 0.5)
    model.train()
    for bucket in by_category.values():
        positives = int(bucket["positive"])
        negatives = int(bucket["negative"])
        bucket["mean_positive_score"] = float(bucket["positive_score"]) / max(1, positives)
        bucket["mean_negative_score"] = float(bucket["negative_score"]) / max(1, negatives)
        bucket["mean_matched_iou"] = float(bucket["matched_iou"]) / max(1, positives)
        bucket["mean_prompt_iou"] = (
            float(bucket["prompt_iou"]) / int(bucket["prompt_iou_count"])
            if evaluate_prompt_box_iou and int(bucket["prompt_iou_count"]) else None
        )
        bucket["recall50"] = int(bucket["recall50"]) / max(1, positives)
        bucket["recall75"] = int(bucket["recall75"]) / max(1, positives)
        bucket["presence_recall_at_0_5"] = int(bucket["presence_true_positive_at_0_5"]) / max(1, positives)
        bucket["presence_false_positive_rate_at_0_5"] = int(bucket["presence_false_positive_at_0_5"]) / max(1, negatives)
    all_iou = [float(values["mean_matched_iou"]) for values in by_category.values() if int(values["positive"]) > 0]
    all_prompt_iou = [
        float(values["mean_prompt_iou"])
        for values in by_category.values()
        if values["mean_prompt_iou"] is not None
    ]
    return {
        "categories": by_category,
        "macro_positive_iou": sum(all_iou) / max(1, len(all_iou)),
        "macro_prompt_iou": (
            sum(all_prompt_iou) / len(all_prompt_iou) if all_prompt_iou else None
        ),
        "prompt_box_iou_enabled": evaluate_prompt_box_iou,
    }


def run_coco_map_eval(
    model: PromptableDetector,
    dataset: CocoFewShotEpisodeDataset,
    device: torch.device,
    query_batch_size: int,
    seed: int,
    max_query_images: int | None = None,
    include_prompt_box_as_target: bool = True,
) -> dict[str, object]:
    """Compute COCO AP/AR over same-image prompted episodes.

    Each episode uses its own query image and one annotated instance from that
    image as the prompt. By default the prompt annotation is included alongside
    the other same-class instances in the COCO ground truth.
    """
    if query_batch_size <= 0:
        raise ValueError("query_batch_size must be positive")
    if max_query_images is not None and max_query_images <= 0:
        raise ValueError("max_query_images must be positive when specified")
    from pycocotools.coco import COCO
    from torch.utils.data import Subset

    image_ids = sorted({image_id for _, image_id, _ in dataset.eval_records})
    if max_query_images is not None and len(image_ids) > max_query_images:
        image_ids = sorted(
            image_ids,
            key=lambda image_id: ((seed * 1_000_003 + image_id) % 2**32, image_id),
        )[:max_query_images]
        image_ids = sorted(image_ids)
    selected_image_ids = set(image_ids)
    selected_indices = [
        index for index, (_, image_id, _) in enumerate(dataset.eval_records)
        if image_id in selected_image_ids
    ]
    if not selected_indices:
        return {
            "mAP": None, "mAP50": None, "mAP75": None, "mAR100": None,
            "query_images": 0, "query_image_category_pairs": 0,
            "prompt_box_included_as_target": include_prompt_box_as_target,
            "evaluation_protocol": "same-image prompt; promptable validation episodes only",
        }

    loader = DataLoader(
        Subset(dataset, selected_indices),
        batch_size=query_batch_size,
        shuffle=False,
        collate_fn=collate_few_shot,
        num_workers=0,
        pin_memory=True,
    )
    source_coco = dataset.coco
    image_metadata = {
        int(image_id): dict(source_coco.loadImgs([int(image_id)])[0])
        for image_id in image_ids
    }
    categories = source_coco.loadCats(source_coco.getCatIds())
    gt_annotations: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    used_categories: set[int] = set()
    category_episode_counts: dict[int, int] = {}
    next_annotation_id = 1
    evaluation_started = time.perf_counter()
    model_was_training = model.training
    model.eval()
    with torch.inference_mode():
        for batch in tqdm(loader, desc="same-image COCO eval", leave=False):
            batch = _move_batch(batch, device)
            outputs = model.forward_same_image(batch["query_images"], batch["support_boxes"])
            scores = outputs["pred_logits"].sigmoid()[..., 0]
            predicted_boxes = outputs["pred_boxes"]
            for index, category_id in enumerate(batch["category_ids"]):
                category_id = int(category_id)
                image_id = int(batch["query_image_ids"][index])
                metadata = image_metadata[image_id]
                width, height = int(metadata["width"]), int(metadata["height"])
                used_categories.add(category_id)
                category_episode_counts[category_id] = category_episode_counts.get(category_id, 0) + 1
                target_count = int(batch["query_eval_counts"][index])
                gt_boxes = batch["query_eval_boxes"][index, :target_count]
                if include_prompt_box_as_target:
                    gt_boxes = torch.cat([gt_boxes, batch["support_boxes"][index:index + 1]], dim=0)
                for box in gt_boxes:
                    cx, cy, box_width, box_height = box.detach().float().cpu().tolist()
                    x = max(0.0, (cx - box_width / 2) * width)
                    y = max(0.0, (cy - box_height / 2) * height)
                    right = min(float(width), (cx + box_width / 2) * width)
                    bottom = min(float(height), (cy + box_height / 2) * height)
                    if right <= x or bottom <= y:
                        continue
                    gt_annotations.append({
                        "id": next_annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [x, y, right - x, bottom - y],
                        "area": (right - x) * (bottom - y),
                        "iscrowd": 0,
                    })
                    next_annotation_id += 1

                boxes_xyxy = box_cxcywh_to_xyxy(predicted_boxes[index]).clamp(0, 1)
                boxes_xyxy[:, 0::2] *= width
                boxes_xyxy[:, 1::2] *= height
                image_scores = scores[index]
                keep = torch.isfinite(image_scores) & torch.isfinite(boxes_xyxy).all(dim=1)
                kept_scores = image_scores[keep]
                kept_boxes = boxes_xyxy[keep]
                if kept_scores.numel() > 100:
                    kept_scores, top_indices = kept_scores.topk(100)
                    kept_boxes = kept_boxes[top_indices]
                for box, score in zip(kept_boxes.cpu().tolist(), kept_scores.cpu().tolist(), strict=True):
                    x0, y0, x1, y1 = box
                    if x1 <= x0 or y1 <= y0:
                        continue
                    predictions.append({
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [x0, y0, x1 - x0, y1 - y0],
                        "score": float(score),
                    })

    if model_was_training:
        model.train()
    elapsed_seconds = time.perf_counter() - evaluation_started
    if not gt_annotations or not predictions:
        return {
            "mAP": None, "mAP50": None, "mAP75": None, "mAR100": None,
            "query_images": len(image_ids),
            "query_image_category_pairs": len(selected_indices),
            "prompt_box_included_as_target": include_prompt_box_as_target,
            "elapsed_seconds": elapsed_seconds,
            "evaluation_protocol": "same-image prompt; promptable validation episodes only",
        }

    coco_gt = COCO()
    coco_gt.dataset = {
        "info": {},
        "licenses": [],
        "images": [image_metadata[image_id] for image_id in image_ids],
        "categories": [category for category in categories if int(category["id"]) in used_categories],
        "annotations": gt_annotations,
    }
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(predictions)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = image_ids
    evaluator.params.catIds = sorted(used_categories)
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    evaluator.accumulate()
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.summarize()

    precision = evaluator.eval["precision"]
    recall = evaluator.eval["recall"]
    iou_thresholds = evaluator.params.iouThrs

    def mean_valid(values: np.ndarray) -> float:
        valid = values[values > -1]
        return float(valid.mean()) if valid.size else -1.0

    category_metrics: dict[int, dict[str, object]] = {}
    for category_index, category_id in enumerate(sorted(used_categories)):
        category_precision = precision[:, :, category_index, :, :]
        category_recall = recall[:, category_index, :, :]
        iou50_index = int(np.argmin(np.abs(iou_thresholds - 0.50)))
        iou75_index = int(np.argmin(np.abs(iou_thresholds - 0.75)))
        category_gt_count = sum(
            int(annotation["category_id"]) == category_id for annotation in gt_annotations
        )
        category_metrics[category_id] = {
            "name": next(
                str(category["name"])
                for category in categories
                if int(category["id"]) == category_id
            ),
            "episodes": category_episode_counts[category_id],
            "ground_truth_objects": category_gt_count,
            "AP": mean_valid(category_precision[:, :, 0, -1]),
            "AP50": mean_valid(category_precision[iou50_index, :, 0, -1]),
            "AP75": mean_valid(category_precision[iou75_index, :, 0, -1]),
            "AP_small": mean_valid(category_precision[:, :, 1, -1]),
            "AP_medium": mean_valid(category_precision[:, :, 2, -1]),
            "AP_large": mean_valid(category_precision[:, :, 3, -1]),
            "AR1": mean_valid(category_recall[:, 0, 0]),
            "AR10": mean_valid(category_recall[:, 0, 1]),
            "AR100": mean_valid(category_recall[:, 0, 2]),
        }

    return {
        "categories": category_metrics,
        "mAP": float(evaluator.stats[0]),
        "mAP50": float(evaluator.stats[1]),
        "mAP75": float(evaluator.stats[2]),
        "mAP_small": float(evaluator.stats[3]),
        "mAP_medium": float(evaluator.stats[4]),
        "mAP_large": float(evaluator.stats[5]),
        "mAR1": float(evaluator.stats[6]),
        "mAR10": float(evaluator.stats[7]),
        "mAR100": float(evaluator.stats[8]),
        "metric_scale": "0_to_1",
        "query_images": len(image_ids),
        "query_image_category_pairs": len(selected_indices),
        "prompt_box_included_as_target": include_prompt_box_as_target,
        "elapsed_seconds": elapsed_seconds,
        "query_pairs_per_second": (
            len(selected_indices) / elapsed_seconds if elapsed_seconds > 0 else 0.0
        ),
        "evaluation_protocol": "COCOeval bbox on same-image prompted episodes; image categories without valid prompt episodes are omitted",
    }


def _cxcywh_pixels(box: torch.Tensor, width: int, height: int) -> list[float]:
    cx, cy, box_width, box_height = box.detach().float().cpu().tolist()
    return [(cx - box_width / 2) * width, (cy - box_height / 2) * height,
            (cx + box_width / 2) * width, (cy + box_height / 2) * height]


def _load_support_visuals(
    dataset: CocoFewShotEpisodeDataset,
    batch: dict,
    query_index: int,
    class_name: str,
) -> list[tuple[int, Image.Image, list[float], str]]:
    """Load the exact support examples and labeled prompt boxes used for one query."""
    support_ids = [int(image_id) for image_id in batch["support_image_ids"][query_index]]
    if support_ids != [int(batch["query_image_ids"][query_index])]:
        raise ValueError("same-image episodes must use the query image as their support image")
    support_coco = dataset.coco
    support_dir = dataset.image_dir
    examples = []
    for shot_number, support_id in enumerate(support_ids, start=1):
        metadata = support_coco.loadImgs([support_id])[0]
        support_image = Image.open(support_dir / metadata["file_name"]).convert("RGB")
        support_width, support_height = support_image.size
        support_box = _cxcywh_pixels(
            batch["support_boxes"][query_index], support_width, support_height
        )
        label = f"Prompt {shot_number} | {class_name} | image {support_id}"
        examples.append((support_id, support_image, support_box, label))
    return examples


def _save_query_support_figure(
    output_path: Path,
    summary: str,
    query_image: Image.Image,
    target_boxes: list[list[float]],
    prediction_boxes: list[list[float]],
    prediction_scores: list[float],
    support_examples: list[tuple[int, Image.Image, list[float], str]],
    plot_mode: str = "overlay",
    red_prediction_count: int | None = None,
) -> None:
    """Save query GT/prediction alongside the exact support images and prompt boxes."""
    if plot_mode not in {"split", "overlay", "compact"}:
        raise ValueError(f"Unknown inference plot mode: {plot_mode!r}")
    if red_prediction_count is None:
        red_prediction_count = len(target_boxes)
    if plot_mode == "overlay":
        figure, (original_axis, axis) = plt.subplots(1, 2, figsize=(16, 7))
        original_axis.imshow(query_image)
        original_axis.set_title("Original query image")
        original_axis.axis("off")
        axis.imshow(query_image)
        axis.set_title(f"Prompt, all query GT, {len(prediction_boxes)} predictions")
        axis.axis("off")
        prompt_legend_labels: set[str] = set()
        for _, _, prompt_box, prompt_name in support_examples:
            x0, y0, x1, y1 = prompt_box
            is_negative_prompt = "negative" in prompt_name.lower()
            prompt_legend_label = "negative prompt" if is_negative_prompt else "positive prompt"
            legend_label = prompt_legend_label if prompt_legend_label not in prompt_legend_labels else None
            prompt_legend_labels.add(prompt_legend_label)
            axis.add_patch(patches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                fill=False, edgecolor="#ff9f1c" if is_negative_prompt else "cyan", linewidth=2,
                label=legend_label,
            ))
        for index, (x0, y0, x1, y1) in enumerate(target_boxes):
            axis.add_patch(patches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                fill=False, edgecolor="lime", linewidth=2,
                label="query GT" if index == 0 else None,
            ))
        for index, (box, score) in enumerate(zip(prediction_boxes, prediction_scores, strict=True)):
            x0, y0, x1, y1 = box
            axis.add_patch(patches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                fill=False,
                edgecolor="red" if index < red_prediction_count else "#b87333",
                linewidth=2 if index < red_prediction_count else 0.8,
                alpha=1.0 if index < red_prediction_count else 0.3,
                label=("top predictions" if index == 0 else None) if index < red_prediction_count else (
                    "extra predictions" if index == red_prediction_count else None
                ),
            ))
        axis.legend(loc="upper right")
        figure.suptitle(summary)
        figure.tight_layout()
        figure.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close(figure)
        return

    if plot_mode == "compact":
        support_count = len(support_examples)
        figure = plt.figure(figsize=(13, 6.2 if support_count <= 2 else 7.2))
        outer_grid = figure.add_gridspec(
            3, 1, height_ratios=(2.0, 0.24, 1.0), hspace=0.22
        )
        query_grid = outer_grid[0].subgridspec(1, 2, wspace=0.04)
        original_axis, overlay_axis = (
            figure.add_subplot(query_grid[0, index]) for index in range(2)
        )
        original_axis.imshow(query_image)
        original_axis.set_title("Query image")
        original_axis.axis("off")
        overlay_axis.imshow(query_image)
        overlay_axis.set_title("Query GT + top predictions")
        overlay_axis.axis("off")
        for x0, y0, x1, y1 in target_boxes:
            overlay_axis.add_patch(patches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                fill=False, edgecolor="lime", linewidth=1.8,
            ))
        for index, (box, _) in enumerate(zip(prediction_boxes, prediction_scores, strict=True)):
            x0, y0, x1, y1 = box
            overlay_axis.add_patch(patches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                fill=False,
                edgecolor="red" if index < red_prediction_count else "#b87333",
                linewidth=1.8 if index < red_prediction_count else 0.8,
                alpha=1.0 if index < red_prediction_count else 0.3,
            ))
        legend_axis = figure.add_subplot(outer_grid[1])
        legend_axis.axis("off")
        legend_handles = [
            patches.Patch(fill=False, edgecolor="lime", linewidth=1.8, label="query GT"),
            patches.Patch(fill=False, edgecolor="red", linewidth=1.8, label="top predictions"),
            patches.Patch(fill=False, edgecolor="cyan", linewidth=1.8, label="support prompt"),
        ]
        if any("negative" in label.lower() for _, _, _, label in support_examples):
            legend_handles[-1] = patches.Patch(
                fill=False, edgecolor="cyan", linewidth=1.8, label="positive prompt"
            )
            legend_handles.append(patches.Patch(
                fill=False, edgecolor="#ff9f1c", linewidth=1.8, label="negative prompt"
            ))
        legend_axis.legend(
            handles=legend_handles, loc="center", ncol=len(legend_handles), frameon=False,
            fontsize=8,
        )
        if support_examples:
            support_grid = outer_grid[2].subgridspec(1, support_count, wspace=0.06)
            for index, (_, support_image, support_box, label) in enumerate(support_examples):
                axis = figure.add_subplot(support_grid[0, index])
                axis.imshow(support_image)
                axis.set_title(label, fontsize=8, pad=1)
                axis.axis("off")
                x0, y0, x1, y1 = support_box
                prompt_color = "#ff9f1c" if "negative" in label.lower() else "cyan"
                axis.add_patch(patches.Rectangle(
                    (x0, y0), x1 - x0, y1 - y0,
                    fill=False, edgecolor=prompt_color, linewidth=1.8,
                ))
        else:
            axis = figure.add_subplot(outer_grid[2])
            axis.axis("off")
        figure.suptitle(summary, y=0.99)
        figure.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close(figure)
        return

    support_count = len(support_examples)
    figure = plt.figure(figsize=(20, 9 if support_count <= 4 else 10))
    outer_grid = figure.add_gridspec(2, 1, height_ratios=(2.0, 1.25), hspace=0.18)
    query_grid = outer_grid[0].subgridspec(1, 4, wspace=0.04)
    query_axes = [figure.add_subplot(query_grid[0, index]) for index in range(4)]
    top_score = prediction_scores[0] if prediction_scores else None
    titles = (
        "Query image", "Query ground truth",
        f"Top {len(prediction_boxes)} predictions" + (f" (best {top_score:.3f})" if top_score is not None else ""),
        "GT + predictions",
    )
    for axis, title in zip(query_axes, titles, strict=True):
        axis.imshow(query_image)
        axis.set_title(title)
        axis.axis("off")
    for x0, y0, x1, y1 in target_boxes:
        query_axes[1].add_patch(
            patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="lime", linewidth=2)
        )
        query_axes[3].add_patch(
            patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="lime", linewidth=2, label="query GT")
        )
    for index, (box, score) in enumerate(zip(prediction_boxes, prediction_scores, strict=True)):
        x0, y0, x1, y1 = box
        is_target_count_prediction = index < red_prediction_count
        color = "red" if is_target_count_prediction else "#b87333"
        linewidth = 2 if is_target_count_prediction else 0.8
        alpha = 1.0 if is_target_count_prediction else 0.3
        query_axes[2].add_patch(patches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            fill=False, edgecolor=color, linewidth=linewidth, alpha=alpha,
            label=("top predictions" if index == 0 else None) if is_target_count_prediction else (
                "extra predictions" if index == red_prediction_count else None
            ),
        ))
        query_axes[3].add_patch(patches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            fill=False, edgecolor=color, linewidth=linewidth, alpha=alpha,
            label=("top predictions" if index == 0 else None) if is_target_count_prediction else (
                "extra predictions" if index == red_prediction_count else None
            ),
        ))
    query_axes[2].legend(loc="upper right")
    query_axes[3].legend(loc="upper right")

    if support_examples:
        support_grid = outer_grid[1].subgridspec(1, support_count, wspace=0.08)
        for index, (_, support_image, support_box, label) in enumerate(support_examples):
            axis = figure.add_subplot(support_grid[0, index])
            axis.imshow(support_image)
            axis.set_title(label, fontsize=9)
            axis.axis("off")
            x0, y0, x1, y1 = support_box
            prompt_color = "#ff9f1c" if "negative" in label.lower() else "cyan"
            axis.add_patch(
                patches.Rectangle(
                    (x0, y0), x1 - x0, y1 - y0,
                    fill=False, edgecolor=prompt_color, linewidth=2, label=label,
                )
            )
    else:
        axis = figure.add_subplot(outer_grid[1])
        axis.text(0.5, 0.5, "No support examples", ha="center", va="center")
        axis.axis("off")

    figure.suptitle(summary)
    figure.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(figure)


def _save_episode_figures(
    figure_dir: Path, class_name: str, filename: str, image: Image.Image,
    prediction_boxes: list[list[float]], prediction_scores: list[float],
    target_boxes: list[list[float]],
    support_examples: list[tuple[int, Image.Image, list[float], str]],
    plot_mode: str = "overlay",
    red_prediction_count: int | None = None,
) -> None:
    safe_name, stem = class_name.replace(" ", "_"), Path(filename).stem
    if plot_mode == "overlay":
        overlay_dir = figure_dir / "overlay" / safe_name
        overlay_dir.mkdir(parents=True, exist_ok=True)
        _save_query_support_figure(
            overlay_dir / f"{stem}.png",
            f"{class_name} | prompt, all query GT, and all predictions",
            image,
            target_boxes,
            prediction_boxes,
            prediction_scores,
            support_examples,
            plot_mode="overlay",
            red_prediction_count=red_prediction_count,
        )
        return
    if plot_mode != "split":
        raise ValueError(f"Unknown inference plot mode: {plot_mode!r}")

    red_count = (
        len(target_boxes)
        if red_prediction_count is None
        else min(len(prediction_boxes), red_prediction_count)
    )
    view_dirs = {
        name: figure_dir / name / safe_name
        for name in ("original", "prediction", "prediction_gt")
    }
    for directory in view_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    image.save(view_dirs["original"] / f"{stem}.png")
    fig, axis = plt.subplots(figsize=(8, 6))
    axis.imshow(image)
    for index, (prediction, score) in enumerate(zip(prediction_boxes, prediction_scores, strict=True)):
        axis.add_patch(patches.Rectangle(
            (prediction[0], prediction[1]), prediction[2] - prediction[0], prediction[3] - prediction[1],
            fill=False, edgecolor="red" if index < red_count else "#b87333",
            linewidth=2 if index < red_count else 0.8,
            alpha=1.0 if index < red_count else 0.3,
            label=("top predictions" if index == 0 else None) if index < red_count else (
                "extra predictions" if index == red_count else None
            ),
        ))
    axis.axis("off")
    axis.legend(loc="upper right")
    fig.savefig(view_dirs["prediction"] / f"{stem}.png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(8, 6))
    axis.imshow(image)
    for index, target in enumerate(target_boxes):
        axis.add_patch(patches.Rectangle(
            (target[0], target[1]), target[2] - target[0], target[3] - target[1],
            fill=False, edgecolor="lime", linewidth=2,
            label="ground truth" if index == 0 else None,
        ))
    for index, (prediction, score) in enumerate(zip(prediction_boxes, prediction_scores, strict=True)):
        axis.add_patch(patches.Rectangle(
            (prediction[0], prediction[1]), prediction[2] - prediction[0], prediction[3] - prediction[1],
            fill=False, edgecolor="red" if index < red_count else "#b87333",
            linewidth=2 if index < red_count else 0.8,
            alpha=1.0 if index < red_count else 0.3,
            label=("top predictions" if index == 0 else None) if index < red_count else (
                "extra predictions" if index == red_count else None
            ),
        ))
    axis.axis("off")
    axis.legend(loc="upper right")
    fig.savefig(view_dirs["prediction_gt"] / f"{stem}.png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    support_dir = figure_dir / "support_query" / safe_name
    support_dir.mkdir(parents=True, exist_ok=True)
    _save_query_support_figure(
        support_dir / f"{stem}.png",
        f"{class_name} | support examples with query GT and prediction",
        image,
        target_boxes,
        prediction_boxes,
        prediction_scores,
        support_examples,
        plot_mode="split",
        red_prediction_count=red_prediction_count,
    )


def _box_iou_cxcywh(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first_xyxy = torch.stack([first[0] - first[2] / 2, first[1] - first[3] / 2, first[0] + first[2] / 2, first[1] + first[3] / 2])
    second_xyxy = torch.stack([second[0] - second[2] / 2, second[1] - second[3] / 2, second[0] + second[2] / 2, second[1] + second[3] / 2])
    intersection_lt = torch.maximum(first_xyxy[:2], second_xyxy[:2])
    intersection_rb = torch.minimum(first_xyxy[2:], second_xyxy[2:])
    intersection = (intersection_rb - intersection_lt).clamp_min(0).prod()
    area_first = (first_xyxy[2:] - first_xyxy[:2]).clamp_min(0).prod()
    area_second = (second_xyxy[2:] - second_xyxy[:2]).clamp_min(0).prod()
    return intersection / (area_first + area_second - intersection).clamp_min(1e-6)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--train-image-dir", type=Path)
    parser.add_argument("--train-annotation-file", type=Path)
    parser.add_argument("--validation-image-dir", type=Path)
    parser.add_argument("--validation-support-image-dir", type=Path)
    parser.add_argument("--validation-annotation-file", type=Path)
    parser.add_argument("--validation-support-annotation-file", type=Path)
    parser.add_argument("--test-image-dir", type=Path)
    parser.add_argument("--test-support-image-dir", type=Path)
    parser.add_argument("--test-annotation-file", type=Path)
    parser.add_argument("--test-support-annotation-file", type=Path)
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--max-shots", type=int, default=5)
    parser.add_argument("--negative-ratio", type=float, default=0.5)
    parser.add_argument("--negative-eval-per-class", type=int, default=100)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-repeat", type=int, default=4,
                        help="Optimizer updates to run on each sampled batch before loading the next; 1 matches the old loop.")
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--checkpoint-every-eval", action=argparse.BooleanOptionalAction, default=True,
                        help="Save a step-numbered model checkpoint after each evaluation (default: enabled).")
    parser.add_argument("--max-eval-checkpoints", type=int, default=3,
                        help="Keep this many newest step-numbered evaluation checkpoints.")
    parser.add_argument("--log-every", type=int, default=100, help="Print averaged training losses every N steps.")
    parser.add_argument("--train-sanity", action=argparse.BooleanOptionalAction, default=True,
                        help="Evaluate fixed training episodes and save qualitative plots at every evaluation step (default: enabled).")
    parser.add_argument("--train-sanity-samples", type=_parse_train_sanity_samples, default=8,
                        help="Train batches per evaluation (integer), or a fraction/percent of the train loader (e.g. 0.02 or 2%%).")
    parser.add_argument("--train-sanity-deterministic", action=argparse.BooleanOptionalAction, default=True,
                        help="Use the same seeded train sanity episodes at every eval and across reruns (default: enabled).")
    parser.add_argument("--validation-sanity", action=argparse.BooleanOptionalAction, default=True,
                        help="Save qualitative plots for a fixed seen-validation subset at every evaluation step (default: enabled).")
    parser.add_argument("--validation-sanity-samples", type=_parse_train_sanity_samples, default=8,
                        help="Seen-validation batches to plot (integer), or a fraction/percent of the validation loader.")
    parser.add_argument("--validation-sanity-deterministic", action=argparse.BooleanOptionalAction, default=True,
                        help="Use the same seeded seen-validation episodes at every eval and across reruns (default: enabled).")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--include-prompt-box-in-loss", action=argparse.BooleanOptionalAction, default=True,
        help="Also train query detection against the prompted instance box (default: enabled).",
    )
    parser.add_argument(
        "--evaluate-prompt-box-iou", action=argparse.BooleanOptionalAction, default=True,
        help="Report IoU of the top prediction against the prompted box separately (default: enabled).",
    )
    parser.add_argument(
        "--coco-map-include-prompt-box", action=argparse.BooleanOptionalAction, default=True,
        help="Count the prompted instance as a GT detection for same-image COCO mAP (default: enabled).",
    )
    parser.add_argument("--coco-eval-batch-size", type=int, default=8)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--resolution", type=int, default=560)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler", choices=("cosine", "constant"), default="cosine",
                        help="Learning-rate schedule after optional linear warmup.")
    parser.add_argument("--warmup-steps", type=_parse_schedule_steps, default=0.05,
                        help="Warmup steps as an integer count or fraction of total steps (e.g. 500, 0.05, or 5%%).")
    parser.add_argument("--warmup-start-factor", type=float, default=0.1,
                        help="Warmup starts at this fraction of the base learning rate.")
    parser.add_argument("--min-lr-ratio", type=float, default=0.01,
                        help="Cosine schedule final LR as a fraction of --lr.")
    parser.add_argument("--figure-dir", type=Path, default=Path("logs/promptable_fewshot_figures"))
    parser.add_argument("--max-figures-per-class", type=int, default=10)
    parser.add_argument(
        "--inference-plot-mode", choices=("overlay", "split"), default="overlay",
        help="overlay: one image with prompt, all query GT, and predictions (default); split: save separate views in folders.",
    )
    parser.add_argument(
        "--inference-extra-predictions", type=int, default=0,
        help="Draw this many extra low-ranked predictions beyond the query GT count (default: 0).",
    )
    parser.add_argument(
        "--selection-metric", choices=("episodic_iou", "coco_map"), default="episodic_iou",
        help="Validation metric used to select the best checkpoint.",
    )
    parser.add_argument(
        "--coco-selection-images", type=int, default=500,
        help="Unique validation images used for same-image COCO mAP checkpoint selection (default: 500).",
    )
    parser.add_argument("--output", type=Path, default=Path("logs/promptable_fewshot.pt"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args_with_python_config(parser)
    if args.log_every <= 0:
        parser.error("--log-every must be positive")
    if args.steps < 2:
        parser.error("--steps must be at least 2 for a learning-rate schedule")
    if args.batch_repeat <= 0:
        parser.error("--batch-repeat must be positive")
    if args.inference_extra_predictions < 0:
        parser.error("--inference-extra-predictions must be non-negative")
    if args.max_eval_checkpoints <= 0:
        parser.error("--max-eval-checkpoints must be positive")
    if args.coco_selection_images <= 0:
        parser.error("--coco-selection-images must be positive")
    if not 0.0 <= args.warmup_start_factor <= 1.0:
        parser.error("--warmup-start-factor must be in [0, 1]")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        parser.error("--min-lr-ratio must be in [0, 1]")
    warmup_steps = _resolve_warmup_steps(args.warmup_steps, args.steps)
    required_paths = (
        "train_image_dir", "train_annotation_file", "validation_image_dir",
        "validation_annotation_file", "test_image_dir", "test_annotation_file",
    )
    missing_paths = [name for name in required_paths if getattr(args, name) is None]
    if missing_paths:
        parser.error(f"missing required dataset paths: {', '.join('--' + name.replace('_', '-') for name in missing_paths)}")
    seed_everything(args.seed)
    device = torch.device(args.device)

    train_dataset = CocoFewShotEpisodeDataset(
        args.train_image_dir,
        args.train_annotation_file,
        resolution=args.resolution,
        episodes=args.episodes,
        max_shots=args.max_shots,
        negative_ratio=args.negative_ratio,
        seed=args.seed,
        training=True,
    )
    val_dataset = CocoFewShotEpisodeDataset(
        args.validation_image_dir,
        args.validation_annotation_file,
        support_image_dir=args.validation_support_image_dir,
        resolution=args.resolution,
        episodes=1,
        max_shots=args.max_shots,
        negative_eval_per_class=args.negative_eval_per_class,
        support_annotation_file=args.validation_support_annotation_file,
        seed=args.seed,
        training=False,
    )
    test_dataset = CocoFewShotEpisodeDataset(
        args.test_image_dir,
        args.test_annotation_file,
        support_image_dir=args.test_support_image_dir,
        resolution=args.resolution,
        episodes=1,
        max_shots=args.max_shots,
        negative_eval_per_class=args.negative_eval_per_class,
        support_annotation_file=args.test_support_annotation_file,
        seed=args.seed + 1,
        training=False,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_few_shot,
        num_workers=4, pin_memory=True, generator=torch.Generator().manual_seed(args.seed + 1),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_few_shot,
        num_workers=4, pin_memory=True, generator=torch.Generator().manual_seed(args.seed + 2),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_few_shot,
        num_workers=4, pin_memory=True, generator=torch.Generator().manual_seed(args.seed + 3),
    )
    train_sanity_loader = (
        _build_fixed_diagnostic_loader(train_dataset, args.batch_size, args.train_sanity_samples, args.seed)
        if args.train_sanity else None
    )
    validation_sanity_loader = (
        _build_fixed_diagnostic_loader(
            val_dataset, args.batch_size, args.validation_sanity_samples, args.seed + 1
        )
        if args.validation_sanity else None
    )
    if train_sanity_loader is not None:
        print(
            f"train sanity enabled: {len(train_sanity_loader)} fixed batches per eval; "
            f"plots under {args.figure_dir / 'train_sanity'}"
        )
    if validation_sanity_loader is not None:
        print(
            f"seen validation plots enabled: {len(validation_sanity_loader)} fixed batches per eval; "
            f"plots under {args.figure_dir / 'validation_seen'}"
        )

    model = build_model(args.resolution, args.num_queries, args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    iterator = iter(train_loader)
    best_score = float("-inf")
    metrics: dict[str, object] = {
        "seed": args.seed,
        "train_config": vars(args) | {"device": args.device, "resolved_warmup_steps": warmup_steps},
    }
    model.train()
    progress = tqdm(range(args.steps), desc="train", unit="step")
    interval_values = {
        key: 0.0
        for key in (
            "loss", "objectness", "objectness_positive", "objectness_negative",
            "bbox", "giou", "matched_iou", "positive_score", "negative_score",
            "pred_cx", "pred_cy", "pred_w", "pred_h",
            "target_cx", "target_cy", "target_w", "target_h",
        )
    }
    interval_steps = 0
    train_log_path = args.output.with_suffix(".train.jsonl")
    train_sanity_log_path = args.output.with_suffix(".train_sanity.jsonl")
    plot_metrics_csv = args.output.with_suffix(".plot_metrics.csv")
    plot_output_dir = args.figure_dir / "training_curves"
    plot_train_records: list[dict[str, float | int]] = []
    train_log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"{train_log_path=}")
    print(
        f"lr schedule={args.lr_scheduler} base_lr={args.lr:.3g} warmup_steps={warmup_steps}/"
        f"{args.steps} warmup_start_factor={args.warmup_start_factor:g} min_lr_ratio={args.min_lr_ratio:g}"
    )
    print(
        f"batch repeat={args.batch_repeat}: each sampled batch gets up to "
        f"{args.batch_repeat} optimizer updates before the next batch"
    )
    print(f"include prompt box in query detection loss={args.include_prompt_box_in_loss}")
    current_batch = None
    for step in progress:
        lr = _scheduled_lr(
            args.lr, step, args.steps, warmup_steps, args.warmup_start_factor,
            args.min_lr_ratio, args.lr_scheduler,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        if current_batch is None or step % args.batch_repeat == 0:
            try:
                current_batch = _move_batch(next(iterator), device)
            except StopIteration:
                iterator = iter(train_loader)
                current_batch = _move_batch(next(iterator), device)
        batch = current_batch
        outputs = model.forward_same_image(batch["query_images"], batch["support_boxes"])
        loss_target_boxes = batch["query_boxes"]
        loss_target_mask = batch["query_target_mask"]
        if args.include_prompt_box_in_loss:
            # Same-image mode: add the prompt instance to this image's query targets.
            # TODO: For batched support/query mode, add separate support-image losses
            # from support_outputs and full support annotations; this switch alone
            # must not add a support box as a target on a different query image.
            prompt_is_present = batch["present"].flatten() > 0.5
            loss_target_boxes = torch.cat(
                [loss_target_boxes, batch["support_boxes"][:, None, :]], dim=1
            )
            loss_target_mask = torch.cat(
                [loss_target_mask, prompt_is_present[:, None]], dim=1
            )
        losses = model.loss(
            outputs,
            loss_target_boxes,
            batch["present"],
            target_mask=loss_target_mask,
        )
        loss = losses["loss_objectness"] + 5 * losses["loss_bbox"] + 2 * losses["loss_giou"]
        scalar_stats = torch.stack(
            [
                loss.detach(), losses["loss_objectness"].detach(),
                losses["loss_objectness_positive"], losses["loss_objectness_negative"],
                losses["loss_bbox"].detach(), losses["loss_giou"].detach(), losses["matched_iou"],
                losses["positive_score"], losses["negative_score"],
            ]
        )
        target_boxes = loss_target_boxes[loss_target_mask]
        target_box_mean = target_boxes.mean(dim=0) if target_boxes.numel() else torch.zeros(4, device=device)
        cpu_stats = torch.cat((scalar_stats, losses["pred_box_mean"], target_box_mean)).cpu().tolist()
        stat_names = (
            "loss", "objectness", "objectness_positive", "objectness_negative", "bbox", "giou",
            "matched_iou", "positive_score", "negative_score", "pred_cx", "pred_cy", "pred_w",
            "pred_h", "target_cx", "target_cy", "target_w", "target_h",
        )
        loss_values = dict(zip(stat_names, cpu_stats, strict=True))
        progress.set_postfix(
            loss=f"{loss_values['loss']:.4f}",
            obj=f"{loss_values['objectness']:.4f}",
            bbox=f"{loss_values['bbox']:.4f}",
            giou=f"{loss_values['giou']:.4f}",
            iou=f"{loss_values['matched_iou']:.3f}",
            refresh=False,
        )
        for name, value in loss_values.items():
            interval_values[name] += value
        interval_steps += 1
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        if (step + 1) % args.log_every == 0 or step + 1 == args.steps:
            averages = {name: value / interval_steps for name, value in interval_values.items()}
            record = {"step": step + 1, **averages, "lr": lr}
            plot_train_records.append(record)
            tqdm.write(
                f"train step={step + 1}/{args.steps} "
                f"loss={averages['loss']:.4f} obj={averages['objectness']:.4f} "
                f"pos_obj={averages['objectness_positive']:.4f} neg_obj={averages['objectness_negative']:.4f} "
                f"bbox={averages['bbox']:.4f} giou={averages['giou']:.4f} "
                f"iou={averages['matched_iou']:.3f} score+={averages['positive_score']:.3f} "
                f"score-={averages['negative_score']:.3f} "
                f"box=({averages['pred_cx']:.2f},{averages['pred_cy']:.2f},"
                f"{averages['pred_w']:.2f},{averages['pred_h']:.2f}) "
                f"gt=({averages['target_cx']:.2f},{averages['target_cy']:.2f},"
                f"{averages['target_w']:.2f},{averages['target_h']:.2f}) "
                f"lr={record['lr']:.2e}"
                # f"log={train_log_path}"
            )
            with train_log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
            interval_values = {name: 0.0 for name in interval_values}
            interval_steps = 0
        if (step + 1) % args.eval_every == 0:
            val_result = run_episode_eval(
                model, val_loader, device,
                evaluate_prompt_box_iou=args.evaluate_prompt_box_iou,
            )
            coco_selection = run_coco_map_eval(
                model,
                val_dataset,
                device,
                query_batch_size=args.coco_eval_batch_size,
                seed=args.seed,
                max_query_images=args.coco_selection_images,
                include_prompt_box_as_target=args.coco_map_include_prompt_box,
            )
            metrics[f"validation_step_{step + 1}"] = {
                "episodic": val_result,
                "coco_selection": coco_selection,
                "selection_note": (
                    "same-image COCO mAP on a deterministic promptable episode subset"
                    if args.selection_metric == "coco_map"
                    else "same-image episodic top-prediction IoU"
                ),
            }
            if train_sanity_loader is not None:
                current_train_sanity_loader = train_sanity_loader
                if not args.train_sanity_deterministic:
                    current_train_sanity_loader = _build_fixed_diagnostic_loader(
                        train_dataset,
                        args.batch_size,
                        args.train_sanity_samples,
                        args.seed + step + 1,
                    )
                sanity_result = run_train_sanity_eval(
                    model,
                    current_train_sanity_loader,
                    device,
                    args.figure_dir / "train_sanity" / f"step_{step + 1:06d}",
                    step + 1,
                    plot_mode=args.inference_plot_mode,
                    extra_predictions=args.inference_extra_predictions,
                    evaluate_prompt_box_iou=args.evaluate_prompt_box_iou,
                )
                metrics[f"train_sanity_step_{step + 1}"] = sanity_result
                with train_sanity_log_path.open("a", encoding="utf-8") as log_file:
                    log_file.write(json.dumps(sanity_result) + "\n")
                    log_file.flush()
                print(
                    f"train sanity step={step + 1} samples={sanity_result['sample_count']} "
                    f"positive_iou={sanity_result['mean_positive_iou']} "
                    f"prompt_iou={sanity_result['mean_prompt_iou']} "
                    f"positive_score={sanity_result['mean_positive_score']} "
                    f"negative_score={sanity_result['mean_negative_score']} "
                    f"plots={args.figure_dir / 'train_sanity' / f'step_{step + 1:06d}'}"
                )
            if validation_sanity_loader is not None:
                current_validation_sanity_loader = validation_sanity_loader
                if not args.validation_sanity_deterministic:
                    current_validation_sanity_loader = _build_fixed_diagnostic_loader(
                        val_dataset,
                        args.batch_size,
                        args.validation_sanity_samples,
                        args.seed + step + 1,
                    )
                validation_plot_result = run_train_sanity_eval(
                    model,
                    current_validation_sanity_loader,
                    device,
                    args.figure_dir / "validation_seen" / f"step_{step + 1:06d}",
                    step + 1,
                    plot_mode=args.inference_plot_mode,
                    extra_predictions=args.inference_extra_predictions,
                    evaluate_prompt_box_iou=args.evaluate_prompt_box_iou,
                )
                metrics[f"validation_plots_step_{step + 1}"] = validation_plot_result
                print(
                    f"seen validation plots step={step + 1} "
                    f"samples={validation_plot_result['sample_count']} "
                    f"positive_iou={validation_plot_result['mean_positive_iou']} "
                    f"prompt_iou={validation_plot_result['mean_prompt_iou']} "
                    f"plots={args.figure_dir / 'validation_seen' / f'step_{step + 1:06d}'}"
                )
            save_rfdetr_training_plots(
                plot_train_records,
                metrics,
                plot_metrics_csv,
                plot_output_dir,
            )
            print(f"updated RF-DETR metric plots under {plot_output_dir}")
            if args.selection_metric == "coco_map":
                score = (
                    float(coco_selection["mAP"])
                    if coco_selection["mAP"] is not None
                    else float("-inf")
                )
            else:
                score = float(val_result["macro_positive_iou"])
            print(
                f"validation step={step + 1} selection_metric={args.selection_metric} "
                f"score={score:.4f} episodic_iou={val_result['macro_positive_iou']:.4f} "
                f"coco_mAP={coco_selection['mAP']}"
            )
            if args.checkpoint_every_eval:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                checkpoint_path = args.output.with_name(
                    f"{args.output.stem}.step_{step + 1:06d}{args.output.suffix}"
                )
                torch.save(model.state_dict(), checkpoint_path)
                checkpoint_pattern = f"{args.output.stem}.step_*{args.output.suffix}"
                checkpoints = sorted(args.output.parent.glob(checkpoint_pattern))
                for stale_checkpoint in checkpoints[:-args.max_eval_checkpoints]:
                    stale_checkpoint.unlink()
                metrics["eval_checkpoints"] = [
                    str(path) for path in checkpoints[-args.max_eval_checkpoints:]
                ]
                print(
                    f"eval checkpoint={checkpoint_path} "
                    f"retention={args.max_eval_checkpoints}"
                )
            if score > best_score:
                best_score = score
                args.output.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), args.output.with_name(f"{args.output.stem}.best{args.output.suffix}"))
        model.train()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    best_path = args.output.with_name(f"{args.output.stem}.best{args.output.suffix}")
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
        metrics["selected_checkpoint"] = str(best_path)
    else:
        metrics["selected_checkpoint"] = str(args.output)
    metrics["seen_validation_final"] = run_episode_eval(
        model, val_loader, device,
        evaluate_prompt_box_iou=args.evaluate_prompt_box_iou,
    )
    metrics["seen_validation_coco"] = run_coco_map_eval(
        model,
        val_dataset,
        device,
        query_batch_size=args.coco_eval_batch_size,
        seed=args.seed,
        include_prompt_box_as_target=args.coco_map_include_prompt_box,
    )
    metrics["held_out_test"] = run_episode_eval(
        model, test_loader, device, figure_dir=args.figure_dir / "heldout_test",
        max_figures_per_class=args.max_figures_per_class,
        plot_mode=args.inference_plot_mode,
        extra_predictions=args.inference_extra_predictions,
        evaluate_prompt_box_iou=args.evaluate_prompt_box_iou,
    )
    metrics["held_out_coco"] = run_coco_map_eval(
        model,
        test_dataset,
        device,
        query_batch_size=args.coco_eval_batch_size,
        seed=args.seed + 1,
        include_prompt_box_as_target=args.coco_map_include_prompt_box,
    )
    metrics["metric_protocol"] = (
        "same-image visual prompting; one instance box is the prompt and the remaining "
        "same-class instances in that image are targets"
    )
    metrics["include_prompt_box_in_loss"] = args.include_prompt_box_in_loss
    metrics["evaluate_prompt_box_iou"] = args.evaluate_prompt_box_iou
    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str))
    print(f"checkpoint={args.output} metrics={metrics_path}")


if __name__ == "__main__":
    main()
