"""Train and evaluate cross-image visual few-shot COCO detection."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torch.utils.data import DataLoader
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


def run_episode_eval(
    model: PromptableDetector,
    loader: DataLoader,
    device: torch.device,
    figure_dir: Path | None = None,
    max_figures_per_class: int = 10,
) -> dict[str, object]:
    model.eval()
    by_category: dict[int, dict[str, object]] = {}
    figure_counts: dict[int, int] = {}
    dataset = loader.dataset
    category_names = {item["id"]: item["name"] for item in dataset.coco.loadCats(dataset.coco.getCatIds())}
    with torch.inference_mode():
        for raw_batch in tqdm(loader, desc="eval", leave=False):
            batch = _move_batch(raw_batch, device)
            outputs = model.forward_support_query(
                batch["support_images"],
                batch["support_boxes"],
                batch["query_images"],
                support_group_ids=batch["support_group_ids"],
                query_to_support=batch["query_to_support"],
            )
            scores = outputs["pred_logits"].sigmoid()[..., 0]
            gt_present = batch["present"].flatten() > 0.5
            for index, category_id in enumerate(batch["category_ids"]):
                bucket = by_category.setdefault(category_id, {"examples": 0, "positive": 0, "negative": 0, "positive_score": 0.0, "negative_score": 0.0, "matched_iou": 0.0, "ground_truth_objects": 0, "presence_true_positive_at_0_5": 0, "presence_false_positive_at_0_5": 0, "recall50": 0, "recall75": 0})
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
                        pred_xyxy = _cxcywh_pixels(pred_box, width, height)
                        gt_xyxy = _cxcywh_pixels(gt_box, width, height)
                        _save_episode_figures(
                            figure_dir, category_names[category_id], metadata["file_name"], image,
                            pred_xyxy, gt_xyxy, score,
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
        bucket["recall50"] = int(bucket["recall50"]) / max(1, positives)
        bucket["recall75"] = int(bucket["recall75"]) / max(1, positives)
        bucket["presence_recall_at_0_5"] = int(bucket["presence_true_positive_at_0_5"]) / max(1, positives)
        bucket["presence_false_positive_rate_at_0_5"] = int(bucket["presence_false_positive_at_0_5"]) / max(1, negatives)
    all_iou = [float(values["mean_matched_iou"]) for values in by_category.values() if int(values["positive"]) > 0]
    return {"categories": by_category, "macro_positive_iou": sum(all_iou) / max(1, len(all_iou))}


def run_coco_map_eval(
    model: PromptableDetector,
    dataset: CocoFewShotEpisodeDataset,
    device: torch.device,
    max_shots: int,
    query_batch_size: int,
    seed: int,
    max_query_images: int | None = None,
) -> dict[str, object]:
    """Evaluate class-conditioned detections with standard COCO AP/AR metrics.

    A deterministic support set is selected for each category. Its image IDs are
    excluded from that category's query set to prevent image-level leakage. The
    query set otherwise contains every COCO image, including images with no
    instances of the evaluated category.
    """
    if query_batch_size <= 0:
        raise ValueError("query_batch_size must be positive")
    if max_query_images is not None and max_query_images <= 0:
        raise ValueError("max_query_images must be positive when specified")
    coco_gt = dataset.coco
    all_image_ids = sorted(coco_gt.getImgIds())
    categories = sorted(coco_gt.getCatIds())
    category_names = {
        item["id"]: item["name"] for item in coco_gt.loadCats(categories)
    }
    category_metrics: dict[int, dict[str, object]] = {}
    support_ids_by_category: dict[int, list[int]] = {}
    for category_id in categories:
        support_image_index = (
            dataset.support_category_images
            if dataset.has_separate_support
            else dataset.category_images
        )
        positive_image_ids = sorted(support_image_index.get(category_id, []))
        minimum_support = 1 if dataset.has_separate_support else 2
        if len(positive_image_ids) < minimum_support:
            continue
        available_shots = len(positive_image_ids) if dataset.has_separate_support else len(positive_image_ids) - 1
        support_count = min(max_shots, available_shots)
        support_ids_by_category[category_id] = sorted(
            positive_image_ids,
            key=lambda image_id: (seed * 1_000_003 + image_id) % 2**32,
        )[:support_count]
    excluded_support_ids = {
        image_id
        for category_support_ids in support_ids_by_category.values()
        for image_id in category_support_ids
    }
    query_ids = (
        all_image_ids
        if dataset.has_separate_support
        else [image_id for image_id in all_image_ids if image_id not in excluded_support_ids]
    )
    if max_query_images is not None and len(query_ids) > max_query_images:
        query_set = set(query_ids)
        selected: set[int] = set()
        per_category_quota = max(1, max_query_images // (2 * max(1, len(categories))))
        for category_id in categories:
            candidates = [
                image_id for image_id in dataset.category_images.get(category_id, [])
                if image_id in query_set
            ]
            selected.update(candidates[:per_category_quota])
        ordered_ids = [image_id for image_id in query_ids if image_id in selected]
        if len(ordered_ids) < max_query_images:
            ordered_ids.extend(
                image_id for image_id in query_ids
                if image_id not in selected
            )
        query_ids = ordered_ids[:max_query_images]
    evaluation_started = time.perf_counter()
    model_was_training = model.training
    model.eval()

    with torch.inference_mode():
        for category_id in categories:
            support_ids = support_ids_by_category.get(category_id)
            if not support_ids:
                continue
            support_count = len(support_ids)
            support_images: list[torch.Tensor] = []
            support_boxes: list[torch.Tensor] = []
            for support_id in support_ids:
                support_annotations = (
                    dataset.support_image_annotations
                    if dataset.has_separate_support
                    else dataset.image_annotations
                )
                annotations = [
                    ann for ann in support_annotations[support_id]
                    if ann["category_id"] == category_id
                ]
                annotation = min(annotations, key=lambda ann: ann["id"])
                support_tensor, width, height = dataset._load_tensor(
                    support_id, support=dataset.has_separate_support
                )
                support_images.append(support_tensor)
                support_boxes.append(
                    dataset._normalize_box(annotation["bbox"], width, height)
                )

            category_results: list[dict[str, float | int]] = []
            group_ids = torch.full((support_count,), category_id, dtype=torch.long, device=device)
            support_images_tensor = torch.stack(support_images).to(device)
            support_boxes_tensor = torch.stack(support_boxes).to(device)
            encoded_support = model.encode_support(
                support_images_tensor, support_boxes_tensor, support_group_ids=group_ids
            )
            for start in range(0, len(query_ids), query_batch_size):
                batch_ids = query_ids[start : start + query_batch_size]
                query_tensors: list[torch.Tensor] = []
                query_sizes: list[tuple[int, int]] = []
                for image_id in batch_ids:
                    query_tensor, width, height = dataset._load_tensor(image_id)
                    query_tensors.append(query_tensor)
                    query_sizes.append((width, height))
                query_count = len(batch_ids)
                outputs = model.forward_support_query(
                    support_images_tensor,
                    support_boxes_tensor,
                    torch.stack(query_tensors).to(device),
                    support_group_ids=group_ids,
                    query_to_support=torch.full(
                        (query_count,), category_id, dtype=torch.long, device=device
                    ),
                    encoded_support=encoded_support,
                )
                scores = outputs["pred_logits"].sigmoid()[..., 0]
                boxes = box_cxcywh_to_xyxy(outputs["pred_boxes"])
                for batch_index, (image_id, (width, height)) in enumerate(
                    zip(batch_ids, query_sizes, strict=True)
                ):
                    image_boxes = boxes[batch_index].clamp(0, 1)
                    image_boxes[:, 0::2] *= width
                    image_boxes[:, 1::2] *= height
                    image_scores = scores[batch_index]
                    keep = torch.isfinite(image_scores) & torch.isfinite(image_boxes).all(dim=1)
                    image_boxes = image_boxes[keep].cpu()
                    image_scores = image_scores[keep].cpu()
                    if image_scores.numel() > 100:
                        image_scores, top_indices = image_scores.topk(100)
                        image_boxes = image_boxes[top_indices]
                    for box, score in zip(image_boxes.tolist(), image_scores.tolist(), strict=True):
                        x0, y0, x1, y1 = box
                        if x1 <= x0 or y1 <= y0:
                            continue
                        category_results.append(
                            {
                                "image_id": image_id,
                                "category_id": category_id,
                                "bbox": [x0, y0, x1 - x0, y1 - y0],
                                "score": score,
                            }
                        )

            if not category_results:
                category_metrics[category_id] = {
                    "name": category_names[category_id],
                    "support_images": support_count,
                    "support_image_ids": support_ids,
                    "query_images": len(query_ids),
                    "AP": 0.0,
                    "AP50": 0.0,
                    "AP75": 0.0,
                    "AP_small": 0.0,
                    "AP_medium": 0.0,
                    "AP_large": 0.0,
                    "AR1": 0.0,
                    "AR10": 0.0,
                    "AR100": 0.0,
                }
                continue
            coco_dt = coco_gt.loadRes(category_results)
            evaluator = COCOeval(coco_gt, coco_dt, "bbox")
            evaluator.params.imgIds = query_ids
            evaluator.params.catIds = [category_id]
            evaluator.params.maxDets = [1, 10, 100]
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
            category_metrics[category_id] = {
                "name": category_names[category_id],
                "support_images": support_count,
                "support_image_ids": support_ids,
                "query_images": len(query_ids),
                "AP": float(evaluator.stats[0]),
                "AP50": float(evaluator.stats[1]),
                "AP75": float(evaluator.stats[2]),
                "AP_small": float(evaluator.stats[3]),
                "AP_medium": float(evaluator.stats[4]),
                "AP_large": float(evaluator.stats[5]),
                "AR1": float(evaluator.stats[6]),
                "AR10": float(evaluator.stats[7]),
                "AR100": float(evaluator.stats[8]),
            }

    if model_was_training:
        model.train()
    elapsed_seconds = time.perf_counter() - evaluation_started
    def macro(metric_name: str) -> float:
        values = [
            float(value[metric_name])
            for value in category_metrics.values()
            if float(value[metric_name]) >= 0
        ]
        return sum(values) / max(1, len(values))

    return {
        "categories": category_metrics,
        "mAP": macro("AP"),
        "mAP50": macro("AP50"),
        "mAP75": macro("AP75"),
        "mAP_small": macro("AP_small"),
        "mAP_medium": macro("AP_medium"),
        "mAP_large": macro("AP_large"),
        "mAR1": macro("AR1"),
        "mAR10": macro("AR10"),
        "mAR100": macro("AR100"),
        "metric_scale": "0_to_1",
        "query_image_category_pairs": len(query_ids) * len(category_metrics),
        "elapsed_seconds": elapsed_seconds,
        "query_pairs_per_second": (
            len(query_ids) * len(category_metrics) / elapsed_seconds
            if elapsed_seconds > 0
            else 0.0
        ),
        "evaluation_protocol": (
            "COCOeval bbox; dedicated support images are disjoint from all query images"
            if dataset.has_separate_support
            else "COCOeval bbox; union of all category support images excluded from one common query-image set"
        ),
    }


def _cxcywh_pixels(box: torch.Tensor, width: int, height: int) -> list[float]:
    cx, cy, box_width, box_height = box.detach().float().cpu().tolist()
    return [(cx - box_width / 2) * width, (cy - box_height / 2) * height,
            (cx + box_width / 2) * width, (cy + box_height / 2) * height]


def _save_episode_figures(
    figure_dir: Path, class_name: str, filename: str, image: Image.Image,
    prediction: list[float], target: list[float], score: float,
) -> None:
    safe_name, stem = class_name.replace(" ", "_"), Path(filename).stem
    view_dirs = {name: figure_dir / name / safe_name for name in ("original", "prediction", "prediction_gt")}
    for directory in view_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    image.save(view_dirs["original"] / f"{stem}.png")
    for view in ("prediction", "prediction_gt"):
        fig, axis = plt.subplots(figsize=(8, 6))
        axis.imshow(image)
        if view == "prediction_gt":
            axis.add_patch(patches.Rectangle((target[0], target[1]), target[2] - target[0], target[3] - target[1], fill=False, edgecolor="green", linewidth=2, label="ground truth"))
        axis.add_patch(patches.Rectangle((prediction[0], prediction[1]), prediction[2] - prediction[0], prediction[3] - prediction[1], fill=False, edgecolor="red", linewidth=2, label=f"prediction {score:.2f}"))
        axis.axis("off")
        axis.legend(loc="upper right")
        fig.savefig(view_dirs[view] / f"{stem}.png", bbox_inches="tight", dpi=120)
        plt.close(fig)


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
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--coco-eval-batch-size", type=int, default=8)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--resolution", type=int, default=560)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--figure-dir", type=Path, default=Path("logs/promptable_fewshot_figures"))
    parser.add_argument("--max-figures-per-class", type=int, default=10)
    parser.add_argument("--selection-metric", choices=("coco_map", "episodic_iou"), default="coco_map")
    parser.add_argument("--coco-selection-images", type=int, default=500)
    parser.add_argument("--output", type=Path, default=Path("logs/promptable_fewshot.pt"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args_with_python_config(parser)
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
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_few_shot, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_few_shot, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_few_shot, num_workers=4, pin_memory=True)

    model = build_model(args.resolution, args.num_queries, args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    iterator = iter(train_loader)
    best_score = float("-inf")
    metrics: dict[str, object] = {"seed": args.seed, "train_config": vars(args) | {"device": args.device}}
    model.train()
    for step in tqdm(range(args.steps), desc="train"):
        try:
            batch = _move_batch(next(iterator), device)
        except StopIteration:
            iterator = iter(train_loader)
            batch = _move_batch(next(iterator), device)
        outputs = model.forward_support_query(
            batch["support_images"],
            batch["support_boxes"],
            batch["query_images"],
            support_group_ids=batch["support_group_ids"],
            query_to_support=batch["query_to_support"],
        )
        losses = model.loss(outputs, batch["query_boxes"], batch["present"])
        loss = losses["loss_objectness"] + 5 * losses["loss_bbox"] + 2 * losses["loss_giou"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        if (step + 1) % args.eval_every == 0:
            val_result = run_episode_eval(model, val_loader, device)
            selection_result = run_coco_map_eval(
                model,
                val_dataset,
                device,
                args.max_shots,
                args.coco_eval_batch_size,
                args.seed,
                max_query_images=args.coco_selection_images,
            )
            metrics[f"validation_step_{step + 1}"] = {
                "episodic": val_result,
                "coco_selection": selection_result,
            }
            score = (
                float(selection_result["mAP"])
                if args.selection_metric == "coco_map"
                else float(val_result["macro_positive_iou"])
            )
            print(
                f"validation step={step + 1} selection_metric={args.selection_metric} "
                f"score={score:.4f}"
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
    metrics["seen_validation_final"] = run_episode_eval(model, val_loader, device)
    metrics["held_out_test"] = run_episode_eval(
        model, test_loader, device, figure_dir=args.figure_dir / "heldout_test",
        max_figures_per_class=args.max_figures_per_class,
    )
    metrics["seen_validation_coco"] = run_coco_map_eval(
        model, val_dataset, device, args.max_shots, args.coco_eval_batch_size, args.seed
    )
    metrics["held_out_coco"] = run_coco_map_eval(
        model, test_dataset, device, args.max_shots, args.coco_eval_batch_size, args.seed + 1
    )
    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str))
    print(f"checkpoint={args.output} metrics={metrics_path}")


if __name__ == "__main__":
    main()
