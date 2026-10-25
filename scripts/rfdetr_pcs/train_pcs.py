"""Full train/validation/test loop for RF-DETR PCS episodes.

Supports same-image prompting and separate support/query episodes. Unlike the
diagnostic trainers, this script trains over the configured episode stream and
evaluates on seen validation and held-out test annotations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rfdetr_pcs.coco_episodes import (  # noqa: E402
    CocoFewShotEpisodeDataset,
    collate_few_shot,
    collate_support_query,
)
from rfdetr_pcs.config import TrainConfig  # noqa: E402
from rfdetr_pcs.datasets.aug_configs import (  # noqa: E402
    AUG_CONFIG,
    AUG_CONSERVATIVE,
    AUG_AGGRESSIVE,
    AUG_AERIAL,
    AUG_INDUSTRIAL,
)
from rfdetr_pcs.model_presets import build_base_model_config  # noqa: E402
from rfdetr_pcs.conditioning import (  # noqa: E402
    class_ids_for_mode, conditioning_mode_at, conditioning_schedule, uses_class_embeddings,
)
from rfdetr_pcs.models.pcs import PCSDetector  # noqa: E402
from rfdetr_pcs.evaluation.fsod_eval import FSODEvaluator  # noqa: E402
from rfdetr_pcs.utilities.box_ops import box_cxcywh_to_xyxy  # noqa: E402
from scripts.rfdetr_promptable.train_promptable_fewshot import (  # noqa: E402
    _save_query_support_figure,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.device):
        return str(value)
    return value


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def _apply_gpu_augmentation(
    batch: dict[str, Any], cfg: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    """Apply GPU augmentation (Kornia) to batch images.

    Applies Kornia-based GPU augmentation to query and support images in the batch.
    Follows the pattern from RFDETRDataModule.on_after_batch_transfer().

    Args:
        batch: Dictionary containing batch tensors on device.
        cfg: Training configuration with augmentation settings.
        device: Target device for augmentation.

    Returns:
        Batch with augmented images and updated targets.
    """
    try:
        from rfdetr_pcs.datasets.kornia_transforms import (
            build_kornia_pipeline, build_normalize,
            require_gpu_backend_ready, is_gpu_postprocess,
        )
    except ImportError as e:
        import warnings
        warnings.warn(f"Kornia GPU augmentation not available: {e}. Skipping GPU augmentation.")
        return batch

    augmentation_backend = str(cfg.get("augmentation_backend", "cpu"))
    
    if not is_gpu_postprocess(augmentation_backend):
        return batch
    
    try:
        require_gpu_backend_ready(augmentation_backend, has_cuda=torch.cuda.is_available())
    except (ImportError, RuntimeError) as e:
        import warnings
        warnings.warn(f"Kornia GPU augmentation requirements not met: {e}. Skipping.")
        return batch

    aug_config = cfg.get("aug_config", None)
    if aug_config is None:
        aug_config = AUG_CONFIG
    
    resolution = int(cfg.get("resolution", 560))
    include_masks = bool(cfg.get("segmentation", True))
    
    kornia_pipeline = build_kornia_pipeline(aug_config, resolution, with_masks=include_masks)
    kornia_normalize = build_normalize()
    
    kornia_pipeline.to(device)
    kornia_normalize.to(device)

    if "query_images" in batch:
        images = batch["query_images"]
        boxes = batch.get("query_boxes")
        padding_mask = batch.get("query_padding_mask")
        masks = batch.get("query_masks") if include_masks else None
        
        batch["query_images"] = _apply_kornia_to_batch(
            images, boxes, masks, padding_mask, kornia_pipeline, kornia_normalize, device
        )
    
    if "support_images" in batch:
        images = batch["support_images"]
        boxes = batch.get("support_boxes")
        padding_mask = batch.get("support_padding_mask")
        masks = batch.get("support_masks") if include_masks else None
        
        batch["support_images"] = _apply_kornia_to_batch(
            images, boxes, masks, padding_mask, kornia_pipeline, kornia_normalize, device
        )

    return batch


def _apply_kornia_to_batch(
    images: torch.Tensor,
    boxes: torch.Tensor | None,
    masks: torch.Tensor | None,
    padding_mask: torch.Tensor | None,
    pipeline: Any,
    normalize: Any,
    device: torch.device,
) -> torch.Tensor:
    """Apply Kornia pipeline to a batch of images.

    Args:
        images: Batch of images tensor [B, C, H, W] on device.
        boxes: Batch of boxes tensor or None.
        masks: Batch of masks tensor or None.
        padding_mask: Padding mask tensor or None.
        pipeline: Kornia augmentation pipeline.
        normalize: Kornia normalization function.
        device: Target device.

    Returns:
        Augmented images tensor.
    """
    if images.shape[0] == 0:
        return images

    b, c, h, w = images.shape
    
    if padding_mask is None:
        padding_mask = torch.zeros(b, h, w, dtype=torch.bool, device=device)
    padding_masks = padding_mask.unsqueeze(1).to(torch.float32)

    if masks is not None and masks.numel() > 0:
        image_height, image_width = h, w
        auxiliary_masks = torch.cat((masks, padding_masks), dim=1)
        img_aug, boxes_aug, auxiliary_masks_aug = pipeline(images, boxes, auxiliary_masks)
        padding_mask_aug = auxiliary_masks_aug[:, masks.shape[1]:] > 0.5
    else:
        img_aug, boxes_aug, padding_masks_aug = pipeline(images, boxes, padding_masks)
        padding_mask_aug = padding_masks_aug[:, 0] > 0.5 if padding_masks_aug is not None else None

    img_aug = normalize(img_aug)
    
    return img_aug


def _class_ids(batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    """Return one raw COCO category ID per episode for the learned class prompt."""
    return torch.as_tensor(batch["category_ids"], dtype=torch.long, device=device)


def _same_image_targets(
    batch: dict[str, Any], device: torch.device, include_prompt_box: bool
) -> list[dict[str, torch.Tensor]]:
    boxes = batch["query_boxes"]
    valid = batch["query_target_mask"].clone()
    masks = batch.get("query_masks")
    if include_prompt_box:
        prompt_boxes = batch["support_boxes"]
        prompt_valid = batch["prompt_target_mask"] & ~batch["prompt_padding_mask"]
        boxes = torch.cat((boxes, prompt_boxes), dim=1)
        valid = torch.cat((valid, prompt_valid), dim=1)
        if masks is not None:
            masks = torch.cat((masks, batch["prompt_masks"]), dim=1)
    targets = []
    for index in range(boxes.shape[0]):
        target = {
            "boxes": boxes[index, valid[index]].to(device),
            "labels": torch.zeros(int(valid[index].sum()), dtype=torch.long, device=device),
        }
        if masks is not None:
            target["masks"] = masks[index, valid[index]].to(device)
        targets.append(target)
    return targets


def _query_targets(batch: dict[str, Any], device: torch.device) -> list[dict[str, torch.Tensor]]:
    boxes = batch["query_boxes"]
    valid = batch["query_target_mask"]
    masks = batch.get("query_masks")
    targets = []
    for index in range(boxes.shape[0]):
        target = {
            "boxes": boxes[index, valid[index]].to(device),
            "labels": torch.zeros(int(valid[index].sum()), dtype=torch.long, device=device),
        }
        if masks is not None:
            target["masks"] = masks[index, valid[index]].to(device)
        targets.append(target)
    return targets


def _support_targets(
    batch: dict[str, Any], device: torch.device, include_prompt_box: bool = True
) -> list[dict[str, torch.Tensor]]:
    boxes = batch["support_target_boxes"].flatten(0, 1)
    valid = batch["support_target_mask"].flatten(0, 1)
    if not include_prompt_box and "support_target_prompt_mask" in batch:
        valid = valid & ~batch["support_target_prompt_mask"].flatten(0, 1)
    masks = batch.get("support_target_masks")
    if masks is not None:
        masks = masks.flatten(0, 1)
    targets = []
    for index in range(boxes.shape[0]):
        target = {
            "boxes": boxes[index, valid[index]].to(device),
            "labels": torch.zeros(int(valid[index].sum()), dtype=torch.long, device=device),
        }
        if masks is not None:
            target["masks"] = masks[index, valid[index]].to(device)
        targets.append(target)
    return targets


def _weighted_loss(model: PCSDetector, losses: dict[str, torch.Tensor]) -> torch.Tensor:
    return sum(
        value * model.criterion.weight_dict[name]
        for name, value in losses.items()
        if name in model.criterion.weight_dict
    )


def _build_datasets(cfg: dict[str, Any], mode: str) -> tuple[CocoFewShotEpisodeDataset, ...]:
    resolution = int(cfg["resolution"])
    shots = int(cfg.get("shots", cfg.get("max_prompt_boxes", 1)))
    aug_config = cfg.get("aug_config", None)
    if aug_config is None and bool(cfg.get("use_default_augmentation", True)):
        aug_config = AUG_CONFIG
    common = {
        "resolution": resolution,
        "max_shots": shots,
        "negative_ratio": float(cfg.get("negative_ratio", 0.0)),
        "negative_prompt_ratio": float(cfg.get("negative_prompt_ratio", 0.0)),
        "negative_class_ratio": float(cfg.get("negative_class_ratio", 0.0)),
        "seed": int(cfg.get("seed", 7)),
        "include_masks": bool(cfg.get("segmentation", True)),
        "aug_config": aug_config,
        "augmentation_backend": str(cfg.get("augmentation_backend", "cpu")),
    }
    separate = mode == "support_query"

    def create(split: str, training: bool, seed_offset: int = 0):
        image_dir = cfg[f"{split}_image_dir"]
        annotation_file = cfg[f"{split}_annotation_file"]
        kwargs: dict[str, Any] = {}
        if separate:
            kwargs = {
                "support_image_dir": cfg[f"{split}_support_image_dir"],
                "support_annotation_file": cfg[f"{split}_support_annotation_file"],
            }
        options = dict(common)
        options.update(
            episodes=int(cfg.get("episodes", 80_000)) if training else 1,
            seed=int(cfg.get("seed", 7)) + seed_offset,
            training=training,
        )
        if not training:
            options["negative_ratio"] = 0.0
            options["negative_class_ratio"] = float(cfg.get(f"{split}_negative_class_ratio", 0.0))
            options["aug_config"] = None
            options["augmentation_backend"] = "cpu"
        return CocoFewShotEpisodeDataset(image_dir, annotation_file, **kwargs, **options)

    return create("train", True), create("validation", False, 1), create("test", False, 2)


def _limit_dataset(dataset: CocoFewShotEpisodeDataset, limit: int, seed: int) -> Subset:
    count = min(len(dataset), limit) if limit > 0 else len(dataset)
    if count == len(dataset):
        indices = list(range(count))
    else:
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(len(dataset)), count))
    return Subset(dataset, indices)


def _box_iou(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first_xyxy = box_cxcywh_to_xyxy(first).clamp(0, 1)
    second_xyxy = box_cxcywh_to_xyxy(second).clamp(0, 1)
    left_top = torch.maximum(first_xyxy[..., :2], second_xyxy[..., :2])
    right_bottom = torch.minimum(first_xyxy[..., 2:], second_xyxy[..., 2:])
    intersection = (right_bottom - left_top).clamp_min(0).prod(-1)
    first_area = (first_xyxy[..., 2:] - first_xyxy[..., :2]).clamp_min(0).prod(-1)
    second_area = (second_xyxy[..., 2:] - second_xyxy[..., :2]).clamp_min(0).prod(-1)
    return intersection / (first_area + second_area - intersection).clamp_min(1e-8)


def _cxcywh_pixels(box: torch.Tensor, width: int, height: int) -> list[float]:
    cx, cy, bw, bh = box.detach().float().cpu().tolist()
    return [(cx - bw / 2) * width, (cy - bh / 2) * height,
            (cx + bw / 2) * width, (cy + bh / 2) * height]


def _selected_mask_logits(
    pred_masks: torch.Tensor | dict[str, torch.Tensor],
    batch_index: int,
    query_indices: torch.Tensor,
) -> torch.Tensor:
    """Return selected mask logits for dense or RF-DETR sparse mask outputs."""
    if torch.is_tensor(pred_masks):
        return pred_masks[batch_index, query_indices]
    spatial = pred_masks["spatial_features"][batch_index]
    queries = pred_masks["query_features"][batch_index, query_indices]
    return torch.einsum("chw,nc->nhw", spatial, queries) + pred_masks["bias"]


def _save_batch_figures(
    model: PCSDetector,
    output: dict[str, Any],
    targets: list[dict[str, torch.Tensor]],
    batch: dict[str, Any],
    dataset: CocoFewShotEpisodeDataset,
    mode: str,
    figure_dir: Path,
    step: int,
    max_figures: int,
    plot_mode: str,
    extra_predictions: int,
) -> None:
    if max_figures <= 0:
        return
    figure_dir.mkdir(parents=True, exist_ok=True)
    scores = output["pred_logits"].sigmoid()[..., 0]
    if "presence_logit_dec" in output:
        scores = scores * output["presence_logit_dec"].sigmoid()[:, None]
    boxes = output["pred_boxes"]
    plotted = 0
    for index in range(len(targets)):
        if plotted >= max_figures:
            break
        query_id = int(batch["query_image_ids"][index])
        query_meta = dataset.coco.loadImgs([query_id])[0]
        query_image = Image.open(dataset.image_dir / query_meta["file_name"]).convert("RGB")
        width, height = query_image.size
        target_boxes = [_cxcywh_pixels(box, width, height) for box in targets[index]["boxes"]]
        target_count = len(target_boxes)
        prediction_count = min(
            scores.shape[1], max(2, target_count) + max(0, extra_predictions)
        )
        prediction_scores, prediction_indices = scores[index].topk(prediction_count)
        prediction_boxes = [
            _cxcywh_pixels(boxes[index, query_index], width, height)
            for query_index in prediction_indices
        ]
        prediction_scores = prediction_scores.detach().cpu().tolist()
        support_examples = []
        if mode == "same_image":
            prompt_mask = ~batch["prompt_padding_mask"][index]
            prompt_boxes = batch["support_boxes"][index, prompt_mask]
            prompt_labels = batch["prompt_labels"][index, prompt_mask]
            for prompt_index, (prompt_box, prompt_label) in enumerate(
                zip(prompt_boxes, prompt_labels, strict=True)
            ):
                label = "positive" if int(prompt_label) else "negative"
                support_examples.append((
                    query_id, query_image, _cxcywh_pixels(prompt_box, width, height),
                    f"Same-image {label} prompt {prompt_index + 1}",
                ))
        else:
            for shot, support_id in enumerate(batch["support_image_ids"][index]):
                support_meta = dataset.support_coco.loadImgs([int(support_id)])[0]
                support_image = Image.open(
                    dataset.support_image_dir / support_meta["file_name"]
                ).convert("RGB")
                valid = ~batch["support_padding_mask"][index, shot]
                for prompt_index in torch.where(valid)[0].tolist():
                    prompt_label = int(batch["support_labels"][index, shot, prompt_index])
                    label = "positive" if prompt_label else "negative"
                    support_examples.append((
                        int(support_id), support_image,
                        _cxcywh_pixels(
                            batch["support_boxes"][index, shot, prompt_index],
                            *support_image.size,
                        ),
                        f"Support {shot + 1} | {label} prompt",
                    ))
        category_id = batch["category_ids"][index]
        _save_query_support_figure(
            figure_dir / f"episode_{plotted:03d}_{query_id}.png",
            f"Category {category_id} | {mode} | {target_count} GT instances",
            query_image,
            target_boxes,
            prediction_boxes,
            prediction_scores,
            support_examples,
            plot_mode=plot_mode,
            red_prediction_count=min(target_count, len(prediction_boxes)),
        )

        if bool(dataset.include_masks) and "pred_masks" in output and "masks" in targets[index]:
            pred_masks = output["pred_masks"]
            matches = model.matcher(output, targets)
            source_ids, target_ids = matches[index]
            if source_ids.numel():
                mask_dir = figure_dir / "masks"
                mask_dir.mkdir(exist_ok=True)
                fig, axes = plt.subplots(1, 2, figsize=(11, 5))
                axes[0].imshow(query_image)
                axes[0].set_title("Original query")
                axes[1].imshow(query_image)
                axes[1].set_title("Matched prediction and GT masks")
                for source_id, target_id in zip(source_ids[:3], target_ids[:3], strict=True):
                    truth = targets[index]["masks"][target_id].float()[None, None]
                    truth = F.interpolate(truth, size=(height, width), mode="nearest")[0, 0]
                    pred_logits = _selected_mask_logits(
                        pred_masks, index, source_id.reshape(1)
                    )
                    pred = F.interpolate(
                        pred_logits[:, None], size=(height, width),
                        mode="bilinear", align_corners=False,
                    )[0, 0].sigmoid() > 0.5
                    truth_np = truth.cpu().numpy() > 0.5
                    pred_np = pred.cpu().numpy()
                    axes[1].imshow(np.ma.masked_where(~truth_np, truth_np), cmap="winter", alpha=0.30)
                    axes[1].imshow(np.ma.masked_where(~pred_np, pred_np), cmap="autumn", alpha=0.30)
                for axis in axes:
                    axis.axis("off")
                fig.tight_layout()
                fig.savefig(mask_dir / f"episode_{plotted:03d}_{query_id}.png", dpi=120)
                plt.close(fig)
        plotted += 1


@torch.inference_mode()
def evaluate(
    model: PCSDetector,
    loader: DataLoader,
    dataset: CocoFewShotEpisodeDataset,
    mode: str,
    device: torch.device,
    include_prompt_box: bool,
    figure_dir: Path | None = None,
    figure_limit: int = 0,
    plot_mode: str = "compact",
    extra_predictions: int = 0,
    step: int = 0,
    conditioning_mode: str = "visual_prompt",
    evaluator: FSODEvaluator | None = None,
) -> dict[str, float | int | None]:
    was_training = model.training
    model.eval()
    loss_sums: dict[str, float] = {}
    box_iou_sum = box_iou_count = 0.0
    mask_intersection = mask_union = 0.0
    episodes = target_count = empty_episodes = 0
    presence_correct = presence_count = 0
    plotted = 0
    for raw in tqdm(loader, desc="eval", leave=False):
        batch = _move_batch(raw, device)
        if mode == "same_image":
            output = model(
                batch["query_images"], batch["support_boxes"],
                batch["prompt_labels"], batch["prompt_padding_mask"],
                class_ids=class_ids_for_mode(batch, device, conditioning_mode),
            )
            targets = _same_image_targets(batch, device, include_prompt_box)
        else:
            output = model.forward_support_query(
                batch["support_images"], batch["support_boxes"], batch["query_images"],
                batch["support_labels"], batch["support_padding_mask"],
                class_ids=class_ids_for_mode(batch, device, conditioning_mode),
            )
            targets = _query_targets(batch, device)
        losses = model.losses(output, targets, batch["present"].reshape(-1))
        presence_target = batch["present"].reshape(-1) > 0.5
        presence_prediction = output["presence_logit_dec"].sigmoid() > 0.5
        presence_correct += int((presence_prediction == presence_target).sum())
        presence_count += int(presence_target.numel())
        weighted = _weighted_loss(model, losses)
        batch_size = len(targets)
        loss_sums["loss_total"] = loss_sums.get("loss_total", 0.0) + float(weighted) * batch_size
        for key, value in losses.items():
            loss_sums[key] = loss_sums.get(key, 0.0) + float(value) * batch_size
        matches = model.matcher(output, targets)
        for index, (source_ids, target_ids) in enumerate(matches):
            count = len(targets[index]["boxes"])
            episodes += 1
            target_count += count
            empty_episodes += int(count == 0)
            if source_ids.numel():
                ious = _box_iou(output["pred_boxes"][index, source_ids], targets[index]["boxes"][target_ids])
                box_iou_sum += float(ious.sum())
                box_iou_count += ious.numel()
                pred_masks = output.get("pred_masks")
                if pred_masks is not None and "masks" in targets[index]:
                    truth = targets[index]["masks"][target_ids].bool()
                    predicted = F.interpolate(
                        _selected_mask_logits(pred_masks, index, source_ids)[:, None],
                        size=truth.shape[-2:],
                        mode="bilinear", align_corners=False,
                    )[:, 0].sigmoid() > 0.5
                    mask_intersection += float((predicted & truth).sum())
                    mask_union += float((predicted | truth).sum())
        if figure_dir is not None and plotted < figure_limit:
            figure_count = min(len(targets), figure_limit - plotted)
            _save_batch_figures(
                model, output, targets, batch, dataset, mode, figure_dir,
                step, figure_count, plot_mode, extra_predictions,
            )
            plotted += figure_count

        if evaluator is not None and "query_image_ids" in batch and len(batch["query_image_ids"]) > 0:
            meta_list = dataset.coco.loadImgs([int(qid) for qid in batch["query_image_ids"]])
            meta_map = {int(m["id"]): (m["height"], m["width"]) for m in meta_list}
            image_sizes = [meta_map[int(qid)] for qid in batch["query_image_ids"]]
            evaluator.add_predictions(
                image_ids=batch["query_image_ids"],
                category_ids=batch["category_ids"],
                output=output,
                image_sizes=image_sizes,
                include_masks=bool(dataset.include_masks),
            )

    if was_training:
        model.train()
    denominator = max(1, episodes)
    res: dict[str, float | int | None | dict[str, Any]] = {
        **{key: value / denominator for key, value in loss_sums.items()},
        "matched_box_iou": box_iou_sum / box_iou_count if box_iou_count else None,
        "matched_mask_iou": mask_intersection / mask_union if mask_union else None,
        "matched_boxes": int(box_iou_count),
        "presence_accuracy": presence_correct / max(1, presence_count),
        "target_boxes": int(target_count),
        "episodes": int(episodes),
        "empty_target_episodes": int(empty_episodes),
    }
    if evaluator is not None:
        coco_metrics = evaluator.evaluate()
        res["coco_eval"] = coco_metrics
        for iou_type, stats in coco_metrics.items():
            if isinstance(stats, dict):
                for metric_name, val in stats.items():
                    res[f"{iou_type}_{metric_name}"] = val
    return res


def _build_evaluator(cfg: dict[str, Any], split: str) -> FSODEvaluator | None:
    if not bool(cfg.get(f"{split}_coco_eval", cfg.get("compute_coco_eval", False))):
        return None
    annotation_file = cfg.get(f"{split}_annotation_file")
    if not annotation_file or not Path(annotation_file).exists():
        return None
    novel_ids = set(cfg.get("novel_category_ids", []))
    base_ids = set(cfg.get("base_category_ids", []))
    iou_types = ("bbox", "segm") if bool(cfg.get("segmentation", True)) else ("bbox",)
    return FSODEvaluator(
        annotation_file,
        novel_cat_ids=novel_ids,
        base_cat_ids=base_ids,
        iou_types=iou_types,
    )


def _lr_at(cfg: dict[str, Any], step: int, total_steps: int) -> float:
    base_lr = float(cfg.get("lr", 1e-4))
    warmup = cfg.get("warmup_steps", 0.05)
    warmup_steps = int(round(float(warmup) * total_steps)) if isinstance(warmup, float) else int(warmup)
    warmup_steps = min(max(0, warmup_steps), max(0, total_steps - 1))
    start_factor = float(cfg.get("warmup_start_factor", 0.1))
    min_ratio = float(cfg.get("min_lr_ratio", 0.01))
    if warmup_steps and step < warmup_steps:
        return base_lr * (start_factor + (1.0 - start_factor) * step / warmup_steps)
    if cfg.get("lr_scheduler", "cosine") == "constant":
        return base_lr
    progress = min(1.0, max(0.0, (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)))
    return base_lr * (min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def _save_curves(records: list[dict[str, Any]], output_dir: Path) -> None:
    if not records:
        return

    # ── CSV export ──────────────────────────────────────────────────────
    columns = list(dict.fromkeys(key for row in records for key in row))
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)

    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    steps = [row["step"] for row in records]

    # Helper: extract a metric series, return None if entirely missing.
    def _series(key: str) -> list[float] | None:
        vals = [row.get(key) for row in records]
        if all(v is None for v in vals):
            return None
        return [float(v) if v is not None else float("nan") for v in vals]

    # Style constants matching the reference RF-DETR visualizations.
    COLORS = {
        "train_loss": "#1f77b4",
        "validation_loss": "#ff7f0e",
        "box_iou": "#2ca02c",
        "mask_iou": "#d62728",
        # mAP colours
        "bbox_mAP": "#ff7f0e",
        "bbox_mAP_50": "#ff7f0e",
        "bbox_mAP_75": "#8c564b",
        "bbox_novel_mAP": "#2ca02c",
        "bbox_novel_mAP_50": "#2ca02c",
        "bbox_novel_mAP_75": "#17becf",
        "bbox_base_mAP": "#9467bd",
        "bbox_base_mAP_50": "#9467bd",
        "bbox_base_mAP_75": "#7f7f7f",
        "segm_mAP": "#ff7f0e",
        "segm_mAP_50": "#ff7f0e",
        "segm_mAP_75": "#8c564b",
        "segm_novel_mAP": "#2ca02c",
        "segm_novel_mAP_50": "#2ca02c",
        "segm_novel_mAP_75": "#17becf",
        "segm_base_mAP": "#9467bd",
        "segm_base_mAP_50": "#9467bd",
        "segm_base_mAP_75": "#7f7f7f",
    }

    # ── 1. Loss metrics  ───────────────────────────────────────────────
    fig_loss, ax_loss = plt.subplots(figsize=(12, 4.5))
    for key, label in [("train_loss", "train/loss"), ("validation_loss", "val/loss")]:
        s = _series(key)
        if s is not None:
            ax_loss.plot(steps, s, label=label, color=COLORS.get(key))
    ax_loss.set_xlabel("Optimizer step")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss", fontweight="bold")
    ax_loss.grid(True, alpha=0.25)
    ax_loss.legend(loc="upper right")
    fig_loss.suptitle("RF-DETR PCS Loss Metrics", fontsize=14, fontweight="bold")
    fig_loss.tight_layout()
    fig_loss.savefig(vis_dir / "loss_metrics.png", dpi=140)
    plt.close(fig_loss)

    # ── 2. Training curves (loss + IoU)  ───────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(steps, [row.get("train_loss", float("nan")) for row in records], label="train loss")
    axes[0].plot(steps, [row.get("validation_loss", float("nan")) for row in records], label="validation loss")
    axes[0].set_ylabel("Weighted loss")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].plot(steps, [row.get("validation_box_iou", float("nan")) for row in records], label="validation box IoU")
    if any(row.get("validation_mask_iou") is not None for row in records):
        axes[1].plot(steps, [row.get("validation_mask_iou", float("nan")) for row in records], label="validation mask IoU")
    axes[1].set_xlabel("Optimizer step")
    axes[1].set_ylabel("IoU")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=140)
    plt.close(fig)

    # ── 3. COCOeval mAP metrics  ───────────────────────────────────────
    # Determine which iou_types actually have data.
    has_bbox = _series("bbox_mAP_50") is not None
    has_segm = _series("segm_mAP_50") is not None
    if not has_bbox and not has_segm:
        return  # No COCOeval metrics recorded yet.

    iou_types_present = []
    if has_bbox:
        iou_types_present.append("bbox")
    if has_segm:
        iou_types_present.append("segm")

    n_types = len(iou_types_present)

    # Detection mAP figure: 2 rows (mAP@50 / novel-base; mAP@50:95 / mAP@75)
    fig_map, axes_map = plt.subplots(2, n_types, figsize=(7 * n_types, 8), squeeze=False)

    for col, iou_type in enumerate(iou_types_present):
        type_label = "BBox" if iou_type == "bbox" else "Segm"

        # ── Row 0: mAP@0.50 (all, novel, base) ─────────────────────
        ax = axes_map[0, col]
        for key, label, ls in [
            (f"{iou_type}_mAP_50", f"val/{type_label} mAP@50", "-"),
            (f"{iou_type}_novel_mAP_50", f"val/{type_label} novel mAP@50", "--"),
            (f"{iou_type}_base_mAP_50", f"val/{type_label} base mAP@50", "-."),
        ]:
            s = _series(key)
            if s is not None:
                ax.plot(steps, s, label=label, linestyle=ls, color=COLORS.get(key))
        ax.set_title(f"{type_label} Detection AP@0.50", fontweight="bold")
        ax.set_ylabel("mAP")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

        # ── Row 1: mAP@50:95 and mAP@75 ───────────────────────────
        ax2 = axes_map[1, col]
        for key, label, ls in [
            (f"{iou_type}_mAP", f"val/{type_label} mAP@50:95", "-"),
            (f"{iou_type}_mAP_75", f"val/{type_label} mAP@75", "--"),
            (f"{iou_type}_novel_mAP", f"val/{type_label} novel mAP@50:95", ":"),
            (f"{iou_type}_novel_mAP_75", f"val/{type_label} novel mAP@75", "-."),
            (f"{iou_type}_base_mAP", f"val/{type_label} base mAP@50:95", (0, (5, 5))),
            (f"{iou_type}_base_mAP_75", f"val/{type_label} base mAP@75", (0, (5, 3, 1, 3))),
        ]:
            s = _series(key)
            if s is not None:
                ax2.plot(steps, s, label=label, linestyle=ls, color=COLORS.get(key))
        ax2.set_title(f"{type_label} Detection AP@[50:95, 75]", fontweight="bold")
        ax2.set_xlabel("Optimizer step")
        ax2.set_ylabel("mAP")
        ax2.grid(True, alpha=0.25)
        ax2.legend(fontsize=8)

    fig_map.suptitle("RF-DETR PCS mAP Metrics", fontsize=14, fontweight="bold")
    fig_map.tight_layout()
    fig_map.savefig(vis_dir / "map_metrics.png", dpi=140)
    plt.close(fig_map)

    # ── 4. Combined training_metrics (reference-style 6-panel) ─────
    n_panels = 2 + (2 * n_types)  # loss, IoU, + 2 per iou_type
    n_cols_grid = 2
    n_rows_grid = math.ceil(n_panels / n_cols_grid)
    fig_all, ax_all = plt.subplots(n_rows_grid, n_cols_grid, figsize=(14, 4.5 * n_rows_grid))
    ax_flat = ax_all.flatten()
    panel = 0

    # Panel: Loss
    ax_flat[panel].plot(steps, [row.get("train_loss", float("nan")) for row in records], label="train/loss", linestyle=":", color="#1f77b4")
    ax_flat[panel].plot(steps, [row.get("validation_loss", float("nan")) for row in records], label="val/loss", color="#ff7f0e")
    ax_flat[panel].set_title("Loss", fontweight="bold")
    ax_flat[panel].set_ylabel("Loss")
    ax_flat[panel].grid(True, alpha=0.25)
    ax_flat[panel].legend(fontsize=8)
    panel += 1

    # Panel: IoU
    ax_flat[panel].plot(steps, [row.get("validation_box_iou", float("nan")) for row in records], label="val/box_IoU", color="#2ca02c")
    s_mask = _series("validation_mask_iou")
    if s_mask is not None:
        ax_flat[panel].plot(steps, s_mask, label="val/mask_IoU", color="#d62728")
    ax_flat[panel].set_title("Matched IoU", fontweight="bold")
    ax_flat[panel].set_ylabel("IoU")
    ax_flat[panel].set_ylim(0, 1)
    ax_flat[panel].grid(True, alpha=0.25)
    ax_flat[panel].legend(fontsize=8)
    panel += 1

    # Panels per iou_type: mAP@50 and mAP@50:95
    for iou_type in iou_types_present:
        type_label = "BBox" if iou_type == "bbox" else "Segm"

        ax_cur = ax_flat[panel]
        for key, label, ls in [
            (f"{iou_type}_mAP_50", f"val/{type_label} mAP@50", "-"),
            (f"{iou_type}_novel_mAP_50", f"val/{type_label} novel mAP@50", "--"),
            (f"{iou_type}_base_mAP_50", f"val/{type_label} base mAP@50", "-."),
        ]:
            s = _series(key)
            if s is not None:
                ax_cur.plot(steps, s, label=label, linestyle=ls)
        ax_cur.set_title(f"{type_label} Detection AP@0.50", fontweight="bold")
        ax_cur.set_ylabel("Detection AP@0.50")
        ax_cur.grid(True, alpha=0.25)
        ax_cur.legend(fontsize=7)
        panel += 1

        ax_cur2 = ax_flat[panel]
        for key, label, ls in [
            (f"{iou_type}_mAP", f"val/{type_label} mAP@50:95", "-"),
            (f"{iou_type}_novel_mAP", f"val/{type_label} novel mAP@50:95", "--"),
            (f"{iou_type}_base_mAP", f"val/{type_label} base mAP@50:95", "-."),
        ]:
            s = _series(key)
            if s is not None:
                ax_cur2.plot(steps, s, label=label, linestyle=ls)
        ax_cur2.set_title(f"{type_label} Detection AP@0.50:0.95", fontweight="bold")
        ax_cur2.set_xlabel("Optimizer step")
        ax_cur2.set_ylabel("Detection AP@0.50:0.95")
        ax_cur2.grid(True, alpha=0.25)
        ax_cur2.legend(fontsize=7)
        panel += 1

    # Hide any unused panels
    for idx in range(panel, len(ax_flat)):
        ax_flat[idx].set_visible(False)

    fig_all.suptitle("RF-DETR PCS Training Metrics", fontsize=14, fontweight="bold")
    fig_all.tight_layout()
    fig_all.savefig(vis_dir / "training_metrics.png", dpi=140)
    plt.close(fig_all)


def _save_checkpoint(
    path: Path, model: PCSDetector, optimizer: torch.optim.Optimizer,
    step: int, best_score: float, cfg: dict[str, Any],
) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "step": step, "best_score": best_score, "mode": cfg["mode"],
        "config": _jsonable(cfg),
    }, temporary_path)
    temporary_path.replace(path)


def _retain_checkpoints(output_dir: Path, maximum: int, protected: set[Path]) -> None:
    checkpoints = sorted(output_dir.glob("checkpoint_step_*.pt"))
    keep = set(checkpoints[-max(1, maximum):]) | protected
    for checkpoint in checkpoints:
        if checkpoint not in keep:
            checkpoint.unlink()


def _build_model(cfg: dict[str, Any], device: torch.device, output_dir: Path) -> PCSDetector:
    model_config = build_base_model_config(cfg, device)
    return PCSDetector.from_config(
        model_config, TrainConfig(dataset_dir=".", output_dir=str(output_dir)),
        num_prompt_classes=int(cfg.get("num_prompt_classes", 91)),
        init_class_embeddings_from_detector=bool(cfg.get("init_class_embeddings_from_detector", True)),
        use_class_embeddings=uses_class_embeddings(cfg),
    ).to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()
    namespace: dict[str, Any] = {}
    exec(compile(args.config.read_text(), str(args.config), "exec"), namespace)
    cfg: dict[str, Any] = dict(namespace["cfg"])
    # Validate mode/curriculum eagerly so bad schedules fail before model setup.
    conditioning_schedule(cfg)
    conditioning_mode_at(cfg, 0)
    mode = str(cfg.get("mode", "same_image"))
    if mode not in {"same_image", "support_query"}:
        parser.error("config mode must be 'same_image' or 'support_query'")
    cfg["mode"] = mode
    if args.resume is not None:
        cfg["resume_checkpoint"] = args.resume
    steps = int(cfg.get("steps", 10_000))
    eval_every = int(cfg.get("eval_every", 500))
    batch_repeat = int(cfg.get("batch_repeat", 1))
    if steps < 1 or eval_every < 1 or batch_repeat < 1:
        parser.error("steps, eval_every, and batch_repeat must be positive")
    seed = int(cfg.get("seed", 7))
    seed_everything(seed)
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(cfg["output_dir"])
    figure_dir = Path(cfg.get("figure_dir", output_dir / "figures"))
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    segmentation = bool(cfg.get("segmentation", True))

    train_dataset, validation_dataset, test_dataset = _build_datasets(cfg, mode)
    collate_fn = collate_few_shot if mode == "same_image" else collate_support_query
    batch_size = int(cfg.get("batch_size", 4 if mode == "same_image" else 2))
    num_workers = int(cfg.get("num_workers", 4))
    generator = torch.Generator().manual_seed(seed + 1)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False,
        num_workers=num_workers, pin_memory=device.type == "cuda", collate_fn=collate_fn,
        generator=generator,
    )
    val_subset = _limit_dataset(
        validation_dataset, int(cfg.get("validation_episodes", 0)), seed + 17
    )
    test_subset = _limit_dataset(test_dataset, int(cfg.get("test_episodes", 0)), seed + 19)
    eval_batch_size = int(cfg.get("eval_batch_size", batch_size))
    val_loader = DataLoader(
        val_subset, batch_size=eval_batch_size, shuffle=False, num_workers=0,
        collate_fn=collate_fn, pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_subset, batch_size=eval_batch_size, shuffle=False, num_workers=0,
        collate_fn=collate_fn, pin_memory=device.type == "cuda",
    )
    sanity_count = int(cfg.get("train_sanity_samples", 8))
    sanity_indices = random.Random(seed + 23).sample(
        range(len(train_dataset)), min(len(train_dataset), sanity_count)
    )
    sanity_loader = DataLoader(
        Subset(train_dataset, sanity_indices), batch_size=eval_batch_size,
        shuffle=False, num_workers=0, collate_fn=collate_fn,
    )
    val_plot_count = int(cfg.get("validation_sanity_samples", 8))
    print(
        f"PCS mode={mode} | train episodes={len(train_dataset)} | "
        f"validation episodes={len(val_subset)} | test episodes={len(test_subset)} | "
        f"segmentation={segmentation} | negative_prompt_ratio={cfg.get('negative_prompt_ratio', 0.0)}"
    )

    model = _build_model(cfg, device, output_dir)
    if cfg.get("init_checkpoint"):
        initialization = torch.load(cfg["init_checkpoint"], map_location="cpu", weights_only=False)
        state = initialization.get("model", initialization)
        missing, skipped = model.load_compatible_state_dict(state)
        print(
            f"initialized PCS weights from {cfg['init_checkpoint']}; "
            f"new/missing keys={missing}; skipped keys={skipped}"
        )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg.get("lr", 1e-4)),
        weight_decay=float(cfg.get("weight_decay", 0.05)),
    )
    start_step = 0
    best_score = float("-inf")
    if cfg.get("resume_checkpoint"):
        checkpoint = torch.load(cfg["resume_checkpoint"], map_location=device, weights_only=False)
        missing, skipped = model.load_compatible_state_dict(checkpoint["model"])
        if missing or skipped:
            print(f"checkpoint PCS migration: initialized missing keys={missing}; skipped keys={skipped}")
        if "optimizer" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
            except (ValueError, KeyError) as exc:
                print(f"optimizer state not reusable after PCS head changes; starting fresh ({exc})")
        start_step = int(checkpoint.get("step", 0))
        best_score = float(checkpoint.get("best_score", best_score))
        print(f"resumed {cfg['resume_checkpoint']} at step {start_step}")
    if start_step >= steps:
        parser.error(f"resume checkpoint step {start_step} must be less than configured steps {steps}")

    metrics_path = output_dir / "metrics.json"
    train_log_path = output_dir / "train.jsonl"
    evaluations: list[dict[str, Any]] = []
    curve_records: list[dict[str, Any]] = []
    if cfg.get("resume_checkpoint") and metrics_path.exists():
        prior = json.loads(metrics_path.read_text())
        evaluations = prior.get("evaluations", [])
        curve_records = prior.get("curve_records", [])

    loader_iter = iter(train_loader)
    current_batch = None
    interval_losses: dict[str, float] = {}
    interval_steps = 0
    eval_train_loss_sum = 0.0
    eval_train_loss_steps = 0
    log_every = int(cfg.get("log_every", 100))
    support_loss_weight = float(cfg.get("support_loss_weight", 0.5))
    include_prompt_box = bool(cfg.get("include_prompt_box_in_loss", True))
    plot_mode = str(cfg.get("inference_plot_mode", "compact"))
    extra_predictions = int(cfg.get("inference_extra_predictions", 0))
    best_path = output_dir / "best.pt"
    model.train()
    progress = tqdm(range(start_step, steps), desc=f"PCS train ({mode})", unit="step")
    for step_index in progress:
        active_conditioning_mode = conditioning_mode_at(cfg, step_index)
        lr = _lr_at(cfg, step_index, steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        if current_batch is None or step_index % batch_repeat == 0:
            try:
                raw_batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)
                raw_batch = next(loader_iter)
            current_batch = _move_batch(raw_batch, device)
        batch = current_batch
        if bool(cfg.get("use_gpu_augmentation", False)):
            batch = _apply_gpu_augmentation(batch, cfg, device)
        if mode == "same_image":
            output = model(
                batch["query_images"], batch["support_boxes"],
                batch["prompt_labels"], batch["prompt_padding_mask"],
                class_ids=class_ids_for_mode(batch, device, active_conditioning_mode),
            )
            query_targets = _same_image_targets(batch, device, include_prompt_box)
            query_losses = model.losses(output, query_targets, batch["present"].reshape(-1))
            total_loss = _weighted_loss(model, query_losses)
            support_losses = None
        else:
            output = model.forward_support_query(
                batch["support_images"], batch["support_boxes"], batch["query_images"],
                batch["support_labels"], batch["support_padding_mask"],
                return_support_outputs=support_loss_weight > 0,
                class_ids=class_ids_for_mode(batch, device, active_conditioning_mode),
            )
            support_output = output.pop("support_outputs", None)
            query_targets = _query_targets(batch, device)
            query_losses = model.losses(output, query_targets, batch["present"].reshape(-1))
            total_loss = _weighted_loss(model, query_losses)
            support_losses = None
            if support_output is not None:
                support_targets = _support_targets(batch, device, include_prompt_box)
                support_losses = model.losses(support_output, support_targets)
                total_loss = total_loss + support_loss_weight * _weighted_loss(model, support_losses)

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("clip_grad_norm", 0.1)))
        optimizer.step()
        global_step = step_index + 1
        scalar_losses = {"loss_total": float(total_loss.detach())}
        eval_train_loss_sum += scalar_losses["loss_total"]
        eval_train_loss_steps += 1
        scalar_losses.update({key: float(value.detach()) for key, value in query_losses.items()})
        if support_losses is not None:
            scalar_losses.update({f"support_{key}": float(value.detach()) for key, value in support_losses.items()})
        for key, value in scalar_losses.items():
            interval_losses[key] = interval_losses.get(key, 0.0) + value
        interval_steps += 1
        progress.set_postfix(
            loss=f"{float(total_loss.detach()):.3f}",
            box=f"{float(query_losses.get('loss_bbox', total_loss).detach()):.3f}",
            lr=f"{lr:.1e}",
        )
        if global_step % log_every == 0 or global_step == steps:
            record = {"step": global_step, "lr": lr}
            record.update({key: value / interval_steps for key, value in interval_losses.items()})
            with train_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
                handle.flush()
            tqdm.write("train " + " ".join(
                f"{key}={value:.4f}" for key, value in record.items()
                if key not in {"step", "lr"}
            ) + f" step={global_step}/{steps} lr={lr:.2e}")
            interval_losses = {}
            interval_steps = 0

        if global_step % eval_every == 0 or global_step == steps:
            # Keep a resumable snapshot before validation so an evaluation or
            # plotting error cannot discard the just-finished training interval.
            _save_checkpoint(
                output_dir / "last.pt", model, optimizer, global_step, best_score, cfg
            )
            val_fig_dir = None
            if bool(cfg.get("validation_sanity", True)):
                val_fig_dir = figure_dir / "validation_seen" / f"step_{global_step:08d}"
            val_evaluator = _build_evaluator(cfg, "validation")
            val_result = evaluate(
                model, val_loader, validation_dataset, mode, device,
                include_prompt_box, val_fig_dir,
                val_plot_count if val_fig_dir is not None else 0,
                plot_mode, extra_predictions, global_step,
                conditioning_mode_at(cfg, global_step - 1),
                evaluator=val_evaluator,
            )
            sanity_result = None
            if bool(cfg.get("train_sanity", True)):
                sanity_result = evaluate(
                    model, sanity_loader, train_dataset, mode, device,
                    include_prompt_box,
                    figure_dir / "train_sanity" / f"step_{global_step:08d}",
                    sanity_count, plot_mode, extra_predictions, global_step,
                    conditioning_mode_at(cfg, global_step - 1),
                )
            train_loss = eval_train_loss_sum / max(1, eval_train_loss_steps)
            eval_train_loss_sum = 0.0
            eval_train_loss_steps = 0
            eval_record = {
                "step": global_step,
                "lr": lr,
                "validation": val_result,
                "train_sanity": sanity_result,
            }
            evaluations.append(eval_record)
            curve_record = {
                "step": global_step,
                "lr": lr,
                "train_loss": train_loss,
                "validation_loss": val_result.get("loss_total"),
                "validation_box_iou": val_result.get("matched_box_iou"),
                "validation_mask_iou": val_result.get("matched_mask_iou"),
                "train_sanity_box_iou": sanity_result.get("matched_box_iou") if sanity_result else None,
            }
            # Capture COCOeval metrics for curve plots
            for coco_key in (
                "bbox_mAP", "bbox_mAP_50", "bbox_mAP_75",
                "bbox_mAP_s", "bbox_mAP_m", "bbox_mAP_l",
                "bbox_novel_mAP", "bbox_novel_mAP_50", "bbox_novel_mAP_75",
                "bbox_base_mAP", "bbox_base_mAP_50", "bbox_base_mAP_75",
                "segm_mAP", "segm_mAP_50", "segm_mAP_75",
                "segm_mAP_s", "segm_mAP_m", "segm_mAP_l",
                "segm_novel_mAP", "segm_novel_mAP_50", "segm_novel_mAP_75",
                "segm_base_mAP", "segm_base_mAP_50", "segm_base_mAP_75",
            ):
                if coco_key in val_result:
                    curve_record[coco_key] = val_result[coco_key]
            curve_records.append(curve_record)
            metrics_path.write_text(json.dumps({
                "config": _jsonable(cfg), "evaluations": evaluations,
                "curve_records": curve_records,
            }, indent=2))
            _save_curves(curve_records, output_dir)
            score_key = str(cfg.get("selection_metric", "matched_box_iou"))
            score = val_result.get(score_key)
            if score is None:
                score = -float(val_result.get("loss_total", float("inf")))
            score = float(score)
            improved = score > best_score
            if improved:
                best_score = score
                _save_checkpoint(best_path, model, optimizer, global_step, best_score, cfg)
            checkpoint_path = output_dir / f"checkpoint_step_{global_step:08d}.pt"
            _save_checkpoint(checkpoint_path, model, optimizer, global_step, best_score, cfg)
            _save_checkpoint(output_dir / "last.pt", model, optimizer, global_step, best_score, cfg)
            protected = {best_path} if best_path.exists() else set()
            _retain_checkpoints(output_dir, int(cfg.get("max_checkpoints", 3)), protected)
            coco_msg = ""
            if "bbox_mAP_50" in val_result:
                coco_msg += f" box_mAP50={val_result['bbox_mAP_50']:.3f}"
            if "bbox_novel_mAP_50" in val_result:
                coco_msg += f" novel_mAP50={val_result['bbox_novel_mAP_50']:.3f}"
            if "segm_mAP_50" in val_result:
                coco_msg += f" segm_mAP50={val_result['segm_mAP_50']:.3f}"
            tqdm.write(
                f"eval step={global_step} val_box_iou={val_result.get('matched_box_iou')} "
                f"val_mask_iou={val_result.get('matched_mask_iou')}{coco_msg} "
                f"val_loss={val_result.get('loss_total'):.4f} best={best_score:.4f}"
            )

    if bool(cfg.get("evaluate_test_at_end", True)):
        test_checkpoint = best_path if best_path.exists() else output_dir / "last.pt"
        best_checkpoint = torch.load(test_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(best_checkpoint["model"])
        selected_step = int(best_checkpoint.get("step", steps))
        test_evaluator = _build_evaluator(cfg, "test")
        test_result = evaluate(
            model, test_loader, test_dataset, mode, device, include_prompt_box,
            figure_dir / "test" if bool(cfg.get("test_sanity", True)) else None,
            int(cfg.get("test_sanity_samples", 16)) if bool(cfg.get("test_sanity", True)) else 0,
            plot_mode, extra_predictions, selected_step,
            conditioning_mode_at(cfg, max(0, selected_step - 1)),
            evaluator=test_evaluator,
        )
        (output_dir / "test_metrics.json").write_text(json.dumps({
            "checkpoint": str(test_checkpoint), "step": selected_step, "test": test_result,
        }, indent=2))
        print(f"test metrics (best validation checkpoint, step {selected_step}): {test_result}")


if __name__ == "__main__":
    main()
