"""Overfit a tiny set of separate support/query episodes with batched encoding."""

from __future__ import annotations

import argparse
import json
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

from rfdetr_pcs.coco_episodes import CocoFewShotEpisodeDataset, collate_support_query
from rfdetr_pcs.config import TrainConfig
from rfdetr_pcs.conditioning import class_ids_for_mode, uses_class_embeddings
from rfdetr_pcs.model_presets import build_base_model_config
from rfdetr_pcs.models.pcs import PCSDetector
from rfdetr_pcs.utilities.box_ops import box_cxcywh_to_xyxy
from scripts.rfdetr_promptable.train_promptable_fewshot import _save_query_support_figure


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def target_dicts(batch: dict[str, Any], device: torch.device) -> list[dict[str, torch.Tensor]]:
    boxes = batch["query_boxes"].to(device)
    valid = batch["query_target_mask"].to(device)
    masks = batch.get("query_masks")
    if masks is not None:
        masks = masks.to(device)
    result = []
    for index in range(boxes.shape[0]):
        target: dict[str, torch.Tensor] = {
            "boxes": boxes[index, valid[index]],
            "labels": torch.zeros(int(valid[index].sum()), dtype=torch.long, device=device),
        }
        if masks is not None:
            target["masks"] = masks[index, valid[index]]
        result.append(target)
    return result


def support_target_dicts(
    batch: dict[str, Any], device: torch.device, include_prompt_box: bool = True
) -> list[dict[str, torch.Tensor]]:
    boxes = batch["support_target_boxes"].to(device).flatten(0, 1)
    valid = batch["support_target_mask"].to(device).flatten(0, 1)
    if not include_prompt_box and "support_target_prompt_mask" in batch:
        valid &= ~batch["support_target_prompt_mask"].to(device).flatten(0, 1)
    masks = batch.get("support_target_masks")
    if masks is not None:
        masks = masks.to(device).flatten(0, 1)
    result = []
    for index in range(boxes.shape[0]):
        target: dict[str, torch.Tensor] = {
            "boxes": boxes[index, valid[index]],
            "labels": torch.zeros(int(valid[index].sum()), dtype=torch.long, device=device),
        }
        if masks is not None:
            target["masks"] = masks[index, valid[index]]
        result.append(target)
    return result


def mask_logits(output: dict[str, Any]) -> torch.Tensor | None:
    masks = output.get("pred_masks")
    if isinstance(masks, dict):
        return torch.einsum("bchw,bqc->bqhw", masks["spatial_features"], masks["query_features"]) + masks["bias"]
    return masks if isinstance(masks, torch.Tensor) else None


@torch.no_grad()
def evaluate(
    model: PCSDetector,
    dataset: CocoFewShotEpisodeDataset,
    indices: list[int],
    device: torch.device,
    output_dir: Path,
    step: int,
    conditioning_mode: str = "visual_prompt",
    include_prompt_box: bool = True,
) -> dict[str, float | None]:
    model.eval()
    loader = DataLoader(Subset(dataset, indices), batch_size=2, shuffle=False, num_workers=0,
                        collate_fn=collate_support_query)
    weighted_total = support_loss_total = bbox_total = giou_total = ce_total = 0.0
    box_iou_sum = box_iou_count = 0.0
    support_box_iou_sum = support_box_iou_count = 0.0
    mask_intersection = mask_union = 0.0
    plotted = 0
    figure_dir = output_dir / "figures" / f"step_{step:06d}"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for raw in loader:
        batch = move_batch(raw, device)
        batch_size, query_count = batch["query_images"].shape[:2]
        output = model.forward_support_query(
            batch["support_images"], batch["support_boxes"], batch["query_images"],
            batch["support_labels"], batch["support_padding_mask"], return_support_outputs=True,
            class_ids=class_ids_for_mode(batch, device, conditioning_mode),
        )
        support_output = output.pop("support_outputs")
        targets = target_dicts(batch, device)
        losses = model.losses(output, targets, batch["present"].reshape(-1))
        support_targets = support_target_dicts(batch, device, include_prompt_box)
        support_losses = model.losses(support_output, support_targets)
        weighted_total += float(sum(
            value * model.criterion.weight_dict[name]
            for name, value in losses.items() if name in model.criterion.weight_dict
        )) * query_count
        support_loss_total += float(sum(
            value * model.criterion.weight_dict[name]
            for name, value in support_losses.items() if name in model.criterion.weight_dict
        )) / max(1, batch["support_images"].shape[1]) * query_count
        bbox_total += float(losses.get("loss_bbox", torch.tensor(0.0))) * query_count
        giou_total += float(losses.get("loss_giou", torch.tensor(0.0))) * query_count
        ce_total += float(losses.get("loss_ce", torch.tensor(0.0))) * query_count
        masks = mask_logits(output)
        matches = model.matcher(output, targets)
        for index, (source_ids, target_ids) in enumerate(matches):
            if source_ids.numel():
                pred = box_cxcywh_to_xyxy(output["pred_boxes"][index, source_ids]).clamp(0, 1)
                gt = box_cxcywh_to_xyxy(targets[index]["boxes"][target_ids]).clamp(0, 1)
                left_top = torch.maximum(pred[:, :2], gt[:, :2])
                right_bottom = torch.minimum(pred[:, 2:], gt[:, 2:])
                intersection = (right_bottom - left_top).clamp_min(0).prod(-1)
                pred_area = (pred[:, 2:] - pred[:, :2]).clamp_min(0).prod(-1)
                gt_area = (gt[:, 2:] - gt[:, :2]).clamp_min(0).prod(-1)
                ious = intersection / (pred_area + gt_area - intersection).clamp_min(1e-8)
                box_iou_sum += float(ious.sum())
                box_iou_count += ious.numel()
                if masks is not None and "masks" in targets[index]:
                    truth = targets[index]["masks"][target_ids].bool()
                    predicted = F.interpolate(masks[index, source_ids, None], size=truth.shape[-2:],
                                              mode="bilinear", align_corners=False)[:, 0].sigmoid() > 0.5
                    mask_intersection += float((predicted & truth).sum())
                    mask_union += float((predicted | truth).sum())
        support_matches = model.matcher(support_output, support_targets)
        for index, (source_ids, target_ids) in enumerate(support_matches):
            if not source_ids.numel():
                continue
            pred = box_cxcywh_to_xyxy(support_output["pred_boxes"][index, source_ids]).clamp(0, 1)
            gt = box_cxcywh_to_xyxy(support_targets[index]["boxes"][target_ids]).clamp(0, 1)
            left_top = torch.maximum(pred[:, :2], gt[:, :2])
            right_bottom = torch.minimum(pred[:, 2:], gt[:, 2:])
            intersection = (right_bottom - left_top).clamp_min(0).prod(-1)
            pred_area = (pred[:, 2:] - pred[:, :2]).clamp_min(0).prod(-1)
            gt_area = (gt[:, 2:] - gt[:, :2]).clamp_min(0).prod(-1)
            ious = intersection / (pred_area + gt_area - intersection).clamp_min(1e-8)
            support_box_iou_sum += float(ious.sum())
            support_box_iou_count += ious.numel()

        for index in range(batch_size):
            if plotted >= 4:
                continue
            query_id = int(batch["query_image_ids"][index])
            query_meta = dataset.coco.loadImgs([query_id])[0]
            query_image = Image.open(dataset.image_dir / query_meta["file_name"]).convert("RGB")
            query_width, query_height = query_image.size

            def to_pixels(box: torch.Tensor, width: int, height: int) -> list[float]:
                cx, cy, bw, bh = box.detach().float().cpu().tolist()
                return [(cx - bw / 2) * width, (cy - bh / 2) * height,
                        (cx + bw / 2) * width, (cy + bh / 2) * height]

            gt_boxes = [to_pixels(box, query_width, query_height) for box in targets[index]["boxes"]]
            scores = output["pred_logits"][index].sigmoid()[..., 0]
            if "presence_logit_dec" in output:
                scores = scores * output["presence_logit_dec"][index].sigmoid()
            shown_ids = scores.topk(min(2, scores.numel())).indices.tolist()
            pred_boxes = [to_pixels(output["pred_boxes"][index, item], query_width, query_height)
                          for item in shown_ids]
            pred_scores = [float(scores[item]) for item in shown_ids]
            support_examples = []
            for shot, support_id in enumerate(batch["support_image_ids"][index]):
                support_meta = dataset.support_coco.loadImgs([int(support_id)])[0]
                support_image = Image.open(dataset.support_image_dir / support_meta["file_name"]).convert("RGB")
                for prompt_index in range(batch["support_boxes"].shape[2]):
                    if batch["support_padding_mask"][index, shot, prompt_index]:
                        continue
                    prompt = to_pixels(
                        batch["support_boxes"][index, shot, prompt_index], *support_image.size
                    )
                    prompt_label = int(batch["support_labels"][index, shot, prompt_index])
                    label_name = "positive" if prompt_label else "negative"
                    support_examples.append((
                        int(support_id), support_image, prompt,
                        f"Support {shot + 1} | {label_name} prompt",
                    ))
            _save_query_support_figure(
                figure_dir / f"episode_{plotted:02d}_{query_id}.png",
                f"Category {batch['category_ids'][index]} | separate support -> query",
                query_image, gt_boxes, pred_boxes, pred_scores, support_examples,
                plot_mode="compact", red_prediction_count=len(pred_boxes),
            )
            plotted += 1
    model.train()
    count = max(len(indices), 1)
    return {
        "loss_total": weighted_total / count,
        "support_loss_total": support_loss_total / count,
        "loss_bbox": bbox_total / count,
        "loss_giou": giou_total / count,
        "loss_ce": ce_total / count,
        "matched_box_iou": box_iou_sum / box_iou_count if box_iou_count else None,
        "support_matched_box_iou": (
            support_box_iou_sum / support_box_iou_count if support_box_iou_count else None
        ),
        "matched_mask_iou": mask_intersection / mask_union if mask_union else None,
        "samples": float(len(indices)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    namespace: dict[str, Any] = {}
    exec(compile(Path(args.config).read_text(), args.config, "exec"), namespace)
    cfg = namespace["cfg"]
    seed = int(cfg.get("seed", 73))
    seed_all(seed)
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    resolution = int(cfg.get("resolution", 280))
    segmentation = bool(cfg.get("segmentation", True))
    model_config = build_base_model_config(cfg, device, segmentation=segmentation)
    model = PCSDetector.from_config(
        model_config, TrainConfig(dataset_dir=".", output_dir=str(output_dir)),
        num_prompt_classes=int(cfg.get("num_prompt_classes", 91)),
        init_class_embeddings_from_detector=bool(cfg.get("init_class_embeddings_from_detector", True)),
        use_class_embeddings=uses_class_embeddings(cfg),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 1e-4)), weight_decay=0.05)
    dataset = CocoFewShotEpisodeDataset(
        cfg["query_image_dir"], cfg["query_annotation_file"],
        support_image_dir=cfg["support_image_dir"], support_annotation_file=cfg["support_annotation_file"],
        resolution=resolution, episodes=int(cfg.get("episodes", 1000)),
        max_shots=int(cfg.get("shots", 2)), negative_ratio=0, seed=seed, training=True,
        negative_prompt_ratio=float(cfg.get("negative_prompt_ratio", 0.0)),
        include_masks=segmentation,
    )
    count = min(int(cfg.get("overfit_episodes", 4)), len(dataset))
    indices = list(range(count))
    loader = DataLoader(Subset(dataset, indices), batch_size=min(int(cfg.get("batch_size", 2)), count),
                        shuffle=True, num_workers=0, collate_fn=collate_support_query)
    iterator = iter(loader)
    metrics: list[dict[str, Any]] = []
    steps = int(cfg.get("steps", 1000))
    eval_every = int(cfg.get("eval_every", 100))
    model.train()
    progress = tqdm(range(1, steps + 1), desc=f"Separate support/query overfit: {count} episodes")
    for step in progress:
        try:
            raw = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            raw = next(iterator)
        batch = move_batch(raw, device)
        output = model.forward_support_query(
            batch["support_images"], batch["support_boxes"], batch["query_images"],
            batch["support_labels"], batch["support_padding_mask"], return_support_outputs=True,
            class_ids=class_ids_for_mode(batch, device, str(cfg.get("conditioning_mode", "visual_prompt"))),
        )
        support_output = output.pop("support_outputs")
        losses = model.losses(output, target_dicts(batch, device), batch["present"].reshape(-1))
        include_prompt_box = bool(cfg.get("include_prompt_box_in_loss", True))
        support_losses = model.losses(
            support_output, support_target_dicts(batch, device, include_prompt_box)
        )
        query_total = sum(value * model.criterion.weight_dict[name]
                          for name, value in losses.items() if name in model.criterion.weight_dict)
        support_total = sum(value * model.criterion.weight_dict[name]
                            for name, value in support_losses.items() if name in model.criterion.weight_dict)
        total = query_total + float(cfg.get("support_loss_weight", 0.5)) * support_total
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("clip_grad_norm", 0.1)))
        optimizer.step()
        progress.set_postfix(loss=f"{float(total.detach()):.3f}")
        if step % eval_every == 0 or step == steps:
            record = {"step": step, **evaluate(
                model, dataset, indices, device, output_dir, step,
                str(cfg.get("conditioning_mode", "visual_prompt")),
                bool(cfg.get("include_prompt_box_in_loss", True)),
            )}
            metrics.append(record)
            (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
            checkpoint_path = output_dir / f"checkpoint_step_{step:06d}.pt"
            torch.save({"model": model.state_dict(), "step": step}, checkpoint_path)
            max_checkpoints = max(1, int(cfg.get("max_checkpoints", 3)))
            checkpoints = sorted(output_dir.glob("checkpoint_step_*.pt"))
            for stale_checkpoint in checkpoints[:-max_checkpoints]:
                stale_checkpoint.unlink()
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot([row["step"] for row in metrics], [row["matched_box_iou"] for row in metrics], label="box IoU")
            ax.plot([row["step"] for row in metrics], [row["matched_mask_iou"] for row in metrics], label="mask IoU")
            ax.plot([row["step"] for row in metrics], [row["support_matched_box_iou"] for row in metrics],
                    label="support box IoU")
            ax.set(xlabel="optimizer step", ylabel="IoU", ylim=(0, 1))
            ax.grid(True, alpha=.25); ax.legend(); fig.tight_layout()
            fig.savefig(output_dir / "training_iou.png", dpi=140); plt.close(fig)
            progress.write(f"step={step} separate_support_query_metrics={record}")


if __name__ == "__main__":
    main()
