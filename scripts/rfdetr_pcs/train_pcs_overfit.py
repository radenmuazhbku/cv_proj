"""Curriculum overfit diagnostic for same-image RF-DETR PCS."""

from __future__ import annotations

import argparse
import json
import re
import random
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rfdetr_pcs.coco_episodes import CocoFewShotEpisodeDataset, collate_few_shot
from rfdetr_pcs.config import RFDETRBaseConfig, TrainConfig
from rfdetr_pcs.conditioning import class_ids_for_mode, conditioning_mode_at, uses_class_embeddings
from rfdetr_pcs.models.pcs import PCSDetector
from rfdetr_pcs.utilities.box_ops import box_cxcywh_to_xyxy
from scripts.rfdetr_promptable.train_promptable_fewshot import _save_query_support_figure


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def box_xyxy(box: torch.Tensor, width: int, height: int) -> tuple[float, float, float, float]:
    cx, cy, bw, bh = box.detach().float().cpu().tolist()
    return ((cx - bw / 2) * width, (cy - bh / 2) * height,
            bw * width, bh * height)


def _target_dicts(batch: dict[str, Any], device: torch.device, include_prompt: bool) -> list[dict[str, torch.Tensor]]:
    boxes = batch["query_boxes"].to(device)
    valid = batch["query_target_mask"].to(device)
    masks = batch.get("query_masks")
    if masks is not None:
        masks = masks.to(device)
    if include_prompt:
        prompt_boxes = batch["support_boxes"].to(device)
        prompt_valid = batch.get("prompt_target_mask")
        if prompt_valid is None:
            prompt_valid = batch["present"].reshape(-1, 1).expand(prompt_boxes.shape[:2]) > 0.5
        else:
            prompt_valid = prompt_valid.to(device=device, dtype=torch.bool)
        padding = batch.get("prompt_padding_mask")
        if padding is not None:
            prompt_valid &= ~padding.to(device=device, dtype=torch.bool)
        boxes = torch.cat([boxes, prompt_boxes], dim=1)
        valid = torch.cat([valid, prompt_valid], dim=1)
        if masks is not None:
            masks = torch.cat([masks, batch["prompt_masks"].to(device)], dim=1)
    targets = []
    for index in range(boxes.shape[0]):
        count = int(valid[index].sum())
        target = {
            "boxes": boxes[index, valid[index]],
            "labels": torch.zeros(count, dtype=torch.long, device=device),
        }
        if masks is not None:
            target["masks"] = masks[index, valid[index]]
        targets.append(target)
    return targets


@torch.no_grad()
def save_diagnostics(
    model: PCSDetector,
    dataset: CocoFewShotEpisodeDataset,
    indices: list[int],
    device: torch.device,
    out_dir: Path,
    step: int,
    include_prompt: bool,
    conditioning_mode: str = "visual_prompt",
) -> dict[str, float | None]:
    model.eval()
    loader = DataLoader(
        Subset(dataset, indices), batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_few_shot
    )
    sums: dict[str, float] = {
        "loss_total": 0.0, "loss_bbox": 0.0, "loss_giou": 0.0,
        "loss_ce": 0.0, "loss_presence": 0.0,
    }
    presence_correct = presence_count = 0
    mask_intersection = 0.0
    mask_union = 0.0
    box_iou_sum = 0.0
    box_match_count = 0
    records = 0
    empty_remainder_records = 0
    figure_dir = out_dir / "figures" / f"step_{step:06d}"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_index = 0
    for raw in loader:
        batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in raw.items()}
        if "is_empty_remainder" in batch:
            empty_remainder_records += int(batch["is_empty_remainder"].sum())
        output = model(batch["query_images"], batch["support_boxes"],
                       batch.get("prompt_labels"), batch.get("prompt_padding_mask"),
                       class_ids=class_ids_for_mode(batch, device, conditioning_mode))
        targets = _target_dicts(batch, device, include_prompt)
        losses = model.losses(output, targets, batch["present"].reshape(-1))
        presence_predictions = output["presence_logit_dec"].sigmoid() > 0.5
        presence_truth = batch["present"].reshape(-1) > 0.5
        presence_correct += int((presence_predictions == presence_truth).sum())
        presence_count += int(presence_truth.numel())
        for name in sums:
            if name == "loss_total":
                weighted = sum(value * model.criterion.weight_dict[key]
                               for key, value in losses.items() if key in model.criterion.weight_dict)
                sums[name] += float(weighted)
            elif name in losses:
                sums[name] += float(losses[name])

        predictions = output["pred_boxes"]
        scores = output["pred_logits"].sigmoid()[..., 0]
        if "presence_logit_dec" in output:
            scores = scores * output["presence_logit_dec"].sigmoid()[:, None]
        pred_masks = None
        mask_output = output.get("pred_masks")
        if isinstance(mask_output, dict):
            pred_masks = torch.einsum(
                "bchw,bqc->bqhw", mask_output["spatial_features"], mask_output["query_features"]
            ) + mask_output["bias"]
        elif isinstance(mask_output, torch.Tensor):
            pred_masks = mask_output

        matches = model.matcher(output, targets)
        for batch_index, (src_idx, tgt_idx) in enumerate(matches):
            if src_idx.numel():
                pred_xyxy = box_cxcywh_to_xyxy(predictions[batch_index, src_idx]).clamp(0, 1)
                target_xyxy = box_cxcywh_to_xyxy(targets[batch_index]["boxes"][tgt_idx]).clamp(0, 1)
                lt = torch.maximum(pred_xyxy[:, :2], target_xyxy[:, :2])
                rb = torch.minimum(pred_xyxy[:, 2:], target_xyxy[:, 2:])
                intersection = (rb - lt).clamp_min(0).prod(-1)
                pred_area = (pred_xyxy[:, 2:] - pred_xyxy[:, :2]).clamp_min(0).prod(-1)
                target_area = (target_xyxy[:, 2:] - target_xyxy[:, :2]).clamp_min(0).prod(-1)
                ious = intersection / (pred_area + target_area - intersection).clamp_min(1e-8)
                box_iou_sum += float(ious.sum())
                box_match_count += ious.numel()
            if pred_masks is not None:
                if src_idx.numel() == 0 or "masks" not in targets[batch_index]:
                    continue
                target_binary = targets[batch_index]["masks"][tgt_idx].bool()
                pred_logits_i = torch.nn.functional.interpolate(
                    pred_masks[batch_index, src_idx][:, None],
                    size=target_binary.shape[-2:], mode="bilinear", align_corners=False,
                )[:, 0]
                pred_binary = pred_logits_i.sigmoid() > 0.5
                intersection = (pred_binary & target_binary).sum().item()
                union = (pred_binary | target_binary).sum().item()
                mask_intersection += intersection
                mask_union += union

        for batch_index in range(batch["query_images"].shape[0]):
            if plot_index >= 8:
                continue
            image_id = int(batch["query_image_ids"][batch_index])
            meta = dataset.coco.loadImgs([image_id])[0]
            image = Image.open(dataset.image_dir / meta["file_name"]).convert("RGB")
            width, height = image.size
            target_boxes = targets[batch_index]["boxes"]
            topk = min(max(1, target_boxes.shape[0]), predictions.shape[1])
            _, query_ids = scores[batch_index].topk(topk)
            def corners(box: torch.Tensor) -> list[float]:
                cx, cy, box_w, box_h = box.detach().float().cpu().tolist()
                return [(cx - box_w / 2) * width, (cy - box_h / 2) * height,
                        (cx + box_w / 2) * width, (cy + box_h / 2) * height]

            target_boxes_px = [corners(box) for box in target_boxes]
            query_ids_list = query_ids.detach().cpu().tolist()
            prediction_boxes_px = [corners(predictions[batch_index, query_id]) for query_id in query_ids_list]
            prediction_scores = [float(scores[batch_index, query_id]) for query_id in query_ids_list]
            prompt_boxes_px = [
                corners(box) for prompt_index, box in enumerate(batch["support_boxes"][batch_index])
                if not bool(batch["prompt_padding_mask"][batch_index, prompt_index])
            ]
            prompt_labels = batch["prompt_labels"][batch_index][
                ~batch["prompt_padding_mask"][batch_index]
            ].tolist()
            category_id = int(batch["category_ids"][batch_index])
            _save_query_support_figure(
                figure_dir / f"episode_{plot_index:02d}_{image_id}.png",
                f"COCO category {category_id} | {len(target_boxes_px)} target boxes",
                image,
                target_boxes_px,
                prediction_boxes_px,
                prediction_scores,
                [(image_id, image, box,
                  f"Same-image {'positive' if prompt_labels[prompt_index] else 'negative'} prompt "
                  f"{prompt_index + 1} | category {category_id}")
                 for prompt_index, box in enumerate(prompt_boxes_px)],
                plot_mode="overlay",
                red_prediction_count=len(target_boxes_px),
            )
            if pred_masks is not None and target_boxes.shape[0] > 0:
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                axes[0].imshow(image)
                axes[0].set_title("Original image")
                axes[1].imshow(image)
                axes[1].set_title("Prediction and GT mask overlay")
                best_query = int(scores[batch_index].argmax())
                pred = torch.nn.functional.interpolate(
                    pred_masks[batch_index, best_query][None, None], size=(height, width), mode="bilinear",
                    align_corners=False,
                )[0, 0].sigmoid().cpu().numpy() > 0.5
                axes[1].imshow(np.ma.masked_where(~pred, pred), cmap="autumn", alpha=0.35)
                if "masks" in targets[batch_index] and targets[batch_index]["masks"].shape[0]:
                    gt_mask = targets[batch_index]["masks"][0].float()[None, None]
                    gt_mask = torch.nn.functional.interpolate(gt_mask, size=(height, width), mode="nearest")[0, 0]
                    gt_mask = gt_mask.cpu().numpy().astype(bool)
                    axes[1].imshow(np.ma.masked_where(~gt_mask, gt_mask), cmap="winter", alpha=0.25)
                for axis in axes:
                    axis.axis("off")
                fig.tight_layout()
                fig.savefig(figure_dir / f"episode_{plot_index:02d}_{image_id}_masks.png", dpi=130)
                plt.close(fig)
            plot_index += 1
        records += batch["query_images"].shape[0]
    model.train()
    metrics: dict[str, float | None] = {key: value / max(1, len(loader)) for key, value in sums.items()}
    metrics["matched_box_iou"] = box_iou_sum / box_match_count if box_match_count else None
    metrics["matched_mask_iou"] = mask_intersection / mask_union if mask_union else None
    metrics["presence_accuracy"] = presence_correct / max(1, presence_count)
    metrics["samples"] = float(records)
    metrics["empty_remainder_samples"] = float(empty_remainder_records)
    return metrics


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
    max_checkpoints = max(1, int(cfg.get("max_checkpoints", 3)))
    segmentation = bool(cfg.get("segmentation", False))
    resolution = int(cfg.get("resolution", 280))

    model_config = RFDETRBaseConfig(
        pretrain_weights=cfg.get("pretrain_weights"),
        resolution=resolution,
        positional_encoding_size=int(cfg.get("positional_encoding_size", 37)),
        num_queries=int(cfg.get("num_queries", 20)),
        num_select=int(cfg.get("num_queries", 20)),
        num_classes=int(cfg.get("pretrained_class_count", 90)),
        group_detr=1,
        two_stage=False,
        bbox_reparam=bool(cfg.get("bbox_reparam", False)),
        segmentation_head=segmentation,
        device=str(device),
    )
    train_config = TrainConfig(dataset_dir=".", output_dir=str(output_dir))
    model = PCSDetector.from_config(
        model_config, train_config,
        num_prompt_classes=int(cfg.get("num_prompt_classes", 91)),
        init_class_embeddings_from_detector=bool(cfg.get("init_class_embeddings_from_detector", True)),
        use_class_embeddings=uses_class_embeddings(cfg),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 1e-4)), weight_decay=0.05)
    resume_checkpoint = cfg.get("resume_checkpoint")
    if resume_checkpoint is not None:
        checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        missing, skipped = model.load_compatible_state_dict(checkpoint["model"])
        if missing or skipped:
            print(f"checkpoint PCS migration: initialized missing keys={missing}; skipped keys={skipped}")
        if "optimizer" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
            except (ValueError, KeyError) as exc:
                print(f"optimizer state not reusable after PCS head changes; starting fresh ({exc})")
        if cfg.get("resume_lr") is not None:
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = float(cfg["resume_lr"])
        resume_step = int(checkpoint.get("step", 0))
    else:
        resume_step = 0

    dataset = CocoFewShotEpisodeDataset(
        cfg["train_image_dir"], cfg["train_annotation_file"], resolution=resolution,
        episodes=1000, max_shots=int(cfg.get("max_prompt_boxes", 1)),
        negative_ratio=float(cfg.get("negative_ratio", 0.0)), seed=seed, training=True,
        negative_prompt_ratio=float(cfg.get("negative_prompt_ratio", 0.0)),
        include_masks=segmentation,
    )
    stage_sizes = [int(value) for value in cfg.get("stage_sizes", [4, 100, 1000])]
    stage_steps = [int(value) for value in cfg.get("stage_steps", [1500, 3000, 5000])]
    if len(stage_sizes) != len(stage_steps) or stage_sizes != sorted(stage_sizes):
        raise ValueError("stage_sizes and stage_steps must align; stage_sizes must be ascending.")
    all_metrics: list[dict[str, Any]] = []
    global_step = resume_step
    model.train()
    for stage_index, (sample_count, steps) in enumerate(zip(stage_sizes, stage_steps, strict=True), start=1):
        indices = list(range(min(sample_count, len(dataset))))
        loader = DataLoader(
            Subset(dataset, indices), batch_size=min(int(cfg.get("batch_size", 4)), len(indices)),
            shuffle=True, drop_last=False, num_workers=int(cfg.get("num_workers", 0)),
            collate_fn=collate_few_shot,
        )
        iterator = iter(loader)
        bar = tqdm(range(steps), desc=f"PCS stage {stage_index}: {sample_count} episodes")
        last_eval = None
        for _ in bar:
            try:
                raw = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                raw = next(iterator)
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in raw.items()}
            output = model(batch["query_images"], batch["support_boxes"],
                           batch.get("prompt_labels"), batch.get("prompt_padding_mask"),
                           class_ids=class_ids_for_mode(batch, device, conditioning_mode_at(cfg, global_step)))
            targets = _target_dicts(batch, device, bool(cfg.get("include_prompt_box", True)))
            losses = model.losses(output, targets, batch["present"].reshape(-1))
            total = sum(value * model.criterion.weight_dict[name]
                        for name, value in losses.items() if name in model.criterion.weight_dict)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("clip_grad_norm", 0.1)))
            optimizer.step()
            global_step += 1
            bar.set_postfix(loss=f"{float(total.detach()):.4f}",
                            box=f"{float(losses.get('loss_bbox', total.detach()).detach()):.4f}",
                            mask=f"{float(losses.get('loss_mask_dice', total.detach()).detach()):.4f}" if segmentation else "off")
            log_every = int(cfg.get("eval_every", 100))
            if global_step % log_every == 0 or _ == steps - 1:
                last_eval = save_diagnostics(
                    model, dataset, indices[: min(len(indices), int(cfg.get("sanity_samples", 8)))],
                    device, output_dir / f"stage_{sample_count}", global_step,
                    bool(cfg.get("include_prompt_box", True)),
                    conditioning_mode_at(cfg, global_step - 1),
                )
                record = {"step": global_step, "stage_episodes": sample_count, **last_eval}
                all_metrics.append(record)
                (output_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2))
                curve_metrics = [entry for entry in all_metrics if entry["stage_episodes"] == sample_count]
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot([entry["step"] for entry in curve_metrics],
                        [entry["loss_total"] for entry in curve_metrics], label="weighted loss")
                ax.set_xlabel("optimizer step")
                ax.set_ylabel("loss")
                ax.grid(True, alpha=0.25)
                ax.legend()
                fig.tight_layout()
                fig.savefig(output_dir / f"loss_stage_{sample_count}.png", dpi=140)
                plt.close(fig)
                if segmentation:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot([entry["step"] for entry in curve_metrics],
                            [entry["matched_mask_iou"] for entry in curve_metrics], label="matched mask IoU")
                    ax.set_xlabel("optimizer step")
                    ax.set_ylabel("IoU")
                    ax.set_ylim(0.0, 1.0)
                    ax.grid(True, alpha=0.25)
                    ax.legend()
                    fig.tight_layout()
                    fig.savefig(output_dir / f"mask_iou_stage_{sample_count}.png", dpi=140)
                    plt.close(fig)
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot([entry["step"] for entry in curve_metrics],
                        [entry["matched_box_iou"] for entry in curve_metrics], label="matched box IoU")
                ax.set_xlabel("optimizer step")
                ax.set_ylabel("IoU")
                ax.set_ylim(0.0, 1.0)
                ax.grid(True, alpha=0.25)
                ax.legend()
                fig.tight_layout()
                fig.savefig(output_dir / f"box_iou_stage_{sample_count}.png", dpi=140)
                plt.close(fig)
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "step": global_step, "stage_episodes": sample_count},
                           output_dir / f"checkpoint_step_{global_step:06d}.pt")
                checkpoints = sorted(output_dir.glob("checkpoint_step_*.pt"))
                boundary_steps = set(cfg.get("keep_checkpoint_steps", []))
                recent_steps = {
                    int(re.search(r"(\d+)$", path.stem).group(1))
                    for path in checkpoints[-max_checkpoints:]
                }
                for checkpoint_path in checkpoints:
                    match = re.search(r"(\d+)$", checkpoint_path.stem)
                    if match and int(match.group(1)) not in boundary_steps | recent_steps:
                        checkpoint_path.unlink()
                bar.write(f"step={global_step} train_subset_metrics={last_eval}")


if __name__ == "__main__":
    main()
