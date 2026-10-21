"""Standard Few-Shot Object Detection & Instance Segmentation Evaluator.

Supports COCO-format evaluation via standard COCOeval for:
- Bounding Box Detection (bbox mAP, AP50, AP75)
- Instance Segmentation (segm mAP, AP50, AP75)
- Class-split breakdowns: Novel (nAP), Base (bAP), All (mAP)
"""

from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mask_utils

from rfdetr_pcs.utilities.box_ops import box_cxcywh_to_xyxy


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


class FSODEvaluator:
    """Evaluates few-shot object detection and segmentation predictions against COCO ground truth."""

    def __init__(
        self,
        gt_annotation_file: str | Path,
        novel_cat_ids: set[int] | None = None,
        base_cat_ids: set[int] | None = None,
        iou_types: tuple[str, ...] = ("bbox", "segm"),
        score_threshold: float = 0.01,
        max_dets: int = 100,
    ) -> None:
        self.gt_annotation_file = Path(gt_annotation_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self.coco_gt = COCO(str(self.gt_annotation_file))
        
        all_cat_ids = set(self.coco_gt.getCatIds())
        self.novel_cat_ids = novel_cat_ids if novel_cat_ids is not None else set()
        self.base_cat_ids = base_cat_ids if base_cat_ids is not None else (all_cat_ids - self.novel_cat_ids)
        self.iou_types = iou_types
        self.score_threshold = score_threshold
        self.max_dets = max_dets

        self.reset()

    def reset(self) -> None:
        self.predictions: list[dict[str, Any]] = []

    def add_predictions(
        self,
        image_ids: list[int] | torch.Tensor,
        category_ids: list[int] | torch.Tensor,
        output: dict[str, Any],
        image_sizes: list[tuple[int, int]],
        include_masks: bool = False,
    ) -> None:
        """Convert a batch of model predictions into COCO evaluation format."""
        pred_logits = output["pred_logits"]
        pred_boxes = output["pred_boxes"]
        presence_logits = output.get("presence_logit_dec")
        pred_masks = output.get("pred_masks") if include_masks else None

        # Compute per-query detection scores
        scores = pred_logits.sigmoid()[..., 0]
        if presence_logits is not None:
            # Broadcast episode presence probability [B, 1] to queries [B, Q]
            presence_prob = presence_logits.sigmoid().reshape(scores.shape[0], 1)
            scores = scores * presence_prob

        batch_size = scores.shape[0]
        for b in range(batch_size):
            img_id = int(image_ids[b])
            cat_id = int(category_ids[b])
            orig_h, orig_w = image_sizes[b]

            b_scores = scores[b]
            valid_mask = b_scores >= self.score_threshold
            if not valid_mask.any():
                continue

            indices = torch.where(valid_mask)[0]
            if len(indices) > self.max_dets:
                topk = b_scores[indices].topk(self.max_dets)
                indices = indices[topk.indices]

            selected_scores = b_scores[indices].detach().cpu().tolist()
            selected_boxes_cxcywh = pred_boxes[b, indices].detach()
            selected_boxes_xyxy = box_cxcywh_to_xyxy(selected_boxes_cxcywh).clamp(0, 1)

            # Convert to absolute pixel [x, y, w, h]
            x1 = selected_boxes_xyxy[:, 0] * orig_w
            y1 = selected_boxes_xyxy[:, 1] * orig_h
            x2 = selected_boxes_xyxy[:, 2] * orig_w
            y2 = selected_boxes_xyxy[:, 3] * orig_h
            w = (x2 - x1).clamp_min(0)
            h = (y2 - y1).clamp_min(0)

            boxes_xywh = torch.stack([x1, y1, w, h], dim=1).cpu().tolist()

            encoded_masks = None
            if include_masks and pred_masks is not None:
                mask_logits = _selected_mask_logits(pred_masks, b, indices)
                # Resize logits to original image dimension
                upscaled_masks = F.interpolate(
                    mask_logits[:, None],
                    size=(orig_h, orig_w),
                    mode="bilinear",
                    align_corners=False,
                )[:, 0].sigmoid() > 0.5
                upscaled_np = upscaled_masks.cpu().numpy().astype(np.uint8)
                # Encode binary masks to RLE
                encoded_masks = [
                    mask_utils.encode(np.asfortranarray(upscaled_np[i]))
                    for i in range(len(indices))
                ]
                for rle in encoded_masks:
                    rle["counts"] = rle["counts"].decode("utf-8")

            for k in range(len(indices)):
                entry: dict[str, Any] = {
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": [round(val, 2) for val in boxes_xywh[k]],
                    "score": float(selected_scores[k]),
                }
                if encoded_masks is not None:
                    entry["segmentation"] = encoded_masks[k]
                self.predictions.append(entry)

    def evaluate(self) -> dict[str, Any]:
        """Compute standard COCOeval metrics over all accumulated predictions."""
        if not self.predictions:
            return {"bbox": {}, "segm": {}, "empty": True}

        metrics: dict[str, Any] = {}

        # Save predictions to memory via COCO.loadRes
        with contextlib.redirect_stdout(io.StringIO()):
            coco_dt = self.coco_gt.loadRes(self.predictions)

        for iou_type in self.iou_types:
            if iou_type == "segm" and not any("segmentation" in p for p in self.predictions):
                continue

            type_metrics: dict[str, float] = {}
            eval_subsets = [("all", None)]
            if self.novel_cat_ids:
                eval_subsets.append(("novel", list(self.novel_cat_ids)))
            if self.base_cat_ids:
                eval_subsets.append(("base", list(self.base_cat_ids)))

            for subset_name, subset_cats in eval_subsets:
                with contextlib.redirect_stdout(io.StringIO()):
                    coco_eval = COCOeval(self.coco_gt, coco_dt, iouType=iou_type)
                    if subset_cats is not None:
                        coco_eval.params.catIds = subset_cats
                    coco_eval.evaluate()
                    coco_eval.accumulate()
                    coco_eval.summarize()

                stats = coco_eval.stats
                prefix = f"{subset_name}_" if subset_name != "all" else ""
                type_metrics.update({
                    f"{prefix}mAP": float(stats[0]),
                    f"{prefix}mAP_50": float(stats[1]),
                    f"{prefix}mAP_75": float(stats[2]),
                    f"{prefix}mAP_s": float(stats[3]),
                    f"{prefix}mAP_m": float(stats[4]),
                    f"{prefix}mAP_l": float(stats[5]),
                })

            metrics[iou_type] = type_metrics

        return metrics
