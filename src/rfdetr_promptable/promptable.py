"""Prompt-conditioned detection built on the isolated RF-DETR implementation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torchvision.ops import roi_align
from torchvision.transforms import functional as TF

from rfdetr_promptable.config import ModelConfig, TrainConfig
from rfdetr_promptable.models.lwdetr import LWDETR, build_model_from_config
from rfdetr_promptable.utilities.box_ops import box_cxcywh_to_xyxy
from rfdetr_promptable.utilities.tensors import nested_tensor_from_tensor_list


def _inverse_sigmoid(value: Tensor, eps: float = 1e-5) -> Tensor:
    value = value.clamp(min=eps, max=1.0 - eps)
    return torch.log(value / (1.0 - value))


class PromptableDetector(nn.Module):
    """Binary box-prompted detector with one dynamic decoder query per prompt box."""

    def __init__(self, detector: LWDETR) -> None:
        super().__init__()
        if detector.group_detr != 1 or detector.two_stage:
            raise ValueError("PromptableDetector requires group_detr=1 and two_stage=False.")
        self.detector = detector
        hidden_dim = detector.transformer.d_model
        self.prompt_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.support_roi_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.support_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.objectness = nn.Linear(hidden_dim, 1)
        nn.init.constant_(self.objectness.bias, -2.0)

    @classmethod
    def from_config(cls, model_config: ModelConfig, train_config: TrainConfig | None = None) -> "PromptableDetector":
        if train_config is None:
            train_config = TrainConfig(dataset_dir=".", output_dir=".")
        return cls(build_model_from_config(model_config, train_config))

    def forward(self, samples: Tensor, prompt_boxes: Tensor) -> dict[str, Tensor]:
        """Run prompted detection for normalized ``cxcywh`` prompt boxes."""
        if samples.ndim != 4 or prompt_boxes.ndim != 3 or prompt_boxes.shape[-1] != 4:
            raise ValueError("Expected samples [B,3,H,W] and prompt_boxes [B,P,4].")
        if samples.shape[0] != prompt_boxes.shape[0]:
            raise ValueError("samples and prompt_boxes must have the same batch size.")

        nested = nested_tensor_from_tensor_list(samples)
        features, positions, cross_attn_features = self.detector.backbone(nested)
        srcs, masks = [], []
        for feature in features:
            src, mask = feature.decompose()
            srcs.append(src)
            masks.append(mask)
        cross_attn_srcs = None
        if cross_attn_features is not None:
            cross_attn_srcs = [feature.decompose()[0] for feature in cross_attn_features]

        batch_size, prompt_count, _ = prompt_boxes.shape
        query_feat = self.detector.query_feat.weight[:1].expand(batch_size, prompt_count, -1)
        query_feat = query_feat + self.prompt_encoder(prompt_boxes)
        refpoints = _inverse_sigmoid(prompt_boxes)
        transformer_outputs = self.detector.transformer(
            srcs, masks, positions, refpoints, query_feat, cross_attn_srcs=cross_attn_srcs
        )
        hs, references = transformer_outputs[:2]
        if hs is None or references is None:
            raise RuntimeError("Promptable detection requires decoder layers.")
        hs_last, ref_last = hs[-1], references[-1]
        pred_boxes = (self.detector.bbox_embed(hs_last) + ref_last).sigmoid()
        return {"pred_logits": self.objectness(hs_last), "pred_boxes": pred_boxes}

    def encode_support(
        self,
        support_images: Tensor,
        support_boxes: Tensor,
        support_group_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Encode and group support image/box examples for reuse across queries."""
        if support_images.ndim != 4:
            raise ValueError("support_images must have shape [S,3,H,W].")
        if support_boxes.ndim == 2:
            support_boxes = support_boxes[:, None, :]
        if support_boxes.ndim != 3 or support_boxes.shape[-1] != 4:
            raise ValueError("support_boxes must have shape [S,4] or [S,K,4] in normalized cxcywh format.")
        if support_boxes.shape[0] != support_images.shape[0]:
            raise ValueError("Each support image must have corresponding support boxes.")

        support_nested = nested_tensor_from_tensor_list(support_images)
        support_features, _, _ = self.detector.backbone(support_nested)
        support_map = support_features[-1].tensors
        _, _, feature_height, feature_width = support_map.shape
        support_count, boxes_per_image, _ = support_boxes.shape
        support_boxes_flat = support_boxes.reshape(-1, 4).to(
            device=support_map.device, dtype=support_map.dtype
        )
        support_indices = torch.arange(support_count, device=support_map.device).repeat_interleave(
            boxes_per_image
        )
        input_height, input_width = support_images.shape[-2:]
        rois = torch.cat(
            [
                support_indices[:, None].to(support_boxes_flat.dtype),
                torch.stack(
                    [
                        support_boxes_flat[:, 0] - support_boxes_flat[:, 2] / 2,
                        support_boxes_flat[:, 1] - support_boxes_flat[:, 3] / 2,
                        support_boxes_flat[:, 0] + support_boxes_flat[:, 2] / 2,
                        support_boxes_flat[:, 1] + support_boxes_flat[:, 3] / 2,
                    ],
                    dim=-1,
                )
                * support_boxes_flat.new_tensor([input_width, input_height, input_width, input_height]),
            ],
            dim=-1,
        )
        roi_features = roi_align(
            support_map,
            rois,
            output_size=(3, 3),
            spatial_scale=(feature_width / input_width + feature_height / input_height) / 2,
            aligned=True,
        ).mean(dim=(-1, -2))
        example_tokens = self.support_roi_encoder(roi_features) + self.prompt_encoder(support_boxes_flat)
        if support_group_ids is None:
            support_group_ids = torch.arange(support_count, device=support_map.device)
        else:
            support_group_ids = support_group_ids.to(device=support_map.device, dtype=torch.long)
        if support_group_ids.shape != (support_count,):
            raise ValueError("support_group_ids must have shape [number of support images].")
        shot_tokens = example_tokens.reshape(support_count, boxes_per_image, -1).mean(dim=1)
        group_ids = torch.unique(support_group_ids, sorted=True)
        grouped_tokens = torch.stack(
            [shot_tokens[support_group_ids == group_id].mean(dim=0) for group_id in group_ids]
        )
        return grouped_tokens, group_ids

    def forward_support_query(
        self,
        support_images: Tensor,
        support_boxes: Tensor,
        query_images: Tensor,
        support_group_ids: Tensor | None = None,
        query_to_support: Tensor | None = None,
        encoded_support: tuple[Tensor, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        """Detect support examples in separate query images.

        Args:
            support_images: ``[S,3,H,W]`` images containing the exemplars.
            support_boxes: ``[S,4]`` or ``[S,K,4]`` normalized ``cxcywh`` boxes.
                Rows are support examples; K provides multiple boxes per support image.
            query_images: ``[B,3,H,W]`` target images to search.
            support_group_ids: Optional ``[S]`` IDs grouping support image/box
                examples into concepts. Repeated IDs represent multiple shots of the
                same concept. Defaults to one concept per support image.
            query_to_support: Optional ``[B]`` IDs mapping each query image to one
                support concept. When absent, all support examples are pooled together.
            encoded_support: Optional output of :meth:`encode_support`, allowing the
                support backbone/ROI features to be reused across query batches.
        """
        if query_images.ndim != 4:
            raise ValueError("query_images must have shape [B,3,H,W].")
        query_count = query_images.shape[0]
        if encoded_support is None:
            grouped_tokens, group_ids = self.encode_support(
                support_images, support_boxes, support_group_ids
            )
        else:
            grouped_tokens, group_ids = encoded_support
        if query_to_support is None:
            concept_tokens = grouped_tokens.mean(dim=0, keepdim=True).expand(query_count, -1)
        else:
            if query_to_support.shape != (query_count,):
                raise ValueError("query_to_support must have shape [number of query images].")
            group_to_index = {int(group_id): index for index, group_id in enumerate(group_ids.tolist())}
            try:
                selected = [group_to_index[int(group_id)] for group_id in query_to_support.tolist()]
            except KeyError as exc:
                raise ValueError(f"query_to_support refers to unknown support group {exc.args[0]}.") from exc
            concept_tokens = grouped_tokens[torch.tensor(selected, device=grouped_tokens.device)]

        query_nested = nested_tensor_from_tensor_list(query_images)
        query_features, query_positions, query_cross = self.detector.backbone(query_nested)
        query_srcs, query_masks = [], []
        for feature in query_features:
            src, mask = feature.decompose()
            query_srcs.append(src)
            query_masks.append(mask)
        query_cross_srcs = None if query_cross is None else [feature.decompose()[0] for feature in query_cross]

        prompt_count = self.detector.num_queries
        base_query = self.detector.query_feat.weight[:prompt_count].expand(query_count, prompt_count, -1)
        concept_tokens = concept_tokens[:, None, :].expand(-1, prompt_count, -1)
        # Use independent learned DETR references. Neither support box nor query GT
        # coordinates are reused as query-image locations.
        query_reference = self.detector.refpoint_embed.weight[:prompt_count].unsqueeze(0).expand(
            query_count, -1, -1
        )
        query_geometry = self.prompt_encoder(query_reference)
        dynamic_query = base_query + self.support_fusion(torch.cat([concept_tokens, query_geometry], dim=-1))
        transformer_outputs = self.detector.transformer(
            query_srcs,
            query_masks,
            query_positions,
            _inverse_sigmoid(query_reference),
            dynamic_query,
            cross_attn_srcs=query_cross_srcs,
        )
        hs, references = transformer_outputs[:2]
        if hs is None or references is None:
            raise RuntimeError("Support/query detection requires decoder layers.")
        hs_last, ref_last = hs[-1], references[-1]
        pred_boxes = (self.detector.bbox_embed(hs_last) + ref_last).sigmoid()
        return {"pred_logits": self.objectness(hs_last), "pred_boxes": pred_boxes}

    @torch.inference_mode()
    def predict_support_query(
        self,
        support_images: Tensor,
        support_boxes: Tensor,
        query_images: Tensor,
        support_group_ids: Tensor | None = None,
        query_to_support: Tensor | None = None,
        threshold: float = 0.0,
    ) -> list[dict[str, Tensor]]:
        """Batched few-shot inference; outputs are scaled to each query image size."""
        device = next(self.parameters()).device
        support_images = support_images.to(device)
        query_images = query_images.to(device)
        support_boxes = support_boxes.to(device)
        if support_boxes.ndim == 2:
            support_boxes = support_boxes[:, None, :]
        resolution = self.detector.backbone[0].encoder.shape[-1]
        support_images = TF.resize(support_images, [resolution, resolution])
        query_sizes = [(image.shape[-2], image.shape[-1]) for image in query_images]
        query_images_resized = TF.resize(query_images, [resolution, resolution])
        outputs = self.forward_support_query(
            support_images,
            support_boxes,
            query_images_resized,
            support_group_ids=support_group_ids,
            query_to_support=query_to_support,
        )
        scores_all = outputs["pred_logits"].sigmoid()[..., 0]
        boxes_all = box_cxcywh_to_xyxy(outputs["pred_boxes"])
        results: list[dict[str, Tensor]] = []
        for batch_index, (height, width) in enumerate(query_sizes):
            scores = scores_all[batch_index]
            keep = scores >= threshold
            boxes = boxes_all[batch_index, keep]
            scale = boxes.new_tensor([width, height, width, height])
            results.append({"scores": scores[keep].cpu(), "boxes": (boxes * scale).cpu().clamp_min(0)})
        return results

    def loss(self, outputs: dict[str, Tensor], target_boxes: Tensor, target_present: Tensor) -> dict[str, Tensor]:
        """Compute binary objectness and matched L1/GIoU losses.

        For same-image prompt mode, each provided box is a query and target_present
        corresponds to each prompt. For support/query mode, target_present is one
        concept-presence label per image; the highest-IoU decoder query is matched to
        its GT box and every unmatched query is trained as background.
        """
        logits = outputs["pred_logits"]
        target_present = target_present.to(logits.dtype)
        if target_present.ndim == 1:
            target_present = target_present[:, None]
        if target_present.ndim == 2 and target_present.shape[1] == logits.shape[1]:
            labels = target_present.unsqueeze(-1)
            target_boxes_expanded = target_boxes
            matched_predictions = outputs["pred_boxes"][labels[..., 0] > 0.5]
            matched_targets = target_boxes_expanded[labels[..., 0] > 0.5]
        else:
            batch_size, query_count = logits.shape[:2]
            labels = logits.new_zeros(batch_size, query_count, 1)
            matched_predictions_list = []
            matched_targets_list = []
            gt_present = target_present.reshape(batch_size, -1).max(dim=1).values > 0.5
            for batch_index in range(batch_size):
                if not bool(gt_present[batch_index]):
                    continue
                gt_box = target_boxes[batch_index].reshape(-1, 4)[0]
                predicted_xyxy = box_cxcywh_to_xyxy(outputs["pred_boxes"][batch_index])
                target_xyxy = box_cxcywh_to_xyxy(gt_box[None])
                ious = _generalized_box_iou(predicted_xyxy, target_xyxy).squeeze(-1)
                best_query = int(ious.argmax())
                labels[batch_index, best_query, 0] = 1
                matched_predictions_list.append(outputs["pred_boxes"][batch_index, best_query])
                matched_targets_list.append(gt_box)
            matched_predictions = torch.stack(matched_predictions_list) if matched_predictions_list else logits.new_zeros(0, 4)
            matched_targets = torch.stack(matched_targets_list) if matched_targets_list else logits.new_zeros(0, 4)
        loss_objectness = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        positive = labels[..., 0] > 0.5
        if matched_predictions.numel() > 0:
            loss_bbox = torch.nn.functional.l1_loss(matched_predictions, matched_targets)
            giou = _generalized_box_iou(box_cxcywh_to_xyxy(matched_predictions), box_cxcywh_to_xyxy(matched_targets))
            loss_giou = (1.0 - torch.diag(giou)).mean()
        else:
            loss_bbox = logits.sum() * 0.0
            loss_giou = logits.sum() * 0.0
        return {"loss_objectness": loss_objectness, "loss_bbox": loss_bbox, "loss_giou": loss_giou}

    @torch.inference_mode()
    def predict(
        self, image: Image.Image | np.ndarray | Tensor, prompt_boxes: Sequence[Sequence[float]], threshold: float = 0.5
    ) -> dict[str, Tensor]:
        """Predict refined prompt boxes in source-image pixel ``xyxy`` coordinates."""
        if isinstance(image, Image.Image):
            width, height = image.size
            tensor = TF.to_tensor(image)
        elif isinstance(image, np.ndarray):
            height, width = image.shape[:2]
            tensor = TF.to_tensor(image)
        else:
            tensor = image.float()
            if tensor.ndim != 3 or tensor.shape[0] not in (1, 3):
                raise ValueError("Tensor image must have shape [C,H,W].")
            _, height, width = tensor.shape
            if tensor.max() > 1:
                tensor = tensor / 255.0
        encoder = self.detector.backbone[0].encoder
        resolution = getattr(encoder, "shape", (384, 384))[-1]
        tensor = TF.resize(tensor, [resolution, resolution])
        boxes = torch.as_tensor(prompt_boxes, dtype=torch.float32)
        if boxes.ndim != 2 or boxes.shape[-1] != 4:
            raise ValueError("prompt_boxes must have shape [P,4] in pixel xyxy format.")
        boxes = torch.stack(
            [boxes[:, 0] / width, boxes[:, 1] / height, boxes[:, 2] / width, boxes[:, 3] / height], dim=-1
        )
        boxes = torch.stack(
            [(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2, boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], dim=-1
        )
        device = next(self.parameters()).device
        outputs = self(tensor.unsqueeze(0).to(device), boxes.unsqueeze(0).to(device))
        scores = outputs["pred_logits"].sigmoid()[0, :, 0]
        keep = scores >= threshold
        predicted = box_cxcywh_to_xyxy(outputs["pred_boxes"][0, keep]).cpu()
        scale = torch.tensor([width, height, width, height])
        return {"scores": scores[keep].cpu(), "boxes": (predicted * scale).clamp_min(0)}


def _generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    inter = (rb - lt).clamp_min(0).prod(-1)
    area1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp_min(0).prod(-1)
    area2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp_min(0).prod(-1)
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp_min(1e-6)
    c_lt = torch.minimum(boxes1[:, None, :2], boxes2[None, :, :2])
    c_rb = torch.maximum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    c_area = (c_rb - c_lt).clamp_min(0).prod(-1)
    return iou - (c_area - union) / c_area.clamp_min(1e-6)