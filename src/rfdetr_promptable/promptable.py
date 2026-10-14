"""Prompt-conditioned detection built on the isolated RF-DETR implementation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
from torchvision.ops import roi_align
from torchvision.transforms import functional as TF

from rfdetr_promptable.config import ModelConfig, TrainConfig
from rfdetr_promptable.models.lwdetr import LWDETR, build_model_from_config
from rfdetr_promptable.utilities.box_ops import box_cxcywh_to_xyxy
from rfdetr_promptable.utilities.tensors import NestedTensor, nested_tensor_from_tensor_list


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

    def forward_same_image(self, images: Tensor, prompt_boxes: Tensor) -> dict[str, Tensor]:
        """Use a box on each image as a visual example to detect its other instances.

        The backbone runs once per image. The prompt ROI is pooled from that same
        feature map, then fused into independent detection queries over the map.
        """
        if images.ndim != 4 or prompt_boxes.shape != (images.shape[0], 4):
            raise ValueError("Expected images [B,3,H,W] and prompt_boxes [B,4].")
        nested = nested_tensor_from_tensor_list(images)
        features, positions, cross_attn_features = self.detector.backbone(nested)
        srcs, masks = [], []
        for feature in features:
            src, mask = feature.decompose()
            srcs.append(src)
            masks.append(mask)
        cross_attn_srcs = None if cross_attn_features is None else [f.decompose()[0] for f in cross_attn_features]

        feature_map = features[-1].tensors
        batch_size, _, feature_height, feature_width = feature_map.shape
        boxes = prompt_boxes.to(device=feature_map.device, dtype=feature_map.dtype)
        input_height, input_width = images.shape[-2:]
        xyxy = torch.stack(
            [boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2,
             boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2], dim=-1
        )
        rois = torch.cat(
            [torch.arange(batch_size, device=feature_map.device, dtype=boxes.dtype)[:, None],
             xyxy * boxes.new_tensor([input_width, input_height, input_width, input_height])], dim=-1
        )
        roi_features = roi_align(
            feature_map, rois, output_size=(3, 3),
            spatial_scale=(feature_width / input_width + feature_height / input_height) / 2,
            aligned=True,
        ).mean(dim=(-1, -2))
        support_token = self.support_roi_encoder(roi_features) + self.prompt_encoder(boxes)

        query_count = self.detector.num_queries
        base_query = self.detector.query_feat.weight[:query_count].expand(batch_size, query_count, -1)
        concept = support_token[:, None, :].expand(-1, query_count, -1)
        ref_logits = self.detector.refpoint_embed.weight[:query_count].unsqueeze(0).expand(batch_size, -1, -1)
        geometry = self.prompt_encoder(ref_logits.sigmoid())
        dynamic_query = base_query + self.support_fusion(torch.cat([concept, geometry], dim=-1))
        transformer_outputs = self.detector.transformer(
            srcs, masks, positions, ref_logits, dynamic_query, cross_attn_srcs=cross_attn_srcs
        )
        hs, references = transformer_outputs[:2]
        if hs is None or references is None:
            raise RuntimeError("Same-image prompted detection requires decoder layers.")
        hs_last, ref_last = hs[-1], references[-1]
        pred_boxes = (self.detector.bbox_embed(hs_last) + ref_last).sigmoid()
        return {"pred_logits": self.objectness(hs_last), "pred_boxes": pred_boxes}

    def forward_batched_support_query(
        self, support_images: Tensor, support_boxes: Tensor, query_images: Tensor
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Encode paired support/query images in one batch, then condition query decoding.

        Images remain separate batch elements, so backbone self-attention never sees
        a spatially stitched mega-image. Support and query predictions share one
        backbone pass; the support output supplies an auxiliary prompt-box detection
        loss while the query output supplies the cross-image transfer loss.
        """
        if support_images.shape != query_images.shape or support_boxes.shape != (support_images.shape[0], 4):
            raise ValueError("Paired support/query images must match [B,3,H,W] and support boxes [B,4].")
        batch_size = support_images.shape[0]
        joined = torch.cat([support_images, query_images], dim=0)
        features, positions, cross = self.detector.backbone(nested_tensor_from_tensor_list(joined))

        def split_features(start: int, end: int):
            return [
                NestedTensor(f.tensors[start:end], None if f.mask is None else f.mask[start:end])
                for f in features
            ]

        support_features = split_features(0, batch_size)
        query_features = split_features(batch_size, batch_size * 2)
        support_positions = [p[:batch_size] for p in positions]
        query_positions = [p[batch_size:] for p in positions]
        if cross is None:
            support_cross = query_cross = None
        else:
            support_cross = [NestedTensor(f.tensors[:batch_size], None if f.mask is None else f.mask[:batch_size]) for f in cross]
            query_cross = [NestedTensor(f.tensors[batch_size:], None if f.mask is None else f.mask[batch_size:]) for f in cross]

        support_map = support_features[-1].tensors
        _, _, feature_h, feature_w = support_map.shape
        boxes = support_boxes.to(device=support_map.device, dtype=support_map.dtype)
        input_h, input_w = support_images.shape[-2:]
        xyxy = torch.stack(
            [boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2,
             boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2], dim=-1
        )
        rois = torch.cat(
            [torch.arange(batch_size, device=boxes.device, dtype=boxes.dtype)[:, None],
             xyxy * boxes.new_tensor([input_w, input_h, input_w, input_h])], dim=-1
        )
        roi = roi_align(
            support_map, rois, output_size=(3, 3),
            spatial_scale=(feature_w / input_w + feature_h / input_h) / 2, aligned=True,
        ).mean(dim=(-1, -2))
        token = self.support_roi_encoder(roi) + self.prompt_encoder(boxes)

        support_outputs = self._decode_with_prompt_token(support_features, support_positions, support_cross, token)
        query_outputs = self._decode_with_prompt_token(query_features, query_positions, query_cross, token)
        return support_outputs, query_outputs

    def _decode_with_prompt_token(self, features, positions, cross_features, prompt_token: Tensor) -> dict[str, Tensor]:
        srcs, masks = [], []
        for feature in features:
            src, mask = feature.decompose()
            srcs.append(src)
            masks.append(mask)
        cross_srcs = None if cross_features is None else [f.decompose()[0] for f in cross_features]
        batch_size = prompt_token.shape[0]
        query_count = self.detector.num_queries
        base = self.detector.query_feat.weight[:query_count].expand(batch_size, query_count, -1)
        concept = prompt_token[:, None, :].expand(-1, query_count, -1)
        ref_logits = self.detector.refpoint_embed.weight[:query_count].unsqueeze(0).expand(batch_size, -1, -1)
        geometry = self.prompt_encoder(ref_logits.sigmoid())
        query = base + self.support_fusion(torch.cat([concept, geometry], dim=-1))
        hs, references = self.detector.transformer(
            srcs, masks, positions, ref_logits, query, cross_attn_srcs=cross_srcs
        )[:2]
        if hs is None or references is None:
            raise RuntimeError("Prompt-conditioned detection requires decoder layers.")
        boxes = (self.detector.bbox_embed(hs[-1]) + references[-1]).sigmoid()
        return {"pred_logits": self.objectness(hs[-1]), "pred_boxes": boxes}

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
        # LW-DETR stores learned references as unconstrained logits (they are
        # initialized to zero). Use their sigmoid only as normalized geometry
        # for the support prompt encoder, and pass the original logits to the
        # transformer. Applying inverse-sigmoid here clamps zero logits to an
        # almost-zero coordinate and collapses every decoded box at the origin.
        query_reference_logits = self.detector.refpoint_embed.weight[:prompt_count].unsqueeze(0).expand(
            query_count, -1, -1
        )
        query_geometry = self.prompt_encoder(query_reference_logits.sigmoid())
        dynamic_query = base_query + self.support_fusion(torch.cat([concept_tokens, query_geometry], dim=-1))
        transformer_outputs = self.detector.transformer(
            query_srcs,
            query_masks,
            query_positions,
            query_reference_logits,
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

    def loss(
        self,
        outputs: dict[str, Tensor],
        target_boxes: Tensor,
        target_present: Tensor,
        target_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute balanced objectness and one-to-one matched L1/GIoU losses.

        For same-image prompt mode, each provided box is a query and target_present
        corresponds to each prompt. For support/query mode, target_present is one
        concept-presence label per image; Hungarian assignment matches distinct
        decoder queries to every valid GT box and unmatched queries are background.
        """
        logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]
        batch_size, query_count = logits.shape[:2]
        target_present = target_present.to(logits.device)

        # Same-image prompted mode provides one output query per prompt and labels
        # already aligned to those queries. Support/query mode instead uses
        # Hungarian matching between every predicted query and every valid GT box.
        aligned_labels = target_present.ndim == 2 and target_present.shape[1] == query_count
        if aligned_labels:
            labels = target_present.to(logits.dtype).unsqueeze(-1)
            matched_predictions = pred_boxes[labels[..., 0] > 0.5]
            matched_targets = target_boxes.to(pred_boxes.device)[labels[..., 0] > 0.5]
        else:
            target_boxes = target_boxes.to(pred_boxes.device)
            if target_mask is None:
                target_mask = target_present.reshape(batch_size, -1)[:, :1] > 0.5
                if target_boxes.shape[1] > 1:
                    target_mask = target_mask.expand(-1, target_boxes.shape[1])
            target_mask = target_mask.to(device=pred_boxes.device, dtype=torch.bool)
            if target_mask.shape != target_boxes.shape[:2]:
                raise ValueError(
                    f"target_mask must have shape {tuple(target_boxes.shape[:2])}, got {tuple(target_mask.shape)}"
                )
            labels = logits.new_zeros(batch_size, query_count, 1)
            matched_predictions_list = []
            matched_targets_list = []
            for batch_index in range(batch_size):
                gt_boxes = target_boxes[batch_index, target_mask[batch_index]]
                if gt_boxes.numel() == 0:
                    continue
                predictions = pred_boxes[batch_index]
                cost_l1 = torch.cdist(predictions, gt_boxes, p=1)
                cost_giou = 1.0 - _generalized_box_iou(
                    box_cxcywh_to_xyxy(predictions), box_cxcywh_to_xyxy(gt_boxes)
                )
                rows, columns = linear_sum_assignment((cost_l1 + 2.0 * cost_giou).detach().cpu().numpy())
                rows_tensor = torch.as_tensor(rows, device=pred_boxes.device, dtype=torch.long)
                columns_tensor = torch.as_tensor(columns, device=pred_boxes.device, dtype=torch.long)
                labels[batch_index, rows_tensor, 0] = 1
                matched_predictions_list.append(predictions[rows_tensor])
                matched_targets_list.append(gt_boxes[columns_tensor])
            matched_predictions = torch.cat(matched_predictions_list) if matched_predictions_list else pred_boxes.new_zeros(0, 4)
            matched_targets = torch.cat(matched_targets_list) if matched_targets_list else pred_boxes.new_zeros(0, 4)

        # Compute focal loss separately over positives and negatives so adding
        # queries cannot dilute the positive classification signal.
        positive = labels[..., 0] > 0.5
        probability = logits.sigmoid()
        cross_entropy = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        p_t = probability * labels + (1.0 - probability) * (1.0 - labels)
        alpha_t = 0.25 * labels + 0.75 * (1.0 - labels)
        focal = alpha_t * (1.0 - p_t).pow(2.0) * cross_entropy
        positive_loss = focal[..., 0][positive].mean() if positive.any() else logits.sum() * 0.0
        negative_loss = focal[..., 0][~positive].mean() if (~positive).any() else logits.sum() * 0.0
        loss_objectness = positive_loss + negative_loss
        if matched_predictions.numel() > 0:
            loss_bbox = torch.nn.functional.l1_loss(matched_predictions, matched_targets)
            matched_pred_xyxy = box_cxcywh_to_xyxy(matched_predictions)
            matched_target_xyxy = box_cxcywh_to_xyxy(matched_targets)
            giou = _generalized_box_iou(matched_pred_xyxy, matched_target_xyxy)
            loss_giou = (1.0 - torch.diag(giou)).mean()
            intersection_lt = torch.maximum(matched_pred_xyxy[:, :2], matched_target_xyxy[:, :2])
            intersection_rb = torch.minimum(matched_pred_xyxy[:, 2:], matched_target_xyxy[:, 2:])
            intersection = (intersection_rb - intersection_lt).clamp_min(0).prod(dim=-1)
            pred_area = (matched_pred_xyxy[:, 2:] - matched_pred_xyxy[:, :2]).clamp_min(0).prod(dim=-1)
            target_area = (matched_target_xyxy[:, 2:] - matched_target_xyxy[:, :2]).clamp_min(0).prod(dim=-1)
            matched_iou = (intersection / (pred_area + target_area - intersection).clamp_min(1e-6)).detach().mean()
        else:
            loss_bbox = logits.sum() * 0.0
            loss_giou = logits.sum() * 0.0
            matched_iou = logits.sum() * 0.0
        return {
            "loss_objectness": loss_objectness,
            "loss_objectness_positive": positive_loss.detach(),
            "loss_objectness_negative": negative_loss.detach(),
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
            "matched_iou": matched_iou,
            "positive_score": probability[..., 0][positive].detach().mean() if positive.any() else logits.sum().detach() * 0.0,
            "negative_score": probability[..., 0][~positive].detach().mean() if (~positive).any() else logits.sum().detach() * 0.0,
            "pred_box_mean": pred_boxes.detach().mean(dim=(0, 1)),
        }

    @torch.inference_mode()
    def predict_same_image(
        self,
        image: Image.Image | np.ndarray | Tensor,
        prompt_boxes: Sequence[Sequence[float]],
        threshold: float = 0.5,
    ) -> dict[str, Tensor]:
        """Find other prompted-category instances in the same image.

        Prompt boxes use source-image pixel XYXY coordinates. Returned boxes use
        the same coordinate system and are ranked by predicted objectness.
        """
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
        boxes = torch.as_tensor(prompt_boxes, dtype=torch.float32)
        if boxes.shape != (1, 4):
            raise ValueError("Same-image inference currently expects one pixel XYXY prompt box.")
        x0, y0, x1, y1 = boxes.unbind(dim=-1)
        normalized = torch.stack(
            [((x0 + x1) / (2 * width)), ((y0 + y1) / (2 * height)),
             ((x1 - x0) / width), ((y1 - y0) / height)], dim=-1
        )
        resolution = self.detector.backbone[0].encoder.shape[-1]
        tensor = TF.resize(tensor, [resolution, resolution]).unsqueeze(0)
        device = next(self.parameters()).device
        outputs = self.forward_same_image(tensor.to(device), normalized.to(device))
        scores = outputs["pred_logits"][0, :, 0].sigmoid()
        boxes_xyxy = box_cxcywh_to_xyxy(outputs["pred_boxes"][0])
        keep = scores >= threshold
        scale = boxes_xyxy.new_tensor([width, height, width, height])
        return {"scores": scores[keep].cpu(), "boxes": (boxes_xyxy[keep] * scale).cpu().clamp_min(0)}

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
