"""Prompt-conditioned detection built on the isolated RF-DETR implementation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
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

    def loss(self, outputs: dict[str, Tensor], target_boxes: Tensor, target_present: Tensor) -> dict[str, Tensor]:
        """Compute binary objectness, L1 box, and generalized-IoU losses."""
        target_present = target_present.to(outputs["pred_logits"].dtype).unsqueeze(-1)
        logits = outputs["pred_logits"]
        loss_objectness = torch.nn.functional.binary_cross_entropy_with_logits(logits, target_present)
        positive = target_present.squeeze(-1) > 0.5
        if positive.any():
            predicted, target = outputs["pred_boxes"][positive], target_boxes[positive]
            loss_bbox = torch.nn.functional.l1_loss(predicted, target)
            giou = _generalized_box_iou(box_cxcywh_to_xyxy(predicted), box_cxcywh_to_xyxy(target))
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