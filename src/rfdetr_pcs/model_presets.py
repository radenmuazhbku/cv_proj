"""RF-DETR preset construction helpers for PCS training configs."""

from __future__ import annotations

from typing import Any

import torch

from rfdetr_pcs.config import RFDETRBaseConfig


def build_base_model_config(
    cfg: dict[str, Any], device: torch.device, *, segmentation: bool | None = None
) -> RFDETRBaseConfig:
    """Build the Base preset config, optionally disabling its published COCO weights.

    If ``use_pretrained_model`` is true and no explicit checkpoint path is
    configured, leave ``pretrain_weights`` unset so the preset supplies its
    published MS COCO checkpoint. A custom ``pretrain_weights`` path remains
    supported. Set the flag false to explicitly override the preset with None.
    """
    kwargs: dict[str, Any] = {
        "resolution": int(cfg.get("resolution", 560)),
        "positional_encoding_size": int(cfg.get("positional_encoding_size", 37)),
        "num_queries": int(cfg.get("num_queries", 100)),
        "num_select": int(cfg.get("num_queries", 100)),
        "num_classes": int(cfg.get("pretrained_class_count", 90)),
        "group_detr": 1,
        "two_stage": False,
        "bbox_reparam": bool(cfg.get("bbox_reparam", False)),
        "segmentation_head": bool(cfg.get("segmentation", True) if segmentation is None else segmentation),
        "device": str(device),
    }
    if not bool(cfg.get("use_pretrained_model", True)):
        kwargs["pretrain_weights"] = None
    elif cfg.get("pretrain_weights") is not None:
        kwargs["pretrain_weights"] = cfg["pretrain_weights"]
    # Otherwise RFDETRBaseConfig's own pretrain_weights default selects the
    # checkpoint published for that preset (rf-detr-base.pth).
    return RFDETRBaseConfig(**kwargs)
