# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Concrete RF-DETR model variant classes.

All classes inherit from :class:`~rfdetr_promptable.detr.RFDETR` which remains defined in ``rfdetr_promptable.detr``. Backward-compatible
access from ``rfdetr_promptable.detr`` is provided via lazy ``__getattr__`` re-exports, so importing ``rfdetr_promptable.variants`` no longer
depends on a fragile eager ``detr -> variants`` import sequence.
"""

from __future__ import annotations

__all__ = [
    "RFDETRBase",
    "RFDETRKeypointPreview",
    "RFDETRNano",
    "RFDETRSmall",
    "RFDETRMedium",
    "RFDETRLarge",
    "RFDETRLargeDeprecated",
    "RFDETRSeg",
    "RFDETRSegPreview",
    "RFDETRSegNano",
    "RFDETRSegSmall",
    "RFDETRSegMedium",
    "RFDETRSegLarge",
    "RFDETRSegXLarge",
    "RFDETRSeg2XLarge",
]

from typing import Any

from deprecate import deprecated_class

from rfdetr_promptable.config import (
    KeypointTrainConfig,
    ModelConfig,
    RFDETRBaseConfig,
    RFDETRKeypointPreviewConfig,
    RFDETRLargeConfig,
    RFDETRLargeDeprecatedConfig,
    RFDETRMediumConfig,
    RFDETRNanoConfig,
    RFDETRSeg2XLargeConfig,
    RFDETRSegLargeConfig,
    RFDETRSegMediumConfig,
    RFDETRSegNanoConfig,
    RFDETRSegPreviewConfig,
    RFDETRSegSmallConfig,
    RFDETRSegXLargeConfig,
    RFDETRSmallConfig,
    SegmentationTrainConfig,
)
from rfdetr_promptable.detr import RFDETR


@deprecated_class(
    target=None,
    deprecated_in="1.7.0",
    remove_in="2.0.0",
)
class RFDETRBase(RFDETR):
    """Train an RF-DETR Base model.

    Training accepts custom square integer ``resolution`` values. The value must be divisible by ``patch_size *
    num_windows``.
    """

    size = "rfdetr_promptable-base"
    _model_config_class = RFDETRBaseConfig


class RFDETRNano(RFDETR):
    """Train an RF-DETR Nano model.

    Training accepts custom square integer ``resolution`` values. The value must be divisible by ``patch_size *
    num_windows``.
    """

    size = "rfdetr_promptable-nano"
    _model_config_class = RFDETRNanoConfig


class RFDETRKeypointPreview(RFDETR):
    """Train or run inference with the RF-DETR keypoint preview model.

    Training accepts custom square integer ``resolution`` values. The value must be divisible by ``patch_size *
    num_windows``.
    """

    size = "rfdetr_promptable-keypoint-preview"
    _model_config_class = RFDETRKeypointPreviewConfig
    _train_config_class = KeypointTrainConfig


class RFDETRSmall(RFDETR):
    """Train an RF-DETR Small model.

    Training accepts custom square integer ``resolution`` values. The value must be divisible by ``patch_size *
    num_windows``.
    """

    size = "rfdetr_promptable-small"
    _model_config_class = RFDETRSmallConfig


class RFDETRMedium(RFDETR):
    """Train an RF-DETR Medium model.

    Training accepts custom square integer ``resolution`` values. The value must be divisible by ``patch_size *
    num_windows``.
    """

    size = "rfdetr_promptable-medium"
    _model_config_class = RFDETRMediumConfig


@deprecated_class(
    target=None,
    deprecated_in="1.7.0",
    remove_in="2.0.0",
)
class RFDETRLargeDeprecated(RFDETR):
    """Train an RF-DETR Large model using the legacy config.

    Training accepts custom square integer ``resolution`` values. The value must be divisible by ``patch_size *
    num_windows``.
    """

    size = "rfdetr_promptable-large"
    _model_config_class = RFDETRLargeDeprecatedConfig


class RFDETRLarge(RFDETR):
    """Train an RF-DETR Large model.

    Training accepts custom square integer ``resolution`` values. The value must be divisible by ``patch_size *
    num_windows``.
    """

    size = "rfdetr_promptable-large"

    def get_model_config(self, **kwargs: Any) -> ModelConfig:
        return RFDETRLargeConfig(**kwargs)


class RFDETRSeg(RFDETR):
    """Base class for all RF-DETR segmentation models.

    Training accepts custom square integer ``resolution`` values. Most segmentation variants use multiples of 24;
    ``RFDETRSegNano`` uses multiples of 12.
    """

    _train_config_class = SegmentationTrainConfig


@deprecated_class(
    target=None,
    deprecated_in="1.7.0",
    remove_in="2.0.0",
)
class RFDETRSegPreview(RFDETRSeg):
    """Train an RF-DETR Segmentation Preview model.

    Training accepts custom square integer ``resolution`` values. The value must be divisible by ``patch_size *
    num_windows``. Deprecated in v1.7.0, scheduled for removal in v2.0.0.
    """

    size = "rfdetr_promptable-seg-preview"
    _model_config_class = RFDETRSegPreviewConfig


class RFDETRSegNano(RFDETRSeg):
    """Train an RF-DETR Segmentation Nano model.

    Training accepts custom square integer ``resolution`` values. The value must be divisible by ``patch_size *
    num_windows``; this variant uses multiples of 12.
    """

    size = "rfdetr_promptable-seg-nano"
    _model_config_class = RFDETRSegNanoConfig


class RFDETRSegSmall(RFDETRSeg):
    """Train an RF-DETR Segmentation Small model.

    Training accepts custom square integer ``resolution`` values. The value must be divisible by ``patch_size *
    num_windows``; this variant uses multiples of 24.
    """

    size = "rfdetr_promptable-seg-small"
    _model_config_class = RFDETRSegSmallConfig


class RFDETRSegMedium(RFDETRSeg):
    """Train an RF-DETR Segmentation Medium model.

    Training accepts custom square integer ``resolution`` values. The value must be divisible by ``patch_size *
    num_windows``; this variant uses multiples of 24.
    """

    size = "rfdetr_promptable-seg-medium"
    _model_config_class = RFDETRSegMediumConfig


class RFDETRSegLarge(RFDETRSeg):
    """Train an RF-DETR Segmentation Large model.

    Training accepts custom square integer ``resolution`` values. The value must be divisible by ``patch_size *
    num_windows``; this variant uses multiples of 24.
    """

    size = "rfdetr_promptable-seg-large"
    _model_config_class = RFDETRSegLargeConfig


class RFDETRSegXLarge(RFDETRSeg):
    """Train an RF-DETR Segmentation XLarge model.

    Training accepts custom square integer ``resolution`` values. The value must be divisible by ``patch_size *
    num_windows``; this variant uses multiples of 24.
    """

    size = "rfdetr_promptable-seg-xlarge"
    _model_config_class = RFDETRSegXLargeConfig


class RFDETRSeg2XLarge(RFDETRSeg):
    """Train an RF-DETR Segmentation 2XLarge model.

    Training accepts custom square integer ``resolution`` values. The value must be divisible by ``patch_size *
    num_windows``; this variant uses multiples of 24.
    """

    size = "rfdetr_promptable-seg-2xlarge"
    _model_config_class = RFDETRSeg2XLargeConfig
