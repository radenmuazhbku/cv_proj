# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Lightning callbacks for RF-DETR training."""

from rfdetr_pcs.training.callbacks.best_model import BestModelCallback, RFDETREarlyStopping
from rfdetr_pcs.training.callbacks.coco_eval import COCOEvalCallback
from rfdetr_pcs.training.callbacks.drop_schedule import DropPathCallback
from rfdetr_pcs.training.callbacks.ema import RFDETREMACallback
from rfdetr_pcs.training.callbacks.gpu_memory_progress_bar import (
    GPUMemoryRichProgressBar,
    GPUMemoryTQDMProgressBar,
)

__all__ = [
    "BestModelCallback",
    "COCOEvalCallback",
    "DropPathCallback",
    "GPUMemoryRichProgressBar",
    "GPUMemoryTQDMProgressBar",
    "RFDETREMACallback",
    "RFDETREarlyStopping",
]
