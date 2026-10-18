# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""RF-DETR CLI package.

The ``rfdetr`` console script and ``python -m rfdetr`` both invoke :func:`main`, which runs
:class:`~rfdetr_pcs.training.cli.RFDETRCli` (Lightning CLI with jsonargparse).
"""

from rfdetr_pcs.training.cli import main

__all__ = ["main"]
