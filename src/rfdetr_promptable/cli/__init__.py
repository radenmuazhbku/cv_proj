# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""RF-DETR CLI package.

The ``rfdetr_promptable`` console script and ``python -m rfdetr_promptable`` both invoke :func:`main`, which runs
:class:`~rfdetr_promptable.training.cli.RFDETRCli` (Lightning CLI with jsonargparse).
"""

from rfdetr_promptable.training.cli import main

__all__ = ["main"]
