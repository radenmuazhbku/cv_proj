"""Generate RF-DETR training-curve images from a CSVLogger metrics file.

Example:
    .venv/bin/python scripts/plot_training_curves.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rfdetr.visualize.training import plot_loss_metrics, plot_map_metrics, plot_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=Path("logs/metrics.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("logs/visualizations"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_metrics(args.metrics, args.output_dir / "training_metrics.png")
    plot_loss_metrics(args.metrics, args.output_dir / "loss_metrics.png", loss_log_scale=True)
    plot_map_metrics(args.metrics, args.output_dir / "map_metrics.png")
    print(f"Saved training curves to {args.output_dir}")


if __name__ == "__main__":
    main()
