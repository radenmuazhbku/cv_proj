"""Evaluate a trained support/query checkpoint with COCO bbox AP/AR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from rfdetr_promptable.coco_episodes import CocoFewShotEpisodeDataset
from rfdetr_promptable.config import RFDETRBaseConfig
from rfdetr_promptable.promptable import PromptableDetector
from train_promptable_fewshot import run_coco_map_eval


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--query-image-dir", type=Path, required=True)
    parser.add_argument("--query-annotation-file", type=Path, required=True)
    parser.add_argument("--support-image-dir", type=Path)
    parser.add_argument("--support-annotation-file", type=Path, required=True)
    parser.add_argument("--max-shots", type=int, default=5)
    parser.add_argument("--query-batch-size", type=int, default=8)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--resolution", type=int, default=560)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("logs/promptable_coco_metrics.json"))
    args = parser.parse_args()

    support_image_dir = args.support_image_dir or args.query_image_dir
    dataset = CocoFewShotEpisodeDataset(
        args.query_image_dir,
        args.query_annotation_file,
        resolution=args.resolution,
        max_shots=args.max_shots,
        negative_eval_per_class=0,
        support_annotation_file=args.support_annotation_file,
        training=False,
        seed=args.seed,
    )
    config = RFDETRBaseConfig(
        pretrain_weights=None,
        resolution=args.resolution,
        num_queries=args.num_queries,
        num_select=args.num_queries,
        group_detr=1,
        two_stage=False,
        device=args.device,
    )
    model = PromptableDetector.from_config(config).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model.eval()
    # The current COCO dataset wrapper uses one image root for both files. Fail
    # early instead of silently loading support/query image IDs from wrong roots.
    if support_image_dir != args.query_image_dir:
        raise ValueError(
            "support and query images must currently share a directory; pass the same --support-image-dir and --query-image-dir"
        )
    result = run_coco_map_eval(
        model,
        dataset,
        torch.device(args.device),
        args.max_shots,
        args.query_batch_size,
        args.seed,
    )
    payload = {
        "checkpoint": str(args.checkpoint),
        "query_annotation_file": str(args.query_annotation_file),
        "support_annotation_file": str(args.support_annotation_file),
        "metrics": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({key: result[key] for key in ("mAP", "mAP50", "mAP75", "mAR100")}, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
