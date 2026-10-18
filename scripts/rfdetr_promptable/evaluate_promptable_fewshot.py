"""Evaluate a trained same-image prompted checkpoint with COCO bbox AP/AR."""

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
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--annotation-file", type=Path, required=True)
    parser.add_argument("--query-batch-size", type=int, default=8)
    parser.add_argument("--max-query-images", type=int)
    parser.add_argument(
        "--include-prompt-box-as-target", action=argparse.BooleanOptionalAction, default=True,
        help="Include each prompt instance in the COCO GT targets (default: enabled).",
    )
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--resolution", type=int, default=560)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("logs/promptable_coco_metrics.json"))
    args = parser.parse_args()

    dataset = CocoFewShotEpisodeDataset(
        args.image_dir,
        args.annotation_file,
        resolution=args.resolution,
        max_shots=1,
        negative_eval_per_class=0,
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
    result = run_coco_map_eval(
        model,
        dataset,
        torch.device(args.device),
        args.query_batch_size,
        args.seed,
        max_query_images=args.max_query_images,
        include_prompt_box_as_target=args.include_prompt_box_as_target,
    )
    payload = {
        "checkpoint": str(args.checkpoint),
        "annotation_file": str(args.annotation_file),
        "metrics": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("mAP", "mAP50", "mAP75", "mAR100", "query_image_category_pairs", "elapsed_seconds", "query_pairs_per_second")
            },
            indent=2,
        )
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
