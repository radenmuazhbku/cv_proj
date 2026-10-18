"""Re-evaluate saved PCS curriculum checkpoints with box/mask IoU metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from rfdetr_pcs.coco_episodes import CocoFewShotEpisodeDataset
from rfdetr_pcs.config import RFDETRBaseConfig, TrainConfig
from rfdetr_pcs.conditioning import class_ids_for_mode, uses_class_embeddings
from rfdetr_pcs.models.pcs import PCSDetector
from train_pcs_overfit import save_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, required=True)
    args = parser.parse_args()
    scope: dict[str, Any] = {}
    exec(compile(Path(args.config).read_text(), args.config, "exec"), scope)
    cfg = scope["cfg"]
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    resolution = int(cfg.get("resolution", 280))
    segmentation = bool(cfg.get("segmentation", False))
    model_config = RFDETRBaseConfig(
        pretrain_weights=None,
        resolution=resolution,
        positional_encoding_size=int(cfg.get("positional_encoding_size", 37)),
        num_queries=int(cfg.get("num_queries", 20)),
        num_select=int(cfg.get("num_queries", 20)),
        num_classes=int(cfg.get("pretrained_class_count", 90)),
        group_detr=1,
        two_stage=False,
        segmentation_head=segmentation,
        bbox_reparam=bool(cfg.get("bbox_reparam", False)),
        device=str(device),
    )
    model = PCSDetector.from_config(
        model_config, TrainConfig(dataset_dir=".", output_dir=str(Path(args.checkpoint).parent)),
        num_prompt_classes=int(cfg.get("num_prompt_classes", 91)),
        init_class_embeddings_from_detector=False,
        use_class_embeddings=uses_class_embeddings(cfg),
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    missing, skipped = model.load_compatible_state_dict(checkpoint["model"])
    if missing or skipped:
        print(f"checkpoint PCS migration: initialized missing keys={missing}; skipped keys={skipped}")
    dataset = CocoFewShotEpisodeDataset(
        cfg["train_image_dir"], cfg["train_annotation_file"], resolution=resolution,
        episodes=args.episodes, max_shots=int(cfg.get("max_prompt_boxes", 1)),
        negative_ratio=float(cfg.get("negative_ratio", 0.0)), seed=int(cfg.get("seed", 73)),
        negative_prompt_ratio=float(cfg.get("negative_prompt_ratio", 0.0)),
        negative_class_ratio=float(cfg.get("negative_class_ratio", 0.0)),
        training=True, include_masks=segmentation,
    )
    metrics = save_diagnostics(
        model, dataset, list(range(min(args.episodes, len(dataset)))), device,
        Path(args.checkpoint).parent / f"posthoc_eval_{args.episodes}",
        int(checkpoint["step"]), bool(cfg.get("include_prompt_box", True)),
        str(cfg.get("conditioning_mode", "visual_prompt")),
    )
    output = Path(args.checkpoint).parent / f"posthoc_metrics_{args.episodes}.json"
    output.write_text(json.dumps({"checkpoint": args.checkpoint, "episodes": args.episodes, **metrics}, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
