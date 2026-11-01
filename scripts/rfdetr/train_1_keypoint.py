#!/usr/bin/env python3
"""Train a single RF-DETR model jointly predicting boxes, instance masks, and keypoints."""
'''
python scripts/rfdetr/train_1_keypoint.py \
--dataset-dir datasets/scalebar_H1H2 \
--epochs 2 --batch-size 4 \
--grad-accum-steps 1     \
--output-dir logs/train_1_keypoint
'''
import argparse
from pathlib import Path
from typing import Any

from rfdetr import RFDETRKeypointPreview
from rfdetr.datasets._keypoint_schema import infer_coco_keypoint_schema
from rfdetr.training.callbacks.qual_visualizer import QualitativeVisualizerCallback
import logging
logging.getLogger("supervision").setLevel(logging.ERROR)

# Mapping of checkpoint families to architectural defaults so backbones load seamlessly
CHECKPOINT_ARCH_DEFAULTS = {
    "keypoint": {
        "patch_size": 12,
        "num_windows": 2,
        "resolution": 576,
        "positional_encoding_size": 48,
        "dec_layers": 4,
    },
    "nano": {
        "patch_size": 16,
        "num_windows": 2,
        "resolution": 384,
        "positional_encoding_size": 24,
        "dec_layers": 2,
    },
    "small": {
        "patch_size": 16,
        "num_windows": 2,
        "resolution": 512,
        "positional_encoding_size": 32,
        "dec_layers": 3,
    },
    "medium": {
        "patch_size": 16,
        "num_windows": 2,
        "resolution": 576,
        "positional_encoding_size": 36,
        "dec_layers": 4,
    },
    "base": {
        "patch_size": 14,
        "num_windows": 4,
        "resolution": 560,
        "positional_encoding_size": 37,
        "dec_layers": 3,
    },
    "large": {
        "patch_size": 16,
        "num_windows": 2,
        "resolution": 704,
        "positional_encoding_size": 44,
        "dec_layers": 4,
    },
}


def get_checkpoint_arch_defaults(checkpoint_name: str | None) -> dict[str, Any]:
    if not checkpoint_name:
        return {}
    name = Path(checkpoint_name).name.lower()
    for key, defaults in CHECKPOINT_ARCH_DEFAULTS.items():
        if key in name:
            return dict(defaults)
    return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RF-DETR for simultaneous box, mask, and keypoint prediction."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("datasets/rfdetr_dfff"),
        help="Path to Roboflow/COCO dataset directory containing train/ and valid/ splits.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per training step.")
    parser.add_argument("--grad-accum-steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs_keypoint"),
        help="Output directory for checkpoints and logs.",
    )

    # -------------------------------------------------------------------------
    # Available pretrained weight options:
    #   - "rf-detr-keypoint-preview-xlarge.pth" (patch_size=12, res=576)
    #   - "rf-detr-nano.pth" / "rf-detr-seg-nano.pt" (patch_size=16, res=384, dec_layers=2)
    #   - "rf-detr-small.pth" / "rf-detr-seg-small.pt" (patch_size=16, res=512, dec_layers=3)
    #   - "rf-detr-medium.pth" / "rf-detr-seg-medium.pt" (patch_size=16, res=576, dec_layers=4)
    #   - "rf-detr-base.pth" / "rf-detr-base-2.pth" (patch_size=14, res=560, num_windows=4)
    #   - "rf-detr-large.pth" / "rf-detr-seg-large.pt" (patch_size=16, res=704)
    #   - "none" (train from scratch)
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--pretrain-weights",
        type=str,
        default="rf-detr-nano.pth",
        help="Pretrained checkpoint to initialize weights from.",
    )

    # Architecture overrides to match custom or non-default checkpoints
    parser.add_argument(
        "--patch-size",
        type=int,
        default=None,
        help="Patch size for the backbone. Auto-derived from checkpoint family if omitted.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Input image square resolution. Auto-derived from checkpoint family if omitted.",
    )
    parser.add_argument(
        "--num-windows",
        type=int,
        default=None,
        help="Number of window partitions for windowed attention. Auto-derived if omitted.",
    )
    parser.add_argument(
        "--dec-layers",
        type=int,
        default=None,
        help="Number of transformer decoder layers. Auto-derived if omitted.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_ann = args.dataset_dir / "train" / "_annotations.coco.json"
    if not train_ann.exists():
        train_ann_candidates = list(args.dataset_dir.glob("*_annotations*.coco.json"))
        if train_ann_candidates:
            train_ann = train_ann_candidates[0]

    pretrain_weights = None if str(args.pretrain_weights).lower() in ("none", "") else args.pretrain_weights

    # Resolve architectural flags: user explicit args > checkpoint defaults > model defaults
    arch_defaults = get_checkpoint_arch_defaults(pretrain_weights)
    patch_size = args.patch_size or arch_defaults.get("patch_size", 12)
    resolution = args.resolution or arch_defaults.get("resolution", 576)
    num_windows = args.num_windows or arch_defaults.get("num_windows", 2)
    dec_layers = args.dec_layers or arch_defaults.get("dec_layers", 4)
    positional_encoding_size = arch_defaults.get("positional_encoding_size", resolution // patch_size)

    # Ensure resolution is divisible by patch_size * num_windows
    block_size = patch_size * num_windows
    if resolution % block_size != 0:
        adjusted_res = round(resolution / block_size) * block_size
        if adjusted_res == 0:
            adjusted_res = block_size
        print(f"Adjusting resolution {resolution} -> {adjusted_res} to be divisible by {block_size}")
        resolution = adjusted_res
        positional_encoding_size = resolution // patch_size

    print(
        f"Architecture configuration: patch_size={patch_size}, resolution={resolution}, "
        f"num_windows={num_windows}, dec_layers={dec_layers}, positional_encoding_size={positional_encoding_size}"
    )

    model_kwargs: dict[str, Any] = {
        "segmentation_head": True,
        "pretrain_weights": pretrain_weights,
        "patch_size": patch_size,
        "resolution": resolution,
        "num_windows": num_windows,
        "dec_layers": dec_layers,
        "positional_encoding_size": positional_encoding_size,
    }

    if train_ann.exists():
        print(f"Inferring keypoint and class schema from: {train_ann}")
        schema = infer_coco_keypoint_schema(train_ann)
        class_names = schema.class_names
        num_keypoints_per_class = schema.num_keypoints_per_class
        keypoint_oks_sigmas = schema.keypoint_oks_sigmas
        print(f"Classes ({len(class_names)}): {class_names}")
        print(f"Keypoints per class: {num_keypoints_per_class}")

        model_kwargs["num_classes"] = len(class_names)
        model_kwargs["num_keypoints_per_class"] = num_keypoints_per_class
    else:
        print(f"Warning: {train_ann} not found; initializing standard RFDETRKeypointPreview.")
        class_names = None
        keypoint_oks_sigmas = None

    model = RFDETRKeypointPreview(**model_kwargs)

    print(
        f"Starting multi-task (box + mask + keypoint) training on {args.dataset_dir} "
        f"for {args.epochs} epochs -> {args.output_dir}"
    )

    qual_cb = QualitativeVisualizerCallback(
        output_dir=args.output_dir / "visualizations",
        num_images=8,
        threshold=0.3,
        class_names=class_names,
    )

    model.train(
        dataset_dir=str(args.dataset_dir),
        dataset_file="roboflow",
        class_names=class_names,
        keypoint_oks_sigmas=keypoint_oks_sigmas,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        mask_ce_loss_coef=5.0,
        mask_dice_loss_coef=5.0,
        keypoint_l1_loss_coef=1.0,
        keypoint_nll_loss_coef=1.0,
        output_dir=str(args.output_dir),
        callbacks=[qual_cb],
    )


if __name__ == "__main__":
    main()

"""
Comparison command examples:
/bin/bash -c "mkdir -p logs_keypoint/comparisons"
.venv/bin/python scripts/rfdetr/generate_gt_pred_comparisons.py \
    --checkpoint logs_keypoint/checkpoint_best_ema.pth \
    --images-dir datasets/rfdetr_dfff/valid \
    --annotations datasets/rfdetr_dfff/valid/_annotations.coco.json \
    --output-dir logs_keypoint/comparisons/valid --count 100 --batch-size 4
"""
