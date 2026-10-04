"""
uv run python scripts/train_promptable_rfdetr.py   \
--image-dir datasets/mscoco/train2017   \
--annotation-file datasets/mscoco_prompt_split/train.json   \
--steps 100000   --batch-size 1   --resolution 560   \
--device cuda   --output logs/promptable1.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

from rfdetr_promptable.coco_prompt import CocoBoxPromptDataset, collate_prompts
from rfdetr_promptable.config import RFDETRBaseConfig
from rfdetr_promptable.promptable import PromptableDetector
from rfdetr_promptable.utilities.box_ops import box_cxcywh_to_xyxy

from tqdm import tqdm


def build_model(resolution: int, device: str) -> PromptableDetector:
    config = RFDETRBaseConfig(
        pretrain_weights=None,
        resolution=resolution,
        num_queries=1,
        num_select=1,
        group_detr=1,
        two_stage=False,
        device=device,
    )
    model = PromptableDetector.from_config(config)
    return model.to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("datasets/mscoco"))
    parser.add_argument("--split", default="train2017")
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--annotation-file", type=Path)
    parser.add_argument("--validation-image-dir", type=Path)
    parser.add_argument("--validation-annotation-file", type=Path)
    parser.add_argument("--test-image-dir", type=Path)
    parser.add_argument("--test-annotation-file", type=Path)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--max-validation-examples", type=int)
    parser.add_argument("--max-test-examples", type=int)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--negative-iou-threshold", type=float, default=0.1)
    parser.add_argument("--max-examples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--objectness-weight", type=float, default=1.0)
    parser.add_argument("--bbox-weight", type=float, default=5.0)
    parser.add_argument("--giou-weight", type=float, default=2.0)
    parser.add_argument("--resolution", type=int, default=560)
    parser.add_argument("--output", type=Path, default=Path("logs/promptable_rfdetr.pt"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    annotation_file = args.annotation_file or (
        args.data_root / "annotations_trainval2017" / "annotations" / f"instances_{args.split}.json"
    )
    image_dir = args.image_dir or (args.data_root / args.split)
    default_annotation_dir = args.data_root / "annotations_trainval2017" / "annotations"
    validation_image_dir = args.validation_image_dir or args.data_root / "val2017"
    validation_annotation_file = args.validation_annotation_file
    test_image_dir = args.test_image_dir or validation_image_dir
    test_annotation_file = args.test_annotation_file
    dataset = CocoBoxPromptDataset(
        image_dir,
        annotation_file,
        resolution=args.resolution,
        negative_ratio=args.negative_ratio,
        negative_iou_threshold=args.negative_iou_threshold,
        seed=args.seed,
        max_examples=args.max_examples,
    )
    if not dataset:
        raise RuntimeError(f"No annotated images found in {annotation_file}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_prompts)

    model = build_model(args.resolution, args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    model.train()
    iterator = iter(loader)
    best_val_iou = float("-inf")
    metrics: dict[str, object] = {"seed": args.seed, "train": {"steps": args.steps}}

    def run_eval(annotation_path: Path, images_path: Path, max_examples: int | None) -> dict[str, float | int]:
        eval_dataset = CocoBoxPromptDataset(
            images_path,
            annotation_path,
            resolution=args.resolution,
            negative_ratio=0.0,
            seed=args.seed,
            max_examples=max_examples,
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=collate_prompts,
        )
        iou_values: list[float] = []
        score_values: list[float] = []
        model.eval()
        with torch.inference_mode():
            for eval_images, eval_boxes, eval_present in eval_loader:
                eval_images = eval_images.to(args.device)
                eval_boxes = eval_boxes.to(args.device)
                eval_present = eval_present.to(args.device)
                eval_outputs = model(eval_images, eval_boxes)
                probs = eval_outputs["pred_logits"].sigmoid()[..., 0]
                scores = probs[eval_present > 0.5]
                if scores.numel():
                    score_values.extend(scores.float().cpu().tolist())
                    positive_predictions = eval_outputs["pred_boxes"][eval_present > 0.5]
                    positive_targets = eval_boxes[eval_present > 0.5]
                    pred_xyxy = box_cxcywh_to_xyxy(positive_predictions)
                    target_xyxy = box_cxcywh_to_xyxy(positive_targets)
                    iou_matrix = box_iou(
                        pred_xyxy.reshape(-1, 4),
                        target_xyxy.reshape(-1, 4),
                    )
                    iou_values.extend(iou_matrix.diag().float().cpu().tolist())
        model.train()
        if not iou_values:
            raise RuntimeError(f"No positive evaluation examples in {annotation_path}")
        iou_tensor = torch.tensor(iou_values)
        return {
            "examples": len(iou_values),
            "mean_score": sum(score_values) / len(score_values),
            "mean_iou": float(iou_tensor.mean()),
            "median_iou": float(iou_tensor.median()),
            "recall50": float((iou_tensor >= 0.5).float().mean()),
            "recall75": float((iou_tensor >= 0.75).float().mean()),
        }

    if validation_annotation_file is None:
        candidate = default_annotation_dir / "instances_val2017.json"
        validation_annotation_file = candidate if candidate.exists() else None
    validation_enabled = args.eval_every > 0 and validation_annotation_file is not None

    for step in (pbar:=tqdm(range(args.steps))):
        try:
            images, prompt_boxes, present = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images, prompt_boxes, present = next(iterator)
        images, prompt_boxes, present = images.to(args.device), prompt_boxes.to(args.device), present.to(args.device)
        outputs = model(images, prompt_boxes)
        losses = model.loss(outputs, prompt_boxes, present)
        loss = (
            args.objectness_weight * losses["loss_objectness"]
            + args.bbox_weight * losses["loss_bbox"]
            + args.giou_weight * losses["loss_giou"]
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        pbar_str = f"loss={loss.item():.4f}"
        pbar.set_postfix_str(pbar_str)
        if step % args.log_every == 0:
            print(pbar_str)
        if validation_enabled and (step + 1) % args.eval_every == 0:
            # Validate on positives only. The loader uses all eligible seen-class
            # annotations unless max-validation-examples is supplied for a smoke run.
            val_result = run_eval(
                validation_annotation_file,
                validation_image_dir,
                args.max_validation_examples,
            )
            metrics[f"validation_step_{step + 1}"] = val_result
            print(f"validation step={step + 1} mean_iou={val_result['mean_iou']:.4f}")
            if float(val_result["mean_iou"]) > best_val_iou:
                best_val_iou = float(val_result["mean_iou"])
                best_path = args.output.with_name(f"{args.output.stem}.best{args.output.suffix}")
                torch.save(model.state_dict(), best_path)
                metrics["best_checkpoint"] = str(best_path)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    model.eval()

    if validation_annotation_file is not None:
        metrics["seen_validation_final"] = run_eval(
            validation_annotation_file,
            validation_image_dir,
            args.max_validation_examples,
        )
    if test_annotation_file is not None:
        metrics["held_out_test"] = run_eval(test_annotation_file, test_image_dir, args.max_test_examples)
    metrics_path = args.metrics_output or args.output.with_suffix(".metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    sample = dataset[0]
    image_id = sample["image_id"]
    metadata = dataset.coco.loadImgs([image_id])[0]
    image = Image.open(image_dir / metadata["file_name"]).convert("RGB")
    width, height = image.size
    box = sample["prompt_box"]
    pixel_box = [
        float((box[0] - box[2] / 2) * width),
        float((box[1] - box[3] / 2) * height),
        float((box[0] + box[2] / 2) * width),
        float((box[1] + box[3] / 2) * height),
    ]
    prediction = model.predict(image, [pixel_box], threshold=0.0)
    print(f"inference boxes={prediction['boxes'].shape[0]} checkpoint={args.output}")
    print(f"metrics={metrics_path}")


if __name__ == "__main__":
    main()
