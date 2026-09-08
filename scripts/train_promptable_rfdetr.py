"""Train and smoke-test isolated box-promptable RF-DETR on MS COCO."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from rfdetr_promptable.coco_prompt import CocoBoxPromptDataset, collate_prompts
from rfdetr_promptable.config import RFDETRNanoConfig
from rfdetr_promptable.promptable import PromptableDetector


def build_model(resolution: int, device: str) -> PromptableDetector:
    config = RFDETRNanoConfig(
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
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--negative-iou-threshold", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--objectness-weight", type=float, default=1.0)
    parser.add_argument("--bbox-weight", type=float, default=5.0)
    parser.add_argument("--giou-weight", type=float, default=2.0)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--output", type=Path, default=Path("logs/promptable_rfdetr.pt"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    annotation_file = args.annotation_file or (
        args.data_root / "annotations_trainval2017" / "annotations" / f"instances_{args.split}.json"
    )
    image_dir = args.image_dir or (args.data_root / args.split)
    dataset = CocoBoxPromptDataset(
        image_dir,
        annotation_file,
        resolution=args.resolution,
        negative_ratio=args.negative_ratio,
        negative_iou_threshold=args.negative_iou_threshold,
        seed=args.seed,
    )
    if not dataset:
        raise RuntimeError(f"No annotated images found in {annotation_file}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_prompts)

    model = build_model(args.resolution, args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    model.train()
    iterator = iter(loader)
    for step in range(args.steps):
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
        print(f"step={step + 1} loss={loss.item():.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    model.eval()
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


if __name__ == "__main__":
    main()
