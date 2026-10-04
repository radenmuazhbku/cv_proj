"""Support/query and same-image inference for the promptable detector."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from rfdetr_promptable.config import RFDETRBaseConfig
from rfdetr_promptable.promptable import PromptableDetector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--support-example",
        type=str,
        nargs=6,
        action="append",
        required=True,
        metavar=("IMAGE", "X0", "Y0", "X1", "Y1", "GROUP"),
        help="pixel xyxy box plus concept GROUP; repeat for multi-shot support examples",
    )
    parser.add_argument("--query-image", type=Path, action="append", required=True)
    parser.add_argument("--query-group", type=int, action="append", help="support GROUP id per query; defaults to first group")
    parser.add_argument("--resolution", type=int, default=560)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("logs/support_query_predictions.json"))
    args = parser.parse_args()

    device = torch.device(args.device)
    model = PromptableDetector.from_config(
        RFDETRBaseConfig(
            pretrain_weights=None,
            resolution=args.resolution,
            num_queries=args.num_queries,
            num_select=args.num_queries,
            group_detr=1,
            two_stage=False,
            device=args.device,
        )
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    support_images, support_boxes, groups = [], [], []
    for raw_example in args.support_example:
        image_path = Path(raw_example[0])
        coords = [float(value) for value in raw_example[1:5]]
        group = int(raw_example[5])
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        support_images.append(TF.resize(TF.to_tensor(image), [args.resolution, args.resolution]))
        x0, y0, x1, y1 = coords
        support_boxes.append([(x0 + x1) / (2 * width), (y0 + y1) / (2 * height), (x1 - x0) / width, (y1 - y0) / height])
        groups.append(group)

    query_images, query_sizes = [], []
    for image_path in args.query_image:
        image = Image.open(image_path).convert("RGB")
        query_sizes.append((image.width, image.height))
        query_images.append(TF.resize(TF.to_tensor(image), [args.resolution, args.resolution]))
    query_groups = args.query_group or [groups[0]] * len(query_images)
    if len(query_groups) != len(query_images):
        parser.error("provide one --query-group per --query-image")
    unknown_groups = set(query_groups) - set(groups)
    if unknown_groups:
        parser.error(f"query groups have no support examples: {sorted(unknown_groups)}")
    # Grouped support expects per-support-image group IDs; multiple examples may share a group.
    support_group_ids = torch.tensor(groups, dtype=torch.long, device=device)
    query_to_support = torch.tensor(query_groups, dtype=torch.long, device=device)
    predictions = model.predict_support_query(
        torch.stack(support_images),
        torch.tensor(support_boxes, dtype=torch.float32, device=device),
        torch.stack(query_images),
        support_group_ids=support_group_ids,
        query_to_support=query_to_support,
        threshold=args.threshold,
    )
    result = []
    for path, (width, height), prediction in zip(args.query_image, query_sizes, predictions, strict=True):
        result.append({"query_image": str(path), "width": width, "height": height, "boxes_xyxy": prediction["boxes"].tolist(), "scores": prediction["scores"].tolist()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"support_examples={len(support_images)} query_images={len(query_images)} output={args.output}")


if __name__ == "__main__":
    main()
