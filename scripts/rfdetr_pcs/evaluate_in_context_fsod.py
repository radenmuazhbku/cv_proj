"""Training-Free In-Context Learning (ICL) Few-Shot Object Detection & Segmentation.

Evaluates a frozen PCS model on novel categories using K-shot support exemplars
as in-context visual prompts, without any parameter updates or fine-tuning at test time.

Supported Datasets:
- MS COCO (60 Base / 20 Novel)
- PASCAL VOC (15 Base / 5 Novel across Split 1, Split 2, Split 3)

Evaluation:
- Standard COCOeval for bounding boxes (nAP, nAP50, nAP75, bAP, mAP)
- Standard COCOeval for instance masks (segm nAP, segm nAP50, segm nAP75)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rfdetr_pcs.config import TrainConfig
from rfdetr_pcs.evaluation.fsod_eval import FSODEvaluator
from rfdetr_pcs.model_presets import build_base_model_config
from rfdetr_pcs.models.pcs import PCSDetector
from pycocotools.coco import COCO


class StandardTestImageDataset(Dataset):
    """Loads raw test query images for training-free in-context evaluation."""

    def __init__(self, image_dir: Path, annotation_file: Path, resolution: int = 560):
        self.image_dir = image_dir
        self.coco = COCO(str(annotation_file))
        self.image_ids = sorted(self.coco.getImgIds())
        self.resolution = resolution

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img_id = self.image_ids[idx]
        meta = self.coco.loadImgs([img_id])[0]
        img_path = self.image_dir / meta["file_name"]
        pil_img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = pil_img.size

        tensor = TF.resize(TF.to_tensor(pil_img), [self.resolution, self.resolution])
        return {
            "image": tensor,
            "image_id": img_id,
            "orig_size": (orig_h, orig_w),
        }


def collate_test_queries(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": torch.stack([item["image"] for item in batch]),
        "image_ids": [item["image_id"] for item in batch],
        "orig_sizes": [item["orig_size"] for item in batch],
    }


def load_in_context_support_exemplars(
    support_coco: COCO,
    support_image_dir: Path,
    category_id: int,
    shots: int,
    resolution: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Load K support images and boxes for category_id as in-context visual prompts."""
    ann_ids = support_coco.getAnnIds(catIds=[category_id], iscrowd=False)
    if not ann_ids:
        return None

    anns = support_coco.loadAnns(ann_ids)
    if len(anns) > shots:
        anns = anns[:shots]

    support_images = []
    support_boxes = []

    for ann in anns:
        img_meta = support_coco.loadImgs([ann["image_id"]])[0]
        img_path = support_image_dir / img_meta["file_name"]
        pil_img = Image.open(img_path).convert("RGB")
        w, h = pil_img.size

        tensor = TF.resize(TF.to_tensor(pil_img), [resolution, resolution])
        x, y, bw, bh = ann["bbox"]
        cx = (x + bw / 2.0) / w
        cy = (y + bh / 2.0) / h
        norm_bw = bw / w
        norm_bh = bh / h

        support_images.append(tensor)
        support_boxes.append(torch.tensor([[cx, cy, norm_bw, norm_bh]], dtype=torch.float32))

    # Shape: [1, S, 3, H, W]
    support_images_tensor = torch.stack(support_images).unsqueeze(0)
    # Shape: [1, S, 1, 4]
    support_boxes_tensor = torch.stack(support_boxes).unsqueeze(0)
    support_labels = torch.ones(support_boxes_tensor.shape[:-1], dtype=torch.long)
    support_padding_mask = torch.zeros(support_boxes_tensor.shape[:-1], dtype=torch.bool)

    return support_images_tensor, support_boxes_tensor, support_labels, support_padding_mask


@torch.inference_mode()
def evaluate_in_context_fsod(
    model: PCSDetector,
    support_coco: COCO,
    support_image_dir: Path,
    test_loader: DataLoader,
    target_category_ids: list[int],
    shots: int,
    device: torch.device,
    resolution: int,
    evaluator: FSODEvaluator,
    include_masks: bool = True,
    score_threshold: float = 0.05,
) -> dict[str, Any]:
    """Perform training-free in-context evaluation across all target categories."""
    model.eval()

    total_categories = len(target_category_ids)
    for cat_idx, cat_id in enumerate(target_category_ids, 1):
        support_data = load_in_context_support_exemplars(
            support_coco, support_image_dir, cat_id, shots, resolution
        )
        if support_data is None:
            continue

        sup_imgs, sup_boxes, sup_labels, sup_pad = [x.to(device) for x in support_data]
        sup_count = sup_imgs.shape[1]

        for batch in tqdm(test_loader, desc=f"ICL Cat {cat_id} ({cat_idx}/{total_categories})", leave=False):
            query_imgs = batch["images"].unsqueeze(0).to(device)  # [1, B, 3, H, W]
            batch_size = query_imgs.shape[1]

            output = model.forward_support_query(
                support_images=sup_imgs,
                support_boxes=sup_boxes,
                query_images=query_imgs,
                support_labels=sup_labels,
                support_padding_mask=sup_pad,
                return_support_outputs=False,
            )

            # Assign predicted category to the queried in-context class
            cat_tensor = torch.full((batch_size,), cat_id, dtype=torch.long, device=device)
            evaluator.add_predictions(
                image_ids=batch["image_ids"],
                category_ids=cat_tensor,
                output=output,
                image_sizes=batch["orig_sizes"],
                include_masks=include_masks,
            )

    return evaluator.evaluate()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint (.pt or .pth)")
    parser.add_argument("--config", type=Path, help="Optional config file to build model")
    parser.add_argument("--support-annotation", type=Path, required=True, help="Few-shot K-shot support JSON")
    parser.add_argument("--support-image-dir", type=Path, required=True, help="Support images directory")
    parser.add_argument("--test-annotation", type=Path, required=True, help="Test set ground-truth JSON")
    parser.add_argument("--test-image-dir", type=Path, required=True, help="Test images directory")
    parser.add_argument("--shots", type=int, default=10, help="Number of in-context support shots K")
    parser.add_argument("--resolution", type=int, default=560)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--novel-categories", nargs="+", type=int, default=[], help="Novel class IDs for nAP breakdown")
    parser.add_argument("--base-categories", nargs="+", type=int, default=[], help="Base class IDs for bAP breakdown")
    parser.add_argument("--target-categories", nargs="+", type=int, default=[], help="Categories to evaluate (default: all support classes)")
    parser.add_argument("--max-test-images", type=int, default=0, help="Limit number of test images for fast testing")
    parser.add_argument("--segmentation", action="store_true", default=True, help="Evaluate instance segmentation masks")
    parser.add_argument("--output-json", type=Path, default=Path("eval_icl_results.json"))
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Loading checkpoint {args.checkpoint} onto {device}...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)

    # Build model preset
    model_cfg = {
        "resolution": args.resolution,
        "segmentation": args.segmentation,
        "num_queries": 100,
        "positional_encoding_size": 37,
        "device": str(device),
    }
    if args.config and args.config.exists():
        namespace: dict[str, Any] = {}
        exec(compile(args.config.read_text(), str(args.config), "exec"), namespace)
        model_cfg.update(namespace.get("cfg", {}))

    base_config = build_base_model_config(model_cfg, device)
    model = PCSDetector.from_config(
        base_config, TrainConfig(dataset_dir=".", output_dir="."),
        num_prompt_classes=int(model_cfg.get("num_prompt_classes", 91)),
        init_class_embeddings_from_detector=False,
    ).to(device)

    missing, skipped = model.load_compatible_state_dict(state_dict)
    print(f"Model loaded. (missing={len(missing)}, skipped={len(skipped)})")

    # Load Support & Test
    print(f"Loading support annotations from {args.support_annotation}...")
    support_coco = COCO(str(args.support_annotation))
    test_dataset = StandardTestImageDataset(args.test_image_dir, args.test_annotation, args.resolution)
    if args.max_test_images > 0:
        test_dataset.image_ids = test_dataset.image_ids[: args.max_test_images]
    print(f"Evaluating across {len(test_dataset)} test images...")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_test_queries,
        pin_memory=device.type == "cuda",
    )

    novel_cat_ids = set(args.novel_categories) if args.novel_categories else set()
    base_cat_ids = set(args.base_categories) if args.base_categories else set()
    target_cats = args.target_categories if args.target_categories else sorted(support_coco.getCatIds())

    iou_types = ("bbox", "segm") if args.segmentation else ("bbox",)
    evaluator = FSODEvaluator(
        args.test_annotation,
        novel_cat_ids=novel_cat_ids,
        base_cat_ids=base_cat_ids,
        iou_types=iou_types,
    )

    print(f"\n--- Starting Training-Free In-Context Evaluation (K={args.shots}) ---")
    print(f"Target categories ({len(target_cats)}): {target_cats}")
    if novel_cat_ids:
        print(f"Novel categories ({len(novel_cat_ids)}): {sorted(novel_cat_ids)}")

    results = evaluate_in_context_fsod(
        model=model,
        support_coco=support_coco,
        support_image_dir=args.support_image_dir,
        test_loader=test_loader,
        target_category_ids=target_cats,
        shots=args.shots,
        device=device,
        resolution=args.resolution,
        evaluator=evaluator,
        include_masks=args.segmentation,
    )

    print("\n================== In-Context Few-Shot Benchmark Results ==================")
    for iou_type, type_results in results.items():
        if isinstance(type_results, dict):
            print(f"\n[{iou_type.upper()}] Metrics:")
            for k, v in type_results.items():
                print(f"  {k:20s}: {v:.4f}")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()

