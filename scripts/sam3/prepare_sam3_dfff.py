#!/usr/bin/env python3
"""Filter DFFF COCO splits into SAM3 support/query episodes.

Examples:
  python scripts/prepare_sam3_dfff.py --dataset-root datasets/dfff_coco
  python scripts/prepare_sam3_dfff.py --dataset-root datasets/dfff_coco \
      --support-split set1 --query-split external --max-queries-per-class 20
  python scripts/prepare_sam3_dfff.py --dataset-root datasets/dfff_coco \
      --support-split set1 --query-split internal --copy-images
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cv_proj.sam3_dfff import DEFAULT_CLASSES, SPLITS, prepare_manifest, read_coco


def print_split_inventory(dataset_root: Path) -> None:
    """List images, target-class boxes, and mask availability per split."""
    for split in SPLITS:
        annotation_path = dataset_root / split / "annotations/instances.json"
        if not annotation_path.is_file():
            print(f"{split}: missing {annotation_path}")
            continue
        data = read_coco(annotation_path)
        categories = {item["id"]: item["name"] for item in data["categories"]}
        target_ids = {
            item["id"]
            for item in data["categories"]
            if item["name"].strip().casefold() in {name.casefold() for name in DEFAULT_CLASSES}
        }
        target_annotations = [
            item for item in data["annotations"] if item["category_id"] in target_ids
        ]
        counts = {
            categories[category_id]: sum(
                annotation["category_id"] == category_id for annotation in target_annotations
            )
            for category_id in sorted(target_ids)
        }
        segmentation_count = sum("segmentation" in item for item in target_annotations)
        print(
            f"{split}: images={len(data['images'])}, annotations={len(data['annotations'])}, "
            f"nasal bone/NT boxes={counts}, polygon/RLE masks={segmentation_count}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/dfff_coco"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sam3_dfff"))
    parser.add_argument("--support-split", choices=SPLITS, default="set1")
    parser.add_argument("--query-split", choices=SPLITS, default="internal")
    parser.add_argument("--classes", nargs="+", default=list(DEFAULT_CLASSES))
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--max-queries-per-class", type=int)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy only selected support/query images into output-dir/images for a portable offline bundle",
    )
    parser.add_argument(
        "--list-only", action="store_true", help="Print the split inventory without writing manifests"
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        parser.error(f"Dataset root does not exist: {dataset_root}")
    print_split_inventory(dataset_root)
    if args.list_only:
        return

    manifest, evaluation = prepare_manifest(
        dataset_root=dataset_root,
        output_dir=args.output_dir,
        support_split=args.support_split,
        query_split=args.query_split,
        class_names=tuple(args.classes),
        shots=args.shots,
        max_queries_per_class=args.max_queries_per_class,
        seed=args.seed,
        copy_images=args.copy_images,
    )
    print(f"\nWrote inference-only manifest: {args.output_dir.resolve() / 'manifest.json'}")
    print(f"Wrote evaluation labels (do not load during inference): {args.output_dir.resolve() / 'evaluation.json'}")
    for class_name, records in manifest["classes"].items():
        print(
            f"{class_name}: supports={len(records['supports'])}, queries={len(records['queries'])}, "
            f"labeled_queries={len(evaluation['classes'][class_name])}"
        )


if __name__ == "__main__":
    main()