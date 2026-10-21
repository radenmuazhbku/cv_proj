"""Generate standard Few-Shot Object Detection (FSOD) splits for MS COCO and PASCAL VOC.

Follows the benchmark protocol from TFA (Wang et al. ICML 2020 / fsdet) & Kang et al.:
1. MS COCO:
   - 60 Base / 20 Novel categories (20 novel = overlapping PASCAL VOC classes).
   - Base training set: all train2017 images with novel annotations stripped.
   - K-shot support sets: exactly K instances per category sampled from train2017.
   - Test set: 100% full val2017 (5,000 images).
2. PASCAL VOC:
   - 3 standard splits (15 Base / 5 Novel each).
   - Base training set: trainval with novel annotations stripped.
   - K-shot support sets: exactly K instances per category.
   - Test set: 100% full val/test set.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

# ==============================================================================
# MS COCO Category Definitions
# ==============================================================================
# 20 Novel Categories (corresponding to PASCAL VOC categories)
COCO_NOVEL_NAMES = [
    "airplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "couch",
    "cow", "dining table", "dog", "horse", "motorcycle",
    "person", "potted plant", "sheep", "train", "tv",
]

# Standard COCO 80 category IDs for the 20 Novel categories:
COCO_NOVEL_CAT_IDS = {
    1,   # person
    2,   # bicycle
    3,   # car
    4,   # motorcycle
    5,   # airplane
    6,   # bus
    7,   # train
    9,   # boat
    16,  # bird
    17,  # cat
    18,  # dog
    19,  # horse
    20,  # sheep
    21,  # cow
    44,  # bottle
    62,  # chair
    63,  # couch
    64,  # potted plant
    67,  # dining table
    72,  # tv
}

# ==============================================================================
# PASCAL VOC Category Definitions & Splits
# ==============================================================================
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

VOC_SPLITS = {
    1: {
        "novel": ["bird", "bus", "cow", "motorbike", "sofa"],
        "base": [c for c in VOC_CLASSES if c not in ["bird", "bus", "cow", "motorbike", "sofa"]],
    },
    2: {
        "novel": ["aeroplane", "bottle", "cow", "horse", "sofa"],
        "base": [c for c in VOC_CLASSES if c not in ["aeroplane", "bottle", "cow", "horse", "sofa"]],
    },
    3: {
        "novel": ["boat", "cat", "motorbike", "sheep", "sofa"],
        "base": [c for c in VOC_CLASSES if c not in ["boat", "cat", "motorbike", "sheep", "sofa"]],
    },
}


def filter_dataset_by_categories(
    data: dict[str, Any],
    allowed_cat_ids: set[int],
    keep_empty_images: bool = False,
) -> dict[str, Any]:
    """Filter annotations to allowed categories; drop images with no allowed annotations unless keep_empty_images is True."""
    allowed_categories = [cat for cat in data["categories"] if cat["id"] in allowed_cat_ids]
    allowed_annotations = [ann for ann in data["annotations"] if ann["category_id"] in allowed_cat_ids]
    
    if keep_empty_images:
        selected_images = list(data["images"])
    else:
        active_image_ids = {ann["image_id"] for ann in allowed_annotations}
        selected_images = [img for img in data["images"] if img["id"] in active_image_ids]

    return {
        "info": data.get("info", {}),
        "licenses": data.get("licenses", []),
        "images": selected_images,
        "annotations": allowed_annotations,
        "categories": allowed_categories,
    }


def sample_k_shot_dataset(
    data: dict[str, Any],
    target_cat_ids: set[int],
    k_shot: int,
    seed: int,
    min_box_area: float = 32.0,
) -> dict[str, Any]:
    """Sample exactly K instances for each category in target_cat_ids."""
    rng = random.Random(seed)
    
    # Group annotations by category
    cat_to_anns: dict[int, list[dict[str, Any]]] = {cid: [] for cid in target_cat_ids}
    for ann in data["annotations"]:
        cid = ann["category_id"]
        if cid in target_cat_ids:
            # Check min area and non-crowd
            if ann.get("iscrowd", 0) == 0 and ann.get("area", 0) >= min_box_area:
                cat_to_anns[cid].append(ann)

    selected_anns: list[dict[str, Any]] = []
    selected_image_ids: set[int] = set()

    for cid in sorted(target_cat_ids):
        anns = cat_to_anns[cid]
        if len(anns) < k_shot:
            raise ValueError(f"Category ID {cid} has only {len(anns)} valid instances, fewer than K={k_shot}")
        
        # Deterministic shuffle for this category
        cat_rng = random.Random(seed + cid * 1000)
        chosen_indices = cat_rng.sample(range(len(anns)), k_shot)
        for idx in chosen_indices:
            ann = anns[idx]
            selected_anns.append(ann)
            selected_image_ids.add(ann["image_id"])

    images_dict = {img["id"]: img for img in data["images"]}
    selected_images = [images_dict[img_id] for img_id in sorted(selected_image_ids) if img_id in images_dict]
    selected_categories = [cat for cat in data["categories"] if cat["id"] in target_cat_ids]

    return {
        "info": {
            **data.get("info", {}),
            "description": f"Few-Shot Object Detection {k_shot}-shot dataset (seed={seed})",
            "k_shot": k_shot,
            "seed": seed,
        },
        "licenses": data.get("licenses", []),
        "images": selected_images,
        "annotations": selected_anns,
        "categories": selected_categories,
    }


def generate_coco_splits(
    coco_train_json: Path,
    coco_val_json: Path,
    output_dir: Path,
    shots: list[int],
    seeds: list[int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading COCO train annotations from {coco_train_json}...")
    with coco_train_json.open("r", encoding="utf-8") as f:
        train_data = json.load(f)

    all_cat_ids = {cat["id"] for cat in train_data["categories"]}
    novel_cat_ids = {cid for cid in all_cat_ids if cid in COCO_NOVEL_CAT_IDS}
    base_cat_ids = all_cat_ids - novel_cat_ids

    print(f"COCO Total categories: {len(all_cat_ids)} | Base: {len(base_cat_ids)} | Novel: {len(novel_cat_ids)}")

    # 1. Base Training Set (all base categories, novel stripped)
    base_train_path = output_dir / "coco_base_train.json"
    if not base_train_path.exists():
        print(f"Generating base training dataset -> {base_train_path}...")
        base_train_data = filter_dataset_by_categories(train_data, base_cat_ids, keep_empty_images=False)
        with base_train_path.open("w", encoding="utf-8") as f:
            json.dump(base_train_data, f)
        print(f"  Base train: {len(base_train_data['images'])} images, {len(base_train_data['annotations'])} annotations")
    else:
        print(f"Base training dataset already exists: {base_train_path}")

    # 2. Few-shot K-shot splits (for both Novel-only and Class-Balanced Base+Novel)
    for k in shots:
        for seed in seeds:
            # Balanced: K instances for all 80 classes
            balanced_path = output_dir / f"coco_fsod_all_{k}shot_seed{seed}.json"
            if not balanced_path.exists():
                balanced_data = sample_k_shot_dataset(train_data, all_cat_ids, k, seed)
                with balanced_path.open("w", encoding="utf-8") as f:
                    json.dump(balanced_data, f)
                print(f"  Wrote {balanced_path.name}: {len(balanced_data['images'])} imgs, {len(balanced_data['annotations'])} anns ({len(all_cat_ids)} classes x {k})")

            # Novel-only: K instances for 20 novel classes
            novel_path = output_dir / f"coco_fsod_novel_{k}shot_seed{seed}.json"
            if not novel_path.exists():
                novel_data = sample_k_shot_dataset(train_data, novel_cat_ids, k, seed)
                with novel_path.open("w", encoding="utf-8") as f:
                    json.dump(novel_data, f)
                print(f"  Wrote {novel_path.name}: {len(novel_data['images'])} imgs, {len(novel_data['annotations'])} anns ({len(novel_cat_ids)} classes x {k})")

    # 3. Validation splits (keep full images, partition annotations for convenience)
    print(f"\nProcessing COCO validation annotations from {coco_val_json}...")
    with coco_val_json.open("r", encoding="utf-8") as f:
        val_data = json.load(f)

    # Full val is untouched
    val_full_path = output_dir / "coco_val_full.json"
    if not val_full_path.exists():
        with val_full_path.open("w", encoding="utf-8") as f:
            json.dump(val_data, f)
        print(f"  Wrote {val_full_path.name}: full {len(val_data['images'])} images")

    # Novel-only annotations over FULL test images (images without novel objects are kept as true negatives!)
    val_novel_path = output_dir / "coco_val_novel.json"
    if not val_novel_path.exists():
        val_novel_data = filter_dataset_by_categories(val_data, novel_cat_ids, keep_empty_images=True)
        with val_novel_path.open("w", encoding="utf-8") as f:
            json.dump(val_novel_data, f)
        print(f"  Wrote {val_novel_path.name}: {len(val_novel_data['images'])} images (kept full), {len(val_novel_data['annotations'])} novel annotations")


def generate_voc_splits(
    voc_trainval_json: Path,
    voc_val_json: Path,
    output_dir: Path,
    shots: list[int],
    seeds: list[int],
    voc_train_json: Path | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # If voc_train_json is given, sample few-shot support sets strictly from train (0 overlap with val)
    support_source_json = voc_train_json if (voc_train_json and voc_train_json.exists()) else voc_trainval_json
    print(f"\nLoading VOC train annotations for few-shot sampling from {support_source_json}...")
    with support_source_json.open("r", encoding="utf-8") as f:
        support_source_data = json.load(f)

    with voc_trainval_json.open("r", encoding="utf-8") as f:
        trainval_data = json.load(f)

    cat_name_to_id = {cat["name"]: cat["id"] for cat in trainval_data["categories"]}

    for split_id, split_cfg in VOC_SPLITS.items():
        novel_names = set(split_cfg["novel"])
        base_names = set(split_cfg["base"])
        novel_ids = {cat_name_to_id[name] for name in novel_names if name in cat_name_to_id}
        base_ids = {cat_name_to_id[name] for name in base_names if name in cat_name_to_id}
        all_ids = novel_ids | base_ids

        split_dir = output_dir / f"split_{split_id}"
        split_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- VOC Split {split_id} ---")
        print(f"Novel ({len(novel_names)}): {sorted(novel_names)}")
        print(f"Base ({len(base_names)}): {sorted(base_names)}")

        # 1. Base training dataset
        base_train_path = split_dir / f"voc_split{split_id}_base_train.json"
        if not base_train_path.exists():
            base_data = filter_dataset_by_categories(trainval_data, base_ids, keep_empty_images=False)
            with base_train_path.open("w", encoding="utf-8") as f:
                json.dump(base_data, f)
            print(f"  Wrote {base_train_path.name}: {len(base_data['images'])} imgs, {len(base_data['annotations'])} anns")

        # 2. Few-shot splits
        for k in shots:
            for seed in seeds:
                balanced_path = split_dir / f"voc_split{split_id}_all_{k}shot_seed{seed}.json"
                if not balanced_path.exists():
                    balanced_data = sample_k_shot_dataset(support_source_data, all_ids, k, seed)
                    with balanced_path.open("w", encoding="utf-8") as f:
                        json.dump(balanced_data, f)
                    print(f"  Wrote {balanced_path.name}: {len(balanced_data['images'])} imgs, {len(balanced_data['annotations'])} anns")

                novel_path = split_dir / f"voc_split{split_id}_novel_{k}shot_seed{seed}.json"
                if not novel_path.exists():
                    novel_data = sample_k_shot_dataset(support_source_data, novel_ids, k, seed)
                    with novel_path.open("w", encoding="utf-8") as f:
                        json.dump(novel_data, f)
                    print(f"  Wrote {novel_path.name}: {len(novel_data['images'])} imgs, {len(novel_data['annotations'])} anns")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["coco", "voc", "all"], default="all")
    parser.add_argument("--coco-train", type=Path, default=Path("datasets/mscoco/annotations_trainval2017/annotations/instances_train2017.json"))
    parser.add_argument("--coco-val", type=Path, default=Path("datasets/mscoco/annotations_trainval2017/annotations/instances_val2017.json"))
    parser.add_argument("--coco-output", type=Path, default=Path("datasets/mscoco_fsod_splits"))
    parser.add_argument("--voc-trainval", type=Path, default=Path("datasets/VOC2012/annotations/voc_2012_trainval.json"))
    parser.add_argument("--voc-train", type=Path, default=Path("datasets/VOC2012/annotations/voc_2012_train.json"))
    parser.add_argument("--voc-val", type=Path, default=Path("datasets/VOC2012/annotations/voc_2012_val.json"))
    parser.add_argument("--voc-output", type=Path, default=Path("datasets/voc_fsod_splits"))
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 2, 3, 5, 10, 30])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    args = parser.parse_args()

    if args.dataset in {"coco", "all"}:
        if args.coco_train.exists() and args.coco_val.exists():
            generate_coco_splits(args.coco_train, args.coco_val, args.coco_output, args.shots, args.seeds)
        else:
            print(f"Warning: COCO annotations not found at {args.coco_train} or {args.coco_val}. Skipping COCO.")

    if args.dataset in {"voc", "all"}:
        if args.voc_trainval.exists() and args.voc_val.exists():
            voc_shots = [s for s in args.shots if s <= 10]
            generate_voc_splits(args.voc_trainval, args.voc_val, args.voc_output, voc_shots, args.seeds, voc_train_json=args.voc_train)
        else:
            print(f"Warning: VOC annotations not found at {args.voc_trainval}. Skipping VOC.")


if __name__ == "__main__":
    main()
