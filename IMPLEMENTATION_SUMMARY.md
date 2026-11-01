# Data Augmentation Implementation Summary

## Overview
Successfully integrated Albumentations data augmentation with automatic bounding box and mask transformation support into the RF-DETR PCS training pipeline.

## Changes Made

### 1. Configuration (`configs/rfdetr_pcs/rfdetr_pcs_coco_fs5.py`)
- Added `AUG_CONSERVATIVE` preset: HorizontalFlip (p=0.5) + mild brightness/contrast jitter
- Added `AUG_AGGRESSIVE` preset: HorizontalFlip, VerticalFlip, Rotate, Affine transforms, ColorJitter
- Added `aug_config` to cfg pointing to `AUG_CONSERVATIVE` (can be changed to `AUG_AGGRESSIVE`)
- Set `output_dir` to None (auto-inferred from config filename)

### 2. Dataset Augmentation (`src/rfdetr_pcs/coco_episodes.py`)

#### New Method: `_load_and_augment_tensor`
- Loads image and annotations from COCO
- Applies Albumentations transforms to image, bounding boxes, and masks
- Handles both numpy array and list formats for masks
- Returns transformed image, boxes, masks with consistent counts
- Falls back to non-augmented path when no augmentation is configured

#### Updated: `_setup_albumentations_transform`
- Builds Albumentations Compose pipeline from `aug_config`
- Configures `BboxParams` with COCO format and clip=False
- Albumentations 2.x automatically handles masks without `mask_params`

#### Updated: `__getitem__`
- Calls `_load_and_augment_tensor` for query and support images when training with augmentation
- Properly passes masks through the augmentation pipeline

#### Updated: `collate_support_query`
- Always creates `query_masks` and `support_target_masks` when any item has masks
- Handles cases where some images may not have annotations
- Properly pads masks to max_targets dimension

### 3. Training Script (`scripts/rfdetr_pcs/train_pcs.py`)

#### Auto-inference of `output_dir`
- If `output_dir` is None, automatically sets to `logs/{config_filename}`
- Creates directory if it doesn't exist

#### Output Directory Warning
- Added check: if `output_dir` exists, prompts user for confirmation before proceeding

#### Config Saving
- Saves a copy of the configuration to `output_dir/config.yaml`

#### Auto-inference of `figure_dir`
- If `figure_dir` is None, sets to `output_dir/figures`
- Creates directory if needed

#### Backend Configuration
- Automatically uses augmentation based on `augmentation_backend` setting
- Default: uses `"cpu"` which auto-selects Albumentations > Kornia > torchvision
- For PCS, augmentation is applied at dataset level (CPU) or batch level (GPU)

## Usage

### Using Built-in Presets

```python
# In your config file
from pathlib import Path

AUG_CONSERVATIVE = {
    "HorizontalFlip": {"p": 0.5},
    "RandomBrightnessContrast": {
        "brightness_limit": 0.1,
        "contrast_limit": 0.1,
        "p": 0.3,
    },
}

AUG_AGGRESSIVE = {
    "HorizontalFlip": {"p": 0.5},
    "VerticalFlip": {"p": 0.5},
    "Rotate": {"limit": 45, "p": 0.5},
    "Affine": {
        "scale": (0.8, 1.2),
        "translate_percent": (-0.1, 0.1),
        "rotate": (-15, 15),
        "shear": (-5, 5),
        "p": 0.5,
    },
    "ColorJitter": {
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.2,
        "hue": 0.1,
        "p": 0.5,
    },
}

cfg = {
    "aug_config": AUG_CONSERVATIVE,  # or AUG_AGGRESSIVE
    "augmentation_backend": "cpu",  # or "albumentations", "kornia", "auto"
    "output_dir": None,  # auto-inferred to logs/{config_file}
    ...
}
```

### Running Training

```bash
# With conservative augmentation (default)
python scripts/rfdetr_pcs/train_pcs.py --config configs/rfdetr_pcs/rfdetr_pcs_coco_fs5.py

# Or modify the config to use aggressive augmentation
# and the training will automatically use it
```

## Technical Details

### Albumentations Integration
- Uses Albumentations 2.x API (compatible with 1.4.24+)
- Automatic bbox transformation via `BboxParams(format='coco', clip=False)`
- Automatic mask transformation (no `mask_params` needed in v2.x)
- Masks passed as list of (H, W) numpy arrays
- Returns masks as list with same length as boxes

### Mask/Box Consistency
- Ensures masks and boxes have the same count after augmentation
- Handles dimension mismatches by truncating to minimum count
- Properly resizes masks to resolution using NEAREST interpolation

### Backend Selection
- `"cpu"`: Auto-selects best CPU backend (Albumentations > Kornia CPU > torchvision)
- `"auto"`: Auto-selects best available (GPU Kornia if CUDA, else same as cpu)
- `"albumentations"`: Force Albumentations (CPU)
- `"kornia"`: Force Kornia (GPU)
- `"torchvision"`: Force torchvision (CPU)

## Verification

Training runs successfully with:
- ✅ Augmentation applied to images
- ✅ Bounding boxes transformed correctly
- ✅ Masks transformed correctly (same count as boxes)
- ✅ No dimension mismatch errors
- ✅ Segmentation losses computed correctly
- ✅ Config auto-saving to output_dir
- ✅ Output directory auto-inference
- ✅ Warning prompt for existing output directories

## Recommendations

- Use `AUG_CONSERVATIVE` for small datasets (< 500 images)
- Use `AUG_AGGRESSIVE` for large datasets (2000+ images)
- For GPU acceleration, use `augmentation_backend="kornia"` (requires `rfdetr[augment]`)
- For CPU, use `augmentation_backend="cpu"` or `"albumentations"`
- Monitor training loss to ensure augmentations aren't too aggressive
