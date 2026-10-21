"""Same-image PCS fit using the full prompt split, staged by episode count.

Run with scripts/rfdetr_pcs/train_pcs_overfit.py. This config follows the
same-image input protocol: a box prompt and same-class targets come from one
query image. The current PCS script is an overfit diagnostic, not the long-run
train/validation/test loop used by the promptable trainer.
"""

from pathlib import Path

cfg = {
    "seed": 7,
    "train_image_dir": Path("datasets/mscoco/train2017"),
    "train_annotation_file": Path("datasets/mscoco_prompt_split/train.json"),
    "resolution": 560,
    "positional_encoding_size": 37,
    "num_queries": 100,
    "bbox_reparam": False,
    "segmentation": True,
    # The prompt instance is also included in the query loss, like the current
    # same-image promptable config.
    "include_prompt_box": True,
    # PCS negative_ratio creates empty-remainder episodes; it is not a negative
    # box-label ratio. Keep this at zero for the positive-prompt baseline.
    "negative_ratio": 0.0,
    "negative_prompt_ratio": 0.0,
    "stage_sizes": [10, 100, 1_000],
    "stage_steps": [100, 1_000, 10_000],
    "batch_size": 4,
    "num_workers": 0,
    "lr": 1e-4,
    "clip_grad_norm": 0.1,
    "eval_every": 100,
    "sanity_samples": 8,
    "max_checkpoints": 3,
    "keep_checkpoint_steps": [100, 1_000, 10_000],
    "device": "cuda",
    "output_dir": Path("logs/rfdetr_pcs_fewshot_2_same_image"),
}
