"""Separate-image few-shot PCS fit using the full prompt split.

Each query image is conditioned on two positive box prompts from distinct
support images. The trainer also applies an auxiliary detection/segmentation
loss to support images. This is the current support/query overfit diagnostic,
not the long-run train/validation/test loop used by the promptable trainer.
"""

from pathlib import Path

cfg = {
    "seed": 7,
    "query_image_dir": Path("datasets/mscoco/train2017"),
    "query_annotation_file": Path("datasets/mscoco_prompt_split/train.json"),
    "support_image_dir": Path("datasets/mscoco/val2017"),
    "support_annotation_file": Path("datasets/mscoco_prompt_split/seen_support.json"),
    "resolution": 560,
    "positional_encoding_size": 37,
    "num_queries": 100,
    "bbox_reparam": False,
    "segmentation": True,
    "shots": 2,
    "negative_prompt_ratio": 0.0,
    "overfit_episodes": 4,
    "steps": 5000,
    "batch_size": 2,
    "lr": 1e-4,
    "support_loss_weight": 0.5,
    "clip_grad_norm": 0.1,
    "eval_every": 250,
    "max_checkpoints": 3,
    "device": "cuda",
    "output_dir": Path("logs/rfdetr_pcs_fewshot_2_support_query"),
}
