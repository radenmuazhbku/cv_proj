"""Short known-class prompt warmup before a weights-only visual PCS run."""

from pathlib import Path
from runpy import run_path

cfg = run_path("configs/rfdetr_pcs/pcs_train_fewshot_2_same_image.py")["cfg"]
cfg.update({
    "conditioning_mode": "class_embedding",
    "steps": 10_000,
    "eval_every": 1_000,
    "negative_class_ratio": 0.1,
    "output_dir": Path("logs/rfdetr_pcs_same_image_class_warmup"),
    "figure_dir": Path("logs/rfdetr_pcs_same_image_class_warmup/figures"),
})
