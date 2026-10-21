"""Visual-only PCS initialized from a completed known-class warmup checkpoint."""

from pathlib import Path
from runpy import run_path

cfg = run_path("configs/rfdetr_pcs/pcs_train_fewshot_2_same_image_visual.py")["cfg"]
cfg.update({
    "init_checkpoint": Path("logs/rfdetr_pcs_same_image_class_warmup/best.pt"),
    "output_dir": Path("logs/rfdetr_pcs_train_same_image_visual_from_class"),
    "figure_dir": Path("logs/rfdetr_pcs_train_same_image_visual_from_class/figures"),
})
