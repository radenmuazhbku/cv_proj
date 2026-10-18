"""Visual-only multi-support PCS initialized from known-class warmup weights."""

from pathlib import Path
from runpy import run_path

cfg = run_path("configs/rfdetr_pcs/pcs_train_fewshot_2_support_query_visual.py")["cfg"]
cfg.update({
    "init_checkpoint": Path("logs/rfdetr_pcs_support_query_class_warmup/best.pt"),
    "output_dir": Path("logs/rfdetr_pcs_train_support_query_visual_from_class"),
    "figure_dir": Path("logs/rfdetr_pcs_train_support_query_visual_from_class/figures"),
})
