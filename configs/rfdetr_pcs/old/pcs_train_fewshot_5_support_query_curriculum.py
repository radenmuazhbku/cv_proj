"""Single-run multi-support curriculum: known-class warmup, then visual PCS."""

from pathlib import Path
from runpy import run_path

cfg = run_path("configs/rfdetr_pcs/pcs_train_fewshot_5_support_query_visual.py")["cfg"]
cfg.update({
    # The class token is used for the first phase, then the model conditions
    # only on the support image/box tokens for the remainder of training.
    "conditioning_mode": "visual_prompt",
    "conditioning_schedule": [
        {"mode": "class_embedding", "steps": 100_000},
        {"mode": "visual_prompt", "steps": 400_000},
    ],
    "steps": 400_000,
    "eval_every": 50_000,
    # Do not include each positive support box as a support-image detection
    # target; train the support auxiliary pass to find the remaining instances.
    "include_prompt_box_in_loss": False,
    "output_dir": Path("logs/rfdetr_pcs_train_fewshot_5_support_query_curriculum"),
    "figure_dir": Path("logs/rfdetr_pcs_train_fewshot_5_support_query_curriculum/figures"),
})
