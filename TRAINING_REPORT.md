# RF-DETR Nano training report: `scripts/train_1.py`

This report summarizes the completed run in `logs/`. Although the requested `hparams.yaml` is present, it contains only `{}`; the resolved settings below therefore come from `logs/training_config.json`.

## Run configuration

| Setting | Value |
| --- | --- |
| Model | RF-DETR Nano (`dinov2_windowed_small`) |
| Dataset | `datasets/rfdetr_dfff` |
| Classes | 9: CM, IT, NT, midbrain, nasal bone, nasal skin, nasal tip, palate, thalami |
| Split sizes | 812 train / 560 validation / 156 test images |
| Training duration | 100 epochs, steps 0–5,099 |
| Batch configuration | Batch size 4, gradient accumulation 4 (effective batch size 16) |
| Learning rate | `1e-4`; step scheduler with `lr_drop=100` |
| Optimizer | AdamW, weight decay `1e-4` |
| Input resolution | 384 × 384 |
| Augmentation | Multi-scale, expanded scales, scale jitter; CPU backend |
| EMA | Enabled; decay `0.993`, update interval 1 |
| Evaluation | Every epoch, maximum 500 detections/image |
| Checkpoint selection | Best mAP; `logs/checkpoint_best_ema.pth` available |

## Validation results

| Metric | Best value | Best epoch | Final epoch (99) |
| --- | ---: | ---: | ---: |
| F1 | 0.7053 | 39 | 0.6891 |
| mAP@0.50 | 0.7664 | 29 | 0.7153 |
| mAP@0.50:0.95 | 0.4487 | 49 | 0.4195 |
| mAP@0.75 | 0.4863 | 30 | 0.4404 |
| mean average recall | 0.6636 | 58 | 0.6566 |
| Precision | 0.6497 | 31 | 0.5952 |
| Recall | 0.8788 | 93 | 0.8519 |

The best mAP@0.50:0.95 occurs at epoch 49 (step 2,549). Final-epoch validation quality is lower than this peak, while recall remains high; precision decreases more noticeably. This makes `checkpoint_best_ema.pth`, rather than the final checkpoint, the appropriate default for qualitative samples and deployment evaluation.

## Training-loss summary

| Measurement | Value |
| --- | ---: |
| First logged total loss | 6.0687 |
| Final logged total loss | 2.9935 |
| Minimum total loss | 2.9207 (epoch 97) |

Loss substantially decreased during the run, but the validation metrics peak earlier and subsequently fluctuate/decline. Use the best EMA checkpoint for test-set reporting, and consider early stopping or a shorter schedule near the best validation epoch in a follow-up experiment.

## Reproducible visualizations

Generate the full curves, loss-only curve, and mAP curve from the CSV log:

```bash
.venv/bin/python scripts/plot_training_curves.py
```

Generate annotated detection samples from the best EMA checkpoint:

```bash
.venv/bin/python scripts/generate_sample_detections.py --count 8 --threshold 0.35
```

Outputs are written under `logs/visualizations/`. The curve script uses `rfdetr.visualize.training`; the sample script loads the checkpoint and uses RF-DETR's `predict()` output with Supervision annotators.
