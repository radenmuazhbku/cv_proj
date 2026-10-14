# Same-image box-prompted detection

The active `train_promptable_fewshot.py` path now trains the same-image task: an image must contain at least two instances of a category; one instance box is the visual prompt, and the remaining same-category boxes are the detection targets. The prompt instance is excluded from the loss targets.

## Model and loss

`PromptableDetector.forward_same_image` runs the image backbone once, pools the prompt ROI from that image's feature map, and fuses the resulting visual/geometry token into the detector queries. Hungarian matching assigns distinct predicted boxes to every remaining target box; unmatched queries receive background objectness loss. The prompt and query features share the same backbone pass.

The COCO episode dataset filters to image/category pairs with at least two usable boxes. Train, validation, and test all use this same protocol. The old cross-image episode implementation is preserved in `archive/promptable_cross_image_20261014/`.

## Comparison experiments

`scripts/rfdetr_promptable/compare_concat_vs_batched.py` compares two cross-image extensions:

- `concat`: support and query images are resized independently and tiled side by side in a wide canvas. The support box prompts detections in the query tile; loss targets are query-tile instances.
- `batch`: support and query images remain separate batch elements through one shared backbone call. The support ROI token conditions query decoding. An auxiliary support loss trains the prompt path to detect the other same-class objects in the support image.

Both modes use the same box detector and matcher. These runs test conditioning and localization plumbing; they are not SAM3 and do not produce segmentation masks.

## Reproduced sanity runs

The fixed same-image overfit run and the concat/batch comparisons save metrics CSV/JSON, curves, model checkpoints, and support/query panels under:

- `logs/same_image_overfit/`
- `logs/prompt_conditioning_comparison/`
- `logs/prompt_conditioning_comparison_560/`

Run the same-image overfit diagnostic with:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/rfdetr_promptable/diagnose_same_image_overfit.py \
  --steps 1200 --episodes 4 --resolution 280 --num-queries 20 \
  --output-dir logs/same_image_overfit
```

Run the configured tiny-split training workflow with:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/rfdetr_promptable/train_promptable_fewshot.py \
  --config configs/rfdetr_promptable/promptable_fewshot_2_tiny.py
```

Compare concat and batch conditioning with:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/rfdetr_promptable/compare_concat_vs_batched.py \
  --steps 800 --episodes 4 --resolution 280 --queries 20 \
  --output-dir logs/prompt_conditioning_comparison
```

The fixed-episode trials are sanity checks, not estimates of validation performance. Results recorded on the RTX 4090:

| Trial | Resolution / episodes / steps | Final matched IoU | Peak allocated GPU memory |
| --- | --- | ---: | ---: |
| Same-image overfit | 280 / 4 / 1,200 | 0.806 (from 0.005 at step 1) | Not recorded |
| Spatial concat | 280 / 4 / 800 | 0.696 | 1.531 GiB |
| Paired batch | 280 / 4 / 800 | 0.719 | 1.544 GiB |
| Spatial concat | 560 / 4 / 600 | 0.540 (best 0.652) | 4.592 GiB |
| Paired batch | 560 / 4 / 600 | 0.651 (best 0.739) | 4.631 GiB |

The concat/batch memory difference was within about 1% in these runs. The paired batch avoids self-attention across support and query pixels and was slightly faster at 560px, but these measurements do **not** demonstrate a meaningful peak-memory reduction. RF-DETR's DINOv2 backbone uses scaled dot-product attention, so attention-map memory is not the dominant quadratic allocation assumed by a naive attention estimate; both modes process a similar total number of image tokens. A lower-memory sequential/checkpointed support encoder would need a separate measurement.
