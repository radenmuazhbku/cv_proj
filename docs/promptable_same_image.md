# Same-image box-prompted detection

The active `train_promptable_fewshot.py` path now trains the same-image task: an image must contain at least two instances of a category; one instance box is the visual prompt, and the remaining same-category boxes are the detection targets. The prompt instance is excluded from the loss targets.

The main trainer supports `batch_repeat`: it runs several optimizer updates on the current sampled batch before fetching the next batch. The CLI default is `4`; the tiny config sets it to `10`. Set it to `1` to recover the previous one-update-per-batch sampling loop. `steps` continues to count optimizer updates, so at repeat `R` the trainer consumes roughly one new batch every `R` steps. Increase repeat when you want each sampled batch to receive more fitting; compare train sanity and held-out metrics to choose a useful value.

## Model and loss

`PromptableDetector.forward_same_image` runs the image backbone once, pools the prompt ROI from that image's feature map, and fuses the resulting visual/geometry token into the detector queries. Hungarian matching assigns distinct predicted boxes to every remaining target box; unmatched queries receive background objectness loss. The prompt and query features share the same backbone pass.

The old cross-image episode implementation is preserved in `archive/promptable_cross_image_20261014/`.

## Input, shots, episodes, and loaders

### One training example

The active trainer reads the RGB image and COCO bounding boxes from `train_image_dir` and `train_annotation_file`. It ignores crowd annotations and boxes no wider or taller than one pixel. An eligible example is an image/category pair with at least two valid boxes for that category.

For each example, the dataset chooses one category and one eligible image containing that category. One of that category's boxes becomes the prompt; the other boxes of the same category in that image become target boxes. Boxes for other categories in the image are ignored. The prompt instance is excluded from the target set. The model receives the resized image and the normalized prompt box (`cx, cy, width, height`); the remaining boxes are supervision for the loss. Support and query are the same image in this path.

This is **one-shot** training: one prompted instance box per episode. The `max_shots` config field does not increase the number of support examples in the active same-image `__getitem__` path. Separate validation/test support directories and annotation files also do not change the active episodic evaluator into cross-image few-shot evaluation. The generic support-query and COCO mAP helper code is not called by the current main training/evaluation loop.

### What `episodes` means

For training, `episodes` sets the dataset's virtual length, not the number of unique source images or a guarantee that every image is visited. Each dataset index deterministically selects a category, eligible image, and prompt box using `seed + index`; the DataLoader shuffles those indices. Different indices can select the same image/category pair, while each index returns the same episode when revisited. Sampling chooses categories uniformly, then eligible images for the chosen category uniformly, then a prompt instance uniformly.

The training DataLoader uses `batch_size` independent episodes per batch, shuffles episode indices, and cycles through the loader when the configured optimizer `steps` require more updates. `batch_repeat=R` applies up to `R` optimizer updates to one fetched batch before fetching another; `batch_repeat=1` is the old behavior. Thus `steps` counts updates, while the approximate number of batches consumed is `steps / batch_repeat`. A batch dimension is a group of separate episodes; it is not the shot count.

Each batch can contain a different number of target boxes per episode. `collate_few_shot` pads target boxes to the batch maximum and adds a mask so padded entries do not enter the loss. It stacks one image and one prompt box per episode.

### Evaluation episodes

For validation and test, the dataset builds a fixed list containing one positive episode for every eligible category/image pair in that split. Therefore `len(eval_dataset)` is the number of eligible pairs; the `episodes=1` constructor value used for eval does not cap it. The prompt box is chosen deterministically from the pair's boxes using the seed, image ID, and category ID. Every other valid box of that category is a target. The eval DataLoader uses `shuffle=False`, so it visits this fixed list in order, grouped by `batch_size`.

The episodic metric takes the highest-objectness predicted box for each positive episode and measures its best IoU against that episode's target boxes. It averages per category and then computes macro positive IoU. This is not COCO AP and does not evaluate negative image/category pairs. In the current dataset path, `negative_ratio` and `negative_eval_per_class` are stored but do not generate negative episodes; `max_shots` does not make eval multi-shot. Treat those options as inactive for this same-image workflow.

### Train sanity loader

When `train_sanity` is enabled, the trainer selects a deterministic subset of training dataset indices once and evaluates those same episodes at each evaluation interval. An integer `train_sanity_samples` means a number of batches; a float fraction or percent means that share of the training loader. It saves qualitative figures and IoU/objectness summaries, but does not update weights on those sanity batches. The model is trained on a batch repeatedly only through the main training loop's `batch_repeat` setting.

Inference figures use `inference_plot_mode`, selectable as `overlay` or `split` from the config or `--inference-plot-mode`. The default `overlay` mode saves one query image with the prompt box (cyan), all query GT boxes (green), and all decoder predictions (top score in red; other predictions in orange). Since the active same-image path is one-shot, the prompt is on that query image. By default all decoder queries are drawn; set `inference_max_predictions` or `--inference-max-predictions K` to cap the plotted predictions, with `0` meaning no cap. The evaluation IoU still uses only the highest-scoring prediction. `split` keeps the separate `original`, `prediction`, `prediction_gt`, and `support_query` views.

## Comparison experiments

`scripts/rfdetr_promptable/compare_concat_vs_batched.py` compares two cross-image extensions and accepts `--shots N` (default `1`):

- `concat`: all support images and the query are resized independently and tiled horizontally in one wide canvas. Each support box ROI is pooled from that shared canvas, and the shot tokens are averaged to condition detection. The loss includes the unprompted same-class objects in each support tile and the query tile; prompt boxes are excluded.
- `batch`: all support images and the query are appended as separate items in one backbone batch. Their ROI tokens are averaged per episode to condition query decoding. Each support image gets an auxiliary loss on its other same-class objects, plus the query loss.

Every support image is distinct from the query and from the other support shots. It must have at least two instances of the selected category so one can be prompted and the remainder can supervise detection. Query GT is used as a target, never as a model input or prompt. The concat canvas grows with shot count; paired batch keeps images separate but its backbone batch grows to `shots + 1` images per episode.

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

For a config-only approximation of the fixed-episode overfit diagnostic, use:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/rfdetr_promptable/train_promptable_fewshot.py \
  --config configs/rfdetr_promptable/promptable_fewshot_2_tiny_diag.py
```

That config makes a four-episode dataset and one batch of four, then repeats that batch for all 1,200 optimizer steps. It uses the diagnostic's resolution, query count, learning rate, and constant LR schedule, and saves train-sanity figures every 100 steps. It is an approximation: the main dataset sampler can choose duplicate image/category pairs and a different prompt instance, and the main trainer currently fixes AdamW weight decay at `0.05` while `diagnose_same_image_overfit.py` uses `0.01`. The main trainer also runs its validation/test evaluation and checkpoint logic.

The previous random-sampling trainer is preserved at `scripts/rfdetr_promptable/train_promptable_fewshot_rs.py`. To use the new trainer with the old one-update-per-batch behavior, pass `--batch-repeat 1`.

Compare concat and batch conditioning with:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/rfdetr_promptable/compare_concat_vs_batched.py \
  --steps 800 --episodes 4 --resolution 280 --queries 20 \
  --output-dir logs/prompt_conditioning_comparison
```

Use multiple support images per query by setting `--shots`, for example:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/rfdetr_promptable/compare_concat_vs_batched.py \
  --shots 5 --steps 800 --episodes 4 --resolution 280 --queries 20 \
  --output-dir logs/prompt_conditioning_comparison_5shot
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
