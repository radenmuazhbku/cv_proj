# RF-DETR PCS fork

This fork lives under `src/rfdetr_pcs` and leaves both `src/rfdetr` and
`src/rfdetr_promptable` unchanged. It is intended to add box-prompt conditioning
to RF-DETR while reusing its backbone, box heads, matching/loss code, and mask
head.

## Modes

1. **Same-image PCS (parity target):** one image, several positive/negative
   instance-box prompts on that image, and predictions for the remaining
   instances of the prompted concept. Prompt and query ground truth must remain
   separate: only prompt boxes/labels enter model forward; query annotations are
   used by the matcher and losses.
2. **Support/query few-shot (extension):** several support images, each with
   box prompts, condition one or more separate query images. Images are
   concatenated on the image batch axis for one backbone call, then their feature
   maps are split; they are not spatially tiled. Each query attends to all prompt
   tokens from its episode's supports. Query targets are used only by the
   matcher/loss.

## Current implementation status

- [x] Forked RF-DETR source to `src/rfdetr_pcs` and changed its internal import
  namespace. The original package is untouched by this fork operation.
- [x] Copied the current promptable COCO episode reader and COCO helper into the
  PCS namespace as a starting point for data sampling.
- [x] Added `BoxPromptEncoder`: one token per prompt, with ROI appearance, box
  geometry, binary prompt-label embeddings, prompt self-attention, and image
  feature cross-attention.
- [x] Added optional prompt-token cross-attention in RF-DETR decoder layers.
  Existing callers omit prompt tokens and retain their original forward route.
- [x] Added `PCSDetector.forward` for same-image box-prompt inference/training
  outputs. Its decoder queries attend to prompt tokens. When the fork is built
  with RF-DETR's segmentation head, it also returns query-conditioned mask
  outputs in the existing compact mask format.
- [x] Added `PCSDetector.forward_support_query` with inputs shaped
  `[B,S,3,H,W]`, `[B,S,P,4]`, and `[B,Q,3,H,W]`. Support and query images are
  concatenated on the image batch axis for one backbone call, then split back
  into support and query feature maps; each episode's prompt tokens are routed
  to every query in that episode. Images must share padded height and width.
  Outputs are flattened in B-major,
  Q-minor order for the existing matcher and criterion.
- [ ] Add the image-memory prompt encoder/fusion block from the full design; the
  current encoder updates prompt tokens from image features, then the decoder
  consumes those tokens.
- [ ] Add support-feature caching for repeated queries against the same episode.
- [x] Add a separate-image episode dataset, collator, and overfit trainer.
  The dataset currently samples one positive prompt per support image; the model
  and collator accept padded multiple prompts per support. Query and support
  instance masks are retained as targets.
- [x] Add auxiliary detection/segmentation loss on support images so the support
  encoder receives direct localization supervision in addition to query loss.
- [x] Add a same-image overfit trainer, full-subset evaluator, checkpoint
  retention, loss/IoU curves, and prompt/GT/prediction figures. Box plots call
  the shared `_save_query_support_figure` matplotlib helper from the promptable
  trainer; segmentation overlays are saved beside them.
- [x] Add `scripts/rfdetr_pcs/train_pcs.py`, a full train/validation/test loop
  for both same-image and separate support/query episodes, including warmup plus
  cosine learning-rate decay, resumable checkpoints, validation-selected best
  weights, metric logs, and deterministic qualitative samples.
- [x] Add mask targets, Hungarian mask matching, BCE/Dice mask losses, mask IoU,
  and prompt/query mask visualization for the tiny overfit run.
- [x] Add a staged tiny-split config and run the 4 -> 100 -> 1,000 episode
  curriculum on GPU. Full-subset endpoint metrics are in
  `logs/rfdetr_pcs_tiny_overfit/posthoc_metrics_{4,100,1000}.json`.
- [x] Smoke-check separate support/query inference and confirm query loss
  backpropagates into support prompt encoding.
- [x] Add seen-val/test evaluation and deterministic multi-prompt support plots.

## Segmentation design

Use the RF-DETR per-query mask head rather than adding an independent semantic
segmentation branch. Decoder query features are combined with spatial image
features to produce one mask-logit map per query. Prompt-conditioned query
features therefore influence masks already. The next segmentation step is to
also add optional prompt cross-attention to the spatial features in the mask
head, matching SAM3's prompt-aware segmentation path more closely.

Training targets include instance masks aligned with boxes and labels. One
Hungarian assignment is shared by boxes and masks. The diagnostic currently
uses the dataset's polygon/RLE masks and includes the prompted box and mask in
the query targets (`include_prompt_box=True`); turn that off to train only on
the other same-class instances. Query masks never form prompt tokens.

## Important current limitations

The support/query dataset uses one positive prompt per support image by
default; it can add sampled negative prompts when `negative_prompt_ratio` is
greater than zero. The model and collator accept multiple prompts per support
image. The same-image trainer also supports empty-remainder episodes through
`negative_ratio`. Such episodes prompt every
instance of a selected class in the query image, leaving an empty target set;
the loss then trains the model to return no further instances. A short tiny
run with `negative_ratio=0.5` reached 0.791 box IoU and 0.704 mask IoU on its
four-episode evaluation subset. This is empty-target training, not a negative
box prompt: `negative_ratio` changes whether same-class targets remain. A
separate `negative_prompt_ratio` now samples boxes from other annotated classes
in the same image and passes them to the encoder with label 0. Its default is
0.0, preserving the existing all-positive prompt behavior. At 1.0 it samples
an expected one negative box per positive prompt, capped by the available
other-class annotations. Negative prompt boxes are conditioning only; they are
not added to the cat target set. The positive exemplar remains in the target
set when `include_prompt_box=True`.

This uses ground-truth boxes of other categories as clean distractors. TODO:
replace or augment this sampler with SAM3-style iterative negative prompts
mined from the model's own high-confidence false positives, and positive
prompts mined from false negatives.

The first separate-image query-only run reached 0.253 query box IoU and 0.887
mask IoU after 1,000 steps. Adding a support-image auxiliary loss improved the
1,000-step result to 0.523 query box IoU and 0.891 mask IoU. A 5,000-step
continuation reached 0.694 query box IoU (best checkpoint evaluation: 0.709),
0.859 support box IoU, and 0.947 query mask IoU. The support loss fell from
5.66 at step 250 to 0.56 at step 5,000. The query box metric plateaued, so
additional steps alone did not close the support-to-query localization gap.
Compact figures show the clean query, query GT with the top two predictions,
and the support images with prompt boxes.

The same-image staged run used random RF-DETR decoder/box heads and DINOv2
backbone initialization (`pretrain_weights=None`), so it measures the prompt
path's fit behavior rather than pretrained detector quality. Full-subset
endpoint IoUs were 0.648 box / 0.938 mask for 4 episodes, 0.361 / 0.592 for 100,
and 0.308 / 0.472 for 1,000. The 100- and 1,000-example runs did not approach
perfect box fit. Continuing the four-example checkpoint for 5,000 steps raised
box IoU as high as 0.946, but it fluctuated. A further 2,000-step continuation
at a lower learning rate ended at 0.976 box IoU and 0.946 mask IoU on all four
episodes. This confirms the same-image model can nearly fit the tiny set when
given enough optimization; it does not establish generalization.

## Reproducing the overfit runs

Run commands from the repository root with the project's virtual environment
available and the tiny COCO split available at
`datasets/mscoco_prompt_split_tiny/train.json`. These are GPU diagnostics; the
configs select `device="cuda"`. Each run writes its metrics, checkpoints, and
plots to the `output_dir` in its config. Existing output directories are reused,
so move or remove an old output directory before starting a clean run.

### Same-image 4 -> 100 -> 1,000 episode ladder

This trains each stage from the previous stage in one invocation:

```bash
.venv/bin/python scripts/rfdetr_pcs/train_pcs_overfit.py \
  --config configs/rfdetr_pcs/pcs_tiny_overfit.py
```

It runs 1,500, 2,000, and 3,000 steps for stages of 4, 100, and 1,000
episodes, respectively. Results are saved under
`logs/rfdetr_pcs_tiny_overfit/`; the full-subset endpoint evaluations are
`posthoc_metrics_4.json`, `posthoc_metrics_100.json`, and
`posthoc_metrics_1000.json`. This is the run that showed poorer fit as the
episode set grew.

### Empty-remainder negative episodes

```bash
.venv/bin/python scripts/rfdetr_pcs/train_pcs_overfit.py \
  --config configs/rfdetr_pcs/pcs_tiny_overfit_negatives.py
```

This tests episodes where every instance of the selected class is prompted, so
there are no remaining target instances. It does not train negative box labels
on boxes from another class. Output is under
`logs/rfdetr_pcs_tiny_overfit_negatives/`.

To isolate negative box prompts from empty-remainder episodes, run:

```bash
.venv/bin/python scripts/rfdetr_pcs/train_pcs_overfit.py \
  --config configs/rfdetr_pcs/pcs_tiny_overfit_negative_prompts.py
```

That config sets `negative_ratio=0.0` and `negative_prompt_ratio=1.0`. Same-image
figures label positive and negative prompts separately. The separate support /
query sampler accepts the same `negative_prompt_ratio`; sampled negative boxes
are taken from other categories present in each support image.

### Separate support/query episodes

The query-only and support-supervised baselines are separate runs:

```bash
.venv/bin/python scripts/rfdetr_pcs/train_support_query_overfit.py \
  --config configs/rfdetr_pcs/pcs_tiny_support_query_overfit.py
.venv/bin/python scripts/rfdetr_pcs/train_support_query_overfit.py \
  --config configs/rfdetr_pcs/pcs_tiny_support_query_overfit_with_support_loss.py
```

The longer support-supervised run is:

```bash
.venv/bin/python scripts/rfdetr_pcs/train_support_query_overfit.py \
  --config configs/rfdetr_pcs/pcs_tiny_support_query_overfit_long.py
```

Its metrics and compact figures are under
`logs/rfdetr_pcs_tiny_support_query_overfit_long/`. At step 5,000, inspect
`metrics.json` and `figures/step_005000/`; the query box IoU was 0.694 and the
support box IoU was 0.859. This is a transfer/conditioning check: the support
image has its own supervised loss, while query predictions must localize
instances in a different image.

### Same-image four-episode long fit and polish

The long fit resumes the original four-episode checkpoint produced by the
ladder run. Run the ladder first so this file exists:
`logs/rfdetr_pcs_tiny_overfit/checkpoint_step_001500.pt`.

```bash
.venv/bin/python scripts/rfdetr_pcs/train_pcs_overfit.py \
  --config configs/rfdetr_pcs/pcs_tiny_overfit_4_resume_long.py
```

This resumes model and optimizer state for 5,000 more steps, reaching global
step 6,500. Its output is `logs/rfdetr_pcs_tiny_overfit_4_resume_long/` and
contains the checkpoint needed by the lower-rate continuation:
`checkpoint_step_006500.pt`.

Then run the 2,000-step lower-learning-rate polish:

```bash
.venv/bin/python scripts/rfdetr_pcs/train_pcs_overfit.py \
  --config configs/rfdetr_pcs/pcs_tiny_overfit_4_polish.py
```

This resumes the step-6,500 checkpoint at `resume_lr=0.00002` and writes to
`logs/rfdetr_pcs_tiny_overfit_4_polish/`. Its final `metrics.json` row is global
step 8,500 (0.976 matched box IoU, 0.946 matched mask IoU). Figures are saved
under `stage_4/figures/step_XXXXXXXX/`; the box and mask IoU curves are saved
at the output root. To reproduce this exact continuation, keep the prior
checkpoints and use the same dataset split, config values, and seed (73).

### Full prompt-split config variants

These two overfit diagnostic configs mirror the split used by
`configs/rfdetr_promptable/promptable_fewshot_2.py` while selecting the PCS
input protocol. They use the full `mscoco_prompt_split` annotations and save to
separate output directories:

```bash
.venv/bin/python scripts/rfdetr_pcs/train_pcs_overfit.py \
  --config configs/rfdetr_pcs/pcs_fewshot_2_same_image_overfit.py
.venv/bin/python scripts/rfdetr_pcs/train_support_query_overfit.py \
  --config configs/rfdetr_pcs/pcs_fewshot_2_support_query.py
```

The same-image curriculum config is now named
`pcs_fewshot_2_same_image_overfit.py` so it is distinct from the full-training
config. The first trains with one image supplying both the box prompt and query
targets. The second uses query images from `train.json` and two prompted support
images from `seen_support.json`, with an auxiliary support loss. These invoke
the current PCS overfit diagnostic scripts; they do not yet provide the
promptable config's full train/validation/test evaluation schedule.

### Full training runs

Use `scripts/rfdetr_pcs/train_pcs.py` for regular training. These configs use
the same COCO class split as the promptable few-shot config, train with the
segmentation head enabled, and evaluate on seen validation episodes during
training. At the end, test metrics and plots use the checkpoint with the best
validation score. Both runs save `train.jsonl`, `metrics.json`, `metrics.csv`,
`training_curves.png`, `best.pt`, `last.pt`, and periodic checkpoints below
their configured output directory. Set `device` to `"cpu"` if CUDA is not
available, and adjust batch size if the GPU runs out of memory.

Same-image prompting trains from one image: the prompt boxes enter the model,
and remaining same-class instances provide query targets. The prompted positive
box is also included in the target loss by default. Empty-remainder episodes
are enabled at a 0.05 ratio; this is distinct from negative box prompts.

```bash
.venv/bin/python scripts/rfdetr_pcs/train_pcs.py \
  --config configs/rfdetr_pcs/pcs_fewshot_2_same_image.py
```

Separate support/query prompting uses two prompted support images per episode
and a different query image. It applies the query detection/segmentation loss
and an auxiliary support loss weighted by 0.5. The image split files must have
disjoint image IDs between support and query within each split.

```bash
.venv/bin/python scripts/rfdetr_pcs/train_pcs.py \
  --config configs/rfdetr_pcs/pcs_train_fewshot_2_support_query.py
```

Both configs set `segmentation=True`. PCS now scores each detection query with
a scaled dot product against the mean pooled encoded box-prompt tokens, and
uses a dedicated decoder presence query with its own binary presence loss.
Inference ranking multiplies the sigmoid query-prompt score by the sigmoid
presence score, following SAM3's score combination. The RF-DETR instance-mask
head is trained from COCO instance masks in addition to box, prompt-score, and
presence losses. For a
box-only run, set `segmentation=False`; the loader then skips mask targets and
the model omits the mask head. Resume training with `--resume` followed by a
`last.pt` or `checkpoint_step_XXXXXXXX.pt` path. Training logs include the
weighted total and component losses; validation metrics include matched box
IoU and, when segmentation is enabled, matched mask IoU.

In `class_embedding` mode, PCS also takes a raw COCO `category_id` for every request. A learned embedding
for that category is added to the encoded box-prompt tokens, so the decoder and
dot-product scorer receive both visual box hints and class identity. The table
has 91 rows (row 0 reserved) and initializes from RF-DETR's pretrained COCO
classifier weights using the `COCO_CLASSES` category-to-row mapping; it remains
trainable with the other PCS parameters. Training configs use
`negative_class_ratio=0.1` (about 9% absent-class episodes): the request class is absent
from the query image, its target is empty, and the presence target is negative.
The full-trainer configs set `pretrain_weights="rf-detr-base.pth"` and load that
checkpoint before copying its classifier rows. The overfit diagnostics keep
`pretrain_weights=None` to preserve their DINOv2-backbone-only setup.

For separate support/query episodes there is one category ID per episode in
class-embedding mode. The
class token is appended once after merging the box tokens from all support
images, then routed to every query image. The optional support auxiliary pass
gets one class token per support image. This adds a single token per query
decoder sequence and does not replicate image features; the main extra cost is
still encoding the configured support images. Same-image and support/query
forward calls accept `class_ids`; raw outputs return aligned
`prompt_class_ids` with `pred_logits`, `pred_boxes`, and optional `pred_masks`.
For simultaneous requests for different classes, batch one request per class
(repeat the image in the input batch and provide a class ID for each row). A
shared image-encoding path for multiple class requests on the same image is not
implemented yet.

### Independent known-class and visual PCS conditioning

The two conditioning paths are now explicit and independent:

- `conditioning_mode="class_embedding"` appends the trainable known-class token
  initialized from the RF-DETR classifier. This is the text-encoder-free path
  for categories in the fixed COCO vocabulary.
- `conditioning_mode="visual_prompt"` builds conditioning only from support
  image features, labeled boxes, and prompt geometry. The model is constructed
  without a class-embedding table, and trainers do not pass category IDs into
  its forward method. Dataset episode generation may still use category IDs to
  choose same-concept support/query examples and targets; those IDs are
  bookkeeping/supervision only and are not model inputs.

The visual path uses `BoxPromptEncoder`: ROI pooled features from each box are
combined with box coordinates and positive/negative prompt labels, then the
tokens exchange information and attend to the image feature map. Decoder object
queries attend to those tokens. The existing detection and instance-mask heads
are trained from query boxes and COCO masks, so visual conditioning can learn
without class embeddings. Multiple support shots contribute multiple visual
tokens; one episode must represent a single target concept. Same-image visual
configs set `negative_class_ratio=0` because an absent class with no positive
visual exemplar is not a defined visual prompt. Separate support/query configs
can retain absent-query episodes because positive support examples define the
concept. Their `validation_negative_class_ratio` and
`test_negative_class_ratio` settings are independently configurable; the visual
support/query config includes both positive and absent-query evaluation cases.

Older configs that omit `conditioning_mode` retain their former
class-embedding behavior. New PCS-only configs set `visual_prompt` explicitly.
The trainer also accepts a curriculum, for example 10,000 class-conditioned
steps followed by 90,000 visual-only steps:

```python
"conditioning_mode": "visual_prompt",
"conditioning_schedule": [
    {"mode": "class_embedding", "steps": 10_000},
    {"mode": "visual_prompt", "steps": 90_000},
],
```

The class embedding is present and trainable during that curriculum, then no
longer enters the forward path after the switch. Alternatively, finish a
class-conditioned run and start a fresh-optimizer visual PCS run from its model
weights with `init_checkpoint`; this loads compatible detector/prompt weights
without restoring the class embedding or optimizer state. `--resume` remains
the full optimizer-and-step resume option.

Ready-to-run configs:

- `pcs_train_fewshot_2_same_image_visual.py`: same-image visual PCS, no class
  IDs passed to the model.
- `pcs_train_fewshot_2_support_query_visual.py`: two-shot visual support/query
  PCS, including positive supports and optional absent-class queries.
- `pcs_train_same_image_class_to_visual_curriculum.py`: one-run class warmup
  followed by visual-only training.
- `pcs_train_fewshot_2_same_image_class_warmup.py` followed by
  `pcs_train_fewshot_2_same_image_visual_from_class.py`: two-run same-image
  warmup and visual PCS, with weights-only transfer.
- `pcs_train_fewshot_2_support_query_class_warmup.py` followed by
  `pcs_train_fewshot_2_support_query_visual_from_class.py`: corresponding
  two-run multi-support path.

Example commands (run from repository root):

```bash
# Visual prompt only
.venv/bin/python scripts/rfdetr_pcs/train_pcs.py --config configs/rfdetr_pcs/pcs_train_fewshot_2_same_image_visual.py
.venv/bin/python scripts/rfdetr_pcs/train_pcs.py --config configs/rfdetr_pcs/pcs_train_fewshot_2_support_query_visual.py

# One-run class-to-visual curriculum
.venv/bin/python scripts/rfdetr_pcs/train_pcs.py --config configs/rfdetr_pcs/pcs_train_same_image_class_to_visual_curriculum.py

# Two-run class warmup then visual-only PCS (fresh optimizer in phase two)
.venv/bin/python scripts/rfdetr_pcs/train_pcs.py --config configs/rfdetr_pcs/pcs_train_fewshot_2_same_image_class_warmup.py
.venv/bin/python scripts/rfdetr_pcs/train_pcs.py --config configs/rfdetr_pcs/pcs_train_fewshot_2_same_image_visual_from_class.py
.venv/bin/python scripts/rfdetr_pcs/train_pcs.py --config configs/rfdetr_pcs/pcs_train_fewshot_2_support_query_class_warmup.py
.venv/bin/python scripts/rfdetr_pcs/train_pcs.py --config configs/rfdetr_pcs/pcs_train_fewshot_2_support_query_visual_from_class.py
```

The tiny visual-mode overfit diagnostics are configured, but deliberately have
not been run:

```bash
.venv/bin/python scripts/rfdetr_pcs/train_pcs_overfit.py --config configs/rfdetr_pcs/pcs_tiny_visual_prompt_overfit.py
.venv/bin/python scripts/rfdetr_pcs/train_support_query_overfit.py --config configs/rfdetr_pcs/pcs_tiny_visual_prompt_support_query_overfit.py
```

Both visual modes retain optional segmentation training when
`segmentation=True`; the support/query trainer can additionally apply
`support_loss_weight` to support-image box/mask supervision.

Old PCS checkpoints can warm-start the updated model: matching tensors load,
new class/scoring/presence parameters keep their initialization, and incompatible
classifier shapes are skipped with a message. If the old optimizer parameter
groups do not match, training resumes with a fresh optimizer state.

### Remaining parity TODOs

- Replace annotation-sampled negative boxes with SAM3-style hard negatives
  mined from confident false positives; `negative_prompt_ratio` currently
  samples other annotated objects.
- Add shared feature reuse when one image is queried for several class IDs in
  one call; today this duplicates image rows and backbone work.
- Run the configured visual-only 4/100/1,000-episode diagnostic and small
  separate support/query diagnostic, then inspect box/mask quality and
  absent-query presence behavior before long training. These runs were not
  launched as part of the implementation.

The active `pcs_fewshot_2_same_image.py` is the regular trainer config and sets
`steps=100_000`. Its staged curriculum diagnostic remains available as
`pcs_fewshot_2_same_image_overfit.py`, where `stage_steps` controls the three
overfit stages.

