# SAM3 visual-example segmentation on DFFF

This workflow uses the labeled DFFF image as a **visual support prompt** and a
separate, unlabeled-at-inference query image. Since SAM3's image predictor
accepts one image at a time, the script lays out the support image(s) and query
image in separate panels of one square canvas. Positive bounding-box prompts
are placed only on the support panel(s); the query panel receives no box, mask,
or other label. The model is asked for the class text plus the example box(es),
then detections are cropped back to the query panel.

## What is in the current converted data

The COCO files under `datasets/dfff_coco/` describe bounding boxes only: none of
the annotations has a `segmentation` field. The exact target-class inventory is:

| Split | Images | All boxes | Nasal bone boxes | NT boxes | Segmentation masks |
| --- | ---: | ---: | ---: | ---: | ---: |
| set1 | 812 | 6,742 | 722 | 794 | 0 |
| set2 | 560 | 2,067 | 224 | 244 | 0 |
| internal | 156 | 624 | 69 | 72 | 0 |
| external | 156 | 0 | 0 | 0 | 0 |

`external` therefore supports qualitative inference only. There are no query
labels there with which to compute metrics. Start with support `set1` and query
`internal` or `set2` for a labeled evaluation. Avoid evaluating on the same
source split used for support unless explicitly testing a disjoint within-split
partition.

## Prepare/filter examples

List all four split inventories without writing output:

```bash
uv run python scripts/prepare_sam3_dfff.py --dataset-root datasets/dfff_coco --list-only
```

Create a deterministic one-shot support/query manifest for the internal set:

```bash
uv run python scripts/prepare_sam3_dfff.py \
  --dataset-root datasets/dfff_coco \
  --support-split set1 --query-split internal \
  --classes "nasal bone" NT --shots 1 --seed 7
```

By default this writes `outputs/sam3_dfff/manifest.json` and a separate
`outputs/sam3_dfff/evaluation.json`. The inference manifest contains support
boxes, but **does not contain query boxes**. Query boxes live only in the
evaluation file and are not opened until all model calls have finished.
For labeled query splits, the manifest includes all images (not just images
annotated for the requested class); the separate evaluation file records only
positive query boxes. This allows AP to penalize class hallucinations on
class-negative images.

### Direct-from-folder / online-to-this-workspace mode

The default preparation mode records absolute paths to the existing images in
`datasets/dfff_coco`. It does not duplicate them; inference opens each image
directly from that folder. This is the simplest way to run against any of the
four existing splits. For an unlabeled qualitative query set, for example:

```bash
uv run python scripts/prepare_sam3_dfff.py \
  --dataset-root datasets/dfff_coco \
  --support-split set1 --query-split external \
  --classes "nasal bone" NT --max-queries-per-class 20
```

Here “online” means reading the existing source tree on demand; it does not
download DFFF images from a remote service.

### Portable offline image bundle

Add `--copy-images` to copy only selected examples under
`outputs/sam3_dfff/images/{support,query split}/...` while still preserving
the separate evaluation labels. This lets the prepared examples move with the
manifest. The separate SAM3 checkpoint must also be present locally, or already
cached from Hugging Face.

To filter other splits, change `--support-split` and `--query-split` to `set1`,
`set2`, `internal`, or `external`. `--max-queries-per-class N` limits runtime;
omit it to use every labeled query for the requested classes. `--shots N`
chooses N different support images per class (one positive box per image).

## Run SAM3 and save plots

SAM3 prerequisites in the upstream README are Python 3.12+, PyTorch 2.7+, and a
CUDA-compatible GPU (CUDA 12.6+ recommended). Access to the gated SAM3
checkpoint on Hugging Face must be requested and authenticated. Use a local
checkpoint with `--checkpoint` if it is already available:

```bash
uv run python scripts/infer_sam3_dfff.py \
  --manifest outputs/sam3_dfff/manifest.json \
  --output-dir outputs/sam3_dfff/results \
  --device auto
```

The script accepts `--device cuda` or `--device cpu`, `--checkpoint PATH`,
`--score-threshold 0.25`, and `--resolution 1008`. Use
`--prompt-box-expand 1.5` (or `2.`) to scale support-box width and height by
that ratio, expanding equally around the center. An integer such as
`--prompt-box-expand 12` adds 12 pixels on every side; `12px` is also accepted
to make the unit explicit. Expanded coordinates are clipped to the support
image bounds. The default `1.0` applies no expansion. For example:

```bash
# Give the positive support prompt 50% more width and height.
uv run python scripts/infer_sam3_dfff.py --prompt-box-expand 1.5

# Add 12 pixels of margin on each side of every support prompt box.
uv run python scripts/infer_sam3_dfff.py --prompt-box-expand 12
```

CPU execution is supported but SAM3 image inference is large and may be slow.
For each pair the script saves a query binary mask and a PNG showing the
original support box, expanded prompt box, predicted query mask, and (only in
this post-inference visualization) the query ground-truth box. It also writes
`predictions.json`, `metrics.json`, and `metrics.png`.

The support/query pair is packed into a square canvas with aspect-ratio-preserving
panels; at the default 1008 resolution, one-shot support and query panels are
each about 504 pixels wide before inference. More shots provide more positive
visual prompts but reduce pixels per panel, so begin with one shot.

## Evaluation and interpretation

COCO boxes are **not segmentation ground truth**. Therefore the evaluation
reports conventional box metrics plus explicitly labeled box-derived mask
proxies:

- Box AP at IoU 0.50 and 0.75, mean predicted-box IoU, and box recall at 0.50
  and 0.75. AP uses all query images, including images without that class's
  box, so it penalizes hallucinations; mean IoU/recall summarize positive
  queries. Predicted boxes are the tight extents of each predicted mask.
- Proxy mask IoU and Dice against a filled rectangle made from the query's
  ground-truth box, box coverage recall, and predicted-mask fraction outside
  that rectangle.

The proxy scores are useful for detecting empty masks, poor localization, and
large leakage, but they **do not measure anatomical boundary accuracy** and
must not be presented as true segmentation IoU/Dice. Real mask-quality metrics
such as mask AP, Dice against expert contours, and boundary F-score require
pixel-level ground-truth masks. The saved visual overlays are intended for
manual quality checking.

## Implementation notes

- Uses the official SAM3 image model and image processor, guided by its
  single-image visual box-prompt API and the local demonstration in
  `scripts/sam3_test2.py`.
- Default requested classes are the COCO names `nasal bone` and `NT`; the
  language prompt for NT is expanded to “nuchal translucency (NT)”.
- Each query image is disjoint from its support images. Query boxes are excluded
  from the SAM3 call path and loaded only for post-run scoring/plots.
- External images have no annotation workbook targets in the converted COCO
  split, so their results are saved without a numeric evaluation score.