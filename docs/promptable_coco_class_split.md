# Promptable COCO Class-Split Experiment

> **Historical v1 only:** this runbook evaluates same-image box prompts, where the query image supplies its own ground-truth box. It is not cross-image visual few-shot detection and its IoU must not be reported as unseen-class recognition. The frozen v1 implementation is under `experiments/promptable_v1/`. For support-image/box prompts applied to separate query images, use [the few-shot runbook](promptable_fewshot.md).

For the complete current reproduction commands and status, see [the reproducibility report](promptable_reproduction_report.md).

This experiment deliberately separates classes seen during training from classes used at test time.

## Already Built

The isolated implementation lives in `src/rfdetr_promptable`; the original `src/rfdetr` package is unchanged.

- `PromptableDetector`: one dynamic decoder query per supplied bounding-box prompt.
- Binary objectness score and box refinement loss.
- `CocoBoxPromptDataset`: loads an image and a positive COCO box prompt.
- `scripts/train_promptable_rfdetr.py`: training, checkpoint save, and one-image inference smoke test.
- `scripts/evaluate_promptable_rfdetr.py`: prompted IoU evaluation and qualitative figures.
- `scripts/generate_coco_prompt_split.py`: deterministic filtered COCO JSON generation.

This is same-image box prompting: the test image itself supplies the box prompt. It is not yet cross-image visual-exemplar prompting.

The RF-DETR detection task is initialized from scratch with `pretrain_weights=None`. The DINOv2 image backbone may load its own pretrained weights; no RF-DETR detection checkpoint or pretrained RF-DETR detection head is loaded.

## Configuration

Python preset: `configs/promptable_coco_class_split.py` (`cfg` values can be overridden with argparse flags).

The preset selects:

- Up to 10,000 training prompt examples
- 10 training classes
- All validation images containing seen classes
- 10 disjoint held-out classes
- deterministic seed `7`
- resolution `560` with the Base configuration
- 10,000 GPU training steps by default

COCO category IDs in the preset:

- Train: `1, 3, 8, 9, 10, 11, 13, 14, 15, 16`
- Held out: `17, 18, 19, 20, 21, 22, 23, 24, 25, 27`

## Commands

Run from the repository root.

### 1. Generate filtered annotations

```bash
uv run python scripts/generate_coco_prompt_split.py \
  --config configs/promptable_coco_class_split.py
```

Pass any generator options directly to override the preset; for example, add `--seed 11 --train-images 5000`.

Expected output reports training, seen-validation, and unseen-test image/annotation counts. It writes `train.json`, `seen_val.json`, and `unseen_test.json`.

### 2. Train on the seen classes

```bash
uv run python scripts/train_promptable_rfdetr.py \
  --config configs/promptable_same_image_baseline.py \
  --device cuda
```

Any option can be overridden on the command line, for example `--steps 20000`.

For a practical run, use a CUDA device and increase `--steps` substantially, for example `10000` or more.

### 3. Evaluate prompted held-out classes

```bash
uv run python scripts/evaluate_promptable_rfdetr.py \
  --image-dir datasets/mscoco/val2017 \
  --annotation-file datasets/mscoco_prompt_split/unseen_test.json \
  --checkpoint logs/promptable_coco_class_split.pt \
  --resolution 560 \
  --threshold 0.0 \
  --figure-dir logs/promptable_experiment/unseen \
  --device cuda
```

The evaluator supplies each held-out object box as the prompt and reports per-class mean score, mean IoU, and recall@0.5. It also writes original, prediction, and prediction-plus-ground-truth figures.

Run the same evaluation on seen validation classes:

```bash
uv run python scripts/evaluate_promptable_rfdetr.py \
  --image-dir datasets/mscoco/val2017 \
  --annotation-file datasets/mscoco_prompt_split/seen_val.json \
  --checkpoint logs/promptable_coco_class_split.pt \
  --resolution 560 \
  --threshold 0.0 \
  --figure-dir logs/promptable_experiment/seen \
  --device cuda
```

### 4. Run the existing smoke test

```bash
uv run python scripts/train_promptable_rfdetr.py \
  --device cpu \
  --split train2017 \
  --steps 1 \
  --batch-size 1 \
  --resolution 560 \
  --output logs/promptable_rfdetr_smoke.pt
```

## Interpretation

A high IoU on held-out classes measures box-conditioned localization/refinement, not semantic recognition from an unseen class name. Because the ground-truth test box is provided, the experiment tests whether the promptable decoder can use visual image evidence to refine the selected object.

To test true SAM3-like cross-image generalization, the next experiment must use a support image plus support box to create a visual prompt embedding, then apply it to a separate query image without giving the query object box.
