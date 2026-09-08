# Promptable COCO Class-Split Experiment

This experiment deliberately separates classes seen during training from classes used at test time.

## Already Built

The isolated implementation lives in `src/rfdetr_promptable`; the original `src/rfdetr` package is unchanged.

- `PromptableDetector`: one dynamic decoder query per supplied bounding-box prompt.
- Binary objectness score and box refinement loss.
- `CocoBoxPromptDataset`: loads an image and a positive COCO box prompt.
- `scripts/train_promptable_rfdetr.py`: training, checkpoint save, and one-image inference smoke test.
- `scripts/evaluate_promptable_rfdetr.py`: prompted IoU evaluation.
- `scripts/generate_coco_prompt_split.py`: deterministic filtered COCO JSON generation.

This is same-image box prompting: the test image itself supplies the box prompt. It is not yet cross-image visual-exemplar prompting.

## Configuration

Preset: `configs/promptable_coco_class_split.yaml`

The preset selects:

- 100 training images
- 10 training classes
- 100 validation images
- 10 disjoint held-out classes
- deterministic seed `7`
- resolution `384`
- 100 CPU training steps by default

COCO category IDs in the preset:

- Train: `1, 3, 8, 9, 10, 11, 13, 14, 15, 16`
- Held out: `17, 18, 19, 20, 21, 22, 23, 24, 25, 27`

## Commands

Run from the repository root.

### 1. Generate filtered annotations

```bash
uv run python scripts/generate_coco_prompt_split.py \
  --source-annotation datasets/mscoco/annotations_trainval2017/annotations/instances_train2017.json \
  --output-dir datasets/mscoco_prompt_split \
  --train-classes 1 3 8 9 10 11 13 14 15 16 \
  --test-classes 17 18 19 20 21 22 23 24 25 27 \
  --train-images 100 \
  --test-images 100 \
  --seed 7
```

Expected output reports the number of selected images and annotations in `train.json` and `test.json`.

### 2. Train on the seen classes

```bash
uv run python scripts/train_promptable_rfdetr.py \
  --image-dir datasets/mscoco/train2017 \
  --annotation-file datasets/mscoco_prompt_split/train.json \
  --steps 100 \
  --batch-size 1 \
  --resolution 384 \
  --device cpu \
  --output logs/promptable_coco_class_split.pt
```

For a practical run, use a CUDA device and increase `--steps` substantially, for example `10000` or more.

### 3. Evaluate prompted held-out classes

```bash
uv run python scripts/evaluate_promptable_rfdetr.py \
  --image-dir datasets/mscoco/val2017 \
  --annotation-file datasets/mscoco_prompt_split/test.json \
  --checkpoint logs/promptable_coco_class_split.pt \
  --resolution 384 \
  --threshold 0.0 \
  --device cpu
```

The evaluator supplies each held-out object box as the prompt and reports mean prompt score and mean IoU after box refinement.

### 4. Run the existing smoke test

```bash
uv run python scripts/train_promptable_rfdetr.py \
  --device cpu \
  --split train2017 \
  --steps 1 \
  --batch-size 1 \
  --resolution 384 \
  --output logs/promptable_rfdetr_smoke.pt
```

## Interpretation

A high IoU on held-out classes measures box-conditioned localization/refinement, not semantic recognition from an unseen class name. Because the ground-truth test box is provided, the experiment tests whether the promptable decoder can use visual image evidence to refine the selected object.

To test true SAM3-like cross-image generalization, the next experiment must use a support image plus support box to create a visual prompt embedding, then apply it to a separate query image without giving the query object box.
