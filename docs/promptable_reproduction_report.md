# Promptable Detection: Reproduction Report and Runbook

This report consolidates the current implementation status, data protocol, expected outputs, and commands for reproducing the promptable COCO runs. Run commands from the repository root.

## Current implementation and status

- Cross-image visual few-shot code is isolated in `src/rfdetr_promptable`; the original `src/rfdetr` package is unchanged.
- Active few-shot presets are Python modules exporting a `cfg` dictionary. The trainer loads them with `--config`; explicit CLI arguments override config values.
- RF-DETR detection/query/box task weights are initialized from scratch. The DINOv2 backbone loads its pretrained weights at the Base model's 560-pixel resolution.
- Support image features are pooled in the supplied support box. Query images do not receive their ground-truth boxes as model inputs.
- The episodic diagnostic reports localization and presence measures. The final runner also uses `pycocotools.COCOeval` for bbox AP and AR.
- **The 10,000-step experiments have not been run to completion.** Current metric values from one-step/tiny-subset smoke runs are plumbing checks, not accuracy results.
- Checkpoint selection uses seen-validation episodic macro IoU. COCO mAP is computed for the selected checkpoint at the end, not used for checkpoint selection.

## Requirements and dataset layout

Python 3.12+, `uv`, and a CUDA-capable PyTorch environment are expected for the full runs. COCO 2017 images and annotations must already be present locally in this layout:

```text
datasets/mscoco/
├── train2017/
├── val2017/
└── annotations_trainval2017/annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

Set up the project environment and confirm CUDA visibility:

```bash
uv sync
uv run python -c 'import torch; print("torch", torch.__version__); print("cuda", torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA unavailable")'
```

Check the required dataset files before generating splits:

```bash
test -d datasets/mscoco/train2017
test -d datasets/mscoco/val2017
test -f datasets/mscoco/annotations_trainval2017/annotations/instances_train2017.json
test -f datasets/mscoco/annotations_trainval2017/annotations/instances_val2017.json
```

## Reproduce the class split

The seen/training classes are COCO IDs `1, 3, 8, 9, 10, 11, 13, 14, 15, 16`. The disjoint held-out classes are `17, 18, 19, 20, 21, 22, 23, 24, 25, 27`.

Generate the filtered annotation files from the Python config (CLI options override `cfg` values):

```bash
uv run python scripts/generate_coco_prompt_split.py \
  --config configs/promptable_coco_class_split.py
```

Equivalent explicit argparse invocation (without loading a config file):

```bash
uv run python scripts/generate_coco_prompt_split.py \
  --train-source-annotation datasets/mscoco/annotations_trainval2017/annotations/instances_train2017.json \
  --validation-source-annotation datasets/mscoco/annotations_trainval2017/annotations/instances_val2017.json \
  --output-dir datasets/mscoco_prompt_split \
  --train-classes 1 3 8 9 10 11 13 14 15 16 \
  --test-classes 17 18 19 20 21 22 23 24 25 27 \
  --train-images 10000 --seed 7
```

The generator filters annotations by class. The train JSON includes the selected 10,000 train images; each validation/test JSON retains all 5,000 COCO val image records so COCO AP includes images without the evaluated classes. The verified generated counts are:

| Split | Image records | Annotations |
| --- | ---: | ---: |
| `train.json` | 10,000 | 48,168 |
| `seen_val.json` | 5,000 | 15,507 |
| `unseen_test.json` | 5,000 | 2,631 |

Verify a regenerated split:

```bash
uv run python - <<'PY'
import json
from pathlib import Path
for name in ("train.json", "seen_val.json", "unseen_test.json"):
    data = json.loads((Path("datasets/mscoco_prompt_split") / name).read_text())
    print(name, "images=", len(data["images"]), "annotations=", len(data["annotations"]))
PY
```

## Run full few-shot experiments

Each command trains one preset. The Python config supplies dataset paths and defaults; command-line flags can override any configured option. Each run writes a final checkpoint, a best-seen-validation checkpoint, episodic metrics, COCO metrics, and held-out qualitative figures.

### One-shot

```bash
uv run python scripts/train_promptable_fewshot.py \
  --config configs/promptable_fewshot_1shot.py \
  --device cuda
```

### One-to-five-shot training distribution

```bash
uv run python scripts/train_promptable_fewshot.py \
  --config configs/promptable_fewshot_5shot.py \
  --device cuda
```

### Negative-heavy training distribution

```bash
uv run python scripts/train_promptable_fewshot.py \
  --config configs/promptable_fewshot_negative_heavy.py \
  --device cuda
```

To override a preset, append command-line options. For example, reduce the COCO evaluation batch size if GPU memory is constrained:

```bash
uv run python scripts/train_promptable_fewshot.py \
  --config configs/promptable_fewshot_5shot.py \
  --device cuda \
  --coco-eval-batch-size 2
```

The defaults configure 10,000 training steps, evaluation every 1,000 steps, 100 decoder queries, resolution 560, and a batch size of 2. `negative_ratio` is the expected negative-to-positive training episode ratio: `0.5` means one negative per two positives on average; `1.5` emphasizes absent-class rejection.

Outputs follow the config's `output` and `figure_dir`, for example:

```text
logs/promptable_fewshot_1shot.pt
logs/promptable_fewshot_1shot.best.pt
logs/promptable_fewshot_1shot.metrics.json
logs/promptable_fewshot_figures/1shot/heldout_test/
```

The `.metrics.json` records episodic scores and two standard COCO evaluation sections: `seen_validation_coco` and `held_out_coco`. COCO AP values are on the 0–1 scale. COCO AP is the 101-point interpolated average over IoU thresholds 0.50:0.05:0.95; AP50/AP75, scale-specific AP, and AR@1/10/100 are also included. The macro mAP is averaged over the configured class set. A deterministic support set is selected per category; the union of all support-image IDs is excluded from a shared query-image set. All other COCO val images, including absent-class images, are queried. The episodic diagnostic remains separate and is not AP.

The best checkpoint is selected by seen-validation episodic macro IoU, not by COCO mAP. Final COCO evaluation can be lengthy: the current implementation recomputes image features during batched evaluation. `--coco-eval-batch-size` controls the query batch size and memory use.

There is currently no standalone support/query COCO-evaluation command for an already-trained checkpoint; the final COCO evaluation is part of the training runner. To recompute these metrics with the current code, rerun the selected experiment/config.

All active promptable presets are Python `cfg` modules or direct argparse options. The frozen `experiments/promptable_v1/` snapshot retains its historical YAML preset for archival reproducibility; RF-DETR and SAM3 upstream YAML configs are left intact because their existing loaders depend on them.

## Compact smoke run

This smoke command creates tiny derived COCO annotation files under `/tmp`, performs one GPU optimization step, evaluates a few seen and held-out episodes, and exercises COCOeval. The scores are not suitable for reporting as model performance.

Create the tiny files:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

source_dir = Path("datasets/mscoco_prompt_split")
out_dir = Path("/tmp/promptable_smoke")
out_dir.mkdir(parents=True, exist_ok=True)

def make_subset(source_name: str, category_id: int, positive_count: int, empty_count: int) -> None:
    source = json.loads((source_dir / source_name).read_text())
    annotations = [a for a in source["annotations"] if a["category_id"] == category_id]
    positive_ids = list(dict.fromkeys(a["image_id"] for a in annotations))
    selected = set(positive_ids[:positive_count])
    for image in source["images"]:
        if image["id"] not in set(positive_ids) and empty_count:
            selected.add(image["id"])
            empty_count -= 1
    result = {
        **source,
        "images": [im for im in source["images"] if im["id"] in selected],
        "annotations": [a for a in annotations if a["image_id"] in selected],
        "categories": [c for c in source["categories"] if c["id"] == category_id],
    }
    (out_dir / source_name).write_text(json.dumps(result))

make_subset("train.json", 1, positive_count=5, empty_count=2)
make_subset("seen_val.json", 1, positive_count=3, empty_count=1)
make_subset("unseen_test.json", 17, positive_count=3, empty_count=1)
print("wrote smoke annotations to", out_dir)
PY
```

Run one short training/evaluation cycle:

```bash
uv run python scripts/train_promptable_fewshot.py \
  --config configs/promptable_fewshot_1shot.py \
  --train-image-dir datasets/mscoco/train2017 \
  --train-annotation-file /tmp/promptable_smoke/train.json \
  --validation-image-dir datasets/mscoco/val2017 \
  --validation-annotation-file /tmp/promptable_smoke/seen_val.json \
  --test-image-dir datasets/mscoco/val2017 \
  --test-annotation-file /tmp/promptable_smoke/unseen_test.json \
  --episodes 8 --max-shots 1 --negative-eval-per-class 1 \
  --steps 1 --eval-every 1 --batch-size 2 --coco-eval-batch-size 2 \
  --num-queries 10 --resolution 560 --device cuda \
  --max-figures-per-class 1 \
  --figure-dir /tmp/promptable_smoke/figures \
  --output /tmp/promptable_smoke/model.pt
```

The command prints COCOeval's standard summary and writes `/tmp/promptable_smoke/model.metrics.json`. Check that both COCO sections exist:

```bash
uv run python - <<'PY'
import json
from pathlib import Path
metrics = json.loads(Path("/tmp/promptable_smoke/model.metrics.json").read_text())
for key in ("seen_validation_coco", "held_out_coco"):
    result = metrics[key]
    print(key, "mAP=", result["mAP"], "mAP50=", result["mAP50"], "mAP75=", result["mAP75"])
PY
```

## Run cross-image inference

Use an actual checkpoint and a support box in original-image pixel `xyxy` coordinates. Replace the example image paths and box coordinates with a labeled object from your support image. The integer group ID is arbitrary but must match the query's `--query-group`.

```bash
uv run python scripts/infer_promptable.py \
  --checkpoint logs/promptable_fewshot_1shot.best.pt \
  --support-example /path/to/support.jpg 120 60 340 400 17 \
  --support-example /path/to/support_second_shot.jpg 85 45 275 360 17 \
  --query-image /path/to/query_a.jpg --query-group 17 \
  --query-image /path/to/query_b.jpg --query-group 17 \
  --resolution 560 --num-queries 100 --threshold 0.0 --device cuda \
  --output logs/support_query_predictions.json
```

Predicted boxes in the output JSON are pixel `xyxy` coordinates in each query image. To inspect all supported options:

```bash
uv run python scripts/train_promptable_fewshot.py --help
uv run python scripts/infer_promptable.py --help
uv run python scripts/generate_coco_prompt_split.py --help
```

## Run the active same-image baseline

This is the older box-prompt task where the query image itself provides the prompt. It uses a Python cfg file and accepts direct argparse overrides:

```bash
uv run python scripts/train_promptable_rfdetr.py \
  --config configs/promptable_same_image_baseline.py \
  --device cuda
```

Its evaluator reports prompt-conditioned localization diagnostics, not standard COCO AP:

```bash
uv run python scripts/evaluate_promptable_rfdetr.py \
  --image-dir datasets/mscoco/val2017 \
  --annotation-file datasets/mscoco_prompt_split/unseen_test.json \
  --checkpoint logs/promptable_same_image_baseline.best.pt \
  --resolution 560 --threshold 0.0 --device cuda \
  --figure-dir logs/promptable_same_image_baseline_figures/unseen
```

## Historical same-image v1 baseline

The v1 snapshot prompts with the query image's own box and is not cross-image few-shot evaluation. Use it only to reproduce the historical plumbing baseline. Set `PYTHONPATH` to the snapshot package so imports resolve to v1 rather than the current implementation.

Train:

```bash
PYTHONPATH=experiments/promptable_v1/src uv run python experiments/promptable_v1/scripts/train_promptable_rfdetr.py \
  --image-dir datasets/mscoco/train2017 \
  --annotation-file datasets/mscoco_prompt_split/train.json \
  --validation-image-dir datasets/mscoco/val2017 \
  --validation-annotation-file datasets/mscoco_prompt_split/seen_val.json \
  --test-image-dir datasets/mscoco/val2017 \
  --test-annotation-file datasets/mscoco_prompt_split/unseen_test.json \
  --steps 10000 --eval-every 1000 --batch-size 1 --resolution 560 --device cuda \
  --output logs/promptable_v1_same_image.pt
```

Evaluate the v1 checkpoint on held-out classes (the query image's GT box is used as the prompt):

```bash
PYTHONPATH=experiments/promptable_v1/src uv run python experiments/promptable_v1/scripts/evaluate_promptable_rfdetr.py \
  --image-dir datasets/mscoco/val2017 \
  --annotation-file datasets/mscoco_prompt_split/unseen_test.json \
  --checkpoint logs/promptable_v1_same_image.best.pt \
  --resolution 560 --threshold 0.0 --device cuda \
  --figure-dir logs/promptable_v1_same_image_figures/unseen
```

Do not compare these v1 prompted IoUs to few-shot COCO AP/mAP as if they measured the same task.
