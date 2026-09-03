# CV project

Tools for converting the DFFF object-detection annotations to the Microsoft COCO
dataset format.

## Requirements

- Python 3.12 or newer
- `pandas`, `openpyxl`, and `Pillow` (declared in `pyproject.toml`)

With [uv](https://docs.astral.sh/uv/), create the environment and install the
project dependencies with:

```bash
uv sync
```

## Convert Excel annotations to COCO

[`scripts/convert_excel_to_coco.py`](scripts/convert_excel_to_coco.py) accepts
a source directory and a target directory. It reads `ObjectDetection.xlsx` by
default, identifies each image's DFFF parent folder, and produces four separate
COCO splits: `set1`, `set2`, `internal`, and `external`.

```bash
uv run python scripts/convert_excel_to_coco.py <source-folder> <target-folder>
```

For the DFFF data linked into this repository, the default output location is
`datasets/coco_dfff`:

```bash
uv run python scripts/convert_excel_to_coco.py datasets/dfff --overwrite
```

Pass a second positional argument to use another output location:

```bash
uv run python scripts/convert_excel_to_coco.py datasets/dfff path/to/coco_dfff --overwrite
```

Use `--excel` when the annotation workbook has another name or location:

```bash
uv run python scripts/convert_excel_to_coco.py <source-folder> <target-folder> \
  --excel path/to/annotations.xlsx
```

The target directory must not already exist unless `--overwrite` is supplied.
The converter validates that every annotated filename resolves to exactly one
image in one source split and that every bounding box lies inside its image.

## Expected Excel schema

The workbook must have these columns:

| Column | Meaning |
| --- | --- |
| `fname` | Image filename |
| `structure` | Object class name |
| `h_min`, `w_min` | Top and left box coordinates |
| `h_max`, `w_max` | Bottom and right box coordinates |

The source coordinates are interpreted as `(y_min, x_min, y_max, x_max)` and
converted to the COCO `bbox` convention: `[x_min, y_min, width, height]`.

## Output layout

```text
<target-folder>/
├── set1/
│   ├── annotations/instances.json
│   └── images/Standard/...
├── set2/
│   ├── annotations/instances.json
│   └── images/{Standard,Non-standard}/...
├── internal/
│   ├── annotations/instances.json
│   └── images/{Standard,Non-standard}/...
└── external/
    ├── annotations/instances.json
    └── images/{Standard,Non-standard}/...
```

Each `instances.json` contains standard COCO `images`, `annotations`, and
`categories` arrays. Images are copied into the output directory so each split
is self-contained. The Excel workbook has no annotations for External, so its
COCO file correctly contains images and categories but no annotations.

The DFFF conversion generated in this workspace contains 1,684 images, 9,433
annotations, and 9 categories: Set1 (812/6,742), Set2 (560/2,067), Internal
(156/624), and External (156/0), shown as images/annotations.

## Create an RF-DETR training dataset

RF-DETR detects COCO datasets when every output split contains image files and
an `_annotations.coco.json` file. Use
[`scripts/split_coco_for_rfdetr.py`](scripts/split_coco_for_rfdetr.py) with the
three generated COCO folders directly:

```bash
uv run python scripts/split_coco_for_rfdetr.py \
  datasets/coco_dfff/set1 \
  datasets/coco_dfff/set2 \
  datasets/coco_dfff/internal \
  --overwrite
```

The arguments map directly to `train`, `valid`, and `test`, respectively. By
default all images from each source folder are used, and the RF-DETR-ready
dataset is written to `datasets/rfdetr_dfff`.

To randomly sample each source folder, use either one fraction or one exact
count for each of train, valid, and test. `--seed` makes a random draw
reproducible; omit it to get a new draw each run.

```bash
# Select 30% of Set1, 50% of Set2, and 20% of Internal.
uv run python scripts/split_coco_for_rfdetr.py \
  datasets/coco_dfff/set1 datasets/coco_dfff/set2 datasets/coco_dfff/internal \
  --fractions 0.3 0.5 0.2 --seed 7 --overwrite

# Select exact random image counts from the same three folders.
uv run python scripts/split_coco_for_rfdetr.py \
  datasets/coco_dfff/set1 datasets/coco_dfff/set2 datasets/coco_dfff/internal \
  --counts 600 400 100 --seed 7 --overwrite
```

# Run

```
uv run scripts/train_1.py
```