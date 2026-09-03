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
a source directory and a new target directory. It recursively finds images in
the source directory and reads `ObjectDetection.xlsx` by default.

```bash
uv run python scripts/convert_excel_to_coco.py <source-folder> <target-folder>
```

For the DFFF data linked into this repository:

```bash
uv run python scripts/convert_excel_to_coco.py datasets/dfff coco_dfff
```

Use `--excel` when the annotation workbook has another name or location:

```bash
uv run python scripts/convert_excel_to_coco.py <source-folder> <target-folder> \
  --excel path/to/annotations.xlsx
```

The target directory must not already exist. The converter validates that every
annotated filename resolves to exactly one image and that every bounding box
lies inside its image.

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
├── annotations/
│   └── instances.json
└── images/
    ├── 1.png
    └── ...
```

`instances.json` contains standard COCO `images`, `annotations`, and
`categories` arrays. Images are copied into the output directory so the result
is self-contained.

The DFFF conversion generated in this workspace contains 1,131 images, 9,433
annotations, and 9 categories.
