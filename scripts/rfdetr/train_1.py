from rfdetr import RFDETRNano

model = RFDETRNano()

model.train(
    dataset_dir="datasets/rfdetr_dfff",
    epochs=100,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir="logs",
)

'''
/bin/bash -c mkdir -p logs/comparisons
.venv/bin/python scripts/generate_gt_pred_comparisons.py --images-dir datasets/rfdetr_dfff/train --annotations datasets/rfdetr_dfff/train/_annotations.coco.json --output-dir logs/comparisons/train --count 10000 --batch-size 4 > logs/comparisons/train.log 2>&1 && .venv/bin/python scripts/generate_gt_pred_comparisons.py --images-dir datasets/rfdetr_dfff/valid --annotations datasets/rfdetr_dfff/valid/_annotations.coco.json --output-dir logs/comparisons/valid --count 10000 --batch-size 4 > logs/comparisons/valid.log 2>&1 && .venv/bin/python scripts/generate_gt_pred_comparisons.py --images-dir datasets/rfdetr_dfff/test --annotations datasets/rfdetr_dfff/test/_annotations.coco.json --output-dir logs/comparisons/test --count 10000 --batch-size 4 > logs/comparisons/test.log 2>&1
.venv/bin/python scripts/generate_gt_pred_comparisons.py --images-dir datasets/rfdetr_dfff/test --annotations datasets/rfdetr_dfff/test/_annotations.coco.json --output-dir logs/comparisons/test --count 10000 --batch-size 4

/bin/bash -c mkdir -p logs/comparisons
.venv/bin/python scripts/generate_gt_pred_comparisons.py --images-dir datasets/rfdetr_dfff/train --annotations datasets/rfdetr_dfff/train/_annotations.coco.json --output-dir logs/comparisons/train --count 10000 --batch-size 4 > logs/comparisons/train.log 2>&1 && .venv/bin/python scripts/generate_gt_pred_comparisons.py --images-dir datasets/rfdetr_dfff/valid --annotations datasets/rfdetr_dfff/valid/_annotations.coco.json --output-dir logs/comparisons/valid --count 10000 --batch-size 4 > logs/comparisons/valid.log 2>&1 && .venv/bin/python scripts/generate_gt_pred_comparisons.py --images-dir datasets/rfdetr_dfff/test --annotations datasets/rfdetr_dfff/test/_annotations.coco.json --output-dir logs/comparisons/test --count 10000 --batch-size 4 > logs/comparisons/test.log 2>&1
.venv/bin/python scripts/generate_gt_pred_comparisons.py --images-dir datasets/rfdetr_dfff/test --annotations datasets/rfdetr_dfff/test/_annotations.coco.json --output-dir logs/comparisons/test --count 10000 --batch-size 4

'''