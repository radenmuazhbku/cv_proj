from rfdetr import RFDETRBase
# from rfdetr import RFDETRBase

model = RFDETRBase()

model.train(
    # dataset_dir="datasets/split_H1H2",
    dataset_dir="datasets/scalebar_H1H2",
    epochs=5,
    # epochs=10,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir="logs/train_fs_1",
)

'''
/bin/bash -c mkdir -p logs/comparisons
.venv/bin/python scripts/rfdetr/generate_gt_pred_comparisons.py --checkpoint logs/train_fs_1/checkpoint_best_ema.pth --images-dir datasets/split_H1H2/train --annotations datasets/split_H1H2/train/_annotations.coco.json --output-dir logs/comparisons/train --count 100 --batch-size 4 > logs/comparisons/train.log 2>&1 
.venv/bin/python scripts/rfdetr/generate_gt_pred_comparisons.py --checkpoint logs/train_fs_1/checkpoint_best_ema.pth --images-dir datasets/split_H1H2/valid --annotations datasets/split_H1H2/valid/_annotations.coco.json --output-dir logs/comparisons/valid --count 100 --batch-size 4 > logs/comparisons/valid.log 2>&1 
.venv/bin/python scripts/rfdetr/generate_gt_pred_comparisons.py --checkpoint logs/train_fs_1/checkpoint_best_ema.pth --images-dir datasets/split_H1H2/test --annotations datasets/split_H1H2/test/_annotations.coco.json --output-dir logs/comparisons/test --count 100 --batch-size 4 > logs/comparisons/test.log 2>&1
'''

'''
.venv/bin/python scripts/rfdetr/generate_gt_pred_comparisons.py --checkpoint logs/train_fs_1/checkpoint_best_ema.pth --images-dir datasets/scalebar_H1H2/train --annotations datasets/scalebar_H1H2/train/_annotations.coco.json --output-dir logs/compare_scalebar/train --count 100 --batch-size 4 > logs/comparisons/train.log 2>&1 

'''