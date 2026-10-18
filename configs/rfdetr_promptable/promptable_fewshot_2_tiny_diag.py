"""Config-only fixed-batch overfit approximation for the tiny same-image task."""

from pathlib import Path

cfg = dict(
    seed=73,
    train_image_dir=Path("datasets/mscoco/train2017"),
    train_annotation_file=Path("datasets/mscoco_prompt_split_tiny/train.json"),
    validation_image_dir=Path("datasets/mscoco/val2017"),
    validation_annotation_file=Path("datasets/mscoco_prompt_split_tiny/seen_val.json"),
    test_image_dir=Path("datasets/mscoco/val2017"),
    test_annotation_file=Path("datasets/mscoco_prompt_split_tiny/unseen_test.json"),
    # Each item selects one image/class pair with >=2 instances. One instance is
    # the box prompt; the other same-class instances are detection targets.
    # One loader batch contains the entire tiny training set for this run.
    episodes=4,
    max_shots=1,
    negative_ratio=0.0,
    negative_eval_per_class=0,
    steps=1000,
    batch_repeat=1000,
    eval_every=100,
    checkpoint_every_eval=True,
    max_eval_checkpoints=3,
    selection_metric="episodic_iou",
    coco_selection_images=100,
    batch_size=4,
    coco_eval_batch_size=4,
    coco_map_include_prompt_box=True,
    num_queries=20,
    resolution=280,
    lr=1e-4,
    lr_scheduler="constant",
    warmup_steps=0,
    warmup_start_factor=0.1,
    min_lr_ratio=0.01,
    train_sanity=True,
    train_sanity_samples=4,
    include_prompt_box_in_loss=True,
    evaluate_prompt_box_iou=True,
    inference_plot_mode="overlay",
    inference_extra_predictions=5,
    output=Path("logs/2_tiny_diag/model.pt"),
    figure_dir=Path("logs/2_tiny_diag"),
    device="cuda",
    log_every=10,
)
