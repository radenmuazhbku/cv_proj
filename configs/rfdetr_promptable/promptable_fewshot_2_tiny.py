"""Same-image box-prompt pilot for the tiny COCO class split."""

from pathlib import Path

cfg = dict(
    seed=7,
    train_image_dir=Path("datasets/mscoco/train2017"),
    train_annotation_file=Path("datasets/mscoco_prompt_split_tiny/train.json"),
    validation_image_dir=Path("datasets/mscoco/val2017"),
    validation_annotation_file=Path("datasets/mscoco_prompt_split_tiny/seen_val.json"),
    test_image_dir=Path("datasets/mscoco/val2017"),
    test_annotation_file=Path("datasets/mscoco_prompt_split_tiny/unseen_test.json"),
    # Each item selects one image/class pair with >=2 instances. One instance is
    # the box prompt; the other same-class instances are detection targets.
    episodes=1000,
    max_shots=1,
    negative_ratio=0.0,
    negative_eval_per_class=0,
    steps=50_000,
    batch_repeat=10,
    eval_every=1000,
    checkpoint_every_eval=True,
    max_eval_checkpoints=3,
    selection_metric="episodic_iou",
    coco_selection_images=100,
    batch_size=4,
    coco_eval_batch_size=4,
    num_queries=100,
    resolution=560,
    lr=5e-4,
    lr_scheduler="cosine",
    warmup_steps=0.05,
    warmup_start_factor=0.1,
    min_lr_ratio=0.01,
    train_sanity=True,
    train_sanity_samples=8,
    inference_plot_mode="overlay",
    inference_max_predictions=0,
    output=Path("logs/2_tiny/model.pt"),
    figure_dir=Path("logs/2_tiny"),
    device="cuda",
    log_every=100,
)
