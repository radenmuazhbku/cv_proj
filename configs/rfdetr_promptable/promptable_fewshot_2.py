from pathlib import Path

cfg = dict(
seed = 7,
train_image_dir = Path("datasets/mscoco/train2017"),
train_annotation_file = Path("datasets/mscoco_prompt_split/train.json"),
validation_image_dir = Path("datasets/mscoco/val2017"),
validation_support_image_dir = Path("datasets/mscoco/val2017"),
validation_annotation_file = Path("datasets/mscoco_prompt_split/seen_val.json"),
validation_support_annotation_file = Path("datasets/mscoco_prompt_split/seen_support.json"),
test_image_dir = Path("datasets/mscoco/val2017"),
test_support_image_dir = Path("datasets/mscoco/val2017"),
test_annotation_file = Path("datasets/mscoco_prompt_split/unseen_test.json"),
test_support_annotation_file = Path("datasets/mscoco_prompt_split/unseen_support.json"),
episodes = 200_000,
max_shots = 5,
negative_ratio = 1.5,
negative_eval_per_class = 100,
steps = 500_000,
eval_every = 100_000,
checkpoint_every_eval = True,
max_eval_checkpoints = 3,
selection_metric = "coco_map",
coco_selection_images = 500,
batch_size = 4,
coco_eval_batch_size = 4,
num_queries = 100,
resolution = 560,
lr = 5e-4,
lr_scheduler = "cosine",
warmup_steps = 0.05,  # fraction of total steps; integer values are absolute steps
warmup_start_factor = 0.1,
min_lr_ratio = 0.01,
train_sanity = True,
train_sanity_samples = 8,
output = Path("logs/promptable_fewshot_2.pt"),
figure_dir = Path("logs/promptable_fewshot_figures/2"),
device = "cuda",
log_every = 1000,
)



