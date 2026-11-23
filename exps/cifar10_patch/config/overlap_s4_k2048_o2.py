"""Settings matching run_overlap_baseline_ablation.py's overlap=2 run."""
CONFIG = {
    "run_name": "L1_s4_K2048_overlap_o2",
    "mode": "train",
    "trainer": "em",
    "levels": 1,
    "patch_sizes": [4],
    "codebook_sizes": [2048],
    "overlap": 2,
    "discrete": False,
    "train_subset": 100,
    "test_subset": 100,
    "epochs": 5,
    "batch_size": 100,
    "umap_enabled": False,
}
