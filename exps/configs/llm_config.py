"""
Configuration for llm_zero_learner.py
Example config with reasonable defaults for quick testing

Run: python exps/llm_zero_learner.py --config exps/configs/llm_config.py
"""

# LLM
model_name = "HuggingFaceTB/SmolLM-135M"
device = "auto"

# Patchifier
patch_size = 16
embed_mode = "float"
# proj_dim = None  # Auto to LLM hidden dim

# Dataset subset
full_dataset = False
train_subset = 5  # 5 images per class
test_subset = 1   # 1 image per class
subset_ordered = True
subset_seed = None

# KV Cache
# disable_kv_sink = False  # KV sink enabled by default

# Answer extraction
answer_mode = "first-token"  # or "ar" or "probs"
top_p = 0.9
max_tokens = 20

# Logging & Output
log_dir = "./logs"
log_level = "INFO"
save_images = True
# images_dir = None  # Default: <log-dir>/images

# Data
data_dir = "./data"
