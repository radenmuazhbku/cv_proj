"""
Configuration for emb_learner.py - CIFAR10 training
Config 1: Moderate settings for reasonable training speed and accuracy

Run: python exps/emb_learner.py --config exps/configs/config_1.py
"""

# Training hyperparameters
epochs = 100
batch_size = 128
lr = 0.01
momentum = 0.9
weight_decay = 5e-4

# Model architecture
num_classes = 10
hidden_size = 512

# Data loading
data_dir = "./data"
num_workers = 4
pin_memory = True

# Logging and checkpointing
log_interval = 20
log_dir = "./logs"
checkpoint_dir = "./checkpoints"
checkpoint_name = "config_1_last.ckpt"

# Device (None = auto-detect CUDA)
device = None
