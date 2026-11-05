"""
Embedding Learner for CIFAR10
Simple ConvNet training with PyTorch, torchvision, tqdm, and argparse config.

Usage:
    # Train with default arguments
    python exps/emb_learner.py

    # Train with CLI arguments only
    python exps/emb_learner.py --batch-size 256 --epochs 50 --lr 0.01 --data-dir ./my_data

    # Train with config file
    python exps/emb_learner.py --config exps/configs/config_1.py

    # Train with config file + CLI override
    python exps/emb_learner.py --config exps/configs/config_1.py --batch-size 256 --epochs 10

    # Resume from checkpoint (add --checkpoint-path to load, then continue training)
    python exps/emb_learner.py --config exps/configs/config_1.py --checkpoint-path ./checkpoints/config_1_last.ckpt
"""

import argparse
import ast
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm


# ─── Setup ────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a simple ConvNet on CIFAR10"
    )

    # Training
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)

    # Model
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--hidden-size", type=int, default=512)

    # Data
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true", default=True)

    # Logging & Checkpointing
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--checkpoint-name", type=str, default="last.ckpt")

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to Python config file to update args (e.g., config.py)",
    )

    # Device
    parser.add_argument("--device", type=str, default=None)

    # Parse known args first (to get config before merging)
    args, unknown = parser.parse_known_args()

    # Load config file if provided
    if args.config and os.path.exists(args.config):
        _load_config_file(args.config, parser)

    # Parse again with all defaults now potentially updated by config
    args = parser.parse_args()

    # Post-process: resolve device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args


def _load_config_file(config_path, parser):
    """Load a Python config file and update the parser's defaults."""
    config_globals = {"__builtins__": {}}
    config_locals = {}

    config_path = Path(config_path).resolve()

    # Read and execute the config file
    with open(config_path, "r") as f:
        config_code = f.read()

    # Parse the config as an AST to extract top-level assignments
    tree = ast.parse(config_code, filename=str(config_path))

    config_dict = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    key = target.id
                    try:
                        # Evaluate the value expression
                        value_node = ast.Expression(node.value)
                        code = compile(value_node, str(config_path), "eval")
                        value = eval(code, {"__builtins__": {}}, {})
                        config_dict[key] = value
                    except Exception as e:
                        print(f"Warning: Could not evaluate config key '{key}': {e}")

    # Update parser defaults
    for key, value in config_dict.items():
        # Check if the parser has this argument
        for action in parser._actions:
            if "--" + key.replace("_", "-") in action.option_strings:
                action.default = value
                parser._defaults[action.dest] = value
                break


# ─── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(log_dir, name="emb_learner"):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"{name}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return log_file


# ─── Model ─────────────────────────────────────────────────────────────────────

class SimpleConvNet(nn.Module):
    """Simple ConvNet for CIFAR10 (32x32 RGB images)."""

    def __init__(self, num_classes=10, hidden_size=512):
        super().__init__()
        self.features = nn.Sequential(
            # 32x32
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16
            # 16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8
            # 8x8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 4x4
            # 4x4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 2x2
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 2 * 2, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ─── Data ─────────────────────────────────────────────────────────────────────

def get_cifar10_dataloaders(data_dir, batch_size, num_workers, pin_memory):
    """CIFAR10 DataLoaders with standard augmentation."""
    data_dir = Path(data_dir)

    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        ),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        ),
    ])

    # Datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, test_loader


# ─── Training ──────────────────────────────────────────────────────────────────

def train_one_epoch(
    model, train_loader, optimizer, criterion, device, epoch, log_interval
):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}", leave=False)
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if batch_idx % log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            accuracy = 100.0 * correct / total
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "acc": f"{accuracy:.2f}%",
            })

    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(test_loader, desc="Evaluating", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            "loss": f"{total_loss / (batch_idx + 1) if 'batch_idx' in locals() else total_loss:.4f}",
        })

    avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ─── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(
    model, optimizer, epoch, loss, accuracy, checkpoint_dir, checkpoint_name
):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / checkpoint_name
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "accuracy": accuracy,
    }
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"Checkpoint saved to {checkpoint_path}")
    return checkpoint_path


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    setup_logging(args.log_dir)

    logging.info("Parsed arguments:")
    for key, value in vars(args).items():
        logging.info(f"  {key}: {value}")

    # Device
    device = torch.device(args.device)
    logging.info(f"Using device: {device}")

    # DataLoaders
    logging.info("Loading CIFAR10 data...")
    train_loader, test_loader = get_cifar10_dataloaders(
        args.data_dir,
        args.batch_size,
        args.num_workers,
        args.pin_memory,
    )
    logging.info(
        f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}"
    )

    # Model
    model = SimpleConvNet(
        num_classes=args.num_classes,
        hidden_size=args.hidden_size,
    )
    model = model.to(device)
    logging.info(f"Model: {model.__class__.__name__}")

    # Optimizer & Criterion
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    logging.info(f"Optimizer: {optimizer.__class__.__name__}, lr={args.lr}")

    # Training loop
    best_test_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch, args.log_interval
        )

        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )

        logging.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train: loss={train_loss:.4f}, acc={train_acc:.2f}% | "
            f"Test: loss={test_loss:.4f}, acc={test_acc:.2f}%"
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            logging.info(f"New best test accuracy: {best_test_acc:.2f}%")

        # Save checkpoint at last epoch
        if epoch == args.epochs:
            save_checkpoint(
                model, optimizer, epoch, test_loss, test_acc,
                args.checkpoint_dir, args.checkpoint_name
            )

    logging.info(f"Training complete. Best test accuracy: {best_test_acc:.2f}%")


if __name__ == "__main__":
    main()
