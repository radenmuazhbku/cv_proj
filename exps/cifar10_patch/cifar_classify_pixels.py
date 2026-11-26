"""CIFAR-10 raw-pixel classification sanity check.

Flatten each float image (3 x 32 x 32, values in [0,1]) and train one
3072-to-10 linear layer, or an optional one-hidden-layer ReLU MLP. This is a
direct-pixel baseline for comparison with the K-means patch-centroid classifier.

Example:
    .venv/bin/python exps/cifar10_patch/cifar_classify_pixels.py --epochs 30
    .venv/bin/python exps/cifar10_patch/cifar_classify_pixels.py --classifier mlp --hidden-dim 512
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--classifier", choices=("linear", "mlp"), default="linear")
    parser.add_argument("--hidden-dim", type=int, default=1024, help="MLP hidden width")
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--test-subset", type=int, default=None)
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs/cifar_patch/pixels_linear"))
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_cifar(root, train, subset, seed):
    dataset = torchvision.datasets.CIFAR10(str(root), train=train, download=False, transform=T.ToTensor())
    indices = list(range(len(dataset)))
    if train:
        random.Random(seed).shuffle(indices)
    if subset is not None:
        indices = indices[:min(subset, len(indices))]
        dataset = Subset(dataset, indices)
    return dataset


@torch.no_grad()
def evaluate(model, loader, device, name):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    confusion = torch.zeros(10, 10, dtype=torch.long)
    for images, labels in tqdm(loader, desc=f"evaluate {name}", leave=False):
        x = images.to(device).reshape(len(images), -1)
        y = labels.to(device)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        pred = logits.argmax(dim=-1).cpu()
        correct += (pred == labels).sum().item()
        total += len(labels)
        for truth, guess in zip(labels.tolist(), pred.tolist()):
            confusion[truth, guess] += 1
    return {"accuracy": correct / max(total, 1), "nll": loss_sum / max(total, 1),
            "confusion_matrix": confusion.tolist()}


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("epochs, batch-size, and learning-rate must be positive")
    if args.hidden_dim < 1:
        raise ValueError("hidden-dim must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "data_root": str(args.data_root), "log_dir": str(args.log_dir)}
    (args.log_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    print(f"Device: {device}; input: flattened float RGB pixels (3072); classifier: {args.classifier}")

    train_set = load_cifar(args.data_root, True, args.train_subset, args.seed)
    test_set = load_cifar(args.data_root, False, args.test_subset, args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=device.type == "cuda")

    model = (nn.Linear(3072, 10) if args.classifier == "linear" else
             nn.Sequential(nn.Linear(3072, args.hidden_dim), nn.ReLU(), nn.Linear(args.hidden_dim, 10))).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        count = 0
        for images, labels in tqdm(train_loader, desc=f"train {epoch + 1}/{args.epochs}", leave=False):
            x = images.to(device).reshape(len(images), -1)
            y = labels.to(device)
            loss = F.cross_entropy(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            count += len(y)
        print(f"epoch {epoch + 1:03d}/{args.epochs} train_nll={loss_sum / max(count, 1):.4f}")

    results = {
        "config": config,
        "input_dim": 3 * 32 * 32,
        "train": evaluate(model, train_loader, device, "train"),
        "test": evaluate(model, test_loader, device, "test"),
    }
    torch.save({"model": model.state_dict(), "config": config}, args.log_dir / f"{args.classifier}_pixels.pt")
    (args.log_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({"train_accuracy": results["train"]["accuracy"],
                      "test_accuracy": results["test"]["accuracy"],
                      "train_nll": results["train"]["nll"], "test_nll": results["test"]["nll"]}, indent=2))
    print(f"Saved checkpoint and metrics to {args.log_dir.resolve()}")


if __name__ == "__main__":
    main()
