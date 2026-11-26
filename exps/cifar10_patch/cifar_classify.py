"""CIFAR-10 classification from position-wise K-means patch centroids.

Each image becomes a grid of patch codes. By default, a separate K-means
dictionary is fit for each grid position. The nearest-centroid index is looked
up in a learned embedding table (default width 128), producing a `(B,D,T)`
tensor for classification. A shared classifier maps each patch embedding to
10 class logits; logits are averaged over patches. `--use-position-embedding`
adds a learned absolute `(D,T)` table. The MLP option uses one ReLU hidden layer.

This is patch-level K-means: at patch size 16, K-means sees vectors of length
768 and learns visual patch prototypes. It is not whole-image K-means, which
would cluster one length-3072 vector per image into complete-image prototypes.
The independent baseline does not condition a patch code on neighboring codes.
A later causal joint model can factor p(z_1,...,z_P) as a raster-ordered
product of p(z_p | earlier allowed codes). An unordered, non-causal joint is
also possible as a pairwise/energy model, but exact normalized likelihood then
requires a generally expensive partition function; bidirectional context can
instead be used as a discriminative feature model without claiming that joint
likelihood.

Examples:
    .venv/bin/python exps/cifar10_patch/cifar_classify.py --patch-size 16 --k 128 --classifier linear
    .venv/bin/python exps/cifar10_patch/cifar_classify.py --patch-size 8 --k 256 --classifier mlp
    .venv/bin/python exps/cifar10_patch/cifar_classify.py --patch-size 16 --embedding-dim 128
    .venv/bin/python exps/cifar10_patch/cifar_classify.py --patch-size 16 --use-position-embedding
    .venv/bin/python exps/cifar10_patch/cifar_classify.py --shared-codebook --classifier mlp

This initial baseline assigns each patch independently. It does not model
neighbor or sequence dependencies between patch codes.
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
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm.auto import tqdm


CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--patch-size", type=int, choices=(1, 2, 4, 8, 16, 32), default=16)
    parser.add_argument("--k", type=int, default=128, help="centroids per position, or shared codebook size")
    parser.add_argument("--kmeans-samples", type=int, default=100000, help="maximum patches used to fit each codebook")
    parser.add_argument("--kmeans-iters", type=int, default=50)
    parser.add_argument("--position-specific", dest="position_specific", action="store_true", default=False,
                        help="fit a separate codebook at each patch position (default)")
    parser.add_argument("--shared-codebook", dest="position_specific", action="store_false",
                        help="fit one dictionary shared across spatial positions")
    parser.add_argument("--classifier", choices=("linear", "mlp"), default="linear")
    parser.add_argument("--hidden-dim", type=int, default=512, help="MLP hidden width")
    parser.add_argument("--embedding-dim", type=int, default=128, help="learned code embedding width")
    parser.add_argument("--use-position-embedding", action="store_true",
                        help="add learned absolute position embeddings (D,T) to patch embeddings")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=1,
                        help="evaluate the test split every N epochs (always evaluates the final epoch)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--test-subset", type=int, default=None)
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs/cifar_patch/classify"))
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_cifar(root, train, subset, seed=0):
    ds = torchvision.datasets.CIFAR10(str(root), train=train, download=False, transform=T.ToTensor())
    indices = list(range(len(ds)))
    if train:
        random.Random(seed).shuffle(indices)
    if subset is not None:
        indices = indices[:min(subset, len(indices))]
        ds = Subset(ds, indices)
    return ds


def load_images(dataset, name, num_workers):
    images, labels = [], []
    for image, label in tqdm(DataLoader(dataset, batch_size=512, shuffle=False, num_workers=num_workers), desc=f"load {name}"):
        images.append(image)
        labels.append(label)
    return torch.cat(images), torch.cat(labels).long()


def patchify(images, patch_size):
    """Return raster-ordered (N, P, D) patches, D=3*patch_size**2."""
    n, c, h, w = images.shape
    if h % patch_size or w % patch_size:
        raise ValueError(f"Image shape {h}x{w} must be divisible by patch size {patch_size}")
    grid = h // patch_size
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    return patches.permute(0, 2, 3, 1, 4, 5).contiguous().reshape(n, grid * grid, c * patch_size * patch_size)


def squared_distances(x, centers):
    return (x.square().sum(1, keepdim=True) + centers.square().sum(1)[None] - 2 * x @ centers.T).clamp_min(0)


@torch.no_grad()
def fit_kmeans(patches, k, n_iters, seed, position_name):
    """Lloyd K-means over a (samples, patch-vector-dim) tensor on device."""
    n = len(patches)
    if n == 0:
        raise ValueError("Cannot fit K-means on an empty patch pool")
    k = min(k, n)
    gen = torch.Generator(device=patches.device).manual_seed(seed)
    centers = patches[torch.randperm(n, generator=gen, device=patches.device)[:k]].clone()
    for _ in tqdm(range(n_iters), desc=f"K-means {position_name}", leave=False):
        distance_parts = []
        assignment_parts = []
        for start in range(0, n, 8192):
            x = patches[start:start + 8192]
            dist = squared_distances(x, centers)
            values, ids = dist.min(dim=1)
            assignment_parts.append(ids)
            distance_parts.append(values)
        assignments = torch.cat(assignment_parts)
        min_dist = torch.cat(distance_parts)
        sums = torch.zeros_like(centers).index_add(0, assignments, patches)
        counts = torch.zeros(k, device=patches.device).index_add(
            0, assignments, torch.ones(n, device=patches.device)
        )
        active = counts > 0
        means = sums / counts.clamp_min(1)[:, None]
        centers = torch.where(active[:, None], means, centers)
        empty_ids = (~active).nonzero().flatten()
        if len(empty_ids):
            farthest = min_dist.topk(len(empty_ids)).indices
            centers = centers.index_copy(0, empty_ids, patches[farthest])
    return centers

@torch.no_grad()
def fit_kmeans_inplace(patches, k, n_iters, seed, position_name):
    """Lloyd K-means over a (samples, patch-vector-dim) tensor on device."""
    n = len(patches)
    if n == 0:
        raise ValueError("Cannot fit K-means on an empty patch pool")
    k = min(k, n)
    gen = torch.Generator(device=patches.device).manual_seed(seed)
    centers = patches[torch.randperm(n, generator=gen, device=patches.device)[:k]].clone()
    for _ in tqdm(range(n_iters), desc=f"K-means {position_name}", leave=False):
        sums = torch.zeros_like(centers)
        counts = torch.zeros(k, device=patches.device)
        min_dist = torch.full((n,), float("inf"), device=patches.device)
        assignments = torch.empty(n, dtype=torch.long, device=patches.device)
        for start in range(0, n, 8192):
            x = patches[start:start + 8192]
            dist = squared_distances(x, centers)
            values, ids = dist.min(dim=1)
            assignments[start:start + len(x)] = ids
            min_dist[start:start + len(x)] = values
        sums.index_add_(0, assignments, patches)
        counts.index_add_(0, assignments, torch.ones(n, device=patches.device))
        active = counts > 0
        centers[active] = sums[active] / counts[active, None]
        if (~active).any():
            empty_ids = (~active).nonzero().flatten()
            farthest = min_dist.topk(len(empty_ids)).indices
            centers[empty_ids] = patches[farthest]
    return centers

@torch.no_grad()
def fit_codebooks(train_images, patch_size, k, sample_limit, n_iters, position_specific, device, seed):
    patches = patchify(train_images, patch_size)
    n, n_positions, dim = patches.shape
    if position_specific:
        books = []
        for pos in range(n_positions):
            pool = patches[:, pos].to(device)
            if len(pool) > sample_limit:
                gen = torch.Generator(device=device).manual_seed(seed + pos)
                pool = pool[torch.randperm(len(pool), generator=gen, device=device)[:sample_limit]]
            books.append(fit_kmeans(pool, k, n_iters, seed + pos, f"position {pos}/{n_positions}"))
        return books
    pool = patches.reshape(n * n_positions, dim).to(device)
    if len(pool) > sample_limit:
        gen = torch.Generator(device=device).manual_seed(seed)
        pool = pool[torch.randperm(len(pool), generator=gen, device=device)[:sample_limit]]
    book = fit_kmeans(pool, k, n_iters, seed, "shared")
    return [book]


@torch.no_grad()
def encode_patches(images, codebooks, patch_size, position_specific, device, name):
    patches = patchify(images, patch_size)
    n, n_positions, dim = patches.shape
    codes = torch.empty((n, n_positions), dtype=torch.long)
    for start in tqdm(range(0, n, 512), desc=f"encode {name}"):
        batch = patches[start:start + 512].to(device)
        for pos in range(n_positions):
            book = codebooks[pos] if position_specific else codebooks[0]
            ids = squared_distances(batch[:, pos], book).argmin(1)
            codes[start:start + len(batch), pos] = ids.cpu()
    return codes


class PatchClassifier(nn.Module):
    """Trainable code lookup and shared patch classifier."""
    def __init__(self, codebook_sizes, n_positions, embedding_dim, classifier, hidden_dim,
                 position_specific, use_position_embedding=False):
        super().__init__()
        self.position_specific = position_specific
        table_sizes = codebook_sizes if position_specific else codebook_sizes[:1]
        self.code_embeddings = nn.ModuleList([nn.Embedding(k, embedding_dim) for k in table_sizes])
        self.use_position_embedding = use_position_embedding
        if use_position_embedding:
            # Absolute position table in channel-first (D,T) form.
            self.position_embedding = nn.Parameter(torch.empty(embedding_dim, n_positions))
            nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        if classifier == "linear":
            self.net = nn.Linear(embedding_dim, 10)
        else:
            self.net = nn.Sequential(nn.Linear(embedding_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 10))

    def forward(self, codes):
        # (B,T) code IDs -> learned embeddings arranged as (B,D,T).
        embedded = torch.stack([
            self.code_embeddings[pos if self.position_specific else 0](codes[:, pos])
            for pos in range(codes.shape[1])
        ], dim=-1)
        if self.use_position_embedding:
            embedded = embedded + self.position_embedding.unsqueeze(0)
        # Classifier is shared over T after converting to (B,T,D).
        return self.net(embedded.transpose(1, 2)).mean(dim=1)


def evaluate(classifier, codes, labels, device, batch_size, name):
    classifier.eval()
    correct = total = 0
    confusion = torch.zeros(10, 10, dtype=torch.long)
    losses = []
    loader = DataLoader(TensorDataset(codes, labels), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for cb, yb in tqdm(loader, desc=f"evaluate {name}", leave=False):
            ydev = yb.to(device)
            logits = classifier(cb.to(device))
            losses.append(F.cross_entropy(logits, ydev, reduction="sum").item())
            pred = logits.argmax(-1).cpu()
            correct += (pred == yb).sum().item()
            total += len(yb)
            for true, guess in zip(yb.tolist(), pred.tolist()):
                confusion[true, guess] += 1
    return {"accuracy": correct / max(total, 1), "nll": sum(losses) / max(total, 1), "confusion_matrix": confusion.tolist()}


def train_classifier(classifier, train_codes, train_labels, test_codes, test_labels, args, device):
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate)
    loader = DataLoader(TensorDataset(train_codes, train_labels), batch_size=args.batch_size, shuffle=True)
    eval_history = []
    for epoch in range(args.epochs):
        classifier.train()
        loss_total = 0.
        for cb, yb in tqdm(loader, desc=f"classifier epoch {epoch + 1}/{args.epochs}", leave=False):
            yb = yb.to(device)
            loss = F.cross_entropy(classifier(cb.to(device)), yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * len(yb)
        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            train_metrics = evaluate(classifier, train_codes, train_labels, device,
                                     args.batch_size, f"train epoch {epoch + 1}")
            test_metrics = evaluate(classifier, test_codes, test_labels, device,
                                    args.batch_size, f"test epoch {epoch + 1}")
            eval_history.append({"epoch": epoch + 1, "train": train_metrics, "test": test_metrics})
            print(f"epoch {epoch + 1:03d}/{args.epochs} "
                  f"train_loss={loss_total / len(train_codes):.4f} "
                  f"train_acc={train_metrics['accuracy']:.4f} train_nll={train_metrics['nll']:.4f} "
                  f"test_acc={test_metrics['accuracy']:.4f} test_nll={test_metrics['nll']:.4f}")
        else:
            print(f"epoch {epoch + 1:03d}/{args.epochs} train_loss={loss_total / len(train_codes):.4f}")
    return eval_history


def main():
    args = parse_args()
    if min(args.k, args.kmeans_samples, args.kmeans_iters, args.embedding_dim,
           args.hidden_dim, args.eval_every) < 1:
        raise ValueError("kmeans settings, dimensions, and eval-every must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    (args.log_dir / "config.json").write_text(json.dumps({**vars(args), "log_dir": str(args.log_dir), "data_root": str(args.data_root)}, indent=2, default=str))
    print(f"Device: {device}; patch={args.patch_size}; codebook={'per-position' if args.position_specific else 'shared'}; classifier={args.classifier}")

    train_ds = load_cifar(args.data_root, True, args.train_subset, args.seed)
    test_ds = load_cifar(args.data_root, False, args.test_subset, args.seed)
    train_images, train_labels = load_images(train_ds, "train", args.num_workers)
    test_images, test_labels = load_images(test_ds, "test", args.num_workers)
    codebooks = fit_codebooks(train_images, args.patch_size, args.k, args.kmeans_samples,
                              args.kmeans_iters, args.position_specific, device, args.seed)
    train_codes = encode_patches(train_images, codebooks, args.patch_size, args.position_specific, device, "train")
    test_codes = encode_patches(test_images, codebooks, args.patch_size, args.position_specific, device, "test")

    patch_dim = 3 * args.patch_size * args.patch_size
    n_positions = (32 // args.patch_size) ** 2
    codebook_sizes = [len(book) for book in codebooks]
    classifier = PatchClassifier(codebook_sizes, n_positions, args.embedding_dim, args.classifier,
                                 args.hidden_dim, args.position_specific,
                                 args.use_position_embedding).to(device)
    eval_history = train_classifier(classifier, train_codes, train_labels,
                                    test_codes, test_labels, args, device)
    results = {
        "config": {**vars(args), "data_root": str(args.data_root), "log_dir": str(args.log_dir)},
        "patch_count": n_positions,
        "centroid_dim": patch_dim,
        "codebook_sizes": codebook_sizes,
        "classifier_embedding_dim": args.embedding_dim,
        "classifier_input_shape": ["B", args.embedding_dim, n_positions],
        "classifier_pooling": "mean patch logits",
        "test_eval_history": eval_history,
        "train": evaluate(classifier, train_codes, train_labels, device, args.batch_size, "train"),
        "test": evaluate(classifier, test_codes, test_labels, device, args.batch_size, "test"),
    }
    book_tensor = torch.stack(codebooks)
    checkpoint = {
        "codebooks": book_tensor.cpu(),
        "classifier": classifier.state_dict(),
        "config": results["config"],
        "patch_count": n_positions,
        "centroid_dim": patch_dim,
        "embedding_dim": args.embedding_dim,
    }
    torch.save(checkpoint, args.log_dir / "checkpoint.pt")
    torch.save({"train_codes": train_codes, "train_labels": train_labels,
                "test_codes": test_codes, "test_labels": test_labels}, args.log_dir / "patch_codes.pt")
    (args.log_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({"train_accuracy": results["train"]["accuracy"],
                      "test_accuracy": results["test"]["accuracy"],
                      "train_nll": results["train"]["nll"], "test_nll": results["test"]["nll"]}, indent=2))
    print(f"Saved checkpoint and metrics to {args.log_dir.resolve()}")


if __name__ == "__main__":
    main()
