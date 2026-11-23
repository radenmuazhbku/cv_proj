"""Discrete/continuous patch-code model for CIFAR-10. Run from repository root.

Reproduce the two referenced legacy HMP figures with matching hard, continuous,
single-level EM settings (the overlap case used five iterations in its script):

    .venv/bin/python exps/cifar10_patch/train.py --config exps/cifar10_patch/config/overfit_s4_k2048.py
    .venv/bin/python exps/cifar10_patch/train.py --config exps/cifar10_patch/config/overlap_s4_k2048_o2.py

Both runs use 100 train and 100 test images, patch=4, K=2048, hard assignments,
and continuous means. They save their reconstruction grids under logs/cifar_patch.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.utils as vutils
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = {
    "run_name": "cifar_patch", "mode": "train", "trainer": "gradient",
    "data_root": "datasets", "log_root": "logs/cifar_patch", "checkpoint": None,
    "levels": 1, "patch_sizes": [8], "codebook_sizes": [2048], "overlap": 0,
    "discrete": True, "train_subset": None, "test_subset": None,
    "epochs": 10, "batch_size": 128, "learning_rate": 0.1, "temperature": 0.1,
    "project_every": 2, "num_workers": 4, "device": "auto", "seed": 0,
    "samples": 64, "class_id": None, "umap_enabled": True,
    "umap_points": 1000, "umap_neighbors": 15, "umap_patch_scale": 1.5,
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, help="Python config file defining CONFIG = {...}")
    for key, default in DEFAULTS.items():
        if key == "config":
            continue
        flag = "--" + key.replace("_", "-")
        typ = type(default) if default is not None else str
        if isinstance(default, bool):
            p.add_argument(flag, action=argparse.BooleanOptionalAction, default=None)
        elif isinstance(default, list):
            p.add_argument(flag, nargs="+", type=int, default=None)
        else:
            p.add_argument(flag, type=typ, default=None)
    known, _ = p.parse_known_args()
    cfg = dict(DEFAULTS)
    if known.config:
        spec = importlib.util.spec_from_file_location("cifar_patch_user_config", known.config)
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot load config: {known.config}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loaded = getattr(module, "CONFIG", None)
        if not isinstance(loaded, dict):
            raise ValueError("Python config must define CONFIG as a dict")
        unknown = set(loaded) - set(cfg)
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        cfg.update(loaded)
        print(f"Loaded config: {known.config.resolve()}")
    cli, _ = p.parse_known_args()
    cli_values = vars(cli)
    for key in cfg:
        if cli_values.get(key) is not None:
            cfg[key] = cli_values[key]
    if cfg["levels"] not in (1, 2):
        raise ValueError("levels must be 1 or 2")
    if len(cfg["codebook_sizes"]) != cfg["levels"]:
        raise ValueError("codebook_sizes must contain one K per level")
    if len(cfg["patch_sizes"]) != cfg["levels"]:
        raise ValueError("patch_sizes must contain one patch size per level")
    if any(size not in (8, 4) for size in cfg["patch_sizes"]):
        raise ValueError("patch_sizes currently supports 8 and 4")
    if cfg["levels"] == 2 and cfg["patch_sizes"] != [8, 4]:
        raise ValueError("two levels must use patch_sizes [8, 4]")
    return cfg


def load_data(root, train, subset):
    ds = torchvision.datasets.CIFAR10(str(root), train=train, download=False, transform=T.ToTensor())
    if subset is not None:
        indices = list(range(min(int(subset), len(ds))))
        ds = torch.utils.data.Subset(ds, indices)
    return ds


def patchify(x, size, overlap=0):
    n, c, h, w = x.shape
    g = h // size
    if overlap:
        ext = size + 2*overlap
        x = F.pad(x, (overlap, overlap, overlap, overlap), mode="reflect")
        return x.unfold(2, ext, size).unfold(3, ext, size).permute(0, 2, 3, 1, 4, 5).contiguous().reshape(n, g*g, c*ext*ext)
    return x.reshape(n, c, g, size, g, size).permute(0, 2, 4, 1, 3, 5).reshape(n, g*g, c*size*size)


def unpatchify(patches, size, overlap=0):
    n, p, d = patches.shape
    g = 32 // size
    if overlap:
        ext = size + 2*overlap
        patches = patches.reshape(n, g*g, 3, ext, ext)[..., overlap:overlap+size, overlap:overlap+size].reshape(n, g*g, 3*size*size)
    return patches.reshape(n, g, g, 3, size, size).permute(0, 3, 1, 4, 2, 5).reshape(n, 3, 32, 32)


def pairwise(x, centers):
    # Squared distances without materializing an N x K x D tensor.
    return (x.square().sum(1, keepdim=True) + centers.square().sum(1)[None] - 2*x @ centers.T).clamp_min_(0)


class PatchModel(nn.Module):
    def __init__(self, levels, sizes, overlap=0, classes=10):
        super().__init__()
        self.levels, self.sizes, self.overlap, self.classes = levels, sizes, overlap, classes
        self.codebooks = nn.ParameterList()
        self.priors = nn.ParameterList()
        for i, (size, k) in enumerate(zip(sizes, levels)):
            ext = size + 2*overlap
            self.codebooks.append(nn.Parameter(torch.rand(k, 3*ext*ext)))
            pos = (32 // size) ** 2
            self.priors.append(nn.Parameter(torch.zeros(classes, pos, k)))
        self.parent = nn.Parameter(torch.zeros(levels[0], levels[1])) if len(levels) == 2 else None

    @torch.no_grad()
    def encode(self, x):
        codes = []
        for size, centers in zip(self.sizes, self.codebooks):
            patches = patchify(x, size, self.overlap).reshape(-1, centers.shape[1])
            z = pairwise(patches, centers).argmin(1).reshape(x.shape[0], -1)
            codes.append(z)
        return codes

    def code_logits(self, level, labels, parent_codes=None):
        # labels: B; return B x positions x K. Fine-level factorization adds a
        # shared parent->child transition matrix to class/position logits.
        logits = self.priors[level][labels]
        if level == 1:
            g = 32 // self.sizes[level]
            pg = 32 // self.sizes[0]
            mapping = torch.arange(g*g, device=labels.device)
            parent_idx = (mapping // g * self.sizes[level] // self.sizes[0]) * pg + (mapping % g * self.sizes[level] // self.sizes[0])
            logits = logits + self.parent[parent_codes[:, parent_idx]]
        return logits

    @torch.no_grad()
    def reconstruct(self, x):
        codes = self.encode(x)
        # Fine patches tile the image; coarse level is a predictive latent.
        z = codes[-1]
        return unpatchify(self.codebooks[-1][z], self.sizes[-1], self.overlap).clamp(0, 1)

    @torch.no_grad()
    def sample(self, n, device, class_id=None, temperature=1.0):
        labels = (torch.full((n,), class_id, device=device, dtype=torch.long) if class_id is not None
                  else torch.randint(self.classes, (n,), device=device))
        zs = []
        for i, (size, cb) in enumerate(zip(self.sizes, self.codebooks)):
            logits = self.code_logits(i, labels, zs[0] if i == 1 else None)
            probs = (logits / max(temperature, 1e-5)).softmax(-1)
            z = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).reshape(n, -1)
            zs.append(z)
        return unpatchify(self.codebooks[-1][zs[-1]], self.sizes[-1], self.overlap).clamp(0, 1)


def project(model, pool):
    with torch.no_grad():
        for i, size in enumerate(model.sizes):
            candidates = patchify(pool, size, model.overlap).reshape(-1, model.codebooks[i].shape[1])
            centers = model.codebooks[i]
            snapped = []
            for start in range(0, len(centers), 128):
                ids = pairwise(centers[start:start+128], candidates).argmin(1)
                snapped.append(candidates[ids])
            centers.copy_(torch.cat(snapped))


def init_codebooks(model, train_images):
    with torch.no_grad():
        for i, size in enumerate(model.sizes):
            pool = patchify(train_images, size, model.overlap).reshape(-1, model.codebooks[i].shape[1])
            k = model.levels[i]
            ids = torch.randperm(len(pool), device=pool.device)[:k] if k <= len(pool) else torch.arange(k, device=pool.device) % len(pool)
            model.codebooks[i].copy_(pool[ids])


def fit_gradient(model, train_loader, train_images, cfg, device, run_dir):
    opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    history = []
    for epoch in range(cfg["epochs"]):
        model.train(); sums = {"loss": 0., "recon": 0., "prior": 0.}; count = 0
        for x, y in tqdm(train_loader, desc=f"train {epoch+1}/{cfg['epochs']}", leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = x.new_zeros(())
            rec_loss = x.new_zeros(()); prior_loss = x.new_zeros(())
            parents = None
            for i, (size, centers) in enumerate(zip(model.sizes, model.codebooks)):
                patches = patchify(x, size, model.overlap)
                flat = patches.reshape(-1, centers.shape[-1])
                d = pairwise(flat, centers).reshape(x.shape[0], -1, centers.shape[0])
                with torch.no_grad():
                    z = d.argmin(-1)
                # Differentiable soft exemplar reconstruction objective.
                temp = cfg["temperature"]
                soft = (-d / max(temp, 1e-6)).softmax(-1)
                soft_recon = soft @ centers
                level_rec = F.mse_loss(soft_recon, patches)
                logits = model.code_logits(i, y, parents)
                level_prior = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), z.reshape(-1))
                loss = loss + level_rec + 0.1*level_prior
                rec_loss += level_rec.detach(); prior_loss += level_prior.detach()
                parents = z
            loss.backward(); opt.step()
            b = len(x); count += b
            sums["loss"] += loss.item()*b; sums["recon"] += rec_loss.item()*b; sums["prior"] += prior_loss.item()*b
        if cfg["discrete"] and cfg["project_every"] and (epoch + 1) % cfg["project_every"] == 0:
            project(model, train_images)
        row = {k: v/max(count, 1) for k, v in sums.items()}; row["epoch"] = epoch + 1
        history.append(row)
        print(f"epoch {epoch+1:03d}/{cfg['epochs']} loss={row['loss']:.6f} recon={row['recon']:.6f} prior={row['prior']:.4f}")
        with (run_dir/"train.jsonl").open("a") as f: f.write(json.dumps(row)+"\n")
    if cfg["discrete"]:
        project(model, train_images)
    return history


@torch.no_grad()
def fit_em(model, train_images, train_labels, cfg):
    """Layerwise hard-EM/K-means codebook fit, then categorical prior counts."""
    for i, size in enumerate(model.sizes):
        x = patchify(train_images, size, model.overlap).reshape(-1, model.codebooks[i].shape[1])
        cb = model.codebooks[i]
        for it in tqdm(range(cfg["epochs"]), desc=f"EM level {i+1}"):
            sums = torch.zeros_like(cb); counts = torch.zeros(len(cb), device=x.device)
            for start in range(0, len(x), 4096):
                xb = x[start:start+4096]
                z = pairwise(xb, cb).argmin(1)
                sums.index_add_(0, z, xb); counts.index_add_(0, z, torch.ones_like(z, dtype=x.dtype))
            used = counts > 0
            cb[used] = sums[used] / counts[used, None]
            if cfg["discrete"]:
                project(model, train_images)
        # Initialize categorical position/class tables from hard assignments.
        z = model.encode(train_images)[i]
        table = model.priors[i]
        table.fill_(-8.)
        for c in range(10):
            mask = train_labels == c
            if mask.any():
                counts = torch.zeros_like(table[c]).add_(1.)
                for pos in range(z.shape[1]):
                    counts[pos].scatter_add_(0, z[mask, pos], torch.ones_like(z[mask, pos], dtype=counts.dtype))
                table[c].copy_(counts.log())
    if model.parent is not None:
        z0, z1 = model.encode(train_images)
        model.parent.fill_(-8.)
        counts = torch.ones_like(model.parent)
        for pos in range(z1.shape[1]):
            pidx = ((pos // 8) // 2) * 4 + ((pos % 8) // 2)
            counts.index_put_((z0[:, pidx], z1[:, pos]), torch.ones(len(z0), device=z0.device), accumulate=True)
        model.parent.copy_(counts.log())


@torch.no_grad()
def evaluate(model, loader, device):
    total_mse = total_nll = total_class = 0.; n = 0; conf = torch.zeros(10, 10, dtype=torch.long)
    for x, y in tqdm(loader, desc="eval", leave=False):
        x, y = x.to(device), y.to(device); codes = model.encode(x); recon = model.reconstruct(x)
        total_mse += F.mse_loss(recon, x, reduction="sum").item() / (3*32*32)
        n += len(x)
        scores = torch.zeros(len(x), 10, device=device)
        for c in range(10):
            cls = torch.full_like(y, c)
            for i, z in enumerate(codes):
                logits = model.code_logits(i, cls, codes[0] if i == 1 else None).log_softmax(-1)
                scores[:, c] += logits.gather(-1, z.unsqueeze(-1)).squeeze(-1).sum(-1)
        true_logp = scores.log_softmax(-1).gather(1, y[:, None]).sum().item()
        total_nll -= true_logp
        preds = scores.argmax(-1)
        total_class += (preds == y).sum().item()
        for a, b in zip(y.cpu().tolist(), preds.cpu().tolist()): conf[a, b] += 1
    return {"mse": total_mse/n, "psnr": 10*math.log10(1/max(total_mse/n, 1e-12)),
            "class_nll_nats_per_image": total_nll/n, "accuracy": total_class/n,
            "confusion_matrix": conf.tolist()}


@torch.no_grad()
def code_nll(model, loader, device, conditional=True):
    total = count = 0
    for x, y in tqdm(loader, desc="code NLL", leave=False):
        x, y = x.to(device), y.to(device); codes = model.encode(x)
        count += sum(z.numel() for z in codes)
        if conditional:
            seq_logp = torch.zeros(len(x), device=device)
            for i, z in enumerate(codes):
                lp = model.code_logits(i, y, codes[0] if i == 1 else None).log_softmax(-1)
                seq_logp += lp.gather(-1, z[..., None]).squeeze(-1).sum(-1)
            total -= seq_logp.sum().item()
        else:
            # Class is a shared latent for the whole image: marginalize after
            # summing all level/site log probabilities, not independently per site.
            class_seq_logp = []
            for c in range(10):
                labels = torch.full_like(y, c)
                seq_logp = torch.zeros(len(x), device=device)
                for i, z in enumerate(codes):
                    lp = model.code_logits(i, labels, codes[0] if i == 1 else None).log_softmax(-1)
                    seq_logp += lp.gather(-1, z[..., None]).squeeze(-1).sum(-1)
                class_seq_logp.append(seq_logp)
            total -= (torch.logsumexp(torch.stack(class_seq_logp), dim=0) - math.log(10)).sum().item()
    return total/max(count, 1)


def save_grids(model, train_images, test_images, run_dir, cfg, device):
    for split, images in (("train", train_images), ("test", test_images)):
        n = min(16, len(images)); originals = images[:n].to(device)
        recon = model.reconstruct(originals)
        vutils.save_image(torch.cat([originals, recon]), run_dir/f"{split}_reconstruction.png", nrow=n, padding=2)
    for c in range(10):
        samples = model.sample(8, device, class_id=c)
        vutils.save_image(samples, run_dir/f"samples_{c}.png", nrow=8, padding=2)
    vutils.save_image(model.sample(cfg["samples"], device, class_id=cfg["class_id"]), run_dir/"samples.png", nrow=8, padding=2)


@torch.no_grad()
def save_umap(model, train_images, test_images, run_dir, cfg, device):
    try:
        import umap
        from PIL import Image, ImageDraw
    except ImportError as e:
        print(f"UMAP visualization skipped (install umap-learn): {e}"); return
    n = min(cfg["umap_points"], len(train_images)+len(test_images))
    ntr = min(len(train_images), n//2); nte = min(len(test_images), n-ntr)
    ims = torch.cat([train_images[:ntr], test_images[:nte]]).to(device)
    level = len(model.sizes)-1; size = model.sizes[level]
    patches = patchify(ims, size, model.overlap).reshape(-1, model.codebooks[level].shape[1])
    is_train_patch = torch.arange(len(patches), device=device) < ntr * ((32 // size) ** 2)
    # Limit actual thumbnails while fitting UMAP to the requested patch set.
    if len(patches) > cfg["umap_points"]:
        ids = torch.randperm(len(patches), device=device)[:cfg["umap_points"]]
        patches = patches[ids]; is_train_patch = is_train_patch[ids]
    codes = pairwise(patches, model.codebooks[level]).argmin(1)
    emb = umap.UMAP(n_neighbors=cfg["umap_neighbors"], min_dist=0.1, random_state=cfg["seed"]).fit_transform(patches.cpu().numpy())
    w, h = 1800, 1400
    thumb = max(1, round(36 * cfg["umap_patch_scale"]))
    canvas = Image.new("RGB", (w, h), "white"); draw = ImageDraw.Draw(canvas)
    lo, hi = emb.min(0), emb.max(0); xy = (emb-lo)/(hi-lo+1e-8)*([w-60, h-60]) + [30, 30]
    for j, (point, code) in enumerate(zip(xy, codes.tolist())):
        x, y = map(int, point)
        ext = size + 2*model.overlap
        patch = patches[j].reshape(3, ext, ext)[:, model.overlap:model.overlap+size, model.overlap:model.overlap+size].clamp(0, 1).cpu()
        tile = vutils.make_grid(patch, padding=0)
        arr = (tile.permute(1, 2, 0).numpy()*255).astype("uint8")
        im = Image.fromarray(arr).resize((thumb, thumb), resample=Image.Resampling.NEAREST)
        canvas.paste(im, (max(0,min(w-thumb,x-thumb//2)), max(0,min(h-thumb,y-thumb//2))))
        is_train = bool(is_train_patch[j])
        draw.rectangle((x-thumb//2, y-thumb//2, x+thumb//2, y+thumb//2), outline="red" if is_train else "blue", width=1)
        draw.text((x+thumb//2, y-thumb//2), str(code), fill="black")
    canvas.save(run_dir/"patch_umap.png")
    # A codebook contact sheet makes the exemplar/centroid identity explicit.
    ext = size + 2*model.overlap
    codebook = model.codebooks[level].reshape(-1, 3, ext, ext)[:, :, model.overlap:model.overlap+size, model.overlap:model.overlap+size]
    codebook_path = run_dir/("codebook_exemplars.png" if cfg["discrete"] else "codebook_centroids.png")
    vutils.save_image(codebook.clamp(0,1), codebook_path, nrow=16, padding=2)
    (run_dir/"patch_umap_README.txt").write_text(
        "UMAP points are actual CIFAR image patches; red borders are train patches and blue borders test patches. "
        "The integer printed beside each patch is its nearest codebook index. "
        f"{codebook_path.name} shows prototypes by index; discrete prototypes are projected to real training patches.\n")


def main():
    cfg = parse_args()
    cfg["codebook_sizes"] = list(cfg["codebook_sizes"])
    torch.manual_seed(cfg["seed"]); random.seed(cfg["seed"])
    device = torch.device("cuda" if cfg["device"] == "auto" and torch.cuda.is_available() else
                          "cpu" if cfg["device"] == "auto" else cfg["device"])
    sizes = list(cfg["patch_sizes"])
    run_dir = Path(cfg["log_root"])/cfg["run_name"]; run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir/"config.json").write_text(json.dumps(cfg, indent=2, default=str))
    print(f"Resolved config: {json.dumps(cfg, indent=2, default=str)}\nDevice: {device}\nLogs: {run_dir.resolve()}")
    train_ds = load_data(cfg["data_root"], True, cfg["train_subset"])
    test_ds = load_data(cfg["data_root"], False, cfg["test_subset"])
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=device.type == "cuda")
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=device.type == "cuda")
    train_images = torch.stack([train_ds[i][0] for i in range(len(train_ds))]).to(device)
    train_labels = torch.tensor([train_ds[i][1] for i in range(len(train_ds))], device=device)
    test_images = torch.stack([test_ds[i][0] for i in range(len(test_ds))]).to(device)

    
    model = PatchModel(cfg["codebook_sizes"], sizes, cfg["overlap"]).to(device)
    if cfg["checkpoint"]:
        state = torch.load(cfg["checkpoint"], map_location=device, weights_only=False)
        model.load_state_dict(state["model"]); print(f"Loaded checkpoint: {Path(cfg['checkpoint']).resolve()}")
    elif cfg["mode"] == "train":
        init_codebooks(model, train_images)
    else:
        raise ValueError("--checkpoint is required for --mode eval or --mode sample")
    if cfg["mode"] == "train":
        start = time.time()
        if cfg["trainer"] == "gradient": fit_gradient(model, train_loader, train_images, cfg, device, run_dir)
        elif cfg["trainer"] == "em": fit_em(model, train_images, train_labels, cfg)
        else: raise ValueError("trainer must be gradient or em")
        path = run_dir/"model.pt"
        torch.save({"model": model.state_dict(), "config": cfg, "sizes": sizes}, path)
        print(f"Saved checkpoint: {path.resolve()} ({time.time()-start:.1f}s)")
    elif cfg["mode"] not in ("eval", "sample"):
        raise ValueError("mode must be train, eval, or sample")
    if cfg["mode"] in ("train", "eval"):
        metrics = evaluate(model, test_loader, device)
        metrics["unconditional_code_nll_nats_per_patch"] = code_nll(model, test_loader, device, False)
        metrics["class_conditional_code_nll_nats_per_patch"] = code_nll(model, test_loader, device, True)
        (run_dir/"metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"Evaluation: {json.dumps(metrics)}")
        save_grids(model, train_images, test_images, run_dir, cfg, device)
        if cfg["umap_enabled"]: save_umap(model, train_images, test_images, run_dir, cfg, device)
    elif cfg["mode"] == "sample":
        samples = model.sample(cfg["samples"], device, cfg["class_id"])
        vutils.save_image(samples, run_dir/"samples.png", nrow=8, padding=2)
        print(f"Saved samples to {(run_dir/'samples.png').resolve()}")


if __name__ == "__main__":
    main()
