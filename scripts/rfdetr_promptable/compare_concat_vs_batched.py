"""Compare spatial support/query tiling with paired batch-dimension conditioning."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from rfdetr_promptable.coco_prompt import COCO
from rfdetr_promptable.config import RFDETRBaseConfig
from rfdetr_promptable.promptable import PromptableDetector
from rfdetr_promptable.utilities.box_ops import box_cxcywh_to_xyxy


def norm_box(ann, width, height):
    x, y, w, h = ann["bbox"]
    return [(x + w / 2) / width, (y + h / 2) / height, w / width, h / height]


def load_pairs(annotation_file, image_dir, resolution, count, seed, shots):
    coco = COCO(str(annotation_file))
    instances_by_image = {}
    for image_id in coco.getImgIds():
        anns = [a for a in coco.loadAnns(coco.getAnnIds(imgIds=[image_id], iscrowd=False)) if a["bbox"][2] > 1 and a["bbox"][3] > 1]
        by_class = {}
        for ann in anns:
            by_class.setdefault(ann["category_id"], []).append(ann)
        instances_by_image[image_id] = by_class
    eligible = {}
    for class_id in sorted(coco.getCatIds()):
        support_ids = [i for i, groups in instances_by_image.items() if len(groups.get(class_id, [])) >= 2]
        query_ids = [i for i, groups in instances_by_image.items() if groups.get(class_id)]
        if support_ids and len(query_ids) > 1:
            eligible[class_id] = (support_ids, query_ids)
    rng = random.Random(seed)
    if not eligible:
        raise ValueError("No category has distinct support and query images")
    chosen, seen = [], set()
    class_ids = sorted(eligible)
    attempts = 0
    while len(chosen) < count and attempts < count * 100:
        class_id = class_ids[len(chosen) % len(class_ids)]
        support_ids, query_ids = eligible[class_id]
        qid = rng.choice(query_ids)
        attempts += 1
        candidates = [sid for sid in support_ids if sid != qid]
        if len(candidates) < shots:
            continue
        sids = tuple(rng.sample(candidates, shots))
        if (class_id, sids, qid) not in seen:
            seen.add((class_id, sids, qid))
            chosen.append((class_id, sids, qid))
    episodes = []
    categories = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    for class_id, support_ids, query_id in chosen:
        support_metas = coco.loadImgs(list(support_ids))
        query_meta = coco.loadImgs([query_id])[0]
        support_pils = [Image.open(image_dir / meta["file_name"]).convert("RGB") for meta in support_metas]
        query_pil = Image.open(image_dir / query_meta["file_name"]).convert("RGB")
        qw, qh = query_pil.size
        prompt_anns, support_target_anns = [], []
        prompts, support_target_tensors = [], []
        for support_id, support_pil in zip(support_ids, support_pils):
            sw, sh = support_pil.size
            support_anns = instances_by_image[support_id][class_id]
            prompt_i = (support_id + class_id + seed) % len(support_anns)
            prompt_ann = support_anns[prompt_i]
            prompt_anns.append(prompt_ann)
            support_targets = [a for i, a in enumerate(support_anns) if i != prompt_i]
            support_target_anns.append(support_targets)
            prompts.append(norm_box(prompt_ann, sw, sh))
            support_target_tensors.append(torch.tensor([norm_box(a, sw, sh) for a in support_targets]))
        query_targets = instances_by_image[query_id][class_id]
        episodes.append({
            "class_id": class_id, "class_name": categories[class_id],
            "support_ids": list(support_ids), "query_id": query_id,
            "support_files": [meta["file_name"] for meta in support_metas], "query_file": query_meta["file_name"],
            "support_pils": support_pils, "query_pil": query_pil,
            "support_images": torch.stack([TF.resize(TF.to_tensor(pil), [resolution, resolution]) for pil in support_pils]),
            "query_image": TF.resize(TF.to_tensor(query_pil), [resolution, resolution]),
            "prompts": torch.tensor(prompts),
            "support_targets": support_target_tensors,
            "query_targets": torch.tensor([norm_box(a, qw, qh) for a in query_targets]),
            "prompt_anns": prompt_anns, "support_target_anns": support_target_anns, "query_target_anns": query_targets,
        })
    if len(episodes) < count:
        raise ValueError(f"Only found {len(episodes)} eligible support/query episodes")
    return episodes


def composite_episode(episode, resolution):
    query = TF.to_pil_image(episode["query_image"])
    shots = len(episode["support_pils"])
    canvas = Image.new("RGB", (resolution * (shots + 1), resolution), (16, 18, 24))
    for i, image in enumerate(episode["support_images"]):
        canvas.paste(TF.to_pil_image(image), (resolution * i, 0))
    canvas.paste(query, (resolution * shots, 0))
    prompt = episode["prompts"].clone()
    prompt[:, 0] = (torch.arange(shots, dtype=prompt.dtype) + prompt[:, 0]) / (shots + 1)
    prompt[:, 2] /= shots + 1
    tiled_targets = []
    for i, boxes in enumerate(episode["support_targets"]):
        boxes = boxes.clone()
        boxes[:, 0] = (i + boxes[:, 0]) / (shots + 1)
        boxes[:, 2] /= shots + 1
        tiled_targets.append(boxes)
    query_targets = episode["query_targets"].clone()
    query_targets[:, 0] = (shots + query_targets[:, 0]) / (shots + 1)
    query_targets[:, 2] /= shots + 1
    targets = torch.cat([*tiled_targets, query_targets], dim=0)
    return TF.to_tensor(canvas), prompt, targets, canvas


def pad_targets(target_lists, device):
    max_count = max(x.shape[0] for x in target_lists)
    boxes = torch.zeros((len(target_lists), max_count, 4), device=device)
    mask = torch.zeros((len(target_lists), max_count), dtype=torch.bool, device=device)
    for i, x in enumerate(target_lists):
        boxes[i, :len(x)] = x.to(device)
        mask[i, :len(x)] = True
    return boxes, mask


def train_mode(name, episodes, resolution, queries, steps, lr, seed, device, out_dir):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    cfg = RFDETRBaseConfig(pretrain_weights=None, resolution=resolution, num_queries=queries,
                           num_select=queries, group_detr=1, two_stage=False, device=str(device))
    model = PromptableDetector.from_config(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    if name == "concat":
        composites = [composite_episode(ep, resolution) for ep in episodes]
        images = torch.stack([x[0] for x in composites]).to(device)
        prompts = torch.stack([x[1] for x in composites]).to(device)
        targets, mask = pad_targets([x[2] for x in composites], device)
        support_images = query_images = support_boxes = support_targets = support_mask = None
    else:
        shots = episodes[0]["support_images"].shape[0]
        support_images = torch.stack([e["support_images"] for e in episodes]).to(device)
        query_images = torch.stack([e["query_image"] for e in episodes]).to(device)
        support_boxes = torch.stack([e["prompts"] for e in episodes]).to(device)
        targets, mask = pad_targets([e["query_targets"] for e in episodes], device)
        support_targets, support_mask = pad_targets(
            [target for e in episodes for target in e["support_targets"]], device
        )
        images = prompts = None

    rows = []
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    started = time.perf_counter()
    for step in range(1, steps + 1):
        model.train()
        if name == "concat":
            outputs = model.forward_same_image(images, prompts)
            losses = model.loss(outputs, targets, torch.ones((len(episodes), 1), device=device), mask)
            loss = losses["loss_objectness"] + 5 * losses["loss_bbox"] + 2 * losses["loss_giou"]
            query_outputs = outputs
        else:
            support_outputs, query_outputs = model.forward_multi_shot_support_query(support_images, support_boxes, query_images)
            support_loss = model.loss(support_outputs, support_targets,
                                      torch.ones((len(episodes) * shots, 1), device=device), support_mask)
            query_loss = model.loss(query_outputs, targets, torch.ones((len(episodes), 1), device=device), mask)
            loss = (query_loss["loss_objectness"] + 5 * query_loss["loss_bbox"] + 2 * query_loss["loss_giou"]
                    + 0.5 * (support_loss["loss_objectness"] + 5 * support_loss["loss_bbox"] + 2 * support_loss["loss_giou"]))
            losses = query_loss
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1); optimizer.step()
        rows.append({"step": step, "loss": float(loss.detach()), "objectness": float(losses["loss_objectness"].detach()),
                     "bbox": float(losses["loss_bbox"].detach()), "giou": float(losses["loss_giou"].detach()),
                     "matched_iou": float(losses["matched_iou"]),
                     "peak_cuda_gb": torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0})
        if step % 50 == 0 or step == 1 or step == steps:
            print(f"{name} step={step}/{steps} loss={rows[-1]['loss']:.3f} IoU={rows[-1]['matched_iou']:.3f}")
    elapsed = time.perf_counter() - started
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    scores, boxes = None, None
    model.eval()
    with torch.inference_mode():
        if name == "concat":
            prediction = model.forward_same_image(images, prompts)
        else:
            _, prediction = model.forward_multi_shot_support_query(support_images, support_boxes, query_images)
        scores = prediction["pred_logits"].sigmoid()[..., 0].cpu()
        boxes = box_cxcywh_to_xyxy(prediction["pred_boxes"]).cpu()
    for i, ep in enumerate(episodes):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].set_title(f"{len(ep['support_pils'])} support shots + box prompts")
        support_strip = Image.new("RGB", (resolution * len(ep["support_pils"]), resolution))
        for shot, (support_pil, prompt_ann) in enumerate(zip(ep["support_pils"], ep["prompt_anns"])):
            resized = TF.to_pil_image(TF.resize(TF.to_tensor(support_pil), [resolution, resolution]))
            support_strip.paste(resized, (shot * resolution, 0))
            px, py, pw, ph = prompt_ann["bbox"]
            sw, sh = support_pil.size
            axes[0].add_patch(patches.Rectangle((shot * resolution + px * resolution / sw, py * resolution / sh),
                                                pw * resolution / sw, ph * resolution / sh,
                                                fill=False, edgecolor="cyan", linewidth=2))
        axes[0].imshow(support_strip)
        axes[1].imshow(ep["query_pil"]); axes[1].set_title("Query + GT")
        for ann in ep["query_target_anns"]:
            x, y, w, h = ann["bbox"]
            axes[1].add_patch(patches.Rectangle((x, y), w, h, fill=False, edgecolor="lime", linewidth=2))
        if name == "concat":
            shown = composites[i][3]
            axes[2].imshow(shown); axes[2].set_title("Concat predictions")
        else:
            axes[2].imshow(ep["query_pil"]); axes[2].set_title("Batched query predictions")
        h, w = axes[2].images[0].get_array().shape[:2]
        selected = scores[i].topk(min(10, scores.shape[1])).indices
        for q in selected.tolist():
            x0, y0, x1, y1 = boxes[i, q].tolist()
            axes[2].add_patch(patches.Rectangle((x0*w, y0*h), (x1-x0)*w, (y1-y0)*h,
                                                fill=False, edgecolor="red", linewidth=1.5))
        if name == "concat":
            for shot, (prompt_ann, support_pil) in enumerate(zip(ep["prompt_anns"], ep["support_pils"])):
                px, py, pw, ph = prompt_ann["bbox"]
                sw, sh = support_pil.size
                axes[2].add_patch(patches.Rectangle(
                    (shot * resolution + px * resolution / sw, py * resolution / sh),
                    pw * resolution / sw, ph * resolution / sh,
                    fill=False, edgecolor="cyan", linewidth=1.5))
        for ax in axes: ax.axis("off")
        fig.suptitle(f"{name} | {ep['class_name']} | {len(ep['support_ids'])}-shot support {ep['support_ids']} -> query {ep['query_id']}")
        fig.tight_layout(); fig.savefig(out_dir / f"episode_{i:02d}.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4)); ax.plot([x["step"] for x in rows], [x["loss"] for x in rows], label="loss")
    ax2 = ax.twinx(); ax2.plot([x["step"] for x in rows], [x["matched_iou"] for x in rows], color="tab:orange", label="IoU")
    ax.set_xlabel("step"); ax.set_ylabel("loss"); ax2.set_ylabel("matched IoU"); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "training_curves.png", dpi=160); plt.close(fig)
    torch.save(model.state_dict(), out_dir / "model.pt")
    summary = {"mode": name, "steps": steps, "elapsed_seconds": elapsed,
               "peak_cuda_gb": rows[-1]["peak_cuda_gb"], "initial": rows[0], "final": rows[-1],
               "best_matched_iou": max(r["matched_iou"] for r in rows),
               "shots": len(episodes[0]["support_ids"]),
               "episodes": [{"class_id": ep["class_id"], "class_name": ep["class_name"],
                             "support_ids": ep["support_ids"], "query_id": ep["query_id"],
                             "support_files": ep["support_files"], "query_file": ep["query_file"]} for ep in episodes]}
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--annotation-file", type=Path, default=Path("datasets/mscoco_prompt_split_tiny/train.json"))
    p.add_argument("--image-dir", type=Path, default=Path("datasets/mscoco/train2017"))
    p.add_argument("--output-dir", type=Path, default=Path("logs/prompt_conditioning_comparison"))
    p.add_argument("--episodes", type=int, default=4); p.add_argument("--steps", type=int, default=200)
    p.add_argument("--shots", type=int, default=1, help="Distinct support images per query episode.")
    p.add_argument("--resolution", type=int, default=280); p.add_argument("--queries", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--seed", type=int, default=91)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(); device = torch.device(args.device)
    if args.shots < 1:
        p.error("--shots must be at least 1")
    episodes = load_pairs(args.annotation_file, args.image_dir, args.resolution, args.episodes, args.seed, args.shots)
    results = []
    for mode in ("concat", "batch"):
        results.append(train_mode(mode, episodes, args.resolution, args.queries, args.steps,
                                  args.lr, args.seed, device, args.output_dir / mode))
    (args.output_dir / "comparison.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
