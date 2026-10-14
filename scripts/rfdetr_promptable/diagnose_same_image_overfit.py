"""Overfit a fixed set of same-image visual-prompt episodes and save diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import random
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


def make_episodes(annotation_file: Path, image_dir: Path, resolution: int, count: int, seed: int):
    coco = COCO(str(annotation_file))
    pairs = []
    for image_id in sorted(coco.getImgIds()):
        anns = [a for a in coco.loadAnns(coco.getAnnIds(imgIds=[image_id], iscrowd=False)) if a["bbox"][2] > 1 and a["bbox"][3] > 1]
        by_class: dict[int, list[dict]] = {}
        for ann in anns:
            by_class.setdefault(ann["category_id"], []).append(ann)
        for category_id, instances in by_class.items():
            if len(instances) >= 2:
                pairs.append((image_id, category_id, instances))
    if len(pairs) < count:
        raise ValueError(f"Need {count} eligible image/class pairs; found {len(pairs)}")
    random.Random(seed).shuffle(pairs)
    episodes = []
    for image_id, category_id, instances in pairs[:count]:
        metadata = coco.loadImgs([image_id])[0]
        image = Image.open(image_dir / metadata["file_name"]).convert("RGB")
        width, height = image.size
        prompt_index = (image_id + seed + category_id) % len(instances)
        prompt_ann = instances[prompt_index]
        targets = [a for i, a in enumerate(instances) if i != prompt_index]
        normalize = lambda ann: [
            (ann["bbox"][0] + ann["bbox"][2] / 2) / width,
            (ann["bbox"][1] + ann["bbox"][3] / 2) / height,
            ann["bbox"][2] / width,
            ann["bbox"][3] / height,
        ]
        episodes.append({
            "image_id": image_id,
            "category_id": category_id,
            "class_name": coco.loadCats([category_id])[0]["name"],
            "file_name": metadata["file_name"],
            "image": TF.resize(TF.to_tensor(image), [resolution, resolution]),
            "prompt": torch.tensor(normalize(prompt_ann), dtype=torch.float32),
            "targets": torch.tensor([normalize(a) for a in targets], dtype=torch.float32),
            "pil": image,
            "prompt_ann": prompt_ann,
            "target_anns": targets,
        })
    return episodes


def draw_box(ax, box, color, label, width, height):
    x, y, w, h = box
    rect = patches.Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=2)
    ax.add_patch(rect)
    ax.text(x, max(0, y - 4), label, color="white", fontsize=8,
            bbox={"facecolor": color, "alpha": 0.8, "pad": 2})


def save_episode_figure(episode, predictions, scores, output: Path):
    image = episode["pil"]
    width, height = image.size
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, title in zip(axes, ("Prompt image and GT targets", "Prompt image and predictions")):
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(title)
        box = episode["prompt_ann"]["bbox"]
        draw_box(ax, box, "cyan", "prompt instance", width, height)
    for ann in episode["target_anns"]:
        draw_box(axes[0], ann["bbox"], "lime", "target instance", width, height)
    for box, score in zip(predictions, scores, strict=True):
        x0, y0, x1, y1 = box
        draw_box(axes[1], [x0, y0, x1 - x0, y1 - y0], "red", f"pred {score:.2f}", width, height)
    fig.suptitle(f"{episode['class_name']} | image {episode['image_id']}")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-file", type=Path, default=Path("datasets/mscoco_prompt_split_tiny/train.json"))
    parser.add_argument("--image-dir", type=Path, default=Path("datasets/mscoco/train2017"))
    parser.add_argument("--output-dir", type=Path, default=Path("logs/same_image_overfit"))
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--resolution", type=int, default=280)
    parser.add_argument("--num-queries", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--figure-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes = make_episodes(args.annotation_file, args.image_dir, args.resolution, args.episodes, args.seed)
    device = torch.device(args.device)
    config = RFDETRBaseConfig(
        pretrain_weights=None, resolution=args.resolution, num_queries=args.num_queries,
        num_select=args.num_queries, group_detr=1, two_stage=False, device=args.device,
    )
    model = PromptableDetector.from_config(config).to(device)
    if args.resume_from:
        model.load_state_dict(torch.load(args.resume_from, map_location=device, weights_only=True))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    images = torch.stack([e["image"] for e in episodes]).to(device)
    prompts = torch.stack([e["prompt"] for e in episodes]).to(device)
    max_targets = max(e["targets"].shape[0] for e in episodes)
    targets = torch.zeros((len(episodes), max_targets, 4), device=device)
    target_mask = torch.zeros((len(episodes), max_targets), dtype=torch.bool, device=device)
    for i, episode in enumerate(episodes):
        n = episode["targets"].shape[0]
        targets[i, :n] = episode["targets"].to(device)
        target_mask[i, :n] = True

    rows = []
    csv_path = args.output_dir / "metrics.csv"
    if csv_path.exists() and args.resume_from:
        with csv_path.open(newline="", encoding="utf-8") as f:
            rows = [{k: (int(v) if k == "step" else float(v)) for k, v in row.items()} for row in csv.DictReader(f)]
    start_step = int(rows[-1]["step"]) if rows else 0
    final_step = start_step + args.steps
    for step in range(start_step + 1, final_step + 1):
        model.train()
        outputs = model.forward_same_image(images, prompts)
        losses = model.loss(outputs, targets, torch.ones((len(episodes), 1), device=device), target_mask)
        loss = losses["loss_objectness"] + 5 * losses["loss_bbox"] + 2 * losses["loss_giou"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        row = {"step": step, "loss": float(loss.detach()), "objectness": float(losses["loss_objectness"].detach()),
               "bbox": float(losses["loss_bbox"].detach()), "giou": float(losses["loss_giou"].detach()),
               "matched_iou": float(losses["matched_iou"]), "positive_score": float(losses["positive_score"])}
        rows.append(row)
        if step == 1 or step % 10 == 0 or step == final_step:
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
        if step == 1 or step % args.figure_every == 0 or step == final_step:
            model.eval()
            with torch.inference_mode():
                outputs = model.forward_same_image(images, prompts)
                scores = outputs["pred_logits"].sigmoid()[..., 0]
                boxes = box_cxcywh_to_xyxy(outputs["pred_boxes"])
                for i, episode in enumerate(episodes):
                    top_scores, indices = scores[i].topk(min(10, scores.shape[1]))
                    top_boxes = boxes[i, indices]
                    h, w = episode["pil"].height, episode["pil"].width
                    scale = top_boxes.new_tensor([w, h, w, h])
                    pixel_boxes = (top_boxes * scale).cpu().tolist()
                    save_episode_figure(episode, pixel_boxes, top_scores.cpu().tolist(),
                        args.output_dir / "figures" / f"step_{step:05d}" / episode["class_name"] / f"{episode['image_id']:012d}.png")
            print(f"step={step} loss={row['loss']:.4f} obj={row['objectness']:.4f} bbox={row['bbox']:.4f} giou={row['giou']:.4f} matched_iou={row['matched_iou']:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot([r["step"] for r in rows], [r["loss"] for r in rows]); axes[0].set_title("Total loss")
    axes[1].plot([r["step"] for r in rows], [r["matched_iou"] for r in rows]); axes[1].set_title("Matched IoU")
    for ax in axes: ax.set_xlabel("step"); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(args.output_dir / "training_curves.png", dpi=160); plt.close(fig)
    summary = {"config": vars(args) | {"device": str(device)}, "episodes": [
        {k: e[k] for k in ("image_id", "category_id", "class_name", "file_name", "prompt_ann", "target_anns")} for e in episodes
    ], "initial": rows[0], "final": rows[-1], "best_matched_iou": max(r["matched_iou"] for r in rows)}
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, default=str))
    torch.save(model.state_dict(), args.output_dir / "model.pt")


if __name__ == "__main__":
    main()
