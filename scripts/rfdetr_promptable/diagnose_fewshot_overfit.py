"""Overfit a small fixed set of positive few-shot episodes as a training sanity check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

from rfdetr_promptable.coco_episodes import CocoFewShotEpisodeDataset, collate_few_shot
from train_promptable_fewshot import build_model, _move_batch
from rfdetr_promptable.utilities.box_ops import box_cxcywh_to_xyxy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--annotation-file", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=4, help="Number of fixed positive episodes to overfit.")
    parser.add_argument("--max-shots", type=int, default=5)
    parser.add_argument("--resolution", type=int, default=560)
    parser.add_argument("--num-queries", type=int, default=10)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("logs/promptable_fewshot_overfit.pt"))
    args = parser.parse_args()
    if args.episodes < 1 or args.steps < 1 or args.log_every < 1:
        parser.error("--episodes, --steps, and --log-every must be positive")

    torch.manual_seed(args.seed)
    dataset = CocoFewShotEpisodeDataset(
        args.image_dir,
        args.annotation_file,
        resolution=args.resolution,
        episodes=20_000,
        max_shots=args.max_shots,
        seed=args.seed,
        training=True,
    )
    fixed_items = []
    for index in range(len(dataset)):
        episode = dataset[index]
        if episode["present"]:
            fixed_items.append(episode)
            if len(fixed_items) == args.episodes:
                break
    if len(fixed_items) != args.episodes:
        raise RuntimeError(f"Only found {len(fixed_items)} positive episodes")

    device = torch.device(args.device)
    batch = _move_batch(collate_few_shot(fixed_items), device)
    model = build_model(args.resolution, args.num_queries, args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    log_path = args.output.with_suffix(".jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    model.train()
    progress = tqdm(range(1, args.steps + 1), desc="fixed-episode overfit", unit="step")
    for step in progress:
        outputs = model.forward_support_query(
            batch["support_images"],
            batch["support_boxes"],
            batch["query_images"],
            support_group_ids=batch["support_group_ids"],
            query_to_support=batch["query_to_support"],
        )
        losses = model.loss(
            outputs,
            batch["query_boxes"],
            batch["present"],
            target_mask=batch["query_target_mask"],
        )
        loss = losses["loss_objectness"] + 5 * losses["loss_bbox"] + 2 * losses["loss_giou"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        if step % args.log_every == 0 or step == args.steps:
            record = {
                "step": step,
                "loss": float(loss.detach()),
                "objectness": float(losses["loss_objectness"].detach()),
                "objectness_positive": float(losses["loss_objectness_positive"]),
                "objectness_negative": float(losses["loss_objectness_negative"]),
                "bbox": float(losses["loss_bbox"].detach()),
                "giou": float(losses["loss_giou"].detach()),
                "matched_iou": float(losses["matched_iou"]),
                "positive_score": float(losses["positive_score"]),
                "negative_score": float(losses["negative_score"]),
                "pred_box_mean": losses["pred_box_mean"].detach().cpu().tolist(),
                "target_box_mean": batch["query_boxes"][batch["query_target_mask"]].mean(dim=0).cpu().tolist(),
            }
            tqdm.write(json.dumps(record))
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
                stream.flush()
    torch.save(model.state_dict(), args.output)
    model.eval()
    with torch.inference_mode():
        final_outputs = model.forward_support_query(
            batch["support_images"],
            batch["support_boxes"],
            batch["query_images"],
            support_group_ids=batch["support_group_ids"],
            query_to_support=batch["query_to_support"],
        )
        final_scores = final_outputs["pred_logits"].sigmoid()[..., 0]
        final_boxes = box_cxcywh_to_xyxy(final_outputs["pred_boxes"])
        for index in range(batch["query_images"].shape[0]):
            image = batch["query_images"][index].detach().cpu().permute(1, 2, 0).numpy()
            height, width = image.shape[:2]
            figure, axis = plt.subplots(figsize=(7, 7))
            axis.imshow(image)
            for box in box_cxcywh_to_xyxy(batch["query_boxes"][index, batch["query_target_mask"][index]]).cpu():
                x0, y0, x1, y1 = (box * box.new_tensor([width, height, width, height])).tolist()
                axis.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="lime", linewidth=2))
            top_indices = final_scores[index].topk(min(3, final_scores.shape[1])).indices
            for query_index in top_indices:
                box = final_boxes[index, query_index].cpu()
                x0, y0, x1, y1 = (box * box.new_tensor([width, height, width, height])).tolist()
                axis.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="red", linewidth=1.5))
                axis.text(x0, y0, f"{float(final_scores[index, query_index]):.2f}", color="white", backgroundcolor="red")
            axis.set_title("GT green; top 3 predictions red")
            axis.axis("off")
            figure.savefig(args.output.with_name(f"{args.output.stem}_{index:02d}.png"), dpi=130, bbox_inches="tight")
            plt.close(figure)
    print(f"saved checkpoint={args.output} log={log_path} figures={args.output.stem}_*.png")


if __name__ == "__main__":
    main()
