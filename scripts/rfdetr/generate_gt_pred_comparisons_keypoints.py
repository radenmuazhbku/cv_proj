"""Create GT-only, prediction-only, and overlay comparison panels from COCO labels.

Supports boxes, segmentation masks, and keypoints. Each output is a 3-panel JPEG
(Ground Truth | Predictions | Overlay) saved to ``output_dir/<image_filename>``.

Example:
.venv/bin/python scripts/rfdetr/generate_gt_pred_comparisons_keypoints.py \
--checkpoint logs/train_1_keypoint/checkpoint_best_ema.pth \
--images-dir datasets/scalebar_H1H2/test \
--annotations datasets/scalebar_H1H2/test/_annotations.coco.json \
--output-dir logs/comparisons/test --count 10
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from rfdetr import from_checkpoint

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
GT_COLOR = (0, 230, 118)    # green
PRED_COLOR = (255, 82, 82)  # red
GT_KEYPOINT_COLOR = (255, 215, 0)  # gold
PRED_KEYPOINT_COLOR = (0, 220, 255)  # cyan
KEYPOINT_HALO_COLOR = (12, 12, 12)  # near-black outline for contrast
_GT_SV_COLOR = None   # resolved lazily
_PRED_SV_COLOR = None  # resolved lazily


def _sv_color(rgb: tuple[int, int, int]) -> Any:
    """Return an ``sv.Color`` from an RGB tuple."""
    import supervision as sv

    # Supervision's OpenCV annotators write BGR pixels; the source canvases here
    # are RGB arrays, so reverse channels to preserve the requested display color.
    return sv.Color(r=rgb[2], g=rgb[1], b=rgb[0])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("logs/checkpoint_best_ema.pth"))
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True, help="COCO _annotations.coco.json file.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--device", default=None, help="Override checkpoint device, e.g. cuda or cpu.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# COCO loading helpers
# ---------------------------------------------------------------------------

def load_coco_gt(path: Path) -> tuple[dict[str, dict], dict[int, str]]:
    """Load COCO annotations.

    Returns:
        gt_by_file: mapping filename → {"boxes": list[[x1,y1,x2,y2]], "class_ids": list[int],
                                         "masks": list[np.ndarray|None], "keypoints": list[np.ndarray|None]}
        id_to_name: mapping category_id → category name
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    id_to_name: dict[int, str] = {cat["id"]: cat["name"] for cat in data["categories"]}
    file_names: dict[int, str] = {img["id"]: img["file_name"] for img in data["images"]}
    img_sizes: dict[int, tuple[int, int]] = {img["id"]: (img["height"], img["width"]) for img in data["images"]}

    gt_by_file: dict[str, dict] = defaultdict(
        lambda: {"boxes": [], "class_ids": [], "masks": [], "keypoints": [], "num_keypoints": []}
    )

    for ann in data["annotations"]:
        fname = file_names[ann["image_id"]]
        h, w = img_sizes[ann["image_id"]]
        x, y, bw, bh = ann["bbox"]
        gt_by_file[fname]["boxes"].append([x, y, x + bw, y + bh])
        gt_by_file[fname]["class_ids"].append(ann["category_id"])

        # Segmentation mask
        seg = ann.get("segmentation")
        mask_arr: np.ndarray | None = None
        if seg:
            try:
                from pycocotools.mask import frPyObjects, decode, merge  # type: ignore[import-untyped]
                rles = frPyObjects(seg, h, w)
                rle = merge(rles) if isinstance(rles, list) else rles
                mask_arr = decode(rle).astype(bool)
            except Exception:
                # Fall back: try simple polygon fill with supervision
                try:
                    import supervision as sv
                    polys = [np.array(p).reshape(-1, 2).astype(np.int32) for p in seg if len(p) >= 6]
                    if polys:
                        canvas = np.zeros((h, w), dtype=bool)
                        for poly in polys:
                            m = sv.polygon_to_mask(poly, (w, h))
                            canvas |= m.astype(bool)
                        mask_arr = canvas
                except Exception:
                    pass
        gt_by_file[fname]["masks"].append(mask_arr)

        # Keypoints  [x, y, v, x, y, v, ...]
        kps = ann.get("keypoints")
        kp_arr: np.ndarray | None = None
        num_kp = ann.get("num_keypoints", 0)
        if kps and num_kp > 0:
            kp_flat = np.array(kps, dtype=float)
            kp_arr = kp_flat.reshape(-1, 3)  # [K, 3]
        gt_by_file[fname]["keypoints"].append(kp_arr)
        gt_by_file[fname]["num_keypoints"].append(num_kp)

    return gt_by_file, id_to_name


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------

def _build_gt_detections(gt: dict, img_h: int, img_w: int) -> Any:
    """Build an ``sv.Detections`` object from raw COCO GT data."""
    import supervision as sv

    boxes = np.array(gt["boxes"], dtype=np.float32) if gt["boxes"] else np.zeros((0, 4), dtype=np.float32)
    class_ids = np.array(gt["class_ids"], dtype=int)

    det_kwargs: dict[str, Any] = {"xyxy": boxes, "class_id": class_ids}

    masks_list = gt.get("masks", [])
    valid_masks = [m for m in masks_list if m is not None]
    if valid_masks and len(valid_masks) == len(boxes):
        det_kwargs["mask"] = np.stack([
            (m if m is not None else np.zeros((img_h, img_w), dtype=bool))
            for m in masks_list
        ]).astype(bool)

    return sv.Detections(**det_kwargs)


def _build_gt_keypoints(gt: dict) -> Any | None:
    """Build an ``sv.KeyPoints`` from GT keypoint arrays, or ``None`` if no keypoints."""
    import supervision as sv

    kp_arrays = [k for k in gt.get("keypoints", []) if k is not None]
    if not kp_arrays:
        return None

    # Align arrays to the same K (pad with zeros if mixed)
    max_k = max(a.shape[0] for a in kp_arrays)
    padded_xy: list[np.ndarray] = []
    padded_conf: list[np.ndarray] = []
    for arr in kp_arrays:
        k = arr.shape[0]
        pad = max_k - k
        xy = arr[:, :2]
        vis = arr[:, 2].clip(0, 1)
        if pad > 0:
            xy = np.vstack([xy, np.zeros((pad, 2))])
            vis = np.concatenate([vis, np.zeros(pad)])
        padded_xy.append(xy)
        padded_conf.append(vis)

    xy_arr = np.stack(padded_xy, axis=0)  # [N, K, 2]
    conf_arr = np.stack(padded_conf, axis=0)  # [N, K]
    return sv.KeyPoints(xy=xy_arr, confidence=conf_arr)


def _annotate(
    scene: np.ndarray,
    detections: Any,
    keypoints: Any | None,
    label_strings: list[str],
    box_color: Any,
    mask_opacity: float = 0.4,
    keypoint_color: Any | None = None,
) -> np.ndarray:
    """Annotate *scene* with boxes, (optional) masks, labels, and keypoints."""
    import supervision as sv

    if detections is not None and len(detections) > 0:
        if detections.mask is not None:
            try:
                scene = sv.MaskAnnotator(color=box_color, opacity=mask_opacity).annotate(scene, detections)
            except Exception:
                pass
        try:
            scene = sv.BoxAnnotator(color=box_color, thickness=2).annotate(scene, detections)
        except Exception:
            pass
        try:
            scene = sv.LabelAnnotator(color=box_color, text_scale=0.4).annotate(
                scene, detections, labels=label_strings
            )
        except Exception:
            pass

    if keypoints is not None and len(keypoints) > 0:
        # Draw a dark underlay first, then a smaller saturated mark. The outline
        # keeps joints and limbs readable over bright backgrounds and masks, and
        # the dedicated palette separates keypoints from the box/mask colors.
        point_color = keypoint_color or box_color
        halo_color = _sv_color(KEYPOINT_HALO_COLOR)
        # Two-point annotations in this dataset have landmarks but no skeleton
        # topology. Skip edges to avoid Supervision's "No skeleton found with 2
        # vertices" warning; keep the outlined landmarks visible.
        if keypoints.xy.shape[1] > 2:
            try:
                scene = sv.EdgeAnnotator(color=halo_color, thickness=6).annotate(scene, keypoints)
            except Exception:
                pass
            try:
                scene = sv.EdgeAnnotator(color=point_color, thickness=3).annotate(scene, keypoints)
            except Exception:
                pass
        try:
            scene = sv.VertexAnnotator(color=halo_color, radius=8).annotate(scene, keypoints)
        except Exception:
            pass
        try:
            scene = sv.VertexAnnotator(color=point_color, radius=5).annotate(scene, keypoints)
        except Exception:
            pass

    return scene


def add_title(image: np.ndarray, title: str) -> np.ndarray:
    """Add a compact panel title without changing image dimensions."""
    from PIL import Image as PILImage, ImageDraw, ImageFont
    canvas = PILImage.fromarray(image.copy())
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    left, top, right, bottom = draw.textbbox((0, 0), title, font=font)
    draw.rectangle((0, 0, right - left + 10, bottom - top + 8), fill=(0, 0, 0))
    draw.text((5, 4), title, fill=(255, 255, 255), font=font)
    return np.asarray(canvas)


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def _pred_detections_and_keypoints(
    detections: Any,
    threshold: float,
    id_to_name: dict[int, str],
) -> tuple[Any, Any | None, list[str]]:
    """Split a supervision prediction object into (sv.Detections, sv.KeyPoints|None, label_strings).

    ``model.predict()`` returns either ``sv.Detections`` (for box/seg models) or
    ``sv.KeyPoints`` (for keypoint models).  In joint mode it returns ``sv.KeyPoints``
    with metadata stored in ``data["mask"]``.
    """
    import supervision as sv

    if isinstance(detections, sv.KeyPoints):
        # RF-DETR stores detection boxes in KeyPoints.data["xyxy"], while scores
        # and class ids are native KeyPoints fields (not data["scores"]/["labels"]).
        data = detections.data  # type: ignore[attr-defined]
        boxes = data.get("xyxy", data.get("boxes"))
        scores = detections.detection_confidence
        if scores is None:
            scores = data.get("scores")
        labels = detections.class_id
        if labels is None:
            labels = data.get("labels")
        masks = data.get("mask")
        class_names = data.get("class_name")

        if boxes is None:
            return None, detections, []

        boxes_np = np.asarray(boxes)
        scores_np = np.asarray(scores) if scores is not None else np.ones(len(boxes_np))
        labels_np = np.asarray(labels, dtype=int) if labels is not None else np.zeros(len(boxes_np), dtype=int)
        if len(boxes_np) != len(detections):
            raise ValueError(
                f"Keypoint prediction boxes ({len(boxes_np)}) do not align with keypoint instances ({len(detections)})."
            )
        keep = scores_np >= threshold

        boxes_np = boxes_np[keep]
        scores_np = scores_np[keep]
        labels_np = labels_np[keep]

        det_kw: dict[str, Any] = {
            "xyxy": boxes_np,
            "confidence": scores_np,
            "class_id": labels_np,
        }
        if masks is not None:
            masks_np = np.asarray(masks)
            if masks_np.ndim == 4 and masks_np.shape[1] == 1:
                masks_np = masks_np[:, 0]
            det_kw["mask"] = masks_np[keep].astype(bool)

        det_obj = sv.Detections(**det_kw) if len(boxes_np) > 0 else None

        # Filter keypoints to kept indices
        kp_xy = np.asarray(detections.xy)[keep]
        kp_confidence = detections.keypoint_confidence
        kp_conf = np.asarray(kp_confidence)[keep] if kp_confidence is not None else np.ones(kp_xy.shape[:2])
        kp_visible = detections.visible
        visible = np.asarray(kp_visible)[keep] if kp_visible is not None else kp_conf > 0
        kp_obj = (
            sv.KeyPoints(
                xy=kp_xy,
                keypoint_confidence=kp_conf,
                detection_confidence=scores_np,
                class_id=labels_np,
                visible=visible,
            )
            if len(kp_xy) > 0
            else None
        )

        filtered_names = np.asarray(class_names, dtype=object)[keep] if class_names is not None else None
        label_strs = [
            f"{name if name else id_to_name.get(int(cls_id), str(cls_id))} {sc:.2f}"
            for idx, (cls_id, sc) in enumerate(zip(labels_np, scores_np))
            for name in [filtered_names[idx] if filtered_names is not None else ""]
        ]
        return det_obj, kp_obj, label_strs

    elif isinstance(detections, sv.Detections):
        # Regular detection or segmentation model
        keep = (detections.confidence >= threshold) if detections.confidence is not None else np.ones(len(detections), dtype=bool)
        filtered = detections[keep]
        class_ids = filtered.class_id if filtered.class_id is not None else np.zeros(len(filtered), dtype=int)
        confs = filtered.confidence if filtered.confidence is not None else np.ones(len(filtered))
        label_strs = [
            f"{id_to_name.get(int(cid), str(cid))} {sc:.2f}"
            for cid, sc in zip(class_ids, confs)
        ]
        return filtered, None, label_strs

    return None, None, []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if args.count < 1 or args.batch_size < 1:
        raise ValueError("--count and --batch-size must be at least 1")
    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must be between 0 and 1")

    import supervision as sv  # noqa: F401  (imported early to surface ImportError cleanly)

    image_paths = sorted(
        path for path in args.images_dir.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    )
    image_paths = image_paths[: args.count]
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.images_dir}")

    gt_by_file, id_to_name = load_coco_gt(args.annotations)

    model_kwargs: dict[str, Any] = {"device": args.device} if args.device else {}
    model = from_checkpoint(args.checkpoint, **model_kwargs)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gt_color = _sv_color(GT_COLOR)
    pred_color = _sv_color(PRED_COLOR)
    gt_keypoint_color = _sv_color(GT_KEYPOINT_COLOR)
    pred_keypoint_color = _sv_color(PRED_KEYPOINT_COLOR)

    for start in range(0, len(image_paths), args.batch_size):
        batch_paths = image_paths[start : start + args.batch_size]
        images = [np.asarray(Image.open(p).convert("RGB")).copy() for p in batch_paths]
        batch_preds = model.predict(images, threshold=args.threshold, include_source_image=False)

        for path, image, raw_pred in zip(batch_paths, images, batch_preds):
            img_h, img_w = image.shape[:2]
            gt = gt_by_file.get(path.name, {"boxes": [], "class_ids": [], "masks": [], "keypoints": [], "num_keypoints": []})

            # --- Ground-truth ---
            gt_dets = _build_gt_detections(gt, img_h, img_w)
            gt_kps = _build_gt_keypoints(gt)
            gt_labels = [
                f"GT {id_to_name.get(cid, str(cid))}"
                for cid in gt.get("class_ids", [])
            ]
            gt_panel = _annotate(
                image.copy(), gt_dets, gt_kps, gt_labels, gt_color, keypoint_color=gt_keypoint_color
            )
            gt_panel = add_title(gt_panel, f"Ground truth ({len(gt.get('boxes', []))})")

            # --- Predictions ---
            pred_dets, pred_kps, pred_labels = _pred_detections_and_keypoints(raw_pred, args.threshold, id_to_name)
            n_preds = len(pred_dets) if pred_dets is not None else 0
            pred_panel = _annotate(
                image.copy(), pred_dets, pred_kps, pred_labels, pred_color, keypoint_color=pred_keypoint_color
            )
            pred_panel = add_title(pred_panel, f"Predictions ({n_preds})")

            # --- Overlay ---
            overlay = _annotate(
                image.copy(), gt_dets, gt_kps, gt_labels, gt_color,
                mask_opacity=0.3, keypoint_color=gt_keypoint_color,
            )
            overlay = _annotate(
                overlay, pred_dets, pred_kps, pred_labels, pred_color,
                mask_opacity=0.3, keypoint_color=pred_keypoint_color,
            )
            overlay = add_title(overlay, "Overlay: GT box green/kp gold; prediction box red/kp cyan")

            Image.fromarray(np.concatenate((gt_panel, pred_panel, overlay), axis=1)).save(
                args.output_dir / path.name
            )
            print(f"{path.name}: {len(gt.get('boxes', []))} GT boxes, {n_preds} predictions")


if __name__ == "__main__":
    main()
