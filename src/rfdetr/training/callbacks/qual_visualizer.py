# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""QualitativeVisualizerCallback — saves GT/prediction comparison panels during training.

Captures a fixed set of validation images on the first batch of each validation epoch,
then renders Ground Truth, Predictions, and Overlay panels with boxes, masks, and keypoints
to ``output_dir/epoch_{epoch:04d}/{img_index:03d}.jpg``. Also runs on test epoch end.
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from pytorch_lightning import Callback
from torchvision import transforms as T  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# ImageNet inverse normalisation (applied to [C, H, W] tensors that
# were normalised by mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
# ------------------------------------------------------------------
_INV_NORMALIZE = T.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
)

_GT_COLOR = (0, 230, 118)
_PRED_COLOR = (255, 82, 82)
_GT_KEYPOINT_COLOR = (255, 215, 0)
_PRED_KEYPOINT_COLOR = (0, 220, 255)
_KEYPOINT_HALO_COLOR = (12, 12, 12)


def _tensor_to_uint8(image_tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalised ``[C, H, W]`` float tensor to an ``[H, W, 3]`` uint8 RGB array."""
    img = _INV_NORMALIZE(image_tensor.detach().cpu().float())
    return np.ascontiguousarray(
        (np.clip(img.numpy().transpose(1, 2, 0), 0.0, 1.0) * 255).astype(np.uint8)
    )


def _sv_color(rgb: tuple[int, int, int]) -> Any:
    """Convert an RGB color to Supervision's OpenCV-compatible color."""
    import supervision as sv

    # Supervision's OpenCV annotators write BGR pixels; the scenes are RGB arrays.
    return sv.Color(r=rgb[2], g=rgb[1], b=rgb[0])


def _class_label(class_id: int, class_names: list[str] | None) -> str:
    """Resolve a class label while tolerating out-of-range ids."""
    if class_names is not None and 0 <= class_id < len(class_names):
        return class_names[class_id]
    return str(class_id)


def _resize_and_pad_masks(
    masks: torch.Tensor | None,
    height: int,
    width: int,
    scene_height: int,
    scene_width: int,
) -> np.ndarray | None:
    """Resize instance masks to the unpadded image size and pad to scene dimensions."""
    if masks is None:
        return None
    from PIL import Image as PILImage

    masks_np = masks.detach().cpu().numpy()
    if masks_np.ndim == 4 and masks_np.shape[1] == 1:
        masks_np = masks_np[:, 0]
    if masks_np.ndim != 3 or masks_np.shape[0] == 0:
        return None

    resized = np.zeros((len(masks_np), scene_height, scene_width), dtype=bool)
    copy_height = min(height, scene_height)
    copy_width = min(width, scene_width)
    for index, mask in enumerate(masks_np):
        binary_mask = mask > 0.5 if np.issubdtype(mask.dtype, np.floating) else mask.astype(bool)
        if mask.shape != (height, width):
            mask_image = PILImage.fromarray(binary_mask.astype(np.uint8) * 255)
            binary_mask = np.asarray(mask_image.resize((width, height), resample=PILImage.Resampling.NEAREST)) > 0
        resized[index, :copy_height, :copy_width] = binary_mask[:copy_height, :copy_width]
    return resized


def _make_detections(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray | None,
) -> Any | None:
    """Create a Supervision Detections object, or None when no boxes survive."""
    if not len(boxes):
        return None
    import supervision as sv

    kwargs: dict[str, Any] = {"xyxy": boxes.astype(np.float32), "confidence": scores, "class_id": labels}
    if masks is not None and len(masks) == len(boxes):
        kwargs["mask"] = masks
    return sv.Detections(**kwargs)


def _annotate_scene(
    scene: np.ndarray,
    detections: Any | None,
    keypoints: Any | None,
    label_strings: list[str],
    box_color: Any,
    keypoint_color: Any,
    mask_opacity: float = 0.4,
) -> np.ndarray:
    """Draw masks, boxes, labels, and outlined keypoints onto an RGB scene."""
    try:
        import supervision as sv
    except ImportError:
        logger.warning("supervision is not installed; skipping qualitative visualization.")
        return scene

    if detections is not None and len(detections) > 0:
        if detections.mask is not None:
            try:
                scene = sv.MaskAnnotator(color=box_color, opacity=mask_opacity).annotate(scene, detections)
            except Exception:
                logger.debug("Could not draw instance masks.", exc_info=True)
        try:
            scene = sv.BoxAnnotator(color=box_color, thickness=2).annotate(scene, detections)
        except Exception:
            logger.debug("Could not draw boxes.", exc_info=True)
        try:
            scene = sv.LabelAnnotator(color=box_color, text_scale=0.4).annotate(
                scene, detections, labels=label_strings
            )
        except Exception:
            logger.debug("Could not draw labels.", exc_info=True)

    if keypoints is not None and len(keypoints) > 0:
        halo_color = _sv_color(_KEYPOINT_HALO_COLOR)
        # Dark underlay plus saturated edges and vertices mirrors the comparison script.
        for annotator in (
            sv.EdgeAnnotator(color=halo_color, thickness=6),
            sv.EdgeAnnotator(color=keypoint_color, thickness=3),
            sv.VertexAnnotator(color=halo_color, radius=8),
            sv.VertexAnnotator(color=keypoint_color, radius=5),
        ):
            try:
                scene = annotator.annotate(scene, keypoints)
            except Exception:
                logger.debug("Could not draw keypoints.", exc_info=True)
    return scene


def _add_title(image: np.ndarray, title: str) -> np.ndarray:
    """Add a compact black-backed title to a visualization panel."""
    from PIL import Image as PILImage, ImageDraw, ImageFont

    canvas = PILImage.fromarray(image.copy())
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    left, top, right, bottom = draw.textbbox((0, 0), title, font=font)
    draw.rectangle((0, 0, right - left + 10, bottom - top + 8), fill=(0, 0, 0))
    draw.text((5, 4), title, fill=(255, 255, 255), font=font)
    return np.asarray(canvas)


def _build_ground_truth(
    target: dict[str, Any],
    scene_shape: tuple[int, int],
    class_names: list[str] | None,
) -> tuple[Any | None, Any | None, list[str], int]:
    """Convert transformed training targets to pixel-space detections and keypoints."""
    import supervision as sv

    scene_height, scene_width = scene_shape
    size = target.get("size")
    height, width = (int(size[0]), int(size[1])) if size is not None else (scene_height, scene_width)
    boxes = target.get("boxes", torch.zeros((0, 4)))
    labels = target.get("labels", torch.zeros((0,), dtype=torch.long))
    boxes_np = boxes.detach().cpu().numpy().astype(np.float32)
    labels_np = labels.detach().cpu().numpy().astype(int)
    if len(boxes_np):
        cx, cy, box_width, box_height = boxes_np.T
        boxes_np = np.stack(
            (cx - box_width / 2, cy - box_height / 2, cx + box_width / 2, cy + box_height / 2), axis=-1
        )
        boxes_np *= np.asarray([width, height, width, height], dtype=np.float32)
    masks = _resize_and_pad_masks(target.get("masks"), height, width, scene_height, scene_width)
    masks = masks if masks is not None and len(masks) == len(boxes_np) else None
    detections = _make_detections(boxes_np, np.ones(len(boxes_np), dtype=np.float32), labels_np, masks)
    label_strings = [f"GT {_class_label(int(class_id), class_names)}" for class_id in labels_np]

    keypoints_tensor = target.get("keypoints")
    keypoints = None
    if keypoints_tensor is not None and keypoints_tensor.numel() > 0:
        keypoints_np = keypoints_tensor.detach().cpu().numpy().astype(np.float32)
        keypoint_xy = keypoints_np[..., :2] * np.asarray([width, height], dtype=np.float32)
        visible = keypoints_np[..., 2] > 0
        keypoints = sv.KeyPoints(
            xy=keypoint_xy,
            keypoint_confidence=visible.astype(np.float32),
            class_id=labels_np,
            visible=visible,
        )
    return detections, keypoints, label_strings, len(boxes_np)


def _build_predictions(
    result: dict[str, Any],
    target: dict[str, Any],
    scene_shape: tuple[int, int],
    threshold: float,
    class_names: list[str] | None,
) -> tuple[Any | None, Any | None, list[str], int]:
    """Convert postprocessed predictions into the callback scene coordinate system."""
    import supervision as sv

    scene_height, scene_width = scene_shape
    scores = result.get("scores")
    boxes = result.get("boxes")
    labels = result.get("labels")
    if boxes is None or scores is None or labels is None:
        return None, None, [], 0
    scores_np = scores.detach().cpu().numpy().astype(np.float32)
    labels_np = labels.detach().cpu().numpy().astype(int)
    boxes_np = boxes.detach().cpu().numpy().astype(np.float32)
    keep = scores_np >= threshold
    scores_np, labels_np, boxes_np = scores_np[keep], labels_np[keep], boxes_np[keep]

    target_size = target.get("size")
    target_height, target_width = (
        (int(target_size[0]), int(target_size[1])) if target_size is not None else (scene_height, scene_width)
    )
    original_size = target.get("orig_size")
    original_height, original_width = (
        (int(original_size[0]), int(original_size[1])) if original_size is not None else (target_height, target_width)
    )
    scale_x = target_width / max(1, original_width)
    scale_y = target_height / max(1, original_height)
    boxes_np *= np.asarray([scale_x, scale_y, scale_x, scale_y], dtype=np.float32)

    masks_tensor = result.get("masks")
    masks_np = None
    if masks_tensor is not None:
        masks_np = _resize_and_pad_masks(
            masks_tensor[keep], target_height, target_width, scene_height, scene_width
        )
        if masks_np is not None and len(masks_np) != len(boxes_np):
            masks_np = None
    detections = _make_detections(boxes_np, scores_np, labels_np, masks_np)

    raw_keypoints = result.get("keypoints")
    keypoints = None
    if raw_keypoints is not None:
        keypoints_np = raw_keypoints.detach().cpu().numpy().astype(np.float32)[keep]
        keypoints_np[..., 0] *= scale_x
        keypoints_np[..., 1] *= scale_y
        confidence = keypoints_np[..., 2] if keypoints_np.shape[-1] > 2 else np.ones(keypoints_np.shape[:2])
        visible = confidence > 0
        keypoints = sv.KeyPoints(
            xy=keypoints_np[..., :2],
            keypoint_confidence=confidence,
            detection_confidence=scores_np,
            class_id=labels_np,
            visible=visible,
        )
    label_strings = [f"{_class_label(int(class_id), class_names)} {score:.2f}" for class_id, score in zip(labels_np, scores_np)]
    return detections, keypoints, label_strings, len(boxes_np)


class QualitativeVisualizerCallback(Callback):
    """Saves annotated prediction images to disk each validation and test epoch.

    Captures the first ``num_images`` images from the validation dataloader on the
    first batch, then at epoch-end runs them through the postprocessor, renders
    supervision overlays (boxes, masks, keypoints), and writes JPEG files to::

        output_dir/epoch_{epoch:04d}/{img_index:03d}.jpg   # validation
        output_dir/test/{img_index:03d}.jpg                 # test

    Only runs on rank 0 in distributed training.

    Args:
        output_dir:     Root directory for saved visualizations.
        num_images:     Maximum number of images to visualize per epoch.
        threshold:      Confidence score threshold for filtering detections.
        eval_interval:  Run visualization every N validation epochs (default: every epoch).
        class_names:    Optional list of class-name strings (0-indexed, used for labels).
    """

    def __init__(
        self,
        output_dir: str | Path,
        num_images: int = 8,
        threshold: float = 0.3,
        eval_interval: int = 1,
        class_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._output_dir = Path(output_dir)
        self._num_images = num_images
        self._threshold = threshold
        self._eval_interval = max(1, eval_interval)
        self._class_names = class_names

        # Accumulated from the first validation/test batch
        self._val_tensors: list[torch.Tensor] = []   # [C,H,W] cpu tensors
        self._val_results: list[dict[str, Any]] = []  # postprocessed per-image dicts
        self._val_targets: list[dict[str, Any]] = []
        self._val_batch_captured = False

        self._test_tensors: list[torch.Tensor] = []
        self._test_results: list[dict[str, Any]] = []
        self._test_targets: list[dict[str, Any]] = []
        self._test_batch_captured = False

    # ------------------------------------------------------------------
    # Reset per-epoch state
    # ------------------------------------------------------------------

    def on_validation_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        """Reset image accumulation buffers at the start of each validation epoch."""
        self._val_tensors = []
        self._val_results = []
        self._val_targets = []
        self._val_batch_captured = False

    def on_test_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        """Reset image accumulation buffers at the start of each test epoch."""
        self._test_tensors = []
        self._test_results = []
        self._test_targets = []
        self._test_batch_captured = False

    # ------------------------------------------------------------------
    # Batch hooks — capture image tensors and results from first batch
    # ------------------------------------------------------------------

    def on_validation_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Capture images and results from the first validation batch."""
        if self._val_batch_captured:
            return
        if not isinstance(outputs, Mapping):
            return
        self._val_batch_captured = True
        self._capture_batch(batch, outputs, self._val_tensors, self._val_results, self._val_targets)

    def on_test_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Capture images and results from the first test batch."""
        if self._test_batch_captured:
            return
        if not isinstance(outputs, Mapping):
            return
        self._test_batch_captured = True
        self._capture_batch(batch, outputs, self._test_tensors, self._test_results, self._test_targets)

    def _capture_batch(
        self,
        batch: Any,
        outputs: Mapping[str, Any],
        tensor_store: list[torch.Tensor],
        result_store: list[dict[str, Any]],
        target_store: list[dict[str, Any]],
    ) -> None:
        """Extract image tensors and postprocessed results from one batch.

        ``batch`` is a ``(NestedTensor, targets)`` pair; ``outputs`` is the
        ``{"results": [...], "targets": [...]}`` dict returned by the step.
        """
        try:
            samples, batch_targets = batch
            # Support both NestedTensor and plain tensors
            img_tensors: torch.Tensor = getattr(samples, "tensors", samples)
            results: list[dict[str, Any]] = outputs.get("results", [])
            targets: list[dict[str, Any]] = outputs.get("targets", batch_targets)

            n = min(self._num_images, img_tensors.shape[0], len(results), len(targets))
            for i in range(n):
                tensor_store.append(img_tensors[i].detach().cpu())
                # Move each result tensor to cpu
                result_cpu: dict[str, Any] = {}
                for k, v in results[i].items():
                    if isinstance(v, torch.Tensor):
                        result_cpu[k] = v.detach().cpu()
                    else:
                        result_cpu[k] = v
                result_store.append(result_cpu)
                target_cpu = {
                    key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                    for key, value in targets[i].items()
                }
                target_store.append(target_cpu)
        except Exception:
            logger.debug("QualitativeVisualizerCallback: failed to capture batch.\n%s", traceback.format_exc())

    # ------------------------------------------------------------------
    # Epoch-end hooks — render and save
    # ------------------------------------------------------------------

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        """Render and save validation visualizations at the end of each epoch."""
        if trainer.global_rank != 0:
            return
        epoch = trainer.current_epoch
        if epoch % self._eval_interval != 0:
            return
        if not self._val_tensors:
            return
        out_dir = self._output_dir / f"epoch_{epoch:04d}"
        self._save_visualizations(self._val_tensors, self._val_results, self._val_targets, out_dir)

    def on_test_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        """Render and save test visualizations at end of test epoch."""
        if trainer.global_rank != 0:
            return
        if not self._test_tensors:
            return
        out_dir = self._output_dir / "test"
        self._save_visualizations(self._test_tensors, self._test_results, self._test_targets, out_dir)

    def _save_visualizations(
        self,
        tensors: list[torch.Tensor],
        results: list[dict[str, Any]],
        targets: list[dict[str, Any]],
        out_dir: Path,
    ) -> None:
        """Render GT, prediction, and overlay panels as JPEG files in *out_dir*.

        Each image is written as ``{out_dir}/{img_index:03d}.jpg``.
        Errors in individual images are caught and logged so that a single
        bad frame does not crash training.
        """
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("QualitativeVisualizerCallback: cannot create %s: %s", out_dir, exc)
            return

        for idx, (tensor, result, target) in enumerate(zip(tensors, results, targets)):
            try:
                from PIL import Image as PILImage

                image = _tensor_to_uint8(tensor)
                scene_shape = image.shape[:2]
                gt_detections, gt_keypoints, gt_labels, gt_count = _build_ground_truth(
                    target, scene_shape, self._class_names
                )
                pred_detections, pred_keypoints, pred_labels, pred_count = _build_predictions(
                    result, target, scene_shape, self._threshold, self._class_names
                )

                gt_panel = _annotate_scene(
                    image.copy(), gt_detections, gt_keypoints, gt_labels,
                    _sv_color(_GT_COLOR), _sv_color(_GT_KEYPOINT_COLOR),
                )
                gt_panel = _add_title(gt_panel, f"Ground truth ({gt_count})")
                pred_panel = _annotate_scene(
                    image.copy(), pred_detections, pred_keypoints, pred_labels,
                    _sv_color(_PRED_COLOR), _sv_color(_PRED_KEYPOINT_COLOR),
                )
                pred_panel = _add_title(pred_panel, f"Predictions ({pred_count})")

                overlay = _annotate_scene(
                    image.copy(), gt_detections, gt_keypoints, gt_labels,
                    _sv_color(_GT_COLOR), _sv_color(_GT_KEYPOINT_COLOR), mask_opacity=0.3,
                )
                overlay = _annotate_scene(
                    overlay, pred_detections, pred_keypoints, pred_labels,
                    _sv_color(_PRED_COLOR), _sv_color(_PRED_KEYPOINT_COLOR), mask_opacity=0.3,
                )
                overlay = _add_title(overlay, "Overlay: GT box green/kp gold; prediction box red/kp cyan")
                panel = np.concatenate((gt_panel, pred_panel, overlay), axis=1)
                PILImage.fromarray(panel).save(out_dir / f"{idx:03d}.jpg")
            except Exception:
                logger.debug(
                    "QualitativeVisualizerCallback: failed to save image %d.\n%s",
                    idx,
                    traceback.format_exc(),
                )

        logger.info("QualitativeVisualizerCallback: saved %d comparison images to %s", len(tensors), out_dir)

