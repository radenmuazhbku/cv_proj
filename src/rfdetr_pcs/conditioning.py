"""Configuration helpers for class-token and visual PCS conditioning."""

from __future__ import annotations

from typing import Any


VALID_CONDITIONING_MODES = {"class_embedding", "visual_prompt"}


def conditioning_schedule(cfg: dict[str, Any]) -> list[tuple[str, int]]:
    """Return the optional ordered ``(mode, steps)`` curriculum."""
    raw = cfg.get("conditioning_schedule")
    if raw is None:
        return []
    if not isinstance(raw, list) or not raw:
        raise ValueError("conditioning_schedule must be a non-empty list of {mode, steps} entries.")
    result: list[tuple[str, int]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError("Each conditioning_schedule entry must be a mapping.")
        mode, steps = str(entry.get("mode", "")), int(entry.get("steps", 0))
        if mode not in VALID_CONDITIONING_MODES or steps <= 0:
            raise ValueError(f"Invalid conditioning schedule entry: {entry!r}")
        result.append((mode, steps))
    return result


def conditioning_mode_at(cfg: dict[str, Any], step: int) -> str:
    """Resolve the zero-based optimizer step to its conditioning mode."""
    schedule = conditioning_schedule(cfg)
    if schedule:
        elapsed = 0
        for mode, duration in schedule:
            elapsed += duration
            if step < elapsed:
                return mode
        return schedule[-1][0]
    # Preserve the pre-existing trainer behavior for configs that predate the
    # explicit mode switch. New PCS-only configs set visual_prompt explicitly.
    mode = str(cfg.get("conditioning_mode", "class_embedding"))
    if mode not in VALID_CONDITIONING_MODES:
        raise ValueError(f"conditioning_mode must be one of {sorted(VALID_CONDITIONING_MODES)}, got {mode!r}")
    return mode


def uses_class_embeddings(cfg: dict[str, Any]) -> bool:
    """Whether any configured phase needs class-token parameters."""
    schedule = conditioning_schedule(cfg)
    if schedule:
        return any(mode == "class_embedding" for mode, _ in schedule)
    return conditioning_mode_at(cfg, 0) == "class_embedding"


def class_ids_for_mode(batch: dict[str, Any], device, mode: str):
    """Get episode category IDs only for the known-class conditioning path."""
    if mode == "visual_prompt":
        return None
    import torch

    return torch.as_tensor(batch["category_ids"], dtype=torch.long, device=device)
