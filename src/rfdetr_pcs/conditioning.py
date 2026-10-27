"""Configuration helpers for class-token and visual PCS conditioning."""

from __future__ import annotations

import random
from typing import Any


VALID_CONDITIONING_MODES = {"class_embedding", "visual_prompt"}


def conditioning_schedule(cfg: dict[str, Any]) -> list[tuple[str, int]]:
    """Return the optional ordered ``(mode, steps)`` curriculum for discrete stages."""
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


def conditioning_sampling_schedule(cfg: dict[str, Any]) -> list[tuple[dict[str, float], int]]:
    """Return the optional sampling-based curriculum as list of ``({mode: prob}, steps)``.
    
    Each entry defines a phase where conditioning modes are sampled according to
    probability weights. For example:
    
        [
            {"class_embedding": 0.5, "visual_prompt": 0.5}, 10000
            {"class_embedding": 0.2, "visual_prompt": 0.8}, 5000
        ]
    
    means: sample 50% class_embedding and 50% visual_prompt for first 10000 steps,
    then sample 20% class_embedding and 80% visual_prompt for next 5000 steps.
    """
    raw = cfg.get("conditioning_sampling_schedule")
    if raw is None:
        return []
    if not isinstance(raw, list) or not raw:
        raise ValueError("conditioning_sampling_schedule must be a non-empty list of {prob_dict, steps} entries.")
    result: list[tuple[dict[str, float], int]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError("Each conditioning_sampling_schedule entry must be a mapping.")
        steps = int(entry.get("steps", 0))
        if steps <= 0:
            raise ValueError(f"Invalid conditioning_sampling_schedule entry: steps must be > 0, got {entry!r}")
        prob_dict = {k: float(v) for k, v in entry.items() if k != "steps"}
        if not prob_dict:
            raise ValueError(f"Invalid conditioning_sampling_schedule entry: no mode probabilities found in {entry!r}")
        for mode in prob_dict:
            if mode not in VALID_CONDITIONING_MODES:
                raise ValueError(f"Invalid mode {mode!r} in conditioning_sampling_schedule; must be one of {sorted(VALID_CONDITIONING_MODES)}")
        total = sum(prob_dict.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Probabilities in conditioning_sampling_schedule must sum to 1.0, got {total}")
        result.append((prob_dict, steps))
    return result


def conditioning_mode_at(cfg: dict[str, Any], step: int, rng: random.Random | None = None) -> str:
    """Resolve the zero-based optimizer step to its conditioning mode.
    
    Supports two curriculum modes:
    1. Discrete stages via ``conditioning_schedule``: each stage uses a single mode
    2. Sampling-based via ``conditioning_sampling_schedule``: each step samples a mode
       based on probability weights for that phase
    
    Args:
        cfg: Training configuration dictionary.
        step: Zero-based optimizer step.
        rng: Optional random number generator for reproducible sampling. If None,
            uses the global random state.
    
    Returns:
        The conditioning mode string for this step.
    
    Example (discrete stages):
        >>> cfg = {"conditioning_schedule": [{"mode": "class_embedding", "steps": 5000},
        ...                                    {"mode": "visual_prompt", "steps": 5000}]}
        >>> conditioning_mode_at(cfg, 0)
        'class_embedding'
        >>> conditioning_mode_at(cfg, 6000)
        'visual_prompt'
    
    Example (sampling-based):
        >>> cfg = {"conditioning_sampling_schedule": [
        ...     {"class_embedding": 0.5, "visual_prompt": 0.5, "steps": 10000},
        ...     {"class_embedding": 0.2, "visual_prompt": 0.8, "steps": 5000}]}
        >>> conditioning_mode_at(cfg, 0)  # Will sample either 'class_embedding' or 'visual_prompt'
    """
    # Try sampling-based schedule first
    sampling_schedule = conditioning_sampling_schedule(cfg)
    if sampling_schedule:
        return _sample_conditioning_mode(sampling_schedule, step, rng)
    
    # Fall back to discrete schedule
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


def _sample_conditioning_mode(
    schedule: list[tuple[dict[str, float], int]],
    step: int,
    rng: random.Random | None = None
) -> str:
    """Sample a conditioning mode based on the sampling schedule.
    
    Args:
        schedule: List of (probability_dict, steps) tuples.
        step: Current optimizer step.
        rng: Optional random number generator.
    
    Returns:
        Sampled conditioning mode string.
    """
    elapsed = 0
    for prob_dict, duration in schedule:
        elapsed += duration
        if step < elapsed:
            if rng is None:
                rng = random
            modes = list(prob_dict.keys())
            weights = list(prob_dict.values())
            return rng.choices(modes, weights=weights, k=1)[0]
    # Return the last phase's mode if step exceeds all durations
    return rng.choices(list(schedule[-1][0].keys()), weights=list(schedule[-1][0].values()), k=1)[0]


def uses_class_embeddings(cfg: dict[str, Any]) -> bool:
    """Whether any configured phase needs class-token parameters."""
    # Check sampling schedule
    sampling_schedule = conditioning_sampling_schedule(cfg)
    if sampling_schedule:
        return any("class_embedding" in prob_dict for prob_dict, _ in sampling_schedule)
    # Check discrete schedule
    schedule = conditioning_schedule(cfg)
    if schedule:
        return any(mode == "class_embedding" for mode, _ in schedule)
    return conditioning_mode_at(cfg, 0) == "class_embedding"


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
