"""Load Python ``cfg`` dictionaries into argparse defaults."""

from __future__ import annotations

import argparse
import runpy
from pathlib import Path
from typing import Any


def load_python_config(config_path: Path, parser: argparse.ArgumentParser) -> dict[str, Any]:
    """Execute a Python config and validate/coerce its ``cfg`` dictionary."""
    namespace = runpy.run_path(str(config_path))
    cfg = namespace.get("cfg")
    if not isinstance(cfg, dict):
        raise ValueError(f"Python config {config_path} must define a dict named 'cfg'.")

    actions = {action.dest: action for action in parser._actions}
    unknown = set(cfg) - actions.keys()
    if unknown:
        raise ValueError(f"Unknown config keys in {config_path}: {sorted(unknown)}")

    typed_cfg: dict[str, Any] = {}
    for key, value in cfg.items():
        action = actions[key]
        if isinstance(value, str) and action.type is not None:
            value = action.type(value)
        typed_cfg[key] = value
    return typed_cfg


def parse_args_with_python_config(
    parser: argparse.ArgumentParser,
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """Load ``--config`` defaults first, then parse CLI overrides."""
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    config_args, _ = config_parser.parse_known_args(argv)
    if config_args.config is not None:
        try:
            parser.set_defaults(**load_python_config(config_args.config, parser))
        except (OSError, SyntaxError, TypeError, ValueError) as exc:
            parser.error(str(exc))
    return parser.parse_args(argv)
