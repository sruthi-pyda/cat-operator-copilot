"""Loads config/settings.yaml for every feature.

Additive shared helper: it introduces no new schema and changes no existing
field name, so it does not alter the API contract (see docs/DECISIONS.md D007).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"
SAFETY_RULES_PATH = PROJECT_ROOT / "config" / "safety_rules.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_settings(path: Optional[Path] = None) -> dict[str, Any]:
    return load_yaml(path or SETTINGS_PATH)


def resolve_path(value: str) -> Path:
    """Resolve a settings path against the project root unless already absolute."""
    candidate = Path(value)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
