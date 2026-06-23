"""Configuration loading for the V0 LightGBM baseline.

Paths in YAML are resolved relative to the project root (the directory two
levels above this file: ``src/js2024/config.py`` -> project root). Absolute paths
are passed through unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# src/js2024/config.py -> parents[0]=js2024, [1]=src, [2]=project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dict."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {p} must be a mapping, got {type(data).__name__}")
    return data


def resolve_project_path(path_str: str | Path) -> Path:
    """Resolve a possibly-relative path against the project root."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


@dataclass
class LGBMConfig:
    """Typed configuration for the LightGBM V0 baseline."""

    train_path: str
    output_dir: str
    model_dir: str
    start_date_id: int | None
    end_date_id: int | None
    valid_days: int
    gap_days: int
    random_state: int
    n_estimators: int
    learning_rate: float
    num_leaves: int
    subsample: float
    colsample_bytree: float
    early_stopping_rounds: int


def load_lgbm_config(path: str | Path) -> LGBMConfig:
    """Load and validate a YAML config into an :class:`LGBMConfig`."""
    raw = load_yaml_config(path)
    return LGBMConfig(
        train_path=raw["train_path"],
        output_dir=raw["output_dir"],
        model_dir=raw["model_dir"],
        start_date_id=raw.get("start_date_id"),
        end_date_id=raw.get("end_date_id"),
        valid_days=int(raw["valid_days"]),
        gap_days=int(raw.get("gap_days", 0)),
        random_state=int(raw.get("random_state", 42)),
        n_estimators=int(raw.get("n_estimators", 3000)),
        learning_rate=float(raw.get("learning_rate", 0.03)),
        num_leaves=int(raw.get("num_leaves", 64)),
        subsample=float(raw.get("subsample", 0.8)),
        colsample_bytree=float(raw.get("colsample_bytree", 0.8)),
        early_stopping_rounds=int(raw.get("early_stopping_rounds", 100)),
    )
