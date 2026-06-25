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

# src/js2024/modeling/config.py -> [0]=modeling, [1]=js2024, [2]=src, [3]=project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]


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
    # Walk-forward (incremental vs full) options; defaulted so existing YAMLs
    # and the other runners are unaffected.
    test_days: int = 200
    update_method: str = "refit"
    update_cadence: int = 1
    refit_decay: float = 0.9
    continue_rounds: int = 10


def validate_lgbm_config(config: LGBMConfig) -> LGBMConfig:
    """Validate hyperparameter ranges; raise ``ValueError`` with a clear message.

    Returns ``config`` unchanged so it can be used inline.
    """
    errors: list[str] = []
    if config.valid_days <= 0:
        errors.append(f"valid_days must be > 0, got {config.valid_days}")
    if config.gap_days < 0:
        errors.append(f"gap_days must be >= 0, got {config.gap_days}")
    if config.n_estimators <= 0:
        errors.append(f"n_estimators must be > 0, got {config.n_estimators}")
    if config.early_stopping_rounds <= 0:
        errors.append(
            f"early_stopping_rounds must be > 0, got {config.early_stopping_rounds}"
        )
    if config.learning_rate <= 0:
        errors.append(f"learning_rate must be > 0, got {config.learning_rate}")
    if config.num_leaves <= 1:
        errors.append(f"num_leaves must be > 1, got {config.num_leaves}")
    if not (0 < config.subsample <= 1):
        errors.append(f"subsample must be in (0, 1], got {config.subsample}")
    if not (0 < config.colsample_bytree <= 1):
        errors.append(
            f"colsample_bytree must be in (0, 1], got {config.colsample_bytree}"
        )
    if config.test_days <= 0:
        errors.append(f"test_days must be > 0, got {config.test_days}")
    if config.update_cadence < 1:
        errors.append(f"update_cadence must be >= 1, got {config.update_cadence}")
    if config.update_method not in {"refit", "continue", "retrain"}:
        errors.append(
            "update_method must be one of {'refit', 'continue', 'retrain'}, got "
            f"{config.update_method!r}"
        )
    if not (0 <= config.refit_decay <= 1):
        errors.append(f"refit_decay must be in [0, 1], got {config.refit_decay}")
    if config.continue_rounds <= 0:
        errors.append(f"continue_rounds must be > 0, got {config.continue_rounds}")
    if errors:
        raise ValueError("Invalid LGBM config:\n- " + "\n- ".join(errors))
    return config


def load_lgbm_config(path: str | Path) -> LGBMConfig:
    """Load and validate a YAML config into an :class:`LGBMConfig`."""
    raw = load_yaml_config(path)
    config = LGBMConfig(
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
        test_days=int(raw.get("test_days", 200)),
        update_method=str(raw.get("update_method", "refit")),
        update_cadence=int(raw.get("update_cadence", 1)),
        refit_decay=float(raw.get("refit_decay", 0.9)),
        continue_rounds=int(raw.get("continue_rounds", 10)),
    )
    return validate_lgbm_config(config)
