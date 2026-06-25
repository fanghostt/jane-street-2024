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
    device_type: str = "cpu"
    max_bin: int = 255
    gpu_use_dp: bool = False
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
    if config.device_type not in {"cpu", "gpu", "cuda"}:
        errors.append(
            "device_type must be one of {'cpu', 'gpu', 'cuda'}, got "
            f"{config.device_type!r}"
        )
    if config.max_bin <= 1:
        errors.append(f"max_bin must be > 1, got {config.max_bin}")
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
        device_type=str(raw.get("device_type", "cpu")),
        max_bin=int(raw.get("max_bin", 255)),
        gpu_use_dp=bool(raw.get("gpu_use_dp", False)),
        test_days=int(raw.get("test_days", 200)),
        update_method=str(raw.get("update_method", "refit")),
        update_cadence=int(raw.get("update_cadence", 1)),
        refit_decay=float(raw.get("refit_decay", 0.9)),
        continue_rounds=int(raw.get("continue_rounds", 10)),
    )
    return validate_lgbm_config(config)


@dataclass
class GRUConfig:
    """Typed configuration for the GRU walk-forward experiment.

    Shares the data/split/test fields with :class:`LGBMConfig` so the GRU and
    LightGBM runners carve identical, comparable train/test blocks.
    """

    train_path: str
    output_dir: str
    model_dir: str
    start_date_id: int | None
    end_date_id: int | None
    valid_days: int
    gap_days: int
    random_state: int
    # Walk-forward.
    test_days: int = 200
    update_cadence: int = 1
    # Whether to include `time_id` as a model input alongside the 79 features.
    include_time: bool = False
    # "auto" uses CUDA when available, else CPU; or force "cpu"/"cuda".
    device: str = "auto"
    # GRU hyperparameters.
    seq_len: int = 16
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 20
    batch_size: int = 1024
    early_stopping_rounds: int = 5
    finetune_epochs: int = 1
    finetune_lr: float = 5e-4
    train_mode: str = "auto"
    max_materialized_windows_gib: float = 8.0


def validate_gru_config(config: GRUConfig) -> GRUConfig:
    """Validate GRU hyperparameter ranges; raise ``ValueError`` with a clear message."""
    errors: list[str] = []
    if config.valid_days <= 0:
        errors.append(f"valid_days must be > 0, got {config.valid_days}")
    if config.gap_days < 0:
        errors.append(f"gap_days must be >= 0, got {config.gap_days}")
    if config.test_days <= 0:
        errors.append(f"test_days must be > 0, got {config.test_days}")
    if config.update_cadence < 1:
        errors.append(f"update_cadence must be >= 1, got {config.update_cadence}")
    if config.seq_len < 1:
        errors.append(f"seq_len must be >= 1, got {config.seq_len}")
    if config.hidden_size <= 0:
        errors.append(f"hidden_size must be > 0, got {config.hidden_size}")
    if config.num_layers <= 0:
        errors.append(f"num_layers must be > 0, got {config.num_layers}")
    if not (0 <= config.dropout < 1):
        errors.append(f"dropout must be in [0, 1), got {config.dropout}")
    if config.lr <= 0:
        errors.append(f"lr must be > 0, got {config.lr}")
    if config.weight_decay < 0:
        errors.append(f"weight_decay must be >= 0, got {config.weight_decay}")
    if config.epochs <= 0:
        errors.append(f"epochs must be > 0, got {config.epochs}")
    if config.batch_size <= 0:
        errors.append(f"batch_size must be > 0, got {config.batch_size}")
    if config.early_stopping_rounds <= 0:
        errors.append(
            f"early_stopping_rounds must be > 0, got {config.early_stopping_rounds}"
        )
    if config.finetune_epochs <= 0:
        errors.append(f"finetune_epochs must be > 0, got {config.finetune_epochs}")
    if config.finetune_lr <= 0:
        errors.append(f"finetune_lr must be > 0, got {config.finetune_lr}")
    if config.train_mode not in {"auto", "materialize", "stream"}:
        errors.append(
            "train_mode must be one of {'auto', 'materialize', 'stream'}, got "
            f"{config.train_mode!r}"
        )
    if config.max_materialized_windows_gib <= 0:
        errors.append(
            "max_materialized_windows_gib must be > 0, got "
            f"{config.max_materialized_windows_gib}"
        )
    if errors:
        raise ValueError("Invalid GRU config:\n- " + "\n- ".join(errors))
    return config


def gru_params(config: GRUConfig) -> dict[str, Any]:
    """Extract the GRUEstimator ``params`` dict from a :class:`GRUConfig`."""
    return {
        "seq_len": config.seq_len,
        "hidden_size": config.hidden_size,
        "num_layers": config.num_layers,
        "dropout": config.dropout,
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "early_stopping_rounds": config.early_stopping_rounds,
        "finetune_epochs": config.finetune_epochs,
        "finetune_lr": config.finetune_lr,
        "train_mode": config.train_mode,
        "max_materialized_windows_gib": config.max_materialized_windows_gib,
    }


def load_gru_config(path: str | Path) -> GRUConfig:
    """Load and validate a YAML config into a :class:`GRUConfig`."""
    raw = load_yaml_config(path)
    config = GRUConfig(
        train_path=raw["train_path"],
        output_dir=raw["output_dir"],
        model_dir=raw["model_dir"],
        start_date_id=raw.get("start_date_id"),
        end_date_id=raw.get("end_date_id"),
        valid_days=int(raw["valid_days"]),
        gap_days=int(raw.get("gap_days", 0)),
        random_state=int(raw.get("random_state", 42)),
        test_days=int(raw.get("test_days", 200)),
        update_cadence=int(raw.get("update_cadence", 1)),
        include_time=bool(raw.get("include_time", False)),
        device=str(raw.get("device", "auto")),
        seq_len=int(raw.get("seq_len", 16)),
        hidden_size=int(raw.get("hidden_size", 64)),
        num_layers=int(raw.get("num_layers", 1)),
        dropout=float(raw.get("dropout", 0.0)),
        lr=float(raw.get("lr", 1e-3)),
        weight_decay=float(raw.get("weight_decay", 0.0)),
        epochs=int(raw.get("epochs", 20)),
        batch_size=int(raw.get("batch_size", 1024)),
        early_stopping_rounds=int(raw.get("early_stopping_rounds", 5)),
        finetune_epochs=int(raw.get("finetune_epochs", 1)),
        finetune_lr=float(raw.get("finetune_lr", 5e-4)),
        train_mode=str(raw.get("train_mode", "auto")),
        max_materialized_windows_gib=float(raw.get("max_materialized_windows_gib", 8.0)),
    )
    return validate_gru_config(config)


@dataclass
class GRUEvgeniavolkovaConfig:
    """Typed configuration for the evgeniavolkova-style day-batch GRU."""

    train_path: str
    start_date_id: int | None
    end_date_id: int | None
    valid_days: int
    gap_days: int
    random_state: int
    test_days: int = 200
    update_cadence: int = 1
    include_time: bool = True
    device: str = "auto"
    hidden_sizes: list[int] | None = None
    dropout_rates: list[float] | None = None
    hidden_sizes_linear: list[int] | None = None
    dropout_rates_linear: list[float] | None = None
    lr: float = 5e-4
    lr_refit: float = 3e-4
    epochs: int = 1000
    early_stopping_patience: int = 1
    weight_decay: float = 0.01
    grad_clip: float = 1.0


def validate_gru_evgeniavolkova_config(config: GRUEvgeniavolkovaConfig) -> GRUEvgeniavolkovaConfig:
    """Validate GRU evgeniavolkova config ranges and list lengths."""
    errors: list[str] = []
    hidden_sizes = config.hidden_sizes or [500]
    dropout_rates = config.dropout_rates or [0.3]
    hidden_linear = config.hidden_sizes_linear or [500, 300]
    dropout_linear = config.dropout_rates_linear or [0.2, 0.1]

    if config.valid_days <= 0:
        errors.append(f"valid_days must be > 0, got {config.valid_days}")
    if config.gap_days < 0:
        errors.append(f"gap_days must be >= 0, got {config.gap_days}")
    if config.test_days <= 0:
        errors.append(f"test_days must be > 0, got {config.test_days}")
    if config.update_cadence < 1:
        errors.append(f"update_cadence must be >= 1, got {config.update_cadence}")
    if not hidden_sizes or any(h <= 0 for h in hidden_sizes):
        errors.append(f"hidden_sizes must be positive, got {hidden_sizes}")
    if len(dropout_rates) != len(hidden_sizes):
        errors.append("dropout_rates length must match hidden_sizes")
    if any(not (0 <= d < 1) for d in dropout_rates):
        errors.append(f"dropout_rates must be in [0, 1), got {dropout_rates}")
    if not hidden_linear or any(h <= 0 for h in hidden_linear):
        errors.append(f"hidden_sizes_linear must be positive, got {hidden_linear}")
    if len(dropout_linear) != len(hidden_linear):
        errors.append("dropout_rates_linear length must match hidden_sizes_linear")
    if any(not (0 <= d < 1) for d in dropout_linear):
        errors.append(
            f"dropout_rates_linear must be in [0, 1), got {dropout_linear}"
        )
    if config.lr <= 0:
        errors.append(f"lr must be > 0, got {config.lr}")
    if config.lr_refit < 0:
        errors.append(f"lr_refit must be >= 0, got {config.lr_refit}")
    if config.epochs <= 0:
        errors.append(f"epochs must be > 0, got {config.epochs}")
    if config.early_stopping_patience < 0:
        errors.append(
            "early_stopping_patience must be >= 0, got "
            f"{config.early_stopping_patience}"
        )
    if config.weight_decay < 0:
        errors.append(f"weight_decay must be >= 0, got {config.weight_decay}")
    if config.grad_clip <= 0:
        errors.append(f"grad_clip must be > 0, got {config.grad_clip}")
    if errors:
        raise ValueError("Invalid GRU evgeniavolkova config:\n- " + "\n- ".join(errors))
    return config


def gru_evgeniavolkova_params(config: GRUEvgeniavolkovaConfig) -> dict[str, Any]:
    """Extract estimator params from :class:`GRUEvgeniavolkovaConfig`."""
    return {
        "hidden_sizes": list(config.hidden_sizes or [500]),
        "dropout_rates": list(config.dropout_rates or [0.3]),
        "hidden_sizes_linear": list(config.hidden_sizes_linear or [500, 300]),
        "dropout_rates_linear": list(config.dropout_rates_linear or [0.2, 0.1]),
        "lr": config.lr,
        "lr_refit": config.lr_refit,
        "epochs": config.epochs,
        "early_stopping_patience": config.early_stopping_patience,
        "weight_decay": config.weight_decay,
        "grad_clip": config.grad_clip,
    }


def load_gru_evgeniavolkova_config(path: str | Path) -> GRUEvgeniavolkovaConfig:
    """Load and validate a YAML config into :class:`GRUEvgeniavolkovaConfig`."""
    raw = load_yaml_config(path)
    config = GRUEvgeniavolkovaConfig(
        train_path=raw["train_path"],
        start_date_id=raw.get("start_date_id"),
        end_date_id=raw.get("end_date_id"),
        valid_days=int(raw.get("valid_days", 200)),
        gap_days=int(raw.get("gap_days", 0)),
        random_state=int(raw.get("random_state", 42)),
        test_days=int(raw.get("test_days", 200)),
        update_cadence=int(raw.get("update_cadence", 1)),
        include_time=bool(raw.get("include_time", True)),
        device=str(raw.get("device", "auto")),
        hidden_sizes=list(raw.get("hidden_sizes", [500])),
        dropout_rates=list(raw.get("dropout_rates", [0.3])),
        hidden_sizes_linear=list(raw.get("hidden_sizes_linear", [500, 300])),
        dropout_rates_linear=list(raw.get("dropout_rates_linear", [0.2, 0.1])),
        lr=float(raw.get("lr", 5e-4)),
        lr_refit=float(raw.get("lr_refit", 3e-4)),
        epochs=int(raw.get("epochs", 1000)),
        early_stopping_patience=int(raw.get("early_stopping_patience", 1)),
        weight_decay=float(raw.get("weight_decay", 0.01)),
        grad_clip=float(raw.get("grad_clip", 1.0)),
    )
    return validate_gru_evgeniavolkova_config(config)
