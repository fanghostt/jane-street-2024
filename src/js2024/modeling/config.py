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
    # When True, reconstruct day-lagged responders (responder_i_lag_1) from the
    # train frame and add them to the model inputs. Defaulted False so existing
    # V0 configs are unaffected. See js2024.modeling.lag_features.
    use_responder_lags: bool = False
    # Walk-forward (incremental vs full) options; defaulted so existing YAMLs
    # and the other runners are unaffected.
    test_days: int = 200
    update_method: str = "refit"
    # Second-level online taxonomy: when set, the `incremental` variant expands
    # into one labelled run per method (e.g. `lgbm_refit`/`lgbm_continue`/
    # `lgbm_retrain`) instead of a single generic `incremental` row. Lets one
    # config compare every online analog side-by-side while the walk-forward
    # variant stays model-agnostic (full/incremental). When None, the single
    # `update_method` above is used.
    update_methods: list[str] | None = None
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
    if config.update_methods is not None:
        if not config.update_methods:
            errors.append("update_methods, when set, must be a non-empty list")
        bad_methods = [
            m for m in config.update_methods if m not in {"refit", "continue", "retrain"}
        ]
        if bad_methods:
            errors.append(
                "update_methods must each be one of {'refit', 'continue', 'retrain'}, "
                f"got {bad_methods}"
            )
        if len(set(config.update_methods)) != len(config.update_methods):
            errors.append(f"update_methods has duplicates: {config.update_methods}")
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
        use_responder_lags=bool(raw.get("use_responder_lags", False)),
        test_days=int(raw.get("test_days", 200)),
        update_method=str(raw.get("update_method", "refit")),
        update_methods=(
            [str(m) for m in raw["update_methods"]]
            if raw.get("update_methods") is not None
            else None
        ),
        update_cadence=int(raw.get("update_cadence", 1)),
        refit_decay=float(raw.get("refit_decay", 0.9)),
        continue_rounds=int(raw.get("continue_rounds", 10)),
    )
    return validate_lgbm_config(config)


@dataclass
class GRUConfig:
    """Typed configuration for the day-batch GRU."""

    train_path: str
    start_date_id: int | None
    end_date_id: int | None
    valid_days: int
    gap_days: int
    random_state: int
    test_days: int = 200
    update_cadence: int = 1
    include_time: bool = True
    # When True, add reconstructed day-lagged responders (responder_i_lag_1) to
    # the GRU inputs. Defaulted False so existing GRU configs are unaffected.
    use_responder_lags: bool = False
    device: str = "auto"
    # Sequence backbone: gru/lstm (recurrent), transformer (causal attention),
    # tcn (causal dilated conv). All share the day-batch + aux-head protocol.
    model_type: str = "gru"
    num_heads: int = 5      # transformer only; each hidden_size must divide evenly.
    kernel_size: int = 3    # tcn only; causal conv kernel width.
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
    # Optional Weights & Biases tracking. Off by default; when True the suite
    # opens one run per variant and the training loop logs per-epoch metrics.
    # See js2024.modeling.tracking (offline-safe when unauthenticated).
    use_wandb: bool = False
    wandb_project: str = "js2024"
    # When True, run the day-batch forward/backward under bf16 autocast (+TF32)
    # on CUDA. ~1.5x faster on Blackwell GPUs; slightly changes numerics, so it
    # is off by default to keep recorded baselines bit-reproducible.
    use_amp: bool = False


def validate_gru_config(config: GRUConfig) -> GRUConfig:
    """Validate GRU config ranges and list lengths."""
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
    seq_types = {"gru", "lstm", "transformer", "tcn"}
    if config.model_type not in seq_types:
        errors.append(
            f"model_type must be one of {sorted(seq_types)}, got {config.model_type!r}"
        )
    if config.num_heads <= 0:
        errors.append(f"num_heads must be > 0, got {config.num_heads}")
    if config.model_type == "transformer":
        bad = [h for h in hidden_sizes if h % config.num_heads != 0]
        if bad:
            errors.append(
                f"transformer requires each hidden_size divisible by num_heads="
                f"{config.num_heads}; offending sizes: {bad}"
            )
    if config.kernel_size <= 0:
        errors.append(f"kernel_size must be > 0, got {config.kernel_size}")
    if errors:
        raise ValueError("Invalid GRU config:\n- " + "\n- ".join(errors))
    return config


def gru_params(config: GRUConfig) -> dict[str, Any]:
    """Extract estimator params from :class:`GRUConfig`."""
    return {
        "model_type": config.model_type,
        "num_heads": config.num_heads,
        "kernel_size": config.kernel_size,
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
        "use_amp": config.use_amp,
    }


def load_gru_config(path: str | Path) -> GRUConfig:
    """Load and validate a YAML config into :class:`GRUConfig`."""
    raw = load_yaml_config(path)
    config = GRUConfig(
        train_path=raw["train_path"],
        start_date_id=raw.get("start_date_id"),
        end_date_id=raw.get("end_date_id"),
        valid_days=int(raw.get("valid_days", 200)),
        gap_days=int(raw.get("gap_days", 0)),
        random_state=int(raw.get("random_state", 42)),
        test_days=int(raw.get("test_days", 200)),
        update_cadence=int(raw.get("update_cadence", 1)),
        include_time=bool(raw.get("include_time", True)),
        use_responder_lags=bool(raw.get("use_responder_lags", False)),
        device=str(raw.get("device", "auto")),
        model_type=str(raw.get("model_type", "gru")),
        num_heads=int(raw.get("num_heads", 5)),
        kernel_size=int(raw.get("kernel_size", 3)),
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
        use_wandb=bool(raw.get("use_wandb", False)),
        wandb_project=str(raw.get("wandb_project", "js2024")),
        use_amp=bool(raw.get("use_amp", False)),
    )
    return validate_gru_config(config)
