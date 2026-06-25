"""Model registry: the mapping that makes the experiment runner config-driven.

A :class:`ModelSpec` bundles everything the model-agnostic
:func:`~js2024.modeling.walk_forward_suite.run_walk_forward_suite` needs to drive a
model from a YAML config alone: how to parse its typed config, how to build the
estimator, which feature columns it consumes, how to load/prepare its training
frame, and how to describe its protocol in the report.

New models are added by writing a :class:`ModelSpec` and registering it in
:data:`MODEL_REGISTRY` — no new runner script. The YAML's ``model:`` key selects
the spec; ``variants:`` selects which walk-forward variants to run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import polars as pl

from ..data.data import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    WEIGHT_COLUMN,
    get_default_columns,
    load_train_data,
    scan_train_data,
    validate_data_path,
)
from .config import (
    GRUConfig,
    GRUEvgeniavolkovaConfig,
    LGBMConfig,
    gru_params,
    gru_evgeniavolkova_params,
    load_gru_config,
    load_gru_evgeniavolkova_config,
    load_lgbm_config,
    resolve_project_path,
)
from .estimators import Estimator, GRUEstimator, LGBMEstimator
from .features import get_gru_feature_columns, get_v0_feature_columns
from .gru_evgeniavolkova import (
    GRUEvgeniavolkovaEstimator,
    add_gru_evgeniavolkova_aux_targets,
    get_gru_evgeniavolkova_feature_columns,
)


@dataclass(frozen=True)
class ModelSpec:
    """Everything the generic walk-forward suite needs to drive one model."""

    name: str
    title: str
    load_config: Callable[[str | Path], Any]
    feature_columns: Callable[[Any], list[str]]
    make_estimator: Callable[[Any, list[str]], Estimator]
    load_frame: Callable[[Any, list[str]], pl.DataFrame]
    describe: Callable[[Any], list[str]]


def _load_default_frame(config: Any, feature_cols: list[str]) -> pl.DataFrame:
    """Load ids + features + target/weight for the configured date range."""
    train_path = resolve_project_path(config.train_path)
    validate_data_path(train_path)
    columns = get_default_columns(include_target=True, include_weight=True)
    return load_train_data(
        train_path,
        columns=columns,
        start_date_id=config.start_date_id,
        end_date_id=config.end_date_id,
        collect=True,
    )


# --- GRU V0 ---------------------------------------------------------------


def _gru_features(config: GRUConfig) -> list[str]:
    return get_gru_feature_columns(include_time=config.include_time)


def _make_gru(config: GRUConfig, feature_cols: list[str]) -> GRUEstimator:
    return GRUEstimator(
        feature_cols=feature_cols,
        target_col=TARGET_COLUMN,
        weight_col=WEIGHT_COLUMN,
        params=gru_params(config),
        random_state=config.random_state,
        device=config.device,
    )


def _gru_describe(config: GRUConfig) -> list[str]:
    return [
        f"GRU: `seq_len={config.seq_len}`, `hidden_size={config.hidden_size}`, "
        f"`num_layers={config.num_layers}`, `dropout={config.dropout}`, "
        f"`epochs={config.epochs}`, `lr={config.lr}`; fine-tune "
        f"`finetune_epochs={config.finetune_epochs}` @ `lr={config.finetune_lr}`, "
        f"cadence={config.update_cadence}.",
        "inputs: the 79 raw `feature_*`"
        + (" + `time_id`" if config.include_time else "")
        + " (standardized, NaN→mean); `symbol_id` via per-symbol sequencing.",
    ]


# --- GRU evgeniavolkova (public-solution day-batch GRU) -------------------


def _evgeniavolkova_features(config: GRUEvgeniavolkovaConfig) -> list[str]:
    return get_gru_evgeniavolkova_feature_columns(include_time=config.include_time)


def _make_evgeniavolkova(config: GRUEvgeniavolkovaConfig, feature_cols: list[str]) -> GRUEvgeniavolkovaEstimator:
    return GRUEvgeniavolkovaEstimator(
        feature_cols=feature_cols,
        target_col=TARGET_COLUMN,
        weight_col=WEIGHT_COLUMN,
        params=gru_evgeniavolkova_params(config),
        random_state=config.random_state,
        device=config.device,
    )


def _load_evgeniavolkova_frame(config: GRUEvgeniavolkovaConfig, feature_cols: list[str]) -> pl.DataFrame:
    """Load features + the responder columns needed for the evgeniavolkova auxiliaries."""
    train_path = resolve_project_path(config.train_path)
    validate_data_path(train_path)
    available = set(scan_train_data(train_path).collect_schema().names())
    aux_source_cols = [c for c in ("responder_7", "responder_8") if c in available]
    columns = (
        ["date_id", "time_id", "symbol_id"]
        + list(FEATURE_COLUMNS)
        + [WEIGHT_COLUMN, TARGET_COLUMN]
        + aux_source_cols
    )
    df = load_train_data(
        train_path,
        columns=columns,
        start_date_id=config.start_date_id,
        end_date_id=config.end_date_id,
        collect=True,
    )
    for aux_col in ("responder_7", "responder_8"):
        if aux_col not in df.columns:
            print(
                f"[js2024] WARNING: {aux_col} missing; using {TARGET_COLUMN} "
                "as a smoke-only auxiliary placeholder."
            )
            df = df.with_columns(pl.col(TARGET_COLUMN).alias(aux_col))
    return add_gru_evgeniavolkova_aux_targets(df)


def _evgeniavolkova_describe(config: GRUEvgeniavolkovaConfig) -> list[str]:
    return [
        f"GRU (evgeniavolkova): `hidden_sizes={config.hidden_sizes}`, "
        f"`epochs={config.epochs}`, `lr={config.lr}`, `lr_refit={config.lr_refit}`, "
        f"cadence={config.update_cadence}.",
        "day-batch GRU with auxiliary responder heads (public-solution style).",
    ]


# --- LightGBM (static / online reference) ---------------------------------


def _lgbm_features(config: LGBMConfig) -> list[str]:
    return get_v0_feature_columns(include_symbol=True, include_time=True)


def _make_lgbm(config: LGBMConfig, feature_cols: list[str]) -> LGBMEstimator:
    return LGBMEstimator(
        feature_cols=feature_cols,
        target_col=TARGET_COLUMN,
        weight_col=WEIGHT_COLUMN,
        params={
            "n_estimators": config.n_estimators,
            "learning_rate": config.learning_rate,
            "num_leaves": config.num_leaves,
            "subsample": config.subsample,
            "colsample_bytree": config.colsample_bytree,
            "device_type": config.device_type,
            "max_bin": config.max_bin,
            "gpu_use_dp": config.gpu_use_dp,
            "random_state": config.random_state,
        },
        early_stopping_rounds=config.early_stopping_rounds,
        update_method=config.update_method,
        refit_decay=config.refit_decay,
        continue_rounds=config.continue_rounds,
    )


def _lgbm_describe(config: LGBMConfig) -> list[str]:
    return [
        f"LightGBM: `n_estimators={config.n_estimators}`, "
        f"`learning_rate={config.learning_rate}`, `num_leaves={config.num_leaves}`, "
        f"`device_type={config.device_type}`; online `update_method="
        f"{config.update_method!r}`, cadence={config.update_cadence}.",
        "inputs: the V0 raw feature set (`symbol_id` + `time_id` as columns).",
    ]


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "gru": ModelSpec(
        name="gru",
        title="GRU V0",
        load_config=load_gru_config,
        feature_columns=_gru_features,
        make_estimator=_make_gru,
        load_frame=_load_default_frame,
        describe=_gru_describe,
    ),
    "gru_evgeniavolkova": ModelSpec(
        name="gru_evgeniavolkova",
        title="GRU (evgeniavolkova)",
        load_config=load_gru_evgeniavolkova_config,
        feature_columns=_evgeniavolkova_features,
        make_estimator=_make_evgeniavolkova,
        load_frame=_load_evgeniavolkova_frame,
        describe=_evgeniavolkova_describe,
    ),
    "lgbm": ModelSpec(
        name="lgbm",
        title="LightGBM",
        load_config=load_lgbm_config,
        feature_columns=_lgbm_features,
        make_estimator=_make_lgbm,
        load_frame=_load_default_frame,
        describe=_lgbm_describe,
    ),
}


def get_model_spec(name: str) -> ModelSpec:
    """Return the :class:`ModelSpec` for ``name`` or raise a clear ``KeyError``."""
    try:
        return MODEL_REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"Unknown model {name!r}; registered models: {known}") from None
