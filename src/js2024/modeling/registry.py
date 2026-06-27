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
    LGBMConfig,
    gru_params,
    load_gru_config,
    load_lgbm_config,
    resolve_project_path,
)
from .estimators import Estimator, LGBMEstimator
from .features import get_v0_feature_columns
from .gru import (
    GRUEstimator,
    add_gru_aux_targets,
    get_gru_feature_columns,
    resolve_gru_aux_targets,
)
from .lag_features import (
    RESPONDER_COLUMNS,
    add_responder_lags_from_train,
    get_lag_feature_columns,
)
from .market_features import (
    add_engineered_features,
    resolve_market_roll_features,
    selected_columns,
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
    # Optional hook expanding the `incremental` variant into several model-specific
    # sub-runs, each ``(sub_label, estimator)``. Used by LightGBM to surface its
    # online taxonomy (refit/continue/retrain) as distinct, named rows rather than
    # a single generic `incremental`. ``None`` -> one `incremental` run via
    # ``make_estimator``.
    incremental_runs: Callable[[Any, list[str]], list[tuple[str, Estimator]]] | None = None


# --- GRU (day-batch, public-solution style) -------------------------------


def _gru_features(config: GRUConfig) -> list[str]:
    cols = get_gru_feature_columns(include_time=config.include_time)
    if config.use_responder_lags:
        cols = cols + get_lag_feature_columns()
    cols = cols + selected_columns(
        use_market_avg=config.use_market_avg,
        use_symbol_rolling=config.use_symbol_rolling,
        features=resolve_market_roll_features(config.market_roll_subset),
    )
    return cols


def _make_seq(
    config: GRUConfig, feature_cols: list[str], *, model_type: str | None = None
) -> GRUEstimator:
    """Build a day-batch sequence estimator, optionally forcing the backbone.

    ``model_type=None`` honours ``config.model_type`` (gru spec, where lstm is also
    reachable); the transformer/tcn specs pass an explicit backbone that wins.
    """
    params = gru_params(config)
    if model_type is not None:
        params["model_type"] = model_type
    return GRUEstimator(
        feature_cols=feature_cols,
        target_col=TARGET_COLUMN,
        weight_col=WEIGHT_COLUMN,
        params=params,
        aux_cols=resolve_gru_aux_targets(config.aux_target_set),
        random_state=config.random_state,
        device=config.device,
    )


def _make_gru(config: GRUConfig, feature_cols: list[str]) -> GRUEstimator:
    return _make_seq(config, feature_cols)


def _load_gru_frame(config: GRUConfig, feature_cols: list[str]) -> pl.DataFrame:
    """Load features + the responder columns needed for the GRU auxiliaries."""
    train_path = resolve_project_path(config.train_path)
    validate_data_path(train_path)
    available = set(scan_train_data(train_path).collect_schema().names())
    # Real source responders to load: those the chosen aux set references (synthetic
    # responder_9/10 are generated below, not loaded) plus responder_7/8 which
    # add_gru_aux_targets always needs. Order is stable for a deterministic request.
    aux_targets = resolve_gru_aux_targets(config.aux_target_set)
    needed_responders = {c for c in aux_targets if c in RESPONDER_COLUMNS}
    needed_responders.update({"responder_7", "responder_8"})
    needed_responders.discard(TARGET_COLUMN)  # responder_6 is already loaded as the target
    aux_source_cols = [c for c in RESPONDER_COLUMNS if c in needed_responders and c in available]
    # Lag reconstruction needs all 9 responders; merge with the aux sources and
    # the target (responder_6, already requested) so we only request each once.
    already_requested = set(aux_source_cols) | {TARGET_COLUMN}
    lag_source_cols = (
        [c for c in RESPONDER_COLUMNS if c in available and c not in already_requested]
        if config.use_responder_lags
        else []
    )
    columns = (
        ["date_id", "time_id", "symbol_id"]
        + list(FEATURE_COLUMNS)
        + [WEIGHT_COLUMN, TARGET_COLUMN]
        + aux_source_cols
        + lag_source_cols
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
    df = add_gru_aux_targets(df)
    if config.use_responder_lags:
        # Add responder_i_lag_1 features (D-1 responders) after the aux targets so
        # the shift-based auxiliaries are computed on the raw responder columns.
        df = add_responder_lags_from_train(df)
    # Engineered market-avg / per-symbol rolling features (leakage-safe; source
    # columns are part of the standard feature set already loaded).
    df = add_engineered_features(
        df,
        use_market_avg=config.use_market_avg,
        use_symbol_rolling=config.use_symbol_rolling,
        window=config.rolling_window,
        features=resolve_market_roll_features(config.market_roll_subset),
    )
    return df


def _make_transformer(config: GRUConfig, feature_cols: list[str]) -> GRUEstimator:
    return _make_seq(config, feature_cols, model_type="transformer")


def _make_tcn(config: GRUConfig, feature_cols: list[str]) -> GRUEstimator:
    return _make_seq(config, feature_cols, model_type="tcn")


def _seq_describe(config: GRUConfig, backbone: str, extra: str) -> list[str]:
    return [
        f"{backbone} (day-batch): `hidden_sizes={config.hidden_sizes}`, "
        f"`epochs={config.epochs}`, `lr={config.lr}`, `lr_refit={config.lr_refit}`, "
        f"cadence={config.update_cadence}{extra}.",
        f"day-batch {backbone} with auxiliary responder heads (public-solution style).",
    ]


def _gru_describe(config: GRUConfig) -> list[str]:
    return _seq_describe(config, config.model_type.upper(), f", `aux_target_set={config.aux_target_set}`")


def _transformer_describe(config: GRUConfig) -> list[str]:
    return _seq_describe(config, "Transformer", f", `num_heads={config.num_heads}`")


def _tcn_describe(config: GRUConfig) -> list[str]:
    return _seq_describe(config, "TCN", f", `kernel_size={config.kernel_size}`")


# --- LightGBM (static / online reference) ---------------------------------


def _lgbm_features(config: LGBMConfig) -> list[str]:
    cols = get_v0_feature_columns(include_symbol=True, include_time=True)
    if config.use_responder_lags:
        cols = cols + get_lag_feature_columns()
    cols = cols + selected_columns(
        use_market_avg=config.use_market_avg,
        use_symbol_rolling=config.use_symbol_rolling,
        features=resolve_market_roll_features(config.market_roll_subset),
    )
    return cols


def _load_lgbm_frame(config: LGBMConfig, feature_cols: list[str]) -> pl.DataFrame:
    """Load ids + features + target/weight and apply the engineered features.

    Mirrors :func:`js2024.modeling.train_lgbm` so the unified
    ``js2024-run-experiment`` path consumes the same lag / market-avg /
    per-symbol-rolling columns as the standalone ``js2024-train-lgbm`` CLI.
    """
    train_path = resolve_project_path(config.train_path)
    validate_data_path(train_path)
    columns = get_default_columns(include_target=True, include_weight=True)
    if config.use_responder_lags:
        # Need responder_0..8 to reconstruct the lags; de-dup since responder_6
        # is already the target. The lag features (not the raw responders) are the
        # model inputs.
        columns = columns + [c for c in RESPONDER_COLUMNS if c not in columns]
    df = load_train_data(
        train_path,
        columns=columns,
        start_date_id=config.start_date_id,
        end_date_id=config.end_date_id,
        collect=True,
    )
    if config.use_responder_lags:
        # Reconstruct D-1 responders as responder_i_lag_1 features (leakage-safe:
        # the first date_id gets null lags). Done before the suite splits folds.
        df = add_responder_lags_from_train(df)
    # Engineered market-avg / per-symbol rolling features (leakage-safe; source
    # columns are part of the standard feature set already loaded).
    df = add_engineered_features(
        df,
        use_market_avg=config.use_market_avg,
        use_symbol_rolling=config.use_symbol_rolling,
        window=config.rolling_window,
        features=resolve_market_roll_features(config.market_roll_subset),
    )
    return df


def _make_lgbm(
    config: LGBMConfig, feature_cols: list[str], *, update_method: str | None = None
) -> LGBMEstimator:
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
        update_method=update_method or config.update_method,
        refit_decay=config.refit_decay,
        continue_rounds=config.continue_rounds,
    )


def _lgbm_online_methods(config: LGBMConfig) -> list[str]:
    """The online strategies the `incremental` variant expands into for LightGBM."""
    return config.update_methods or [config.update_method]


def _lgbm_incremental_runs(
    config: LGBMConfig, feature_cols: list[str]
) -> list[tuple[str, LGBMEstimator]]:
    """One ``(method, estimator)`` per configured online strategy.

    The sub-label is the bare method name so the suite renders it as
    ``lgbm_refit``/``lgbm_continue``/``lgbm_retrain`` instead of ``lgbm_incremental``.
    """
    return [(m, _make_lgbm(config, feature_cols, update_method=m)) for m in _lgbm_online_methods(config)]


def _lgbm_describe(config: LGBMConfig) -> list[str]:
    methods = _lgbm_online_methods(config)
    online = (
        f"online methods `{methods}`" if config.update_methods is not None
        else f"online `update_method={config.update_method!r}`"
    )
    eng = []
    if config.use_responder_lags:
        eng.append("D-1 responder lags")
    if config.use_market_avg:
        eng.append("cross-sectional market-avg")
    if config.use_symbol_rolling:
        eng.append(f"per-symbol rolling (window={config.rolling_window})")
    inputs = "the V0 raw feature set (`symbol_id` + `time_id` as columns)"
    if eng:
        inputs += " + engineered: " + ", ".join(eng)
    return [
        f"LightGBM: `n_estimators={config.n_estimators}`, "
        f"`learning_rate={config.learning_rate}`, `num_leaves={config.num_leaves}`, "
        f"`device_type={config.device_type}`; {online}, cadence={config.update_cadence}.",
        f"inputs: {inputs}.",
    ]


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "gru": ModelSpec(
        name="gru",
        title="GRU (day-batch)",
        load_config=load_gru_config,
        feature_columns=_gru_features,
        make_estimator=_make_gru,
        load_frame=_load_gru_frame,
        describe=_gru_describe,
    ),
    "transformer": ModelSpec(
        name="transformer",
        title="Transformer (day-batch)",
        load_config=load_gru_config,
        feature_columns=_gru_features,
        make_estimator=_make_transformer,
        load_frame=_load_gru_frame,
        describe=_transformer_describe,
    ),
    "tcn": ModelSpec(
        name="tcn",
        title="TCN (day-batch)",
        load_config=load_gru_config,
        feature_columns=_gru_features,
        make_estimator=_make_tcn,
        load_frame=_load_gru_frame,
        describe=_tcn_describe,
    ),
    "lgbm": ModelSpec(
        name="lgbm",
        title="LightGBM",
        load_config=load_lgbm_config,
        feature_columns=_lgbm_features,
        make_estimator=_make_lgbm,
        load_frame=_load_lgbm_frame,
        describe=_lgbm_describe,
        incremental_runs=_lgbm_incremental_runs,
    ),
}


def get_model_spec(name: str) -> ModelSpec:
    """Return the :class:`ModelSpec` for ``name`` or raise a clear ``KeyError``."""
    try:
        return MODEL_REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"Unknown model {name!r}; registered models: {known}") from None
