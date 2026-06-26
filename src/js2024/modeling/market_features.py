"""Market-average and per-symbol rolling features (leakage-safe).

Two engineered feature families from ``evgeniavolkova/kagglejanestreet`` that the V0
inputs lack:

- **Market average** — for each selected feature, the cross-sectional mean across all
  symbols present at the *same* ``(date_id, time_id)``. Contemporaneous and
  symbol-agnostic; available at inference (the live API delivers every symbol's row
  for a timestamp before predictions are requested), so it is leakage-safe.
- **Per-symbol rolling** — for each selected feature, a trailing rolling mean and std
  over the last ``window`` steps of that symbol's ``(date_id, time_id)``-ordered
  series. The window ends at the current row (inclusive) and never reads the future,
  so it is causal. Only *features* are rolled — never responders (current-day
  responders are unavailable at inference; lagging them was tried and rejected, see
  ``lag_features`` / ``docs/experiments/lag_features_v1.md``).

Feature subset = the 12 highest-importance V0 LGBM features, to keep the added column
count and runtime modest for a first A/B. These columns are already part of the
standard 79 features, so no extra source columns need to be loaded.
"""

from __future__ import annotations

import polars as pl

# Top-12 features by `lgbm_v0_recent700` importance (excluding the time_id input).
MARKET_ROLL_FEATURES: list[str] = [
    "feature_61", "feature_20", "feature_24", "feature_08", "feature_21", "feature_22",
    "feature_25", "feature_31", "feature_30", "feature_07", "feature_38", "feature_05",
]
DEFAULT_ROLLING_WINDOW: int = 1000


def get_market_avg_columns() -> list[str]:
    """Cross-sectional market-average feature names (one per selected feature)."""
    return [f"{f}_mkt" for f in MARKET_ROLL_FEATURES]


def get_rolling_columns() -> list[str]:
    """Per-symbol rolling mean/std feature names (two per selected feature)."""
    cols: list[str] = []
    for f in MARKET_ROLL_FEATURES:
        cols.extend([f"{f}_roll_mean", f"{f}_roll_std"])
    return cols


def selected_columns(*, use_market_avg: bool, use_symbol_rolling: bool) -> list[str]:
    """Names of the engineered columns enabled by the given flags (in apply order)."""
    cols: list[str] = []
    if use_market_avg:
        cols.extend(get_market_avg_columns())
    if use_symbol_rolling:
        cols.extend(get_rolling_columns())
    return cols


def add_engineered_features(
    df: pl.DataFrame,
    *,
    use_market_avg: bool,
    use_symbol_rolling: bool,
    window: int = DEFAULT_ROLLING_WINDOW,
) -> pl.DataFrame:
    """Apply the enabled engineered-feature families to ``df`` (no-op if all off)."""
    if use_market_avg:
        df = add_market_avg(df)
    if use_symbol_rolling:
        df = add_symbol_rolling(df, window=window)
    return df


def _require_features(df: pl.DataFrame) -> None:
    missing = [f for f in MARKET_ROLL_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"market_features: missing feature columns {missing}")


def add_market_avg(df: pl.DataFrame) -> pl.DataFrame:
    """Add ``<feature>_mkt`` = cross-sectional mean over ``(date_id, time_id)``.

    Leakage-safe: uses only contemporaneous feature values across symbols.
    """
    _require_features(df)
    return df.with_columns(
        pl.col(f).mean().over(["date_id", "time_id"]).alias(f"{f}_mkt")
        for f in MARKET_ROLL_FEATURES
    )


def add_symbol_rolling(df: pl.DataFrame, window: int = DEFAULT_ROLLING_WINDOW) -> pl.DataFrame:
    """Add trailing per-symbol rolling mean/std of each selected feature.

    The frame is sorted by ``(symbol_id, date_id, time_id)`` so each symbol's rolling
    window runs over its own time-ordered history and ends at the current row — strictly
    causal (no future leakage). Returns the frame sorted canonically by
    ``(date_id, time_id, symbol_id)``.
    """
    _require_features(df)
    df = df.sort(["symbol_id", "date_id", "time_id"])
    exprs: list[pl.Expr] = []
    for f in MARKET_ROLL_FEATURES:
        exprs.append(
            pl.col(f)
            .rolling_mean(window_size=window, min_samples=1)
            .over("symbol_id")
            .alias(f"{f}_roll_mean")
        )
        exprs.append(
            pl.col(f)
            .rolling_std(window_size=window, min_samples=2)
            .over("symbol_id")
            .fill_null(0.0)
            .alias(f"{f}_roll_std")
        )
    return df.with_columns(exprs).sort(["date_id", "time_id", "symbol_id"])
