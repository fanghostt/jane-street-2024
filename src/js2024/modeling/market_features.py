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

Feature subset (``market_roll_subset``): which V0 features get engineered, ranked by
``lgbm_v0_recent700`` *split* importance (the ranking the original top-12 came from):

- ``top12`` — the 12 highest-importance features (the default; keeps the added column
  count and runtime modest for a first A/B).
- ``top24`` — the 24 highest-importance features.
- ``all``   — all 79 V0 features.

All candidates are already part of the standard 79 features, so no extra source columns
need to be loaded regardless of subset.
"""

from __future__ import annotations

import polars as pl

# All 79 V0 features ranked by `lgbm_v0_recent700` *split* importance (the original
# top-12 below is exactly this list's first 12, in this order, so the `top12` subset is
# byte-for-byte the historical MARKET_ROLL_FEATURES — no A/B drift for existing configs).
MARKET_ROLL_FEATURES_RANKED: list[str] = [
    "feature_61", "feature_20", "feature_24", "feature_08", "feature_21", "feature_22",
    "feature_25", "feature_31", "feature_30", "feature_07", "feature_38", "feature_05",
    "feature_26", "feature_23", "feature_29", "feature_27", "feature_28", "feature_01",
    "feature_58", "feature_60", "feature_04", "feature_37", "feature_47", "feature_33",
    "feature_36", "feature_15", "feature_69", "feature_77", "feature_59", "feature_06",
    "feature_56", "feature_78", "feature_11", "feature_70", "feature_09", "feature_17",
    "feature_50", "feature_74", "feature_72", "feature_67", "feature_14", "feature_73",
    "feature_02", "feature_62", "feature_52", "feature_49", "feature_12", "feature_53",
    "feature_34", "feature_42", "feature_39", "feature_68", "feature_57", "feature_45",
    "feature_00", "feature_32", "feature_76", "feature_35", "feature_75", "feature_66",
    "feature_10", "feature_03", "feature_55", "feature_18", "feature_46", "feature_64",
    "feature_48", "feature_65", "feature_19", "feature_16", "feature_44", "feature_41",
    "feature_51", "feature_13", "feature_71", "feature_54", "feature_40", "feature_43",
    "feature_63",
]

# Named subsets selectable via the `market_roll_subset` config key.
MARKET_ROLL_SUBSETS: dict[str, list[str]] = {
    "top12": MARKET_ROLL_FEATURES_RANKED[:12],
    "top24": MARKET_ROLL_FEATURES_RANKED[:24],
    "all": MARKET_ROLL_FEATURES_RANKED,
}
DEFAULT_MARKET_ROLL_SUBSET: str = "top12"

# Backwards-compatible default feature list (the historical top-12).
MARKET_ROLL_FEATURES: list[str] = MARKET_ROLL_SUBSETS[DEFAULT_MARKET_ROLL_SUBSET]
DEFAULT_ROLLING_WINDOW: int = 1000


def resolve_market_roll_features(subset: str) -> list[str]:
    """Map a ``market_roll_subset`` name to its ordered feature list."""
    try:
        return MARKET_ROLL_SUBSETS[subset]
    except KeyError:
        known = ", ".join(MARKET_ROLL_SUBSETS)
        raise ValueError(
            f"unknown market_roll_subset {subset!r}; choose from {known}"
        ) from None


def _features(features: list[str] | None) -> list[str]:
    return MARKET_ROLL_FEATURES if features is None else features


def get_market_avg_columns(features: list[str] | None = None) -> list[str]:
    """Cross-sectional market-average feature names (one per selected feature)."""
    return [f"{f}_mkt" for f in _features(features)]


def get_rolling_columns(features: list[str] | None = None) -> list[str]:
    """Per-symbol rolling mean/std feature names (two per selected feature)."""
    cols: list[str] = []
    for f in _features(features):
        cols.extend([f"{f}_roll_mean", f"{f}_roll_std"])
    return cols


def selected_columns(
    *, use_market_avg: bool, use_symbol_rolling: bool, features: list[str] | None = None
) -> list[str]:
    """Names of the engineered columns enabled by the given flags (in apply order)."""
    cols: list[str] = []
    if use_market_avg:
        cols.extend(get_market_avg_columns(features))
    if use_symbol_rolling:
        cols.extend(get_rolling_columns(features))
    return cols


def add_engineered_features(
    df: pl.DataFrame,
    *,
    use_market_avg: bool,
    use_symbol_rolling: bool,
    window: int = DEFAULT_ROLLING_WINDOW,
    features: list[str] | None = None,
) -> pl.DataFrame:
    """Apply the enabled engineered-feature families to ``df`` (no-op if all off)."""
    if use_market_avg:
        df = add_market_avg(df, features=features)
    if use_symbol_rolling:
        df = add_symbol_rolling(df, window=window, features=features)
    return df


def _require_features(df: pl.DataFrame, features: list[str]) -> None:
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"market_features: missing feature columns {missing}")


def add_market_avg(df: pl.DataFrame, features: list[str] | None = None) -> pl.DataFrame:
    """Add ``<feature>_mkt`` = cross-sectional mean over ``(date_id, time_id)``.

    Leakage-safe: uses only contemporaneous feature values across symbols.
    """
    feats = _features(features)
    _require_features(df, feats)
    return df.with_columns(
        pl.col(f).mean().over(["date_id", "time_id"]).alias(f"{f}_mkt")
        for f in feats
    )


def add_symbol_rolling(
    df: pl.DataFrame, window: int = DEFAULT_ROLLING_WINDOW, features: list[str] | None = None
) -> pl.DataFrame:
    """Add trailing per-symbol rolling mean/std of each selected feature.

    The frame is sorted by ``(symbol_id, date_id, time_id)`` so each symbol's rolling
    window runs over its own time-ordered history and ends at the current row — strictly
    causal (no future leakage). Returns the frame sorted canonically by
    ``(date_id, time_id, symbol_id)``.
    """
    feats = _features(features)
    _require_features(df, feats)
    df = df.sort(["symbol_id", "date_id", "time_id"])
    exprs: list[pl.Expr] = []
    for f in feats:
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
