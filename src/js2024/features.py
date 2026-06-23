"""V0 feature assembly.

Intentionally minimal: V0 uses the 79 raw features plus the ``symbol_id`` and
``time_id`` identifiers as model inputs. No standardization, no engineered
features, no NaN imputation — LightGBM handles missing values natively. Richer
feature engineering belongs to a later milestone.
"""

from __future__ import annotations

import pandas as pd
import polars as pl

from .data import FEATURE_COLUMNS


def get_v0_feature_columns(include_symbol: bool = True, include_time: bool = True) -> list[str]:
    """Return the V0 model input columns: features (+ symbol_id, time_id)."""
    cols = list(FEATURE_COLUMNS)
    if include_symbol:
        cols.append("symbol_id")
    if include_time:
        cols.append("time_id")
    return cols


def prepare_lgbm_frame(
    df: pl.DataFrame,
    feature_cols: list[str],
    target_col: str,
    weight_col: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Split a polars DataFrame into (X, y, weight) pandas objects for LightGBM.

    Raises
    ------
    ValueError
        If any required column is missing.
    """
    required = list(feature_cols) + [target_col, weight_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"prepare_lgbm_frame: missing columns {missing}")

    pdf = df.select(required).to_pandas()
    X = pdf[feature_cols]
    y = pdf[target_col]
    weight = pdf[weight_col]
    return X, y, weight
