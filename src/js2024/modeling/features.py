"""V0 feature assembly.

Intentionally minimal: V0 uses the 79 raw features plus the ``symbol_id`` and
``time_id`` identifiers as model inputs. No standardization, no engineered
features, no NaN imputation — LightGBM handles missing values natively. Richer
feature engineering belongs to a later milestone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from ..data.data import FEATURE_COLUMNS


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


# --- GRU (sequence model) feature prep -------------------------------------
#
# Neural-net inputs need standardized, NaN-free features. The day-batch GRU
# (see :mod:`js2024.modeling.gru`) fits this per-feature
# standardizer on its train frame; ``symbol_id`` is not a model input here.


def fit_feature_standardizer(
    df: pl.DataFrame, feature_cols: list[str], *, eps: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """Fit NaN-aware per-feature mean/std on ``df`` for GRU standardization.

    Returns ``(mean, std)`` as float32 arrays of length ``len(feature_cols)``;
    ``std`` is floored at ``eps`` so constant/empty columns never divide by zero.
    """
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"fit_feature_standardizer: missing columns {missing}")
    arr = df.select(feature_cols).to_numpy().astype(np.float32)
    if arr.shape[0] == 0:
        raise ValueError("fit_feature_standardizer: empty frame")
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    # All-NaN columns yield nan mean/std; treat as 0 mean / unit std.
    mean = np.nan_to_num(mean, nan=0.0).astype(np.float32)
    std = np.nan_to_num(std, nan=1.0).astype(np.float32)
    std = np.maximum(std, eps).astype(np.float32)
    return mean, std
