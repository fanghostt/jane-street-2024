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
# The GRU consumes per-``symbol_id`` sequences ordered by ``(date_id, time_id)``.
# Unlike LightGBM, a neural net needs standardized, NaN-free inputs, so these
# helpers fit a per-feature standardizer on the train frame and turn rows into
# fixed-length lookback windows. ``symbol_id`` is *not* a model input here:
# per-symbol sequencing already encodes symbol identity.

# Sentinel symbol used by ``build_symbol_windows`` when no ``symbol_id`` column
# is present (treats the whole frame as one sequence — handy for tests).
_SINGLE_SYMBOL = -1


def get_gru_feature_columns(include_time: bool = False) -> list[str]:
    """Return the GRU model input columns: the 79 raw features (+ ``time_id``)."""
    cols = list(FEATURE_COLUMNS)
    if include_time:
        cols.append("time_id")
    return cols


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


def build_symbol_windows(
    df: pl.DataFrame,
    feature_cols: list[str],
    mean: np.ndarray,
    std: np.ndarray,
    seq_len: int,
    *,
    target_col: str | None = None,
    weight_col: str | None = None,
    history: dict[int, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Turn rows into per-symbol lookback windows aligned to ``df``'s row order.

    For every row a ``(seq_len, F)`` window ending at that row is emitted, built
    from that ``symbol_id``'s rows ordered by ``(date_id, time_id)`` plus any
    prior ``history`` for the symbol. Features are standardized with
    ``(x - mean) / std`` and NaNs imputed to 0 (the post-standardization mean).
    Short histories are left-padded with zeros.

    Parameters
    ----------
    history
        Optional ``{symbol_id: (k, F) float32}`` of already-standardized feature
        rows preceding ``df`` (the estimator's cross-day buffer). Prepended per
        symbol before windowing so windows can span day boundaries.

    Returns
    -------
    dict with keys:
        ``"windows"`` -> ``(N, seq_len, F)`` float32, ``N == df.height``, in
        ``df``'s original row order; ``"y"`` / ``"w"`` (float32, length ``N``)
        when ``target_col`` / ``weight_col`` are given.
    """
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    required = list(feature_cols)
    for c in (target_col, weight_col):
        if c is not None:
            required.append(c)
    has_symbol = "symbol_id" in df.columns
    order_cols = [c for c in ("date_id", "time_id") if c in df.columns]
    proj = required + (["symbol_id"] if has_symbol else []) + order_cols
    missing = [c for c in proj if c not in df.columns]
    if missing:
        raise ValueError(f"build_symbol_windows: missing columns {missing}")

    n = df.height
    f = len(feature_cols)
    # Carry a stable original-row index through the per-symbol sort so we can
    # scatter windows back into df's row order.
    pdf = df.select(proj).to_pandas()
    pdf["__row__"] = np.arange(n)
    mean = mean.astype(np.float32)
    std = std.astype(np.float32)

    windows = np.zeros((n, seq_len, f), dtype=np.float32)
    y = np.zeros(n, dtype=np.float32) if target_col is not None else None
    w = np.zeros(n, dtype=np.float32) if weight_col is not None else None

    sort_keys = order_cols if order_cols else ["__row__"]
    group_key = "symbol_id" if has_symbol else None
    groups = pdf.groupby(group_key, sort=False) if group_key else [(_SINGLE_SYMBOL, pdf)]

    for sym, g in groups:
        g = g.sort_values(sort_keys, kind="stable")
        feats = g[feature_cols].to_numpy().astype(np.float32)
        feats = (feats - mean) / std
        feats = np.nan_to_num(feats, nan=0.0)
        rows = g["__row__"].to_numpy()

        hist = None
        if history is not None:
            hist = history.get(int(sym) if has_symbol else _SINGLE_SYMBOL)
        if hist is not None and hist.shape[0] > 0:
            seq = np.concatenate([hist.astype(np.float32), feats], axis=0)
            offset = hist.shape[0]
        else:
            seq = feats
            offset = 0

        for i, orig in enumerate(rows):
            end = offset + i + 1  # window ends at (and includes) this row
            start = max(0, end - seq_len)
            chunk = seq[start:end]
            windows[orig, seq_len - chunk.shape[0]:] = chunk

        if y is not None:
            y[rows] = g[target_col].to_numpy().astype(np.float32)
        if w is not None:
            w[rows] = g[weight_col].to_numpy().astype(np.float32)

    out: dict[str, np.ndarray] = {"windows": windows}
    if y is not None:
        out["y"] = y
    if w is not None:
        out["w"] = w
    return out


def standardized_symbol_tails(
    df: pl.DataFrame,
    feature_cols: list[str],
    mean: np.ndarray,
    std: np.ndarray,
    keep: int,
) -> dict[int, np.ndarray]:
    """Return the last ``keep`` standardized feature rows per ``symbol_id``.

    Used to seed/advance the GRU estimator's cross-day context buffer. NaNs are
    imputed to 0 to match :func:`build_symbol_windows`.
    """
    if keep <= 0:
        return {}
    has_symbol = "symbol_id" in df.columns
    order_cols = [c for c in ("date_id", "time_id") if c in df.columns]
    proj = list(feature_cols) + (["symbol_id"] if has_symbol else []) + order_cols
    pdf = df.select(proj).to_pandas()
    mean = mean.astype(np.float32)
    std = std.astype(np.float32)
    sort_keys = order_cols if order_cols else None
    group_key = "symbol_id" if has_symbol else None
    groups = pdf.groupby(group_key, sort=False) if group_key else [(_SINGLE_SYMBOL, pdf)]

    tails: dict[int, np.ndarray] = {}
    for sym, g in groups:
        if sort_keys:
            g = g.sort_values(sort_keys, kind="stable")
        feats = g[feature_cols].to_numpy().astype(np.float32)[-keep:]
        feats = (feats - mean) / std
        feats = np.nan_to_num(feats, nan=0.0)
        tails[int(sym) if has_symbol else _SINGLE_SYMBOL] = feats
    return tails
