"""Competition metric: sample-weighted zero-mean R².

The Jane Street 2024 competition scores predictions of ``responder_6`` with a
*weighted, zero-mean* R²::

    R2 = 1 - sum(w_i * (y_i - yhat_i)^2) / sum(w_i * y_i^2)

Note the denominator uses ``y_i^2`` (not the variance around a weighted mean),
so this is *not* ``sklearn.metrics.r2_score``. A constant zero prediction gives
exactly 0.0; a perfect prediction gives 1.0; arbitrarily bad predictions go
negative.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _to_1d_float_array(name: str, values: Any) -> np.ndarray:
    """Convert input (numpy array / pandas Series / sequence) to a 1d float64 array."""
    # pandas Series / DataFrame and numpy arrays both expose ``to_numpy``/``np.asarray``.
    arr = np.asarray(values, dtype=np.float64)
    # Accept column vectors like (n, 1) by squeezing the trailing singleton axis.
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional, got shape {arr.shape}")
    return arr


def weighted_zero_mean_r2(y_true: Any, y_pred: Any, weight: Any) -> float:
    """Compute the sample-weighted zero-mean R² used by the competition.

    Parameters
    ----------
    y_true, y_pred, weight
        1-d numpy arrays or pandas Series of equal length.

    Returns
    -------
    float
        ``1 - sum(w * (y - yhat)^2) / sum(w * y^2)``.

    Raises
    ------
    ValueError
        If shapes differ, if any input contains NaN/inf, or if the denominator
        ``sum(w * y^2)`` is <= 0.
    """
    yt = _to_1d_float_array("y_true", y_true)
    yp = _to_1d_float_array("y_pred", y_pred)
    w = _to_1d_float_array("weight", weight)

    if not (yt.shape == yp.shape == w.shape):
        raise ValueError(
            f"Shape mismatch: y_true{yt.shape}, y_pred{yp.shape}, weight{w.shape}"
        )

    for name, arr in (("y_true", yt), ("y_pred", yp), ("weight", w)):
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains NaN or inf values")

    denominator = float(np.sum(w * yt * yt))
    if denominator <= 0:
        raise ValueError(
            f"Denominator sum(w * y_true^2) must be > 0, got {denominator}"
        )

    numerator = float(np.sum(w * (yt - yp) ** 2))
    return 1.0 - numerator / denominator
