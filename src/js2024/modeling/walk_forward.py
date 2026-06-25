"""Model-agnostic walk-forward (streaming) evaluation over a fixed test block.

Both training modes are scored on the **same** trailing test block so the numbers
are directly comparable:

- ``mode="full"``   — predict every test day with the initial model; never update.
- ``mode="incremental"`` — walk the test days in order; on each cadence boundary,
  ``update`` the model with the days revealed *since the last update* **before**
  predicting the next day. Daily cadence reproduces the per-day online loop used by
  ``evgeniavolkova/kagglejanestreet``.

``full`` is just ``incremental`` with zero updates, so a single loop drives both and
the prediction path is identical. A leakage guard asserts no test day's labels are
ever fed to ``update`` before that day has been predicted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from .estimators import Estimator
from .metrics import weighted_zero_mean_r2
from .validation import filter_by_date_range


def _summary(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


@dataclass
class WalkForwardResult:
    """Outcome of a single walk-forward evaluation over the test block."""

    mode: str
    test_start: int
    test_end: int
    n_test_days: int
    n_test_rows: int
    n_updates: int
    score: float
    prediction_summary: dict[str, float] = field(default_factory=dict)
    target_summary: dict[str, float] = field(default_factory=dict)


def walk_forward_evaluate(
    estimator: Estimator,
    df: pl.DataFrame,
    test_start: int,
    test_end: int,
    *,
    mode: str = "incremental",
    update_cadence: int = 1,
    date_col: str = "date_id",
    target_col: str = "responder_6",
    weight_col: str = "weight",
) -> WalkForwardResult:
    """Evaluate ``estimator`` day-by-day over ``[test_start, test_end]``.

    The estimator must already be ``fit`` on the training region (the engine never
    trains it). ``df`` must contain every test day plus, for incremental updates, the
    day immediately preceding ``test_start``.
    """
    if mode not in {"full", "incremental"}:
        raise ValueError(f"mode must be 'full' or 'incremental', got {mode!r}")
    if update_cadence < 1:
        raise ValueError(f"update_cadence must be >= 1, got {update_cadence}")

    test_df = filter_by_date_range(df, date_col, test_start, test_end)
    if test_df.height == 0:
        raise ValueError(
            f"No rows in test block [{test_start}, {test_end}] for '{date_col}'."
        )
    test_dates = sorted(test_df.get_column(date_col).unique().to_list())

    preds_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    w_parts: list[np.ndarray] = []

    n_updates = 0
    predicted_max_date: int | None = None  # leakage guard: last day already predicted
    last_update_boundary = 0  # index up to which days were already used for update

    for i, d in enumerate(test_dates):
        # Update *before* predicting day d, using only already-predicted days.
        if mode == "incremental" and i > 0 and (i % update_cadence == 0):
            upd_lo = test_dates[last_update_boundary]
            upd_hi = test_dates[i - 1]
            # Leakage guard: every update day must already have been predicted.
            assert predicted_max_date is not None and upd_hi <= predicted_max_date, (
                f"Leakage: updating with day {upd_hi} not yet predicted "
                f"(max predicted {predicted_max_date})."
            )
            df_upd = filter_by_date_range(df, date_col, upd_lo, upd_hi)
            estimator.update(df_upd)
            n_updates += 1
            last_update_boundary = i

        df_day = test_df.filter(pl.col(date_col) == d)
        preds_parts.append(np.asarray(estimator.predict(df_day), dtype=np.float64))
        y_parts.append(df_day.get_column(target_col).to_numpy().astype(np.float64))
        w_parts.append(df_day.get_column(weight_col).to_numpy().astype(np.float64))
        predicted_max_date = d

    preds = np.concatenate(preds_parts)
    y_true = np.concatenate(y_parts)
    weight = np.concatenate(w_parts)
    score = weighted_zero_mean_r2(y_true, preds, weight)

    return WalkForwardResult(
        mode=mode,
        test_start=int(test_start),
        test_end=int(test_end),
        n_test_days=len(test_dates),
        n_test_rows=int(test_df.height),
        n_updates=n_updates,
        score=float(score),
        prediction_summary=_summary(preds),
        target_summary=_summary(y_true),
    )
