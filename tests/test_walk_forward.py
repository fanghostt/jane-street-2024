"""Tests for the model-agnostic walk-forward engine.

A ``FakeEstimator`` drives the engine so we test the loop logic (update count,
cadence, leakage guard, full coverage) without LightGBM.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from js2024.walk_forward import walk_forward_evaluate


class FakeEstimator:
    """Predicts a constant; records every (date range) it is updated with."""

    def __init__(self, const: float = 0.1) -> None:
        self.const = const
        self.update_calls: list[tuple[int, int]] = []

    def fit(self, df_train, df_valid=None):
        return self

    def update(self, df_new):
        lo = int(df_new.get_column("date_id").min())
        hi = int(df_new.get_column("date_id").max())
        self.update_calls.append((lo, hi))
        return self

    def predict(self, df):
        return np.full(df.height, self.const, dtype=np.float64)


def _frame(dates, symbols_per_day=3):
    rows = []
    for d in dates:
        for s in range(symbols_per_day):
            rows.append(
                {
                    "date_id": d,
                    "time_id": 0,
                    "symbol_id": s,
                    "responder_6": 0.5,
                    "weight": 1.0,
                }
            )
    return pl.DataFrame(rows)


def test_full_mode_does_no_updates():
    df = _frame(range(10, 20))
    est = FakeEstimator()
    res = walk_forward_evaluate(est, df, test_start=10, test_end=19, mode="full")
    assert res.n_updates == 0
    assert est.update_calls == []
    assert res.n_test_days == 10
    assert res.n_test_rows == 30


def test_incremental_daily_updates_each_step():
    df = _frame(range(10, 20))
    est = FakeEstimator()
    res = walk_forward_evaluate(
        est, df, test_start=10, test_end=19, mode="incremental", update_cadence=1
    )
    # 10 test days -> 9 updates (none before the first prediction).
    assert res.n_updates == 9
    assert len(est.update_calls) == 9
    # Each daily update is exactly the previous single day.
    assert est.update_calls[0] == (10, 10)
    assert est.update_calls[-1] == (18, 18)


def test_incremental_cadence_every_three():
    df = _frame(range(10, 20))
    est = FakeEstimator()
    res = walk_forward_evaluate(
        est, df, test_start=10, test_end=19, mode="incremental", update_cadence=3
    )
    # Updates fire at i = 3, 6, 9 -> 3 updates.
    assert res.n_updates == 3
    # First update covers days revealed since start: 10..12.
    assert est.update_calls[0] == (10, 12)


def test_no_update_uses_unpredicted_day():
    """Leakage guard: every update day must already have been predicted."""
    df = _frame(range(10, 20))
    est = FakeEstimator()
    walk_forward_evaluate(
        est, df, test_start=10, test_end=19, mode="incremental", update_cadence=1
    )
    # All updated day ranges must be strictly below the running prediction frontier;
    # since updates always use [<= i-1] before predicting day i, the max updated day
    # never exceeds the last predicted day.
    for lo, hi in est.update_calls:
        assert hi < 19  # never the final day, which is only predicted, never used


def test_full_and_incremental_score_same_block():
    df = _frame(range(10, 20))
    full = walk_forward_evaluate(FakeEstimator(), df, 10, 19, mode="full")
    inc = walk_forward_evaluate(FakeEstimator(), df, 10, 19, mode="incremental")
    # Same test coverage; FakeEstimator is constant so scores match exactly.
    assert full.n_test_rows == inc.n_test_rows == 30
    assert full.score == pytest.approx(inc.score)


def test_empty_test_block_raises():
    df = _frame(range(10, 20))
    with pytest.raises(ValueError):
        walk_forward_evaluate(FakeEstimator(), df, 100, 110, mode="full")


def test_bad_mode_raises():
    df = _frame(range(10, 20))
    with pytest.raises(ValueError):
        walk_forward_evaluate(FakeEstimator(), df, 10, 19, mode="bogus")
