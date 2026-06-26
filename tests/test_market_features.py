"""Tests for market-average and per-symbol rolling features — focus on leakage."""

import math

import polars as pl
import pytest

from js2024.modeling.market_features import (
    MARKET_ROLL_FEATURES,
    add_market_avg,
    add_symbol_rolling,
    get_market_avg_columns,
    get_rolling_columns,
)

F = MARKET_ROLL_FEATURES[0]  # the feature we drive with known values


def _panel(rows: list[dict]) -> pl.DataFrame:
    """Build a panel with all required feature columns (unused ones zeroed)."""
    base_cols = {f: 0.0 for f in MARKET_ROLL_FEATURES}
    return pl.DataFrame([{**base_cols, **r} for r in rows])


def test_column_accessors():
    assert get_market_avg_columns() == [f"{f}_mkt" for f in MARKET_ROLL_FEATURES]
    assert len(get_rolling_columns()) == 2 * len(MARKET_ROLL_FEATURES)
    assert get_rolling_columns()[:2] == [f"{F}_roll_mean", f"{F}_roll_std"]


def test_missing_feature_raises():
    df = pl.DataFrame({"date_id": [0], "time_id": [0], "symbol_id": [0]})
    with pytest.raises(ValueError, match="missing feature columns"):
        add_market_avg(df)


def test_market_avg_is_cross_sectional_mean():
    # Two symbols at the same (date, time): mkt = their mean, shared by both rows.
    df = _panel([
        {"date_id": 0, "time_id": 0, "symbol_id": 0, F: 2.0},
        {"date_id": 0, "time_id": 0, "symbol_id": 1, F: 4.0},
        {"date_id": 0, "time_id": 1, "symbol_id": 0, F: 10.0},
    ])
    out = add_market_avg(df).sort(["date_id", "time_id", "symbol_id"])
    mkt = out.get_column(f"{F}_mkt").to_list()
    assert mkt[0] == pytest.approx(3.0)   # mean(2, 4)
    assert mkt[1] == pytest.approx(3.0)
    assert mkt[2] == pytest.approx(10.0)  # lone symbol at (0, 1)


def test_rolling_mean_is_trailing():
    rows = [{"date_id": 0, "time_id": t, "symbol_id": 0, F: float(v)}
            for t, v in enumerate([1, 2, 3, 4])]
    out = add_symbol_rolling(_panel(rows), window=2).sort(["date_id", "time_id"])
    rm = out.get_column(f"{F}_roll_mean").to_list()
    assert rm == pytest.approx([1.0, 1.5, 2.5, 3.5])  # trailing window of size 2
    rs = out.get_column(f"{F}_roll_std").to_list()
    assert rs[0] == pytest.approx(0.0)                 # <2 samples -> filled 0
    assert rs[1] == pytest.approx(math.sqrt(0.5))      # std([1,2]) ddof=1


def test_rolling_is_causal_not_leaky():
    """Changing a FUTURE row must not change an earlier row's rolling value."""
    rows = [{"date_id": 0, "time_id": t, "symbol_id": 0, F: float(v)}
            for t, v in enumerate([1, 2, 3, 4])]
    base = add_symbol_rolling(_panel(rows), window=3).sort(["time_id"])
    # Perturb only the LAST time step.
    rows_future = [dict(r) for r in rows]
    rows_future[-1][F] = 999.0
    perturbed = add_symbol_rolling(_panel(rows_future), window=3).sort(["time_id"])

    base_rm = base.get_column(f"{F}_roll_mean").to_list()
    pert_rm = perturbed.get_column(f"{F}_roll_mean").to_list()
    assert base_rm[:3] == pytest.approx(pert_rm[:3])   # earlier rows unchanged
    assert base_rm[3] != pytest.approx(pert_rm[3])     # only the last row moves


def test_rolling_is_per_symbol():
    # Two symbols interleaved in time; each rolls over its OWN series only.
    rows = [
        {"date_id": 0, "time_id": 0, "symbol_id": 0, F: 1.0},
        {"date_id": 0, "time_id": 0, "symbol_id": 1, F: 100.0},
        {"date_id": 0, "time_id": 1, "symbol_id": 0, F: 3.0},
        {"date_id": 0, "time_id": 1, "symbol_id": 1, F: 300.0},
    ]
    out = add_symbol_rolling(_panel(rows), window=2).sort(["symbol_id", "time_id"])
    rm = out.get_column(f"{F}_roll_mean").to_list()
    # symbol 0: [1, 2]; symbol 1: [100, 200] — no cross-contamination.
    assert rm == pytest.approx([1.0, 2.0, 100.0, 200.0])
