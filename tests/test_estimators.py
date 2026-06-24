"""Smoke tests for LGBMEstimator (tiny data, fast)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from js2024.estimators import Estimator, LGBMEstimator

FEATURES = [f"feature_{i:02d}" for i in range(79)]


def _frame(n_days: int = 8, symbols: int = 4, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        for s in range(symbols):
            row = {"date_id": d, "time_id": 0, "symbol_id": s}
            for f in FEATURES:
                row[f] = float(rng.normal())
            # target correlated with feature_00 so the model learns something.
            row["responder_6"] = float(0.7 * row["feature_00"] + 0.1 * rng.normal())
            row["weight"] = 1.0
            rows.append(row)
    return pl.DataFrame(rows)


def _est() -> LGBMEstimator:
    return LGBMEstimator(
        feature_cols=FEATURES + ["symbol_id", "time_id"],
        target_col="responder_6",
        weight_col="weight",
        params={"n_estimators": 20, "num_leaves": 7, "learning_rate": 0.1},
        early_stopping_rounds=5,
        refit_decay=0.9,
    )


def test_lgbm_estimator_is_estimator_protocol():
    assert isinstance(_est(), Estimator)


def test_fit_predict_finite():
    df = _frame()
    est = _est().fit(df.filter(pl.col("date_id") < 6), df.filter(pl.col("date_id") >= 6))
    preds = est.predict(df.filter(pl.col("date_id") == 7))
    assert preds.shape[0] == 4
    assert np.all(np.isfinite(preds))


def test_update_changes_booster():
    df = _frame()
    est = _est().fit(df.filter(pl.col("date_id") < 6))
    before = est._booster
    est.update(df.filter(pl.col("date_id") == 6))
    assert est._booster is not before  # refit returns a new booster


def test_update_before_fit_raises():
    df = _frame()
    with pytest.raises(RuntimeError):
        _est().update(df)


def test_empty_update_is_noop():
    df = _frame()
    est = _est().fit(df.filter(pl.col("date_id") < 6))
    booster = est._booster
    est.update(df.filter(pl.col("date_id") == 999))  # empty
    assert est._booster is booster
