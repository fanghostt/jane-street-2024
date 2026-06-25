"""Smoke + unit tests for GRUEstimator and the windowing helpers (tiny, fast)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

torch = pytest.importorskip("torch")  # skip the whole module if torch is absent

from js2024.modeling.estimators import Estimator, GRUEstimator  # noqa: E402
from js2024.modeling.features import build_symbol_windows  # noqa: E402

FEATURES = [f"feature_{i:02d}" for i in range(8)]


def _frame(n_days: int = 12, symbols: int = 4, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        for s in range(symbols):
            row = {"date_id": d, "time_id": 0, "symbol_id": s}
            for f in FEATURES:
                row[f] = float(rng.normal())
            row["responder_6"] = float(0.7 * row["feature_00"] + 0.1 * rng.normal())
            row["weight"] = 1.0
            rows.append(row)
    return pl.DataFrame(rows)


def _est(seed: int = 42) -> GRUEstimator:
    return GRUEstimator(
        feature_cols=FEATURES,
        target_col="responder_6",
        weight_col="weight",
        params={
            "seq_len": 4,
            "hidden_size": 8,
            "num_layers": 1,
            "epochs": 2,
            "batch_size": 64,
            "early_stopping_rounds": 2,
            "finetune_epochs": 1,
            "lr": 0.01,
        },
        random_state=seed,
    )


def _stream_est(seed: int = 42) -> GRUEstimator:
    return GRUEstimator(
        feature_cols=FEATURES,
        target_col="responder_6",
        weight_col="weight",
        params={
            "seq_len": 4,
            "hidden_size": 8,
            "num_layers": 1,
            "epochs": 1,
            "batch_size": 64,
            "early_stopping_rounds": 1,
            "finetune_epochs": 1,
            "lr": 0.01,
            "train_mode": "stream",
        },
        random_state=seed,
    )


def test_gru_estimator_is_estimator_protocol():
    assert isinstance(_est(), Estimator)


def test_fit_predict_finite_and_shaped():
    df = _frame()
    est = _est().fit(df.filter(pl.col("date_id") < 8), df.filter(pl.col("date_id") >= 8))
    day = df.filter(pl.col("date_id") == 9)
    preds = est.predict(day)
    assert preds.shape == (day.height,)
    assert np.all(np.isfinite(preds))


def test_stream_fit_predict_finite_and_shaped():
    df = _frame()
    est = _stream_est().fit(
        df.filter(pl.col("date_id") < 8), df.filter(pl.col("date_id") >= 8)
    )
    day = df.filter(pl.col("date_id") == 9)
    preds = est.predict(day)
    assert preds.shape == (day.height,)
    assert np.all(np.isfinite(preds))


def test_predict_row_alignment():
    df = _frame()
    est = _est().fit(df.filter(pl.col("date_id") < 8))
    test = df.filter(pl.col("date_id") == 9)
    base = est.predict(test, _advance_buffer=False)

    perm = np.array([3, 0, 2, 1])  # symbols=4 -> 4 rows on a single day
    shuffled = test[perm.tolist()]
    shuf_preds = est.predict(shuffled, _advance_buffer=False)
    # Predictions must follow the rows, not their position.
    assert np.allclose(shuf_preds, base[perm], atol=1e-6)


def test_update_changes_parameters():
    df = _frame()
    est = _est().fit(df.filter(pl.col("date_id") < 6))

    def _flat():
        return torch.cat([p.detach().flatten() for p in est._model.parameters()]).clone()

    before = _flat()
    # Predict day 6 first so the leakage-clean buffer advances, then update on it.
    est.predict(df.filter(pl.col("date_id") == 6))
    est.update(df.filter(pl.col("date_id") == 6))
    after = _flat()
    assert not torch.allclose(before, after)


def test_update_before_fit_raises():
    with pytest.raises(RuntimeError):
        _est().update(_frame())


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        _est().predict(_frame())


def test_empty_update_is_noop():
    df = _frame()
    est = _est().fit(df.filter(pl.col("date_id") < 6))
    before = torch.cat([p.detach().flatten() for p in est._model.parameters()]).clone()
    est.update(df.filter(pl.col("date_id") == 999))  # empty
    after = torch.cat([p.detach().flatten() for p in est._model.parameters()]).clone()
    assert torch.allclose(before, after)


def test_determinism_same_seed():
    df = _frame()
    train = df.filter(pl.col("date_id") < 8)
    test = df.filter(pl.col("date_id") == 9)
    a = _est(seed=7).fit(train).predict(test, _advance_buffer=False)
    b = _est(seed=7).fit(train).predict(test, _advance_buffer=False)
    assert np.allclose(a, b)


# --- build_symbol_windows unit tests ---------------------------------------

def _identity_scaler(f: int):
    return np.zeros(f, dtype=np.float32), np.ones(f, dtype=np.float32)


def test_windows_left_padding_and_order():
    # One symbol, 3 ordered rows, seq_len=2: each window ends at its row.
    df = pl.DataFrame(
        {
            "date_id": [0, 0, 1],
            "time_id": [0, 1, 0],
            "symbol_id": [0, 0, 0],
            "feature_00": [1.0, 2.0, 3.0],
        }
    )
    mean, std = _identity_scaler(1)
    out = build_symbol_windows(df, ["feature_00"], mean, std, seq_len=2)
    w = out["windows"]
    assert w.shape == (3, 2, 1)
    # row0: left-padded with zeros, ends at value 1.
    assert w[0, 0, 0] == 0.0 and w[0, 1, 0] == 1.0
    # row1: [1, 2]; row2: [2, 3].
    assert w[1, 0, 0] == 1.0 and w[1, 1, 0] == 2.0
    assert w[2, 0, 0] == 2.0 and w[2, 1, 0] == 3.0


def test_windows_history_prepend():
    df = pl.DataFrame(
        {"date_id": [1], "time_id": [0], "symbol_id": [0], "feature_00": [3.0]}
    )
    mean, std = _identity_scaler(1)
    history = {0: np.array([[2.0]], dtype=np.float32)}  # one prior standardized row
    out = build_symbol_windows(df, ["feature_00"], mean, std, seq_len=2, history=history)
    w = out["windows"]
    # Window for the single row spans the day boundary: [history(2), current(3)].
    assert w[0, 0, 0] == 2.0 and w[0, 1, 0] == 3.0


def test_windows_nan_imputed_to_zero():
    df = pl.DataFrame(
        {"date_id": [0], "time_id": [0], "symbol_id": [0], "feature_00": [float("nan")]}
    )
    mean, std = _identity_scaler(1)
    out = build_symbol_windows(df, ["feature_00"], mean, std, seq_len=1)
    assert out["windows"][0, 0, 0] == 0.0
