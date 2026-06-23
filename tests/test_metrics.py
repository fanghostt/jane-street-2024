import numpy as np
import pandas as pd
import pytest

from js2024.metrics import weighted_zero_mean_r2


def test_perfect_prediction_returns_one():
    y = np.array([1.0, -2.0, 0.5, 3.0])
    w = np.array([1.0, 2.0, 0.5, 1.0])
    assert weighted_zero_mean_r2(y, y.copy(), w) == pytest.approx(1.0)


def test_all_zero_prediction_returns_zero():
    y = np.array([1.0, -2.0, 0.5, 3.0])
    w = np.array([1.0, 2.0, 0.5, 1.0])
    pred = np.zeros_like(y)
    assert weighted_zero_mean_r2(y, pred, w) == pytest.approx(0.0)


def test_worse_than_zero_baseline_is_negative():
    y = np.array([1.0, 1.0, 1.0])
    w = np.array([1.0, 1.0, 1.0])
    # Predicting the opposite sign is worse than predicting 0.
    pred = np.array([-3.0, -3.0, -3.0])
    assert weighted_zero_mean_r2(y, pred, w) < 0.0


def test_accepts_pandas_series():
    y = pd.Series([1.0, 2.0, 3.0])
    w = pd.Series([1.0, 1.0, 1.0])
    assert weighted_zero_mean_r2(y, y.copy(), w) == pytest.approx(1.0)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        weighted_zero_mean_r2(np.array([1.0, 2.0]), np.array([1.0]), np.array([1.0, 1.0]))


def test_zero_denominator_raises():
    y = np.zeros(3)
    w = np.ones(3)
    with pytest.raises(ValueError):
        weighted_zero_mean_r2(y, np.ones(3), w)


def test_nan_raises():
    y = np.array([1.0, np.nan, 3.0])
    w = np.ones(3)
    with pytest.raises(ValueError):
        weighted_zero_mean_r2(y, np.ones(3), w)


def test_inf_raises():
    y = np.array([1.0, 2.0, 3.0])
    pred = np.array([1.0, np.inf, 3.0])
    w = np.ones(3)
    with pytest.raises(ValueError):
        weighted_zero_mean_r2(y, pred, w)
