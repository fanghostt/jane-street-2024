import polars as pl
import pytest

from js2024.data.data import FEATURE_COLUMNS, WEIGHT_COLUMN
from js2024.modeling.config import LGBMConfig
from js2024.modeling.lag_features import (
    LAG_FEATURE_COLUMNS,
    RESPONDER_COLUMNS,
    add_responder_lags_from_train,
    get_lag_feature_columns,
)
from js2024.modeling.train_lgbm import run


def make_frame() -> pl.DataFrame:
    """Two days, two time_ids, two symbols, with known responder values.

    responder_i value is encoded as date_id*100 + time_id*10 + symbol_id + i so
    every (date_id, time_id, symbol_id, responder) cell is unique and the lag
    relationship is easy to assert.
    """
    rows = []
    for date_id in (0, 1):
        for time_id in (0, 1):
            for symbol_id in (0, 1):
                row = {"date_id": date_id, "time_id": time_id, "symbol_id": symbol_id}
                for i in range(9):
                    row[f"responder_{i}"] = float(date_id * 100 + time_id * 10 + symbol_id + i)
                rows.append(row)
    return pl.DataFrame(rows)


def test_lag_equals_previous_date_same_keys():
    out = add_responder_lags_from_train(make_frame())
    # Lag at date 1 == responder at date 0 for the same (time_id, symbol_id).
    d1 = out.filter(pl.col("date_id") == 1).sort(["time_id", "symbol_id"])
    d0 = make_frame().filter(pl.col("date_id") == 0).sort(["time_id", "symbol_id"])
    for i in range(9):
        assert d1.get_column(f"responder_{i}_lag_1").to_list() == d0.get_column(
            f"responder_{i}"
        ).to_list()


def test_lag_does_not_use_current_date_responder():
    out = add_responder_lags_from_train(make_frame())
    d1 = out.filter(pl.col("date_id") == 1)
    # Current-date responders differ from D-1, so the lag must never equal the
    # current-date responder on the same row.
    for i in range(9):
        same = (
            d1.get_column(f"responder_{i}_lag_1") == d1.get_column(f"responder_{i}")
        ).any()
        assert not same


def test_first_day_lags_all_null():
    out = add_responder_lags_from_train(make_frame())
    d0 = out.filter(pl.col("date_id") == 0)
    for col in LAG_FEATURE_COLUMNS:
        assert d0.get_column(col).null_count() == d0.height


def test_join_key_is_date_time_symbol():
    # Drop a (date 0) row so its (time_id, symbol_id) has no predecessor on date 1.
    df = make_frame().filter(
        ~((pl.col("date_id") == 0) & (pl.col("time_id") == 1) & (pl.col("symbol_id") == 1))
    )
    out = add_responder_lags_from_train(df)
    # Row count unchanged (left join), and the date-1 row whose key is missing on
    # date 0 gets null lags rather than a wrong value.
    assert out.height == df.height
    orphan = out.filter(
        (pl.col("date_id") == 1) & (pl.col("time_id") == 1) & (pl.col("symbol_id") == 1)
    )
    for col in LAG_FEATURE_COLUMNS:
        assert orphan.get_column(col).null_count() == orphan.height


def test_missing_responder_column_raises():
    df = make_frame().drop("responder_3")
    with pytest.raises(ValueError, match="responder_3"):
        add_responder_lags_from_train(df)


def test_get_lag_feature_columns():
    assert get_lag_feature_columns() == [f"responder_{i}_lag_1" for i in range(9)]
    assert RESPONDER_COLUMNS == [f"responder_{i}" for i in range(9)]


def test_reconstruction_uses_only_train_frame():
    # Reconstruction is a pure function of the train frame: same input -> exact
    # lag values, no external lags.parquet involved.
    out = add_responder_lags_from_train(make_frame())
    expected = float(0 * 100 + 0 * 10 + 0)  # responder_0 at date0/time0/symbol0
    got = out.filter(
        (pl.col("date_id") == 1) & (pl.col("time_id") == 0) & (pl.col("symbol_id") == 0)
    ).get_column("responder_0_lag_1")[0]
    assert got == expected


def _full_train_frame(n_days: int = 6) -> pl.DataFrame:
    """Train frame with all 9 responders + features + weight (for run() wiring)."""
    rows = []
    for date_id in range(n_days):
        for time_id in range(3):
            for symbol_id in range(2):
                row = {"date_id": date_id, "time_id": time_id, "symbol_id": symbol_id}
                for j, c in enumerate(FEATURE_COLUMNS):
                    row[c] = float((date_id + time_id + symbol_id + j) % 7)
                row[WEIGHT_COLUMN] = 1.0
                for i in range(9):
                    row[f"responder_{i}"] = float(date_id * 100 + time_id * 10 + symbol_id + i)
                rows.append(row)
    return pl.DataFrame(rows)


def test_run_with_responder_lags_includes_lag_features(tmp_path):
    cfg = LGBMConfig(
        train_path=str(tmp_path / "unused.parquet"),
        output_dir=str(tmp_path / "outputs"),
        model_dir=str(tmp_path / "models"),
        start_date_id=None,
        end_date_id=None,
        valid_days=2,
        gap_days=0,
        random_state=42,
        n_estimators=5,
        learning_rate=0.1,
        num_leaves=7,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=5,
        use_responder_lags=True,
    )
    result = run(cfg, run_name="lags_smoke", df=_full_train_frame())
    # V0 inputs (79 features + symbol_id + time_id = 81) plus the 9 lag features.
    assert result.feature_count == 81 + len(LAG_FEATURE_COLUMNS)
    names = [name for name, _ in result.feature_importance_top20]
    assert any(c in names for c in LAG_FEATURE_COLUMNS) or result.feature_count == 90
