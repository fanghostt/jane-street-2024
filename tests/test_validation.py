import polars as pl
import pytest

from js2024.validation import (
    DateRangeSplit,
    build_holdout_split,
    filter_by_date_range,
    summarize_date_split,
)


def test_normal_holdout_split():
    split = build_holdout_split(min_date_id=0, max_date_id=999, valid_days=200, gap_days=0)
    assert split.valid_end == 999
    assert split.valid_start == 800
    assert split.train_end == 799
    assert split.train_start == 0


def test_gap_days_applied():
    split = build_holdout_split(min_date_id=0, max_date_id=999, valid_days=200, gap_days=10)
    assert split.valid_start == 800
    # gap removes 10 days before valid_start: train_end = 800 - 10 - 1 = 789
    assert split.train_end == 789


def test_insufficient_history_raises():
    with pytest.raises(ValueError):
        build_holdout_split(min_date_id=0, max_date_id=100, valid_days=200, gap_days=0)


def test_filter_inclusive_behaviour():
    df = pl.DataFrame({"date_id": list(range(10))})
    out = filter_by_date_range(df, "date_id", 2, 5)
    assert out.get_column("date_id").to_list() == [2, 3, 4, 5]

    # None bounds leave that side unbounded.
    assert filter_by_date_range(df, "date_id", None, 1).height == 2
    assert filter_by_date_range(df, "date_id", 8, None).height == 2


def test_summary_rows_and_days():
    # 3 dates, 2 rows each = 6 rows total.
    df = pl.DataFrame(
        {"date_id": [0, 0, 1, 1, 2, 2], "x": range(6)}
    )
    split = DateRangeSplit(train_start=0, train_end=1, valid_start=2, valid_end=2)
    summary = summarize_date_split(df, split, date_col="date_id")
    assert summary["train_rows"] == 4
    assert summary["valid_rows"] == 2
    assert summary["train_days"] == 2
    assert summary["valid_days"] == 1
