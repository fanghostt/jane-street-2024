import polars as pl
import pytest

from js2024.data import (
    FEATURE_COLUMNS,
    ID_COLUMNS,
    TARGET_COLUMN,
    WEIGHT_COLUMN,
    get_date_id_range,
    get_default_columns,
    load_train_data,
    validate_data_path,
)


def _make_parquet(tmp_path):
    n = 12
    data = {"date_id": [i // 3 for i in range(n)]}  # date_ids 0..3, 3 rows each
    data["time_id"] = [i % 3 for i in range(n)]
    data["symbol_id"] = [0 for _ in range(n)]
    for c in FEATURE_COLUMNS:
        data[c] = [float(i) for i in range(n)]
    data[WEIGHT_COLUMN] = [1.0 for _ in range(n)]
    data[TARGET_COLUMN] = [0.1 * i for i in range(n)]
    df = pl.DataFrame(data)
    path = tmp_path / "train.parquet"
    df.write_parquet(path)
    return path


def test_default_columns():
    cols = get_default_columns()
    assert cols[: len(ID_COLUMNS)] == ID_COLUMNS
    assert WEIGHT_COLUMN in cols
    assert TARGET_COLUMN in cols
    assert len(cols) == len(ID_COLUMNS) + len(FEATURE_COLUMNS) + 2
    # toggles
    assert TARGET_COLUMN not in get_default_columns(include_target=False)
    assert WEIGHT_COLUMN not in get_default_columns(include_weight=False)


def test_date_filter(tmp_path):
    path = _make_parquet(tmp_path)
    df = load_train_data(path, start_date_id=1, end_date_id=2)
    assert get_date_id_range(df) == (1, 2)
    assert df.height == 6


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_data_path(tmp_path / "nope.parquet")
    with pytest.raises(FileNotFoundError):
        load_train_data(tmp_path / "nope.parquet")


def test_missing_columns_raises(tmp_path):
    path = _make_parquet(tmp_path)
    with pytest.raises(ValueError) as exc:
        load_train_data(path, columns=["date_id", "feature_99"])
    assert "feature_99" in str(exc.value)


def test_collect_false_returns_lazyframe(tmp_path):
    path = _make_parquet(tmp_path)
    lf = load_train_data(path, collect=False)
    assert isinstance(lf, pl.LazyFrame)
    assert isinstance(lf.collect(), pl.DataFrame)
