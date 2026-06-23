import polars as pl
import pytest

from js2024.data import (
    FEATURE_COLUMNS,
    ID_COLUMNS,
    TARGET_COLUMN,
    WEIGHT_COLUMN,
    get_date_id_range,
    get_default_columns,
    get_required_train_columns,
    load_train_data,
    resolve_parquet_scan_path,
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


def test_required_train_columns():
    assert get_required_train_columns() == get_default_columns(
        include_target=True, include_weight=True
    )


def test_resolve_scan_path_single_file(tmp_path):
    path = _make_parquet(tmp_path)
    resolved = resolve_parquet_scan_path(path)
    assert resolved == path
    assert load_train_data(path).height == 12


def test_resolve_scan_path_directory_with_parquet(tmp_path):
    # _make_parquet writes train.parquet directly under tmp_path.
    _make_parquet(tmp_path)
    resolved = resolve_parquet_scan_path(tmp_path)
    assert str(resolved) == str(tmp_path / "*.parquet")
    # load_train_data should read the directory as a glob.
    df = load_train_data(tmp_path)
    assert df.height == 12


def test_resolve_scan_path_partitioned_directory(tmp_path):
    # Mimic Kaggle's train.parquet/partition_id=*/part-0.parquet layout.
    root = tmp_path / "train.parquet"
    for pid in (0, 1):
        part_dir = root / f"partition_id={pid}"
        part_dir.mkdir(parents=True)
        df = pl.DataFrame({"date_id": [pid, pid], "x": [1.0, 2.0]})
        df.write_parquet(part_dir / "part-0.parquet")

    resolved = resolve_parquet_scan_path(root)
    assert str(resolved) == str(root / "**" / "*.parquet")
    out = pl.scan_parquet(resolved).collect()
    assert out.height == 4
    assert set(out.get_column("date_id").to_list()) == {0, 1}


def test_resolve_scan_path_empty_dir_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError) as exc:
        resolve_parquet_scan_path(empty)
    assert "no parquet" in str(exc.value).lower()


def test_resolve_scan_path_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_parquet_scan_path(tmp_path / "nope")
