import pytest

from js2024.data.data_contract import (
    RAW_REQUIRED_FILES,
    check_raw_data_contract,
    contract_ok,
)


def test_contract_passes_single_file(raw_dir_factory):
    raw = raw_dir_factory(partitioned=False)
    report = check_raw_data_contract(raw)
    assert report["exists"] is True
    assert contract_ok(report)
    assert report["missing_train_columns"] == []
    assert report["train_columns_count"] == 3 + 79 + 2  # ids + features + weight + target
    assert report["sample_rows_checked"] > 0
    assert report["date_min"] == 0
    assert report["date_max"] >= report["date_min"]
    names = {f["name"] for f in report["required_files"]}
    assert names == set(RAW_REQUIRED_FILES)
    assert all(f["exists"] for f in report["required_files"])


def test_contract_passes_partitioned_dir(raw_dir_factory):
    raw = raw_dir_factory(partitioned=True)
    report = check_raw_data_contract(raw)
    assert contract_ok(report)
    # The train "file" is actually a partitioned directory.
    train_entry = next(f for f in report["required_files"] if f["name"] == "train.parquet")
    assert train_entry["type"] == "dir"
    assert "**" in report["train_scan_path"]


def test_contract_missing_required_file(raw_dir_factory):
    raw = raw_dir_factory(missing_files=["features.csv"])
    report = check_raw_data_contract(raw)
    assert not contract_ok(report)
    features = next(f for f in report["required_files"] if f["name"] == "features.csv")
    assert features["exists"] is False


def test_contract_missing_train_columns(raw_dir_factory):
    raw = raw_dir_factory(drop_train_columns=["weight", "feature_05"])
    report = check_raw_data_contract(raw)
    assert not contract_ok(report)
    assert "weight" in report["missing_train_columns"]
    assert "feature_05" in report["missing_train_columns"]


def test_contract_nonexistent_dir(tmp_path):
    report = check_raw_data_contract(tmp_path / "does_not_exist")
    assert report["exists"] is False
    assert not contract_ok(report)
    assert report["train_scan_path"] is None
