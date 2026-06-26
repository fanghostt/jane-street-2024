"""CLI + end-to-end tests for the LightGBM V0 baseline.

These never touch the real Kaggle data: the end-to-end test trains a 5-tree
model on the tiny conftest fixture, with all artifacts written under tmp_path.
"""

from __future__ import annotations

import yaml

from js2024.modeling.train_lgbm import main


def _write_config(tmp_path, train_path, **overrides):
    cfg = {
        "train_path": str(train_path),
        "output_dir": str(tmp_path / "outputs"),
        "model_dir": str(tmp_path / "models"),
        "start_date_id": None,
        "end_date_id": None,
        "valid_days": 1,
        "gap_days": 0,
        "random_state": 42,
        "n_estimators": 5,
        "learning_rate": 0.1,
        "num_leaves": 7,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "early_stopping_rounds": 5,
    }
    cfg.update(overrides)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_missing_train_path_returns_1(tmp_path, capsys):
    cfg = _write_config(tmp_path, tmp_path / "does_not_exist.parquet")
    rc = main(["--config", str(cfg)])
    err = capsys.readouterr().err
    assert rc == 1
    assert "[js2024] ERROR" in err


def test_invalid_config_returns_1(tmp_path, write_train, capsys):
    train = write_train()
    cfg = _write_config(tmp_path, train, num_leaves=1)  # invalid
    rc = main(["--config", str(cfg)])
    err = capsys.readouterr().err
    assert rc == 1
    assert "num_leaves" in err


def test_missing_config_file_returns_1(tmp_path, capsys):
    rc = main(["--config", str(tmp_path / "nope.yaml")])
    err = capsys.readouterr().err
    assert rc == 1
    assert "[js2024] ERROR" in err


def test_end_to_end_training(tmp_path, write_train):
    # Tiny fixture: date_ids 0..3; valid_days=1 -> train 0..2, valid 3.
    train = write_train()
    cfg = _write_config(tmp_path, train, valid_days=1, n_estimators=5)

    rc = main(["--config", str(cfg)])
    assert rc == 0

    # Artifacts are named after the config file stem (config.yaml -> "config").
    model_path = tmp_path / "models" / "config.txt"
    oof_path = tmp_path / "outputs" / "oof" / "config_valid_predictions.parquet"
    report_path = tmp_path / "outputs" / "reports" / "config_report.md"
    assert model_path.exists()
    assert oof_path.exists()
    assert report_path.exists()

    import polars as pl

    oof = pl.read_parquet(oof_path)
    assert "prediction" in oof.columns
    assert oof.height > 0
