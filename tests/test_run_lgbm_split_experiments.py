"""Tests for the V0 split-experiment runner. Offline; no real Kaggle data."""

from __future__ import annotations

import yaml

from js2024.runners.run_lgbm_split_experiments import (
    build_grid,
    main,
    make_run_name,
    parse_int_list,
)


def _write_base_config(tmp_path, train_path, **overrides):
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
    path = tmp_path / "base_config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


# --- pure helpers ----------------------------------------------------------


def test_parse_int_list():
    assert parse_int_list("100,200,300") == [100, 200, 300]
    assert parse_int_list(" 5 , 10 ") == [5, 10]


def test_build_grid_3x3():
    grid = build_grid([100, 200, 300], [0, 5, 20])
    assert len(grid) == 9
    # valid_days varies slowest.
    assert grid[0] == (100, 0)
    assert grid[1] == (100, 5)
    assert grid[3] == (200, 0)


def test_make_run_name():
    assert make_run_name("lgbm_v0_recent700", 100, 5) == "lgbm_v0_recent700_v100_g5"


# --- CLI behavior ----------------------------------------------------------


def test_dry_run_does_not_train(tmp_path, write_train, capsys):
    train = write_train()
    base = _write_base_config(tmp_path, train)
    docs_out = tmp_path / "docs.md"
    rc = main(
        [
            "--base-config", str(base),
            "--valid-days", "100,200,300",
            "--gap-days", "0,5,20",
            "--out-dir", str(tmp_path / "out"),
            "--docs-out", str(docs_out),
            "--allow-non-700-start",
            "--dry-run",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert out.count("(valid_days=") == 9  # all 9 run names printed
    assert "Dry run" in out
    assert not docs_out.exists()  # nothing written
    assert not (tmp_path / "out").exists()


def test_limit_restricts_grid(tmp_path, write_train, capsys):
    train = write_train()
    base = _write_base_config(tmp_path, train)
    rc = main(
        [
            "--base-config", str(base),
            "--out-dir", str(tmp_path / "out"),
            "--docs-out", str(tmp_path / "docs.md"),
            "--allow-non-700-start",
            "--dry-run",
            "--limit", "2",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert out.count("(valid_days=") == 2


def test_rejects_non_700_start_without_flag(tmp_path, write_train, capsys):
    train = write_train()
    base = _write_base_config(tmp_path, train, start_date_id=None)
    rc = main(
        [
            "--base-config", str(base),
            "--out-dir", str(tmp_path / "out"),
            "--docs-out", str(tmp_path / "docs.md"),
            "--dry-run",
        ]
    )
    err = capsys.readouterr().err
    assert rc == 1
    assert "start_date_id" in err


def test_end_to_end_single_split(tmp_path, write_train):
    # Tiny fixture: date_ids 0..3. One split, 3 trees.
    train = write_train()
    base = _write_base_config(tmp_path, train)
    out_dir = tmp_path / "out"
    docs_out = tmp_path / "split_docs.md"

    rc = main(
        [
            "--base-config", str(base),
            "--valid-days", "1",
            "--gap-days", "0",
            "--out-dir", str(out_dir),
            "--docs-out", str(docs_out),
            "--allow-non-700-start",
            "--n-estimators", "3",
        ]
    )
    assert rc == 0

    assert (out_dir / "summary.csv").exists()
    assert (out_dir / "summary.md").exists()
    assert docs_out.exists()

    # Per-split artifacts are namespaced by run_name under out_dir.
    run_name = make_run_name("base_config", 1, 0)
    assert (out_dir / "models" / f"{run_name}.txt").exists()
    assert (out_dir / "oof" / f"{run_name}_valid_predictions.parquet").exists()

    docs = docs_out.read_text(encoding="utf-8")
    assert "# LGBM V0 Split Experiments" in docs
    assert "status:** completed" in docs
    assert run_name in docs

    import polars as pl

    summary = pl.read_csv(out_dir / "summary.csv")
    assert summary.height == 1
    assert "score" in summary.columns
    assert summary["valid_days"][0] == 1
