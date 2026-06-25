"""CLI test for the incremental-vs-full runner (tiny fake parquet, no real data)."""

from __future__ import annotations

import numpy as np
import polars as pl

from js2024.runners.run_incremental_vs_full import main

FEATURES = [f"feature_{i:02d}" for i in range(79)]


def _write_fake_train(path, n_days=20, symbols=4, seed=0):
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
    pl.DataFrame(rows).write_parquet(path)


def _write_config(path, train_path, *, test_days=4, valid_days=3):
    path.write_text(
        "\n".join(
            [
                f"train_path: {train_path}",
                "output_dir: outputs",
                "model_dir: models",
                "start_date_id: 0",
                "end_date_id: null",
                f"test_days: {test_days}",
                f"valid_days: {valid_days}",
                "gap_days: 0",
                "update_method: refit",
                "update_cadence: 1",
                "refit_decay: 0.9",
                "random_state: 42",
                "n_estimators: 15",
                "learning_rate: 0.1",
                "num_leaves: 7",
                "subsample: 0.8",
                "colsample_bytree: 0.8",
                "early_stopping_rounds: 5",
            ]
        ),
        encoding="utf-8",
    )


def test_dry_run(tmp_path, capsys):
    train = tmp_path / "train.parquet"
    _write_fake_train(train)
    cfg = tmp_path / "cfg.yaml"
    _write_config(cfg, train)
    rc = main(["--config", str(cfg), "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Dry run" in out


def test_cli_runs_all_variants_and_writes_doc(tmp_path):
    train = tmp_path / "train.parquet"
    _write_fake_train(train)
    cfg = tmp_path / "cfg.yaml"
    _write_config(cfg, train)
    out_dir = tmp_path / "out"
    docs = tmp_path / "doc.md"

    rc = main(
        [
            "--config", str(cfg),
            "--methods", "refit,continue,retrain",
            "--retrain-cadence", "2",  # exercise retrain on the 4-day test block
            "--out-dir", str(out_dir),
            "--docs-out", str(docs),
        ]
    )
    assert rc == 0

    text = docs.read_text(encoding="utf-8")
    for label in ("full", "refit", "continue", "retrain"):
        assert label in text
    assert "fixed test block" in text

    summary = pl.read_csv(out_dir / "summary.csv")
    by = {r["variant"]: r for r in summary.to_dicts()}
    assert set(by) == {"full", "refit", "continue", "retrain"}
    assert by["full"]["n_updates"] == 0
    assert by["refit"]["n_updates"] == 3  # 4 test days, daily -> 3
    assert by["continue"]["n_updates"] == 3
    assert by["retrain"]["n_updates"] == 1  # cadence 2 over 4 days -> 1


def test_cli_subset_of_methods(tmp_path):
    train = tmp_path / "train.parquet"
    _write_fake_train(train)
    cfg = tmp_path / "cfg.yaml"
    _write_config(cfg, train)
    out_dir = tmp_path / "out"
    rc = main(
        ["--config", str(cfg), "--methods", "refit", "--out-dir", str(out_dir),
         "--docs-out", str(tmp_path / "d.md")]
    )
    assert rc == 0
    summary = pl.read_csv(out_dir / "summary.csv")
    assert set(summary.get_column("variant").to_list()) == {"full", "refit"}
