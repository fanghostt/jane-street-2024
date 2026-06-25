"""CLI tests for the config-driven experiment runner (tiny fake parquet)."""

from __future__ import annotations

import numpy as np
import polars as pl

from js2024.runners.run_experiment import main

FEATURES = [f"feature_{i:02d}" for i in range(79)]


def _write_fake_train(path, n_days=12, symbols=4, seed=0):
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


def _write_config(path, train_path):
    path.write_text(
        "\n".join(
            [
                "model: gru",
                "variants: [full, incremental]",
                f"train_path: {train_path}",
                "output_dir: outputs",
                "model_dir: models",
                "start_date_id: 0",
                "end_date_id: null",
                "test_days: 2",
                "valid_days: 2",
                "gap_days: 0",
                "update_cadence: 1",
                "random_state: 42",
                "device: cpu",
                "include_time: false",
                "seq_len: 3",
                "hidden_size: 4",
                "num_layers: 1",
                "dropout: 0.0",
                "lr: 0.01",
                "weight_decay: 0.0",
                "epochs: 1",
                "batch_size: 16",
                "early_stopping_rounds: 1",
                "finetune_epochs: 1",
                "finetune_lr: 0.005",
                "train_mode: stream",
            ]
        ),
        encoding="utf-8",
    )


def test_cli_runs_full_variant_only(tmp_path):
    train = tmp_path / "train.parquet"
    _write_fake_train(train)
    cfg = tmp_path / "cfg.yaml"
    _write_config(cfg, train)
    out_dir = tmp_path / "out"
    docs = tmp_path / "doc.md"

    rc = main(
        [
            "--config",
            str(cfg),
            "--variants",
            "full",
            "--out-dir",
            str(out_dir),
            "--docs-out",
            str(docs),
        ]
    )
    assert rc == 0

    summary = pl.read_csv(out_dir / "summary.csv")
    assert summary.get_column("variant").to_list() == ["gru_full"]
    assert summary.get_column("n_updates").to_list() == [0]
    assert summary.get_column("model").to_list() == ["gru"]
    assert "gru_full" in docs.read_text(encoding="utf-8")


def _write_lgbm_config(path, train_path):
    path.write_text(
        "\n".join(
            [
                "model: lgbm",
                "variants: [full, incremental]",
                f"train_path: {train_path}",
                "output_dir: outputs",
                "model_dir: models",
                "start_date_id: 0",
                "end_date_id: null",
                "test_days: 2",
                "valid_days: 2",
                "gap_days: 0",
                "update_methods: [refit, continue, retrain]",
                "update_cadence: 1",
                "refit_decay: 0.9",
                "continue_rounds: 2",
                "random_state: 42",
                "n_estimators: 20",
                "learning_rate: 0.1",
                "num_leaves: 7",
                "subsample: 1.0",
                "colsample_bytree: 1.0",
                "early_stopping_rounds: 5",
                "device_type: cpu",
            ]
        ),
        encoding="utf-8",
    )


def test_lgbm_incremental_expands_into_named_methods(tmp_path):
    train = tmp_path / "train.parquet"
    _write_fake_train(train, n_days=10, symbols=4)
    cfg = tmp_path / "cfg.yaml"
    _write_lgbm_config(cfg, train)
    out_dir = tmp_path / "out"
    docs = tmp_path / "doc.md"

    rc = main(
        ["--config", str(cfg), "--out-dir", str(out_dir), "--docs-out", str(docs)]
    )
    assert rc == 0

    summary = pl.read_csv(out_dir / "summary.csv")
    variants = summary.get_column("variant").to_list()
    # `incremental` expanded into one named row per online strategy — never the
    # generic `lgbm_incremental`.
    assert variants == ["lgbm_full", "lgbm_refit", "lgbm_continue", "lgbm_retrain"]
    assert "lgbm_incremental" not in variants
    doc_text = docs.read_text(encoding="utf-8")
    assert "lgbm_retrain" in doc_text
    # full never updates; the online variants do (cadence=1 over a 2-day block).
    by_variant = dict(zip(variants, summary.get_column("n_updates").to_list()))
    assert by_variant["lgbm_full"] == 0
    assert by_variant["lgbm_refit"] >= 1


def test_cli_unknown_model_errors(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("model: nope\ntrain_path: x\nvalid_days: 2\n", encoding="utf-8")
    rc = main(["--config", str(cfg)])
    assert rc == 1
