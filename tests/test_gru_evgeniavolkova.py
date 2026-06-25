from __future__ import annotations

import numpy as np
import polars as pl

from js2024.modeling.gru_evgeniavolkova import (
    add_gru_evgeniavolkova_aux_targets,
    get_gru_evgeniavolkova_feature_columns,
)
from js2024.runners.run_experiment import main


FEATURES = [f"feature_{i:02d}" for i in range(79)]


def test_evgeniavolkova_feature_columns_drop_categoricals_and_include_time():
    cols = get_gru_evgeniavolkova_feature_columns(include_time=True)
    assert "feature_09" not in cols
    assert "feature_10" not in cols
    assert "feature_11" not in cols
    assert "time_id" in cols


def test_add_gru_evgeniavolkova_aux_targets():
    df = pl.DataFrame(
        {
            "symbol_id": [0, 0, 0, 0, 0],
            "responder_6": [1.0, 2.0, 3.0, 4.0, 5.0],
            "responder_7": [10.0, 11.0, 12.0, 13.0, 14.0],
            "responder_8": [20.0, 21.0, 22.0, 23.0, 24.0],
        }
    )
    out = add_gru_evgeniavolkova_aux_targets(df)
    assert "responder_9" in out.columns
    assert "responder_10" in out.columns
    # responder_9 = responder_8 + shift(-4) within symbol, fill null with 0.
    assert out.get_column("responder_9").to_list()[0] == 44.0
    assert out.get_column("responder_9").to_list()[-1] == 0.0


def _write_fake_train(path, n_days=7, n_times=3, n_symbols=2, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        for t in range(n_times):
            for s in range(n_symbols):
                row = {"date_id": d, "time_id": t, "symbol_id": s}
                for f in FEATURES:
                    row[f] = float(rng.normal())
                row["responder_6"] = float(0.2 * row["feature_00"] + 0.1 * rng.normal())
                row["responder_7"] = float(0.1 * row["feature_01"] + 0.1 * rng.normal())
                row["responder_8"] = float(0.1 * row["feature_02"] + 0.1 * rng.normal())
                row["weight"] = 1.0
                rows.append(row)
    pl.DataFrame(rows).write_parquet(path)


def _write_config(path, train_path):
    path.write_text(
        "\n".join(
            [
                "model: gru_evgeniavolkova",
                "variants: [full, incremental]",
                f"train_path: {train_path}",
                "start_date_id: 0",
                "end_date_id: null",
                "test_days: 2",
                "valid_days: 2",
                "gap_days: 0",
                "update_cadence: 1",
                "random_state: 42",
                "device: cpu",
                "include_time: true",
                "hidden_sizes: [4]",
                "dropout_rates: [0.0]",
                "hidden_sizes_linear: [4]",
                "dropout_rates_linear: [0.0]",
                "lr: 0.0001",
                "lr_refit: 0.0001",
                "epochs: 1",
                "early_stopping_patience: 1",
                "weight_decay: 0.0",
                "grad_clip: 1.0",
            ]
        ),
        encoding="utf-8",
    )


def test_evgeniavolkova_runs_via_generic_runner(tmp_path):
    train = tmp_path / "train.parquet"
    _write_fake_train(train)
    cfg = tmp_path / "cfg.yaml"
    _write_config(cfg, train)
    out = tmp_path / "out"

    rc = main(["--config", str(cfg), "--variants", "full", "--out-dir", str(out)])

    assert rc == 0
    summary = pl.read_csv(out / "summary.csv")
    assert summary.get_column("variant").to_list() == ["gru_evgeniavolkova_full"]
    assert (out / "manifest.json").exists()
