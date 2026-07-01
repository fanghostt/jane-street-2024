from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from js2024.modeling.gru import (
    GRU_AUX_COLUMNS,
    GRU_AUX_TARGET_SETS,
    GRU_DEFAULT_PARAMS,
    GRUEstimator,
    add_gru_aux_targets,
    get_gru_feature_columns,
    resolve_gru_aux_targets,
)
from js2024.runners.run_experiment import main


FEATURES = [f"feature_{i:02d}" for i in range(79)]


def test_gru_feature_columns_drop_categoricals_and_include_time():
    cols = get_gru_feature_columns(include_time=True)
    assert "feature_09" not in cols
    assert "feature_10" not in cols
    assert "feature_11" not in cols
    assert "time_id" in cols


def test_add_gru_aux_targets():
    df = pl.DataFrame(
        {
            "symbol_id": [0, 0, 0, 0, 0],
            "responder_6": [1.0, 2.0, 3.0, 4.0, 5.0],
            "responder_7": [10.0, 11.0, 12.0, 13.0, 14.0],
            "responder_8": [20.0, 21.0, 22.0, 23.0, 24.0],
        }
    )
    out = add_gru_aux_targets(df)
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
                # All 9 real responders so non-base4 aux sets (all9/all11) load.
                for i in range(9):
                    row[f"responder_{i}"] = float(0.1 * row[f"feature_{i:02d}"] + 0.1 * rng.normal())
                row["responder_6"] = float(0.2 * row["feature_00"] + 0.1 * rng.normal())
                row["weight"] = 1.0
                rows.append(row)
    pl.DataFrame(rows).write_parquet(path)


def test_resolve_aux_base4_matches_legacy_constant():
    # base4 must reproduce the public-solution default verbatim (order + contents)
    # so existing GRU configs stay bit-for-bit identical.
    assert resolve_gru_aux_targets("base4") == GRU_AUX_COLUMNS
    assert GRU_AUX_TARGET_SETS["base4"] == ["responder_10", "responder_9", "responder_8", "responder_7"]


def test_resolve_aux_sets_membership():
    assert resolve_gru_aux_targets("all9") == [f"responder_{i}" for i in range(9)]
    assert resolve_gru_aux_targets("all11") == [f"responder_{i}" for i in range(11)]
    assert resolve_gru_aux_targets("target_family") == [
        "responder_6", "responder_7", "responder_8", "responder_9", "responder_10"
    ]


def test_resolve_aux_unknown_raises():
    with pytest.raises(ValueError, match="unknown aux_target_set"):
        resolve_gru_aux_targets("nope")


def test_estimator_aux_cols_default_and_override():
    cols = get_gru_feature_columns(include_time=True)
    assert GRUEstimator(cols).aux_cols == GRU_AUX_COLUMNS  # default == base4
    est = GRUEstimator(cols, aux_cols=resolve_gru_aux_targets("all9"))
    assert est.aux_cols == [f"responder_{i}" for i in range(9)]


def _build_test_model(architecture, n_features=6, n_aux=4, extra=None):
    from js2024.modeling.gru import _build_model

    params = {
        **GRU_DEFAULT_PARAMS,
        "architecture": architecture,
        "hidden_sizes": [8],
        "dropout_rates": [0.0],
        "hidden_sizes_linear": [8],
        "dropout_rates_linear": [0.0],
    }
    if extra:
        params.update(extra)
    return _build_model(n_features, params, n_aux)


def _forward_shapes(architecture, extra=None):
    torch = pytest.importorskip("torch")
    model = _build_test_model(architecture, extra=extra)
    x = torch.randn(3, 5, 6)  # 3 symbols x 5 time_ids x 6 features
    y, aux, _ = model(x, None)
    return tuple(y.shape), tuple(aux.shape)


def test_architecture_defaults_to_gru_mlp():
    from js2024.modeling.config import GRUConfig

    assert GRUConfig.architecture == "gru_mlp"
    assert GRU_DEFAULT_PARAMS["architecture"] == "gru_mlp"


def test_deep_wide_gru_forward_shape_matches_gru_mlp():
    base = _forward_shapes("gru_mlp")
    dw = _forward_shapes(
        "deep_wide_gru",
        {"wide_hidden_sizes": [8], "wide_dropout_rates": [0.0]},
    )
    assert dw == base


def test_deep_wide_residual_forward_shape_matches_gru_mlp():
    base = _forward_shapes("gru_mlp")
    res = _forward_shapes(
        "deep_wide_residual",
        {"wide_hidden_sizes": [8], "wide_dropout_rates": [0.0]},
    )
    assert res == base


def test_gru_mlp_builds_no_wide_branch():
    # gru_mlp must not allocate wide/fusion modules so old models are untouched.
    pytest.importorskip("torch")
    model = _build_test_model("gru_mlp")
    names = [n for n, _ in model.named_modules()]
    assert not any("wide" in n or "fusion" in n for n in names)
    # the deep_wide variant, by contrast, does create them.
    dw = _build_test_model(
        "deep_wide_gru", extra={"wide_hidden_sizes": [8], "wide_dropout_rates": [0.0]}
    )
    dw_names = [n for n, _ in dw.named_modules()]
    assert any("wide" in n for n in dw_names)
    assert any("fusion" in n for n in dw_names)


def _write_config(path, train_path, *, model="gru", extra=()):
    path.write_text(
        "\n".join(
            [
                f"model: {model}",
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
                *extra,
            ]
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    "model, extra",
    [
        ("gru", ()),
        ("transformer", ("num_heads: 2",)),  # hidden_size 4 must divide num_heads
        ("tcn", ("kernel_size: 2",)),
    ],
)
def test_seq_backbones_run_via_generic_runner(tmp_path, model, extra):
    train = tmp_path / "train.parquet"
    _write_fake_train(train)
    cfg = tmp_path / "cfg.yaml"
    _write_config(cfg, train, model=model, extra=extra)
    out = tmp_path / "out"

    rc = main(["--config", str(cfg), "--variants", "full", "--out-dir", str(out)])

    assert rc == 0
    summary = pl.read_csv(out / "summary.csv")
    assert summary.get_column("variant").to_list() == [f"{model}_full"]
    assert (out / "manifest.json").exists()


@pytest.mark.parametrize(
    "extra",
    [
        (
            "architecture: deep_wide_gru",
            "wide_hidden_sizes: [4]",
            "wide_dropout_rates: [0.0]",
        ),
        (
            "architecture: deep_wide_residual",
            "wide_hidden_sizes: [4]",
            "wide_dropout_rates: [0.0]",
            "wide_residual_scale: 0.1",
        ),
    ],
)
def test_deep_wide_architectures_run_via_generic_runner(tmp_path, extra):
    train = tmp_path / "train.parquet"
    _write_fake_train(train)
    cfg = tmp_path / "cfg.yaml"
    _write_config(cfg, train, extra=extra)
    out = tmp_path / "out"

    rc = main(["--config", str(cfg), "--variants", "full", "--out-dir", str(out)])

    assert rc == 0
    summary = pl.read_csv(out / "summary.csv")
    assert summary.get_column("variant").to_list() == ["gru_full"]


@pytest.mark.parametrize("aux_set", ["base4", "target_family", "all9", "all11"])
def test_aux_target_sets_run_via_generic_runner(tmp_path, aux_set):
    train = tmp_path / "train.parquet"
    _write_fake_train(train)
    cfg = tmp_path / "cfg.yaml"
    _write_config(cfg, train, extra=(f"aux_target_set: {aux_set}",))
    out = tmp_path / "out"

    rc = main(["--config", str(cfg), "--variants", "full", "--out-dir", str(out)])

    assert rc == 0
    summary = pl.read_csv(out / "summary.csv")
    assert summary.get_column("variant").to_list() == ["gru_full"]


def test_online_update_uses_target_loss_regardless_of_aux_set(tmp_path):
    # The online update() path trains on the target (responder_6) loss only, so it
    # runs unchanged for any aux set — including ones with more heads than base4.
    train = tmp_path / "train.parquet"
    _write_fake_train(train)
    df = pl.read_parquet(train)
    df = add_gru_aux_targets(df)
    cols = get_gru_feature_columns(include_time=True)
    est = GRUEstimator(
        cols,
        aux_cols=resolve_gru_aux_targets("all9"),
        params={"epochs": 1, "hidden_sizes": [4], "dropout_rates": [0.0],
                "hidden_sizes_linear": [4], "dropout_rates_linear": [0.0],
                "lr": 1e-4, "lr_refit": 1e-4},
        device="cpu",
    )
    train_days = df.filter(pl.col("date_id") < 5)
    new_days = df.filter(pl.col("date_id") >= 5)
    est.fit(train_days, train_days)
    # update() optimizes the target (responder_6) loss only — it builds no aux loss,
    # so a 9-head aux set fine-tunes online exactly like base4.
    est.update(new_days)
    preds = est.predict(new_days)
    assert preds.shape[0] == new_days.height


def test_aux_sweep_paired_and_summary():
    from js2024.runners.run_aux_sweep import paired_rows, summarize_cells

    runs = [
        {"aux_set": "base4", "seed": 42, "score": 0.010},
        {"aux_set": "base4", "seed": 43, "score": 0.012},
        {"aux_set": "all9", "seed": 42, "score": 0.011},   # +0.001
        {"aux_set": "all9", "seed": 43, "score": 0.013},   # +0.001
    ]
    paired = paired_rows(runs)
    assert {p["aux_set"] for p in paired} == {"all9"}
    assert all(abs(p["delta"] - 0.001) < 1e-9 for p in paired)
    summary = summarize_cells(paired)
    assert len(summary) == 1
    assert summary[0]["aux_set"] == "all9"
    assert summary[0]["n_positive"] == 2
    assert abs(summary[0]["mean_delta"] - 0.001) < 1e-9
