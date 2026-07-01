import pytest

from js2024.modeling.config import (
    GRUConfig,
    PROJECT_ROOT,
    LGBMConfig,
    gru_params,
    load_gru_config,
    load_lgbm_config,
    validate_gru_config,
    validate_lgbm_config,
)


def _valid_config(**overrides) -> LGBMConfig:
    base = dict(
        train_path="data/interim/train_smoke.parquet",
        output_dir="outputs",
        model_dir="models",
        start_date_id=None,
        end_date_id=None,
        valid_days=3,
        gap_days=0,
        random_state=42,
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=20,
    )
    base.update(overrides)
    return LGBMConfig(**base)


def test_smoke_config_loads():
    cfg = load_lgbm_config(PROJECT_ROOT / "configs" / "smoke" / "lgbm_v0_smoke.yaml")
    assert cfg.train_path == "data/interim/train_smoke.parquet"
    assert cfg.valid_days == 3
    assert cfg.gap_days == 0
    assert cfg.n_estimators == 100
    assert cfg.early_stopping_rounds == 20
    assert cfg.start_date_id is None and cfg.end_date_id is None
    assert cfg.device_type == "cpu"
    assert cfg.max_bin == 255


def test_v0_config_loads():
    cfg = load_lgbm_config(PROJECT_ROOT / "configs" / "lgbm_v0.yaml")
    assert cfg.valid_days == 200
    assert cfg.device_type == "gpu"
    assert cfg.max_bin == 255


def test_recent700_config_defaults_to_gpu():
    cfg = load_lgbm_config(PROJECT_ROOT / "configs" / "lgbm_v0_recent700.yaml")
    assert cfg.device_type == "gpu"
    assert cfg.max_bin == 255
    assert cfg.gpu_use_dp is False


def test_use_responder_lags_defaults_false():
    cfg = load_lgbm_config(PROJECT_ROOT / "configs" / "lgbm_v0_recent700.yaml")
    assert cfg.use_responder_lags is False


def test_lags_v1_config_enables_responder_lags():
    cfg = load_lgbm_config(PROJECT_ROOT / "configs" / "lgbm_lags_v1_recent700.yaml")
    assert cfg.use_responder_lags is True
    assert cfg.start_date_id == 700
    assert cfg.valid_days == 200


def test_market_roll_config_enables_engineered_features():
    cfg = load_lgbm_config(PROJECT_ROOT / "configs" / "lgbm_marketroll_v1_recent700.yaml")
    assert cfg.use_market_avg is True
    assert cfg.use_symbol_rolling is True
    assert cfg.rolling_window == 1000
    base = load_lgbm_config(PROJECT_ROOT / "configs" / "lgbm_v0_recent700.yaml")
    assert base.use_market_avg is False and base.use_symbol_rolling is False


def test_gru_config_loads():
    cfg = load_gru_config(PROJECT_ROOT / "configs" / "gru_v0.yaml")
    assert cfg.train_path == "data/raw/train.parquet"
    assert cfg.start_date_id == 700
    assert cfg.hidden_sizes == [500]
    assert cfg.hidden_sizes_linear == [500, 300]
    assert cfg.lr == 0.0005
    assert cfg.lr_refit == 0.0003
    assert cfg.epochs == 1000
    params = gru_params(cfg)
    assert params["hidden_sizes"] == [500]
    assert params["early_stopping_patience"] == 1


def test_validate_gru_rejects_mismatched_dropouts():
    cfg = GRUConfig(
        train_path="x",
        start_date_id=700,
        end_date_id=None,
        valid_days=200,
        gap_days=0,
        random_state=42,
        hidden_sizes=[500, 300],
        dropout_rates=[0.3],
    )
    with pytest.raises(ValueError, match="dropout_rates length"):
        validate_gru_config(cfg)


def test_gru_architecture_defaults_to_gru_mlp():
    cfg = load_gru_config(PROJECT_ROOT / "configs" / "gru_v0.yaml")
    assert cfg.architecture == "gru_mlp"
    params = gru_params(cfg)
    assert params["architecture"] == "gru_mlp"
    # No wide/fusion shapes flow through unless a config opts in.
    assert params["wide_hidden_sizes"] is None
    assert params["fusion_hidden_sizes"] is None


def test_deepwide_config_loads():
    cfg = load_gru_config(PROJECT_ROOT / "configs" / "deepwide_gru_w128.yaml")
    assert cfg.architecture == "deep_wide_gru"
    assert cfg.wide_hidden_sizes == [128, 128]
    assert cfg.wide_dropout_rates == [0.2, 0.1]
    # Same data/aux settings as the gru_marketroll_v1 SOTA candidate.
    assert cfg.aux_target_set == "all9"
    assert cfg.use_market_avg is True and cfg.use_symbol_rolling is True


def test_validate_gru_rejects_mismatched_wide_lengths():
    cfg = GRUConfig(
        train_path="x",
        start_date_id=700,
        end_date_id=None,
        valid_days=200,
        gap_days=0,
        random_state=42,
        architecture="deep_wide_gru",
        wide_hidden_sizes=[128, 128],
        wide_dropout_rates=[0.1],
    )
    with pytest.raises(ValueError, match="wide_dropout_rates length"):
        validate_gru_config(cfg)


def test_validate_gru_rejects_unknown_architecture():
    cfg = GRUConfig(
        train_path="x",
        start_date_id=700,
        end_date_id=None,
        valid_days=200,
        gap_days=0,
        random_state=42,
        architecture="transformer_wide",
    )
    with pytest.raises(ValueError, match="architecture must be one of"):
        validate_gru_config(cfg)


def test_validate_accepts_valid_config():
    cfg = _valid_config()
    assert validate_lgbm_config(cfg) is cfg


@pytest.mark.parametrize(
    "overrides",
    [
        {"valid_days": 0},
        {"gap_days": -1},
        {"n_estimators": 0},
        {"early_stopping_rounds": 0},
        {"learning_rate": 0.0},
        {"num_leaves": 1},
        {"subsample": 0.0},
        {"subsample": 1.5},
        {"colsample_bytree": 0.0},
        {"colsample_bytree": 1.01},
    ],
)
def test_validate_rejects_bad_config(overrides):
    with pytest.raises(ValueError):
        validate_lgbm_config(_valid_config(**overrides))


def test_load_invalid_config_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "train_path: x\noutput_dir: o\nmodel_dir: m\nvalid_days: 3\nnum_leaves: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as exc:
        load_lgbm_config(bad)
    assert "num_leaves" in str(exc.value)


def test_load_missing_required_key_raises(tmp_path):
    # No train_path -> KeyError from the dataclass construction.
    bad = tmp_path / "bad.yaml"
    bad.write_text("output_dir: o\nmodel_dir: m\nvalid_days: 3\n", encoding="utf-8")
    with pytest.raises(KeyError):
        load_lgbm_config(bad)
