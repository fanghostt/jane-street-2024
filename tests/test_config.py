import pytest

from js2024.modeling.config import (
    PROJECT_ROOT,
    LGBMConfig,
    load_lgbm_config,
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
    cfg = load_lgbm_config(PROJECT_ROOT / "configs" / "lgbm_v0_smoke.yaml")
    assert cfg.train_path == "data/interim/train_smoke.parquet"
    assert cfg.valid_days == 3
    assert cfg.gap_days == 0
    assert cfg.n_estimators == 100
    assert cfg.early_stopping_rounds == 20
    assert cfg.start_date_id is None and cfg.end_date_id is None


def test_v0_config_loads():
    cfg = load_lgbm_config(PROJECT_ROOT / "configs" / "lgbm_v0.yaml")
    assert cfg.valid_days == 200


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
