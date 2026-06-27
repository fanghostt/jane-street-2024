"""Tests for the marketroll subset plumbing and the sweep aggregation helpers."""

import pytest

from js2024.modeling.market_features import (
    MARKET_ROLL_FEATURES,
    MARKET_ROLL_SUBSETS,
    resolve_market_roll_features,
    selected_columns,
)
from js2024.runners.run_marketroll_sweep import (
    paired_rows,
    summarize_cells,
)


# --- subset resolution ----------------------------------------------------


def test_top12_is_historical_list():
    # The default subset must stay byte-for-byte the historical top-12 so the
    # existing A/B (and existing configs) are unaffected.
    assert resolve_market_roll_features("top12") == MARKET_ROLL_FEATURES
    assert MARKET_ROLL_SUBSETS["top12"] == MARKET_ROLL_FEATURES


def test_subset_sizes_and_nesting():
    top12 = resolve_market_roll_features("top12")
    top24 = resolve_market_roll_features("top24")
    allf = resolve_market_roll_features("all")
    assert (len(top12), len(top24), len(allf)) == (12, 24, 79)
    # Subsets are prefixes of the same ranking — top12 ⊂ top24 ⊂ all.
    assert top24[:12] == top12
    assert allf[:24] == top24


def test_unknown_subset_raises():
    with pytest.raises(ValueError, match="unknown market_roll_subset"):
        resolve_market_roll_features("top99")


def test_selected_columns_scale_with_subset():
    feats = resolve_market_roll_features("top24")
    cols = selected_columns(use_market_avg=True, use_symbol_rolling=True, features=feats)
    # 1 market-avg + 2 rolling columns per engineered feature.
    assert len(cols) == 24 * 3
    assert cols[:1] == [f"{feats[0]}_mkt"]


def test_subset_flows_into_gru_feature_columns():
    import dataclasses

    from js2024.modeling.config import load_gru_config
    from js2024.modeling.registry import get_model_spec

    spec = get_model_spec("gru")
    cfg = load_gru_config("configs/gru_marketroll_v1.yaml")  # use_market_avg+rolling on
    cfg12 = dataclasses.replace(cfg, market_roll_subset="top12")
    cfg24 = dataclasses.replace(cfg, market_roll_subset="top24")
    extra = len(spec.feature_columns(cfg24)) - len(spec.feature_columns(cfg12))
    # 12 extra engineered features × 3 columns each.
    assert extra == 12 * 3


# --- paired aggregation ---------------------------------------------------


def _runs():
    return [
        {"kind": "baseline", "window": None, "subset": None, "seed": 1, "score": 0.10},
        {"kind": "baseline", "window": None, "subset": None, "seed": 2, "score": 0.20},
        {"kind": "marketroll", "window": 500, "subset": "top12", "seed": 1, "score": 0.13},
        {"kind": "marketroll", "window": 500, "subset": "top12", "seed": 2, "score": 0.19},
    ]


def test_paired_rows_join_same_seed_baseline():
    paired = paired_rows(_runs())
    assert len(paired) == 2
    by_seed = {p["seed"]: p for p in paired}
    assert by_seed[1]["delta"] == pytest.approx(0.03)   # 0.13 - 0.10
    assert by_seed[2]["delta"] == pytest.approx(-0.01)  # 0.19 - 0.20
    assert by_seed[1]["baseline_score"] == pytest.approx(0.10)


def test_paired_rows_skip_marketroll_without_baseline():
    runs = [
        {"kind": "marketroll", "window": 500, "subset": "top12", "seed": 9, "score": 0.5},
    ]
    assert paired_rows(runs) == []


def test_summarize_cells_stats_and_sort():
    paired = paired_rows(_runs()) + [
        {"window": 1000, "subset": "all", "seed": 1,
         "marketroll_score": 0.30, "baseline_score": 0.10, "delta": 0.20},
        {"window": 1000, "subset": "all", "seed": 2,
         "marketroll_score": 0.40, "baseline_score": 0.20, "delta": 0.20},
    ]
    summary = summarize_cells(paired)
    # Best mean delta first: the (1000, all) cell with mean Δ = 0.20.
    assert (summary[0]["window"], summary[0]["subset"]) == (1000, "all")
    assert summary[0]["mean_delta"] == pytest.approx(0.20)
    assert summary[0]["n_positive"] == 2
    # The (500, top12) cell: deltas {0.03, -0.01} -> mean 0.01, 1 positive.
    cell = next(r for r in summary if r["window"] == 500)
    assert cell["mean_delta"] == pytest.approx(0.01)
    assert cell["n_positive"] == 1
    assert cell["n_seeds"] == 2
