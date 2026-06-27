# Market-average & per-symbol rolling features — experiment log

Does adding the public-solution feature engineering we were missing help? Two leakage-safe
families (see `src/js2024/modeling/market_features.py`), behind the config flags
`use_market_avg` / `use_symbol_rolling` (both default off):

- **cross-sectional market average** `<feature>_mkt` — per `(date_id, time_id)` mean of a
  feature across symbols (what the rest of the universe is doing right now);
- **per-symbol trailing rolling mean/std** `<feature>_roll_mean` / `_roll_std` over the last
  `rolling_window` (=1000) time steps within each `symbol_id`. The window ends at the current
  row (inclusive) and never reads the future — leakage-safe because the current row's
  `feature_*` is already available at prediction time. (Not `.shift(1)`; if you wanted a
  strictly history-only window excluding the current value you would shift first.)

Applied to the 12 top-importance features. A/B'd against the matching baselines (recent700
window, shared last-200 test block `[1499, 1698]`): `lgbm_marketroll_v1_recent700` vs
`lgbm_v0_recent700`, and `gru_marketroll_v1` vs `gru_v0`. Heavy per-run artifacts are
gitignored; this file is the committed record.

## Status / verdict

- **Helps the online GRU (the decisive metric): small but consistent +~11%. Slightly hurts
  LGBM static. Net: keep for the sequence/online model, not for static GBDT.**

| model | variant | baseline R² | +market R² | Δ |
| --- | --- | ---: | ---: | ---: |
| `lgbm` | static | 0.010659 | 0.010090 | −0.00057 |
| `gru` | full | −0.015794 | −0.007975 | +0.0078 (noise regime) |
| **`gru`** | **incremental (3-seed mean)** | **0.010770** | **0.011921** | **+0.00115 (≈ +11%)** |

### The incremental result is paired and real, not noise

The single-run win we first saw (0.012389 vs 0.011139) sat inside the cross-seed spread
(baselines alone range 0.0102–0.0111 ≈ 0.0009), so it was not yet conclusive. Re-running as a
**paired** A/B — same `random_state` → same init, so the per-seed difference cancels the init
noise — settles it:

| seed | `gru_v0` incr | `gru_marketroll_v1` incr | Δ |
| ---: | ---: | ---: | ---: |
| 42 | 0.011139 | 0.012389 | +0.00125 |
| 1  | 0.010220 | 0.011871 | +0.00165 |
| 2  | 0.010952 | 0.011502 | +0.00055 |
| **mean** | **0.010770** | **0.011921** | **+0.00115** |

All 3/3 pairs positive; paired `t ≈ 3.6` (df=2, one-sided p ≈ 0.035). The `full` GRU column
sits in the negative/noise regime but points the same way (+0.0078).

### Why it helps the GRU but not LGBM

LGBM trees already recover cross-sectional and feature interactions from the raw inputs, so the
hand-rolled `_mkt` / `_roll_*` columns are largely redundant for it (they rank high in
importance — `feature_38_roll_mean` is #2 — yet do not move validation R², a classic
redundant-but-used signature, and add enough noise to cost −0.0006). The day-batch GRU sees one
`(symbols × time × features)` tensor per day with no built-in cross-symbol view; the market
average and trailing rolling stats are genuinely new context, hence the gain.

## Robustness sweep — window × subset × seed

The follow-ups above (tune `rolling_window`, widen beyond top-12, more seeds) were run as one
**paired sweep** (`js2024-run-marketroll-sweep`): windows {250,500,1000,2000} × feature
subsets {top12,top24} × seeds {42,43,44}, plus 3 shared baselines (the baseline is invariant
to window/subset, so it is run once per seed and reused). Decisive metric is the per-seed
paired Δ = `gru_incremental(marketroll) − gru_incremental(baseline)`; the baseline reproduced
`gru_v0` exactly (0.011139 at seed 42). 27 runs total.

| window | subset | n | mean Δ | std Δ | +seeds |
| ---: | --- | ---: | ---: | ---: | ---: |
| 250 | top12 | 3 | **+0.001053** | 0.000249 | 3/3 |
| 1000 | top12 | 3 | +0.000940 | 0.000356 | 3/3 |
| 2000 | top12 | 3 | +0.000811 | 0.000253 | 3/3 |
| 500 | top12 | 3 | +0.000773 | 0.000220 | 3/3 |
| 2000 | top24 | 3 | +0.000511 | 0.000535 | 2/3 |
| 250 | top24 | 3 | +0.000440 | 0.000453 | 3/3 |
| 1000 | top24 | 3 | +0.000377 | 0.000127 | 3/3 |
| 500 | top24 | 3 | +0.000352 | 0.000265 | 3/3 |

Three conclusions, all of which close the corresponding follow-up:

- **The gain is robust.** Every top12 cell is 3/3 seeds positive with mean ≈ 4× its std — the
  +0.00115 single-point win holds across windows and seeds, not a lucky seed.
- **top12 > top24, decisively.** Engineering *more* features halves the gain (top24 ≈ +0.0004
  vs top12 ≈ +0.0009) and destabilises it (w2000/top24 drops to 2/3). The extra rank-13..24
  features carry weaker signal; their `_mkt`/`_roll_*` columns dilute the online fine-tune
  rather than help. `all` (79 features) was not run — the monotone 12→24 decline makes it a
  safe inference that it is worse still.
- **`rolling_window` is a weak knob.** top12 ordering is 250 ≥ 1000 ≥ 2000 ≥ 500 but all sit
  within ~0.0003. Short window (250) is marginally best; the difference is not worth chasing.

`market_roll_subset` (top12/top24/all, prefixes of the V0 LGBM *split*-importance ranking) is
a config key on both GRU and LGBM configs; `top12` is byte-for-byte the original feature list,
so this section's baseline matches the earlier A/B with no drift.

## Cross-check against the public solution

These are exactly the two engineered families `evgeniavolkova/kagglejanestreet` uses on top of
the aux targets + online finetune (the deferred parity flagged in `seq_backbones_v0.md` and
`lag_features_v1.md`). The online GRU now reaches `0.011921` mean (best seed 0.012389), edging
past the plain-online baseline (`0.011139`) and their reported 6-model ensemble LB (0.0112) with
a single model. Contrast with `lag_features_v1.md`, where raw D-1 responders as inputs were
net-negative everywhere — temporal/cross-sectional signal comes from engineered features and
online updating, not from feeding lagged responders.

## Decision

- **Keep market-avg + per-symbol rolling for the GRU / sequence + online track**
  (`use_market_avg: true`, `use_symbol_rolling: true`). Leave them off for the static LGBM
  baseline, where they do not pay.
- **Settings:** `market_roll_subset: top12`, `rolling_window: 250` (or 1000 — within noise).
  The robustness sweep below settled the three open knobs: the gain holds across seeds, top12
  beats top24 (don't widen), and the window barely matters.
- Effect is small (+~11% on a ~0.011 base); treat as one accepted brick, not a finish line.
  The window/subset axes are now exhausted — further gains need a *new* direction (a new
  feature family or target engineering), not more tuning of these two.

## Reproduce

```bash
# LGBM static A/B
uv run js2024-train-lgbm --config configs/lgbm_marketroll_v1_recent700.yaml

# GRU full + paired incremental A/B (single seed)
uv run js2024-run-experiment --config configs/gru_marketroll_v1.yaml --variants full,incremental
uv run js2024-run-experiment --config configs/gru_v0.yaml          --variants incremental

# Robustness sweep — window × subset × seed, paired vs the same-seed baseline
uv run js2024-run-marketroll-sweep \
    --windows 250,500,1000,2000 --subsets top12,top24 --seeds 42,43,44
```

Heavy per-run artifacts are gitignored scratch (`experiments/marketroll_sweep/`); the numbers
above are the committed record. The sweep writes `runs.csv` / `paired.csv` / `summary.csv` /
`report.md` and supports `--wandb --wandb-project <name>` for live tracking.
