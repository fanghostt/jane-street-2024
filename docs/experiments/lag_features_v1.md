# Responder lag features (responder_i_lag_1) — experiment log

Does feeding the **previous day's responders** as input features help? The competition
serves each day the prior day's responders (`responder_0_lag_1 … responder_8_lag_1`)
via `lags.parquet`. That file is synthetic and does not match this repo's
`train.parquet`, so we reconstruct the lags offline from `train.parquet` with strict
leakage control (shift `date_id` by +1, left-join; first `date_id` gets null lags —
current-day responders are never used as their own lag). See
`src/js2024/modeling/lag_features.py`. The 9 lag columns are appended to the model
inputs behind the config flag `use_responder_lags` (defaulted off).

Two clean A/Bs vs the matching baselines (recent700 window, seed 42, shared last-200
test block `[1499, 1698]`): `lgbm_lags_v1_recent700` vs `lgbm_v0_recent700`, and
`gru_lags_v1` vs `gru_v0`. Heavy per-run artifacts are gitignored; this file is the
committed record.

## Status / verdict

- **Lags hurt — net negative everywhere, not merely "no gain."**

| model | variant | baseline R² | +lags R² | Δ |
| --- | --- | ---: | ---: | ---: |
| `lgbm` | static | 0.010659 | 0.010471 | −0.0002 |
| `gru` | full | −0.015794 | −0.017692 | −0.0019 (noise regime) |
| **`gru`** | **incremental** | **0.011139** | **0.008915** | **−0.0022 (≈ −20%)** |

- The decisive read is the `incremental` column (per-day online finetune): the stable,
  meaningful metric where the GRU actually generalizes. There lags cost ~0.0022
  (−20% relative). The `full` GRU column sits in the negative/noise regime, so its
  swing is weak signal, but it points the same way; LGBM agrees.
- **Why they hurt:** the 9 lag columns add parameters/noise with no incremental
  predictive value over the existing inputs + auxiliary targets, so the model slightly
  overfits. The valid-R² plateau barely moves (~0.021 → ~0.021).

## Cross-check against the public solution

`evgeniavolkova/kagglejanestreet` (the solution this GRU is modelled on) **does not
feed raw D-1 responders as input features.** Its temporal signal comes from elsewhere:

- engineered **auxiliary targets** — `responder_7/8` plus shift-combined `responder_9`
  (`r8 + r8.shift(-4)`) and `responder_10` (`r6 + r6.shift(-20) + r6.shift(-40)`),
  which we already replicate (`gru.add_gru_aux_targets`);
- **per-day online finetune** (lr 3e-4, responder_6 loss) — our `incremental` variant;
- **feature engineering we do not have yet**: cross-sectional market averages per
  `(date_id, time_id)`, and per-`symbol_id` rolling statistics over ~1000 time steps.

So this negative result is consistent with the reference: day-lag responders as inputs
are not how temporal information is injected. Notably our single online GRU
(`gru_incremental` = 0.011139) already matches their final 6-model ensemble LB (0.0112).

## Decision

- **Drop lags.** Keep the wiring (`use_responder_lags`, `lag_features.py`, the
  `*_lags_v1` configs) defaulted off as a recorded, rejected experiment.
- **Higher-leverage next step is the missing feature engineering** — cross-sectional
  market averages and per-symbol rolling stats (the deferred public-solution parity,
  also flagged in `seq_backbones_v0.md`), A/B'd on the `incremental` column.

## Reproduce

```bash
uv run js2024-train-lgbm --config configs/lgbm_lags_v1_recent700.yaml          # LGBM A/B
uv run js2024-run-experiment --config configs/gru_lags_v1.yaml --variants full,incremental
```

`gru_lags_v1.yaml` also exercises optional Weights & Biases tracking (`use_wandb`) and
the bf16 `use_amp` speed flag; neither affects the verdict (both default off elsewhere).
