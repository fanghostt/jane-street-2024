# GRU — experiment log

Consolidated record for the GRU sequence-model work. Two GRU implementations sit
behind the same `Estimator` (`fit`/`update`/`predict`) interface and run through the
**same** fixed-test-block walk-forward engine as the LightGBM baseline, so every
weighted zero-mean R² below is directly comparable to `docs/experiments/lgbm_v0.md`.

Heavy per-run artifacts (models / CSVs / per-run reports) are written under
`experiments/` (gitignored); this file is the only committed GRU experiment doc.

## Status

- **Final model found:** the evgeniavolkova-style **day-batch GRU + auxiliary
  responder heads + per-day online finetune** is the strongest model to date,
  **weighted zero-mean R² = 0.011139** on the shared last-200 test block — above
  LightGBM retrain (0.009956) and the LightGBM static reference (0.010469).
- **Online finetune is decisive:** the same architecture *without* updates scores
  0.002126; per-day finetune lifts it ~5× to 0.011139.
- **The naive GRU fails:** a single GRU over per-`symbol_id` lookback windows
  (`gru_v0`) scores **−0.081** on the same block — worse than predicting zero. The
  per-window framing is the wrong inductive bias for this low-SNR, cross-sectional
  data.
- **Decision:** adopt `gru_evgeniavolkova` (day-batch, aux responders, online
  finetune cadence 1) as the GRU of record; keep `gru_v0` only as a negative
  baseline.
- **Not yet done:** feature-engineering parity (market averages / rolling stats),
  GRU+LightGBM ensemble, prediction clipping (finetune still overshoots [-5, 5]).

## Shared setup

- Metric: sample-weighted zero-mean R², `1 - Σ w(y-ŷ)² / Σ w·y²` (constant-zero ⇒ 0).
- Data start `date_id >= 700`, seed 42, `device: auto` (CUDA when available).
- **Fixed test block 1499–1698** (200 days, 7,435,208 rows), shared by every variant.
  Early stopping uses a train-tail holdout (1299–1498) — **never** the test block, so
  these numbers are leakage-clean and comparable to the LightGBM §3 table.
- All variants run through the config-driven runner:
  `uv run js2024-run-experiment --config <cfg>` (`model:` selects the architecture,
  `variants:` selects `full` / `incremental`).

## 1. `gru_v0` — per-symbol lookback GRU (negative baseline)

`uv run js2024-run-experiment --config configs/gru_v0.yaml`

Each row's prediction comes from a length-`seq_len` (=16) window of that symbol's
`(date_id, time_id)`-ordered feature vectors (standardized, NaN→mean); `symbol_id` is
encoded via the per-symbol sequencing, not as a raw input. A per-symbol context buffer
is advanced in `predict` using **features only** (leakage-clean). `update` runs a few
fine-tuning gradient steps on each revealed chunk.

| variant | cadence | n_updates | R² | note |
| --- | ---: | ---: | ---: | --- |
| gru_full | – | 0 | **−0.081** | worse than constant zero |

**Conclusion:** the per-window framing does not work here. Treating each row as an
independent short sequence discards the cross-sectional (all-symbols-at-once)
structure the signal lives in. Kept only as a baseline; not pursued further.

## 2. `gru_evgeniavolkova` — day-batch GRU + auxiliary responders (final model)

`uv run js2024-run-experiment --config configs/gru_evgeniavolkova_v1.yaml`

Public-solution parity with `evgeniavolkova/kagglejanestreet`: one `date_id` is one
batch, rows are reshaped to `symbols × time_id × features`, and four auxiliary
responder heads (`responder_7/8/9/10`) are trained jointly with the target. Inputs are
all `feature_*` except 09–11, plus `time_id`. Online finetune uses a smaller LR
(`lr_refit=3e-4`); the `incremental` variant finetunes once per day (cadence 1).
Architecture: `hidden_sizes=[500]`, linear head `[500, 300]`, dropout `0.3/0.2/0.1`,
`lr=5e-4`, `epochs=1000` with `early_stopping_patience=1`, `weight_decay=0.01`,
`grad_clip=1.0`.

| variant | cadence | n_updates | R² | pred[min,max] |
| --- | ---: | ---: | ---: | --- |
| gru_evgeniavolkova_full | – | 0 | 0.002126 | [-0.52, 2.99] |
| **gru_evgeniavolkova_incremental** | 1 | 199 | **0.011139** | [-2.71, 7.06] |

**Conclusion:** the day-batch + auxiliary-responder design is what makes a GRU work on
this data, and per-day online finetune is essential (0.002 → 0.011). This is the
neural analog of "with vs without online learning", and unlike the LightGBM case
(where only periodic full retrain helped) the cheap tiny-LR one-pass update is exactly
the right move for the net.

## 3. Cross-model comparison (same fixed test block 1499–1698)

| model | variant | update | n_updates | R² |
| --- | --- | --- | ---: | ---: |
| `gru_v0` | full | – | 0 | −0.081 |
| `lgbm` | full | static | 0 | 0.007832 |
| `gru_evgeniavolkova` | full | – | 0 | 0.002126 |
| `lgbm` | retrain | expanding (cad 50) | 3 | 0.009956 |
| **`gru_evgeniavolkova`** | **incremental** | **finetune (cad 1)** | **199** | **0.011139** |

(LightGBM rows from `docs/experiments/lgbm_v0.md` §3.) The online GRU is the current
best single model; LightGBM retrain is the best non-neural option and the honest
local reference.

## Decision & next steps

- **Adopt `gru_evgeniavolkova` (incremental, cadence 1)** as the GRU of record.
- Next: **feature-engineering parity** (market averages / rolling stats from the
  reference repo); a **GRU + LightGBM ensemble** (the two best models disagree in
  prediction shape); and **prediction clipping** — finetune predictions still reach
  7.06, beyond the target's [-5, 5], so clipping is a cheap candidate (deferred,
  tracked as a config option).
