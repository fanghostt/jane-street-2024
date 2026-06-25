# LGBM V0 — experiment log

Single consolidated record for the V0 raw-feature LightGBM work. Heavy per-run
artifacts (models / OOF / per-run reports) are written under `outputs/` (gitignored);
this file is the only committed experiment doc.

## Status

- **Baseline established** and **stable** across splits (raw features only).
- **Online-learning question answered:** of the LightGBM update strategies, only
  **expanding retrain** beats static training; per-day `refit` / `continue` hurt.
- **Decision:** adopt **retrain** (periodic full retrain on all data seen so far) as
  the V0 update protocol. In practice this just means *retraining the model* on the
  growing window — there is no cheap tree-level "online" shortcut that helps.
- **Not yet done:** repo-style 2-fold CV + 200-day gap protocol, feature engineering,
  GRU (the `Estimator` interface is ready for it), ensemble, prediction clipping.

## Shared setup

- Metric: sample-weighted zero-mean R², `1 - Σ w(y-ŷ)² / Σ w·y²` (constant-zero ⇒ 0,
  so small positive values are real signal on this low-SNR data).
- Inputs: 79 raw features + `symbol_id` + `time_id`. No engineering, no imputation
  (LightGBM handles missing natively).
- Data start `date_id >= 700` (where `time_id` stabilises at 968; verified empirically
  it jumps 849→968 at day 677), matching `evgeniavolkova/kagglejanestreet`.
- Hyperparameters: `n_estimators=3000`, `learning_rate=0.03`, `num_leaves=64`,
  `subsample=0.8`, `colsample_bytree=0.8`, `early_stopping_rounds=100`, `seed=42`.

## 1. Baseline (recent700)

`uv run js2024-train-lgbm --config configs/lgbm_v0_recent700.yaml`

- Train 700–1498 (27,329,544 rows), validate 1499–1698 (200 days, 7,435,208 rows).
- best_iteration 980; **weighted zero-mean R² = 0.010469**.
- Top features: `time_id`, `feature_61`, `feature_20`, `feature_24`, `feature_21`
  (`time_id` is #1 — strong intraday/seasonal structure).
- Note: this run uses the last-200 block as its early-stopping `eval_set`, so it is the
  *reference* number; the leakage-clean number (§3) is slightly lower.

## 2. Split stability

`uv run js2024-run-lgbm-split-experiments --base-config configs/lgbm_v0_recent700.yaml`
(grid `valid_days ∈ {100,200,300}` × `gap_days ∈ {0,5,20}`, 9 runs)

- All 9 splits positive; **mean R² 0.010130, std 0.002112** (min 0.007257, max 0.013343).
- R² **rises monotonically with the validation window** (vd100≈0.0076, vd200≈0.0100,
  vd300≈0.0127) — so the *absolute* number is protocol-dependent; cross-experiment
  comparisons must fix the split.
- Increasing `gap_days` 0→20 costs ≈ −8% within a fixed window (mild near-boundary
  autocorrelation, not large leakage).
- Feature ranking stable: `time_id` #1 in 9/9, `feature_61` top-5 in 9/9.
- Predictions exceed [-5, 5] (upper tail) in 9/9 — clipping deferred.

**Takeaway:** the signal is real and not a one-split artifact; fix the split when
comparing anything.

## 3. Incremental vs full (fixed last-200 test)

`uv run js2024-run-incremental-vs-full --config configs/lgbm_v0_incremental.yaml
--methods refit,continue,retrain`

Test block 1499–1698 (7,435,208 rows), shared by every variant. Early stopping uses a
train-tail holdout (1299–1498), **never** the test block — so `full` here is
leakage-clean and sits a bit below the §1 reference 0.010469.

| variant | update | cadence | n_updates | R² | Δ vs full |
| --- | --- | ---: | ---: | ---: | ---: |
| full | static | – | 0 | 0.007832 | — |
| refit | `Booster.refit` leaves | 1 | 199 | 0.000668 | −0.007164 |
| continue | +trees (init_model) | 1 | 199 | −0.953125 | −0.960957 |
| **retrain** | expanding, from scratch | 50 | 3 | **0.009956** | **+0.002124 (+27%)** |

**Conclusion:** only **retrain** beats static `full`. `refit` drifts off as 199 daily
leaf-refits re-weight all leaves toward single noisy days; `continue` overfits
catastrophically (adds trees per day → prediction range explodes to [-7, 10.6], R²
−0.95). The right "online" move for a tree model is **periodic full retrain**, not a
cheap per-day tweak — fundamentally different from the GRU's tiny-LR one-pass update.

## 4. Retrain cadence sweep

`uv run js2024-run-incremental-vs-full --config configs/lgbm_v0_incremental.yaml
--retrain-cadences 100,50,25`

How much does retraining *more often* help, and is it worth the cost? Same fixed test
block 1499–1698.

| cadence (days) | retrains | R² | Δ vs full |
| ---: | ---: | ---: | ---: |
| — (full) | 0 | 0.007832 | — |
| 100 | 1 | 0.009093 | +0.001261 (+16%) |
| 50 | 3 | 0.009956 | +0.002124 (+27%) |

**Conclusion:** the gain is **monotonic in retrain frequency** but front-loaded. A
*single* retrain (every 100 days) already recovers most of it (+16%); halving the
interval to 50 days (3 retrains) reaches +27% total — i.e. the 2 extra retrains add
only ~+11pp more. Finer than 50 (cadence 25 ≈ 7 expanding retrains) was **too slow to
run** locally (each retrain ≈ a full from-scratch fit; the sweep hit a multi-hour
wall) for a diminishing marginal gain. **Operating point: retrain every ~50 days** is
the sensible cost/benefit choice.

## Decision & next steps

- **Adopt `retrain`** (expanding, periodic full retrain, **~every 50 days** per §4) as
  the V0 update protocol.
  Caveat: a true from-scratch retrain every step is too slow for the live Kaggle
  inference window — which is exactly why the reference solution used a GRU with cheap
  per-day online updates. For *local* validation, retrain is the honest reference.
- Next: **repo-style 2-fold CV + 200-day gap** protocol; then **feature engineering
  parity**; then a **GRU** estimator behind the same `Estimator` (`fit`/`update`/
  `predict`) interface; later ensemble and prediction clipping.
