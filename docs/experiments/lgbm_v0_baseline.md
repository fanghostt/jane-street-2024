# LGBM V0 Baseline

## Purpose

The V0 baseline establishes a **raw-feature LightGBM benchmark** for the Jane
Street 2024 task. It is deliberately minimal:

- Inputs are the 79 raw features plus `symbol_id` and `time_id` — no feature
  engineering, no standardization, no NaN imputation (LightGBM handles missing
  values natively).
- It does **not** chase the leaderboard.
- It does **not** implement GRU, auxiliary targets, online learning, or
  ensembling.

Its job is to be a reproducible, recorded **control group** that every later
milestone (repo parity / feature parity / GRU parity / online learning) is
measured against.

The competition metric is the sample-weighted, zero-mean R²
(`1 - sum(w*(y-yhat)^2) / sum(w*y^2)`): a constant-zero prediction scores exactly
0, so small positive scores are meaningful on this low signal-to-noise data.

## Configs

| config | train_path | start_date_id | end_date_id | valid_days | gap_days | purpose |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| lgbm_v0_smoke.yaml | data/interim/train_smoke.parquet | null | null | 3 | 0 | engineering smoke |
| lgbm_v0_recent700.yaml | data/raw/train.parquet | 700 | null | 200 | 0 | first formal baseline |
| lgbm_v0_full.yaml | data/raw/train.parquet | null | null | 200 | 0 | full-data baseline |

Shared hyperparameters for the two formal configs: `n_estimators=3000`,
`learning_rate=0.03`, `num_leaves=64`, `subsample=0.8`, `colsample_bytree=0.8`,
`early_stopping_rounds=100`, `random_state=42`.

## Recent700 Run

- **status:** completed
- **command:** `uv run js2024-train-lgbm --config configs/lgbm_v0_recent700.yaml`
- **loaded rows:** 34,764,752 (`date_id` range [700, 1698])
- **train date range:** 700 – 1498 (799 days)
- **valid date range:** 1499 – 1698 (200 days)
- **train rows:** 27,329,544
- **valid rows:** 7,435,208
- **feature count:** 81 (`feature_00..feature_78` + `symbol_id` + `time_id`)
- **best iteration:** 980 (early-stopped at 1000, patience 100; valid L2 = 0.59175)
- **weighted zero-mean R²:** **0.010469**
- **predictions:** mean=0.000971 | std=0.08637 | min=-2.6083 | max=5.8182
- **target:** mean=-0.002991 | std=0.79371 | min=-5 | max=5
- **top 20 feature importance:** time_id (3965), feature_61 (2659), feature_20
  (2055), feature_24 (1851), feature_21 (1780), feature_08 (1718), feature_30
  (1584), feature_22 (1578), feature_07 (1561), feature_25 (1546), feature_31
  (1546), feature_05 (1469), feature_38 (1446), feature_29 (1405), feature_23
  (1392), feature_26 (1316), feature_27 (1308), feature_28 (1275), feature_01
  (1217), feature_58 (1028)
- **artifacts (local, gitignored):** `models/lgbm_v0.txt`,
  `outputs/oof/lgbm_v0_valid_predictions.parquet`,
  `outputs/reports/lgbm_v0_report.md`

## Full Run

- status: not run yet
- reason: full run is resource-heavy; the recent700 formal baseline is
  prioritized and reviewed first. The config is committed so it can be run later.

## Interpretation

- The recent700 baseline scores **R² ≈ 0.0105** — small but clearly positive,
  i.e. it beats the trivial constant-zero prediction (which scores exactly 0).
- This reflects a **raw-feature LightGBM baseline only** — no engineered
  features, no temporal model.
- A small R² is normal for this low signal-to-noise financial data; the
  zero-mean metric means even a tiny positive value is real signal.
- `time_id` being the single most important input is expected (strong
  intraday/seasonality structure); a later milestone may revisit how that
  identifier is encoded.
- This number is the reference point for later comparisons: feature parity with
  the `evgeniavolkova/kagglejanestreet` repo, GRU models, online learning, and
  ensembling should each be judged against it.

## Next Steps

1. V0 split experiments (vary `valid_days` / `gap_days`, sanity-check stability).
2. Feature parity with the `evgeniavolkova/kagglejanestreet` repo.
3. Repo-style protocol: `date_id >= 700`, 2-fold, gap test.
4. GRU parity.
5. Online learning parity.
6. Ensemble parity.
