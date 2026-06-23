# LGBM V0 Split Experiments

## Purpose

Verify that the raw-feature LightGBM **recent700** baseline is stable across validation splits — i.e. that its weighted zero-mean R² is not a one-split artifact.

- Baseline reference: recent700, `valid_days=200`, `gap_days=0`, R² = 0.010469.
- Raw features only (`feature_00..feature_78` + `symbol_id` + `time_id`). No feature engineering, no GRU, no online learning, no ensemble.
- Prediction clipping is still deferred.

## Experiment Grid

| start_date_id | valid_days | gap_days |
| ---: | ---: | ---: |
| 700 | 100 | 0 |
| 700 | 100 | 5 |
| 700 | 100 | 20 |
| 700 | 200 | 0 |
| 700 | 200 | 5 |
| 700 | 200 | 20 |
| 700 | 300 | 0 |
| 700 | 300 | 5 |
| 700 | 300 | 20 |

## Results

- **status:** completed

| run_name | valid_days | gap_days | train_range | valid_range | train_rows | valid_rows | best_iter | R² | top_5_features |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| lgbm_v0_recent700_v100_g0 | 100 | 0 | 700–1598 | 1599–1698 | 31,048,600 | 3,716,152 | 636 | 0.007864 | time_id; feature_61; feature_07; feature_24; feature_08 |
| lgbm_v0_recent700_v100_g5 | 100 | 5 | 700–1593 | 1599–1698 | 30,863,712 | 3,716,152 | 693 | 0.007818 | time_id; feature_61; feature_24; feature_08; feature_07 |
| lgbm_v0_recent700_v100_g20 | 100 | 20 | 700–1578 | 1599–1698 | 30,308,080 | 3,716,152 | 503 | 0.007257 | time_id; feature_61; feature_07; feature_08; feature_20 |
| lgbm_v0_recent700_v200_g0 | 200 | 0 | 700–1498 | 1499–1698 | 27,329,544 | 7,435,208 | 980 | 0.010469 | time_id; feature_61; feature_20; feature_24; feature_21 |
| lgbm_v0_recent700_v200_g5 | 200 | 5 | 700–1493 | 1499–1698 | 27,143,688 | 7,435,208 | 507 | 0.009990 | time_id; feature_61; feature_07; feature_08; feature_20 |
| lgbm_v0_recent700_v200_g20 | 200 | 20 | 700–1478 | 1499–1698 | 26,617,096 | 7,435,208 | 406 | 0.009565 | time_id; feature_61; feature_07; feature_08; feature_20 |
| lgbm_v0_recent700_v300_g0 | 300 | 0 | 700–1398 | 1399–1698 | 23,736,328 | 11,028,424 | 569 | 0.013343 | time_id; feature_61; feature_20; feature_07; feature_24 |
| lgbm_v0_recent700_v300_g5 | 300 | 5 | 700–1393 | 1399–1698 | 23,564,024 | 11,028,424 | 401 | 0.012589 | time_id; feature_61; feature_07; feature_20; feature_05 |
| lgbm_v0_recent700_v300_g20 | 300 | 20 | 700–1378 | 1399–1698 | 23,058,728 | 11,028,424 | 612 | 0.012272 | time_id; feature_61; feature_20; feature_24; feature_07 |

## Stability Summary

- runs completed: 9
- mean R²: 0.010130
- std R²: 0.002112
- min R²: 0.007257 (lgbm_v0_recent700_v100_g20)
- max R²: 0.013343 (lgbm_v0_recent700_v300_g0)
- split matching baseline (v200_g0): R² = 0.010469
- all splits positive: True (a constant-zero prediction scores exactly 0)

## Interpretation

- All splits score positive, so the raw-feature LGBM signal is consistently present — it is not a one-split artifact.
- The score depends on the validation window length (vd100≈0.0076, vd200≈0.0100, vd300≈0.0127); R² rises monotonically with `valid_days`. So the *absolute* number is protocol-dependent — cross-experiment comparisons must fix the split.
- Increasing `gap_days` 0→20 changes R² by on average -8.1% within a fixed `valid_days` — a mild penalty, consistent with slight near-boundary temporal autocorrelation rather than large leakage.
- Feature ranking is stable: `time_id` is the #1 feature in 9/9 splits and `feature_61` is top-5 in 9/9.
- Predictions exceed the target's [-5, 5] range in 9/9 splits (upper tail). Prediction clipping remains deferred to a separate PR.

## Next Steps

1. If stable: feature parity with the `evgeniavolkova/kagglejanestreet` repo.
2. If unstable: inspect split-specific target/weight distributions.
3. Later: repo-style 2-fold + 200-day gap test.
4. Later: `clip_predictions` PR.
5. Later: GRU parity.
