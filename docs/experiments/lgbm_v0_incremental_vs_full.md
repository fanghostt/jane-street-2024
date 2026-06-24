# LGBM V0 — incremental (daily refit) vs full training

## Purpose

Compare a statically-trained LightGBM (**full**) against the same model updated **incrementally** (LightGBM leaf-value refit, `update_method=refit`, cadence `1` day) as it walks a fixed final test block. This is the leakage-clean LightGBM analog of the *with vs without online learning* comparison in the evgeniavolkova writeup.

## Protocol

- data start: `date_id >= 700` (her cutoff).
- **fixed test block:** last `200` date_ids = [1499, 1698] (shared by both modes).
- train region: [700, 1498]; early-stopping holdout (from train tail, `valid_days=200`): [1299, 1498] — the test block is **never** used for early stopping.
- raw features only (`feature_00..feature_78` + `symbol_id` + `time_id`); no feature engineering, GRU, auxiliary targets, ensemble, or clipping.

## Results

- **status:** completed

| mode | test_range | test_rows | n_updates | R² |
| --- | --- | ---: | ---: | ---: |
| full | 1499–1698 | 7,435,208 | 0 | 0.007832 |
| incremental | 1499–1698 | 7,435,208 | 199 | 0.000668 |

- **delta (incremental − full):** -0.007164 (-91.5% relative).

## Interpretation

- Daily leaf-value refitting does **not** improve (or hurts) the score here. LightGBM `refit` only re-weights existing leaves; the structural online gains reported for the GRU may need continued boosting / retrain or richer features. Recorded as a negative result.
- The **full** number here is leakage-clean (early stopping uses a train-tail holdout, not the test block), so it may differ slightly from the recent700 baseline R²=0.010469, which used the last-200 block as its eval_set.

## Next steps

1. Try `continue` (init_model) and `retrain` (expanding) update methods.
2. Vary `update_cadence` (1 / 20 / 50) for the cost-vs-benefit curve.
3. Repo-style 2-fold CV + 200-day gap protocol.
4. Feature engineering parity, then the GRU estimator behind the same API.
