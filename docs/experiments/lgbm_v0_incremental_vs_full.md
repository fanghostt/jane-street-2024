# LGBM V0 — incremental vs full training

## Purpose

Compare a statically-trained LightGBM (**full**) against three incremental update strategies as the model walks a fixed final test block: **refit** (leaf-value refit), **continue** (continued boosting), and **retrain** (expanding retrain from scratch). Leakage-clean LightGBM analog of the *with vs without online learning* comparison in the evgeniavolkova writeup.

## Protocol

- data start: `date_id >= 700` (her cutoff).
- **fixed test block:** last `200` date_ids = [1499, 1698] (shared by every variant).
- train region: [700, 1498]; early-stopping holdout (train tail, `valid_days=200`): [1299, 1498] — the test block is **never** used for early stopping.
- update params: `refit_decay=0.9`, `continue_rounds=10`.
- raw features only (`feature_00..feature_78` + `symbol_id` + `time_id`); no feature engineering, GRU, auxiliary targets, ensemble, or clipping.

## Results

- **status:** completed

| variant | cadence | n_updates | R² | Δ vs full | pred[min,max] |
| --- | ---: | ---: | ---: | ---: | --- |
| full | – | 0 | 0.007832 |  | [-3.06, 5.55] |
| refit | 1 | 199 | 0.000668 | -0.007164 | [-2.11, 3.01] |
| continue | 1 | 199 | -0.953125 | -0.960957 | [-7.16, 10.59] |
| retrain | 50 | 3 | 0.009956 | +0.002124 | [-2.57, 5.68] |

- **best variant:** `retrain` (R²=0.009956).

## Interpretation

- The **full** number here is leakage-clean (early stopping uses a train-tail holdout, not the test block), so it may differ slightly from the recent700 baseline R²=0.010469, which used the last-200 block as its eval_set.
- **refit** degrades: each daily leaf-refit re-weights *all* leaves toward one noisy day; 199 cumulative refits drag the fit off. `Booster.refit` is the wrong online analog for a tree model.
- **continue** blows up (R²=-0.953125): adding `10` trees per day on a *single* day's rows compounds over ~200 updates into a heavily over-fit ensemble (prediction range explodes well past the target's [-5, 5]). Daily continued boosting is unstable without shrinkage / a held-out check.
- **retrain** is the strongest (R²=0.009956): expanding retrain reincorporates the early-stopping holdout and every revealed day, so it trains on more data than the static/online variants — the closest analog to her expanding CV.

## Next steps

1. Tune `continue_rounds` / `refit_decay` / `update_cadence` trade-offs.
2. Repo-style 2-fold CV + 200-day gap protocol.
3. Feature engineering parity, then the GRU estimator behind the same API.
