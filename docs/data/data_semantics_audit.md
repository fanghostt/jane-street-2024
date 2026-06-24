# Jane Street 2024 — data semantics audit

Read-only audit of `data/raw/*`. No model is trained, no feature is engineered and no raw data is mutated. Regenerate with `uv run js2024-data-semantics-audit` (see README).

## 1. File inventory

| file | exists | role | local scoring? | notes |
| --- | --- | --- | --- | --- |
| train.parquet | yes | historical training data, contains responders | yes (cut by time) | source of all local validation splits |
| test.parquet | yes | mock test structure / evaluation API input | no | no `responder_6` label; API/inference compatibility only |
| lags.parquet | yes | responder_0..8 lagged by one date_id | no | served at first time_id of the succeeding date; for lag features / online / GRU |
| features.csv | yes | metadata for anonymized features (boolean tags) | no | feature grouping / parity analysis |
| responders.csv | yes | metadata for anonymized responders | no | not the label values themselves |
| sample_submission.csv | yes | submission format (`row_id`,`responder_6`) | no | shape reference for the inference gateway |

**Key conclusions:**

- `train.parquet`: historical training data; contains responders; local validation **must** be cut from it by time.
- `test.parquet`: mock test structure / evaluation API input; does **not** contain `responder_6`; cannot be used for local R² validation.
- `lags.parquet`: `responder_0..8` lagged by one `date_id`, served by the evaluation API at the first `time_id` of the succeeding date; useful for lag features / online learning / GRU, **not** used in V0 raw LGBM.
- `features.csv` / `responders.csv`: anonymized metadata (boolean tags), **not** the label values.

## 2. train.parquet audit

- schema columns: **92**
- row count: **47,127,338**
- has date_id/time_id/symbol_id: yes; has weight: yes
- feature_00..feature_78 present: yes (79/79)
- responder_0..responder_8 present: yes (9/9)
- **responder_6 exists: yes**
- date_id min/max: 0 … 1698 (1699 unique)
- time_id n_unique: 968 (min 0, max 967)
- symbol_id n_unique: 39
- responder_6 summary: mean=-0.002141 | std=0.889852 | min=-5.000000 | max=5.000000
- weight summary: mean=2.009445 | std=1.129388 | min=0.149967 | max=10.240419
- **all responders clipped to [-5, 5]: yes** (observed per-responder min/max: responder_0=[-5.000000,5.000000], responder_1=[-5.000000,5.000000], responder_2=[-5.000000,5.000000], responder_3=[-5.000000,5.000000], responder_4=[-5.000000,5.000000], responder_5=[-5.000000,5.000000], responder_6=[-5.000000,5.000000], responder_7=[-5.000000,5.000000], responder_8=[-5.000000,5.000000])

## 3. test.parquet audit

- schema columns: **85**
- row count: **39**
- has date_id/time_id/symbol_id/weight: yes; has is_scored: yes; has row_id: yes
- feature_00..feature_78 present: yes (79/79)
- **contains responder_6: no**
- date_id min/max: 0 … 0; time_id min/max: 0 … 0; symbol_id n_unique: 39

**Conclusion:** `test.parquet` is a *mock* of the evaluation API input (one served batch: `row_id`, ids, `weight`, `is_scored`, features, but no `responder_6`). It exists for API/inference compatibility, **not** local model evaluation. Because there is no label, no R² / weighted-R² can be computed against it locally.

## 4. lags.parquet audit

- schema columns: **12**
- columns: `date_id`, `time_id`, `symbol_id`, `responder_0_lag_1` … `responder_8_lag_1`
- lag columns present: yes (9/9)
- has date_id/time_id/symbol_id: yes
- row count: **39**
- date_id range: 0 … 0; time_id values: 0 … 0; symbol_id n_unique: 39
- sample row: date_id=0, time_id=0, symbol_id=0, responder_0_lag_1=-0.442215, responder_1_lag_1=-0.322407, responder_2_lag_1=0.143594, …

**Semantics:** at a new `date_id` D, the evaluation API delivers the responders from `date_id` D-1 (all `time_id`s of that prior date) as `responder_*_lag_1`, handed over at the **first `time_id` of D**. They are the only responder information available at inference time — the live API never reveals current-date responders.

**Local reconstruction:** for train-time experiments these lags can be rebuilt from `train.parquet` responders by shifting one `date_id` forward (responders of D-1 become features for D). This must avoid using current- or future-date responders, or it leaks the target. (V0 raw LGBM does not use lags at all.)

## 5. features.csv audit

- row count (features): **79**
- columns: `feature`, `tag_0`, `tag_1`, `tag_2`, `tag_3`, `tag_4`, `tag_5`, `tag_6`, `tag_7`, `tag_8`, `tag_9`, `tag_10`, `tag_11`, `tag_12`, `tag_13`, `tag_14`, `tag_15`, `tag_16`
- tag columns: **17**

| tag | true | false |
| --- | --- | --- |
| tag_0 | 12 | 67 |
| tag_1 | 3 | 76 |
| tag_2 | 10 | 69 |
| tag_3 | 28 | 51 |
| tag_4 | 10 | 69 |
| tag_5 | 10 | 69 |
| tag_6 | 6 | 73 |
| tag_7 | 4 | 75 |
| tag_8 | 6 | 73 |
| tag_9 | 9 | 70 |
| tag_10 | 3 | 76 |
| tag_11 | 3 | 76 |
| tag_12 | 18 | 61 |
| tag_13 | 20 | 59 |
| tag_14 | 18 | 61 |
| tag_15 | 17 | 62 |
| tag_16 | 10 | 69 |

**Features grouped by tag (membership, tags overlap):**

- `tag_0`: feature_20, feature_21, feature_22, feature_23, feature_24, feature_25, feature_26, feature_27, feature_28, feature_29, feature_30, feature_31
- `tag_1`: feature_09, feature_10, feature_11
- `tag_2`: feature_00, feature_01, feature_02, feature_03, feature_04, feature_32, feature_33, feature_34, feature_35, feature_36
- `tag_3`: feature_05, feature_06, feature_07, feature_08, feature_37, feature_38, feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49, feature_50, feature_51, feature_52, feature_53, feature_54, feature_55, feature_56, feature_57, feature_58, feature_59, feature_60
- `tag_4`: feature_18, feature_39, feature_40, feature_41, feature_45, feature_50, feature_51, feature_52, feature_56, feature_65
- `tag_5`: feature_19, feature_42, feature_43, feature_44, feature_46, feature_53, feature_54, feature_55, feature_57, feature_66
- `tag_6`: feature_15, feature_16, feature_17, feature_62, feature_63, feature_64
- `tag_7`: feature_18, feature_19, feature_65, feature_66
- `tag_8`: feature_73, feature_74, feature_75, feature_76, feature_77, feature_78
- `tag_9`: feature_12, feature_13, feature_14, feature_67, feature_68, feature_69, feature_70, feature_71, feature_72
- `tag_10`: feature_70, feature_71, feature_72
- `tag_11`: feature_67, feature_68, feature_69
- `tag_12`: feature_02, feature_04, feature_06, feature_13, feature_16, feature_34, feature_36, feature_40, feature_43, feature_48, feature_51, feature_54, feature_59, feature_63, feature_68, feature_71, feature_75, feature_76
- `tag_13`: feature_01, feature_03, feature_04, feature_07, feature_14, feature_17, feature_33, feature_35, feature_36, feature_41, feature_44, feature_49, feature_52, feature_55, feature_60, feature_64, feature_69, feature_72, feature_77, feature_78
- `tag_14`: feature_00, feature_01, feature_05, feature_12, feature_15, feature_32, feature_33, feature_39, feature_42, feature_47, feature_50, feature_53, feature_58, feature_62, feature_67, feature_70, feature_73, feature_74
- `tag_15`: feature_39, feature_40, feature_41, feature_42, feature_43, feature_44, feature_45, feature_46, feature_47, feature_48, feature_49, feature_61, feature_62, feature_63, feature_64, feature_65, feature_66
- `tag_16`: feature_00, feature_01, feature_02, feature_03, feature_04, feature_05, feature_06, feature_07, feature_08, feature_20

**Spot checks:**
- `feature_09` tags: tag_1
- `feature_10` tags: tag_1
- `feature_11` tags: tag_1
- `feature_20`…`feature_31`: feature_20→tag_0|tag_16; feature_21→tag_0; feature_22→tag_0; feature_23→tag_0; feature_24→tag_0; feature_25→tag_0; feature_26→tag_0; feature_27→tag_0; feature_28→tag_0; feature_29→tag_0; feature_30→tag_0; feature_31→tag_0
- `feature_61` tags: tag_15 (has tags)

**Parity note (evgeniavolkova repo):** the tag columns group features that share anonymized structure, so they are a natural unit for feature-engineering parity — e.g. building per-tag aggregates or ensuring the same features feed the same derived columns. Do **not** infer real financial meaning from tags; they are anonymized metadata only.

## 6. responders.csv audit

- row count (responders): **9**
- columns: `responder`, `tag_0`, `tag_1`, `tag_2`, `tag_3`, `tag_4`
- responder names: responder_0, responder_1, responder_2, responder_3, responder_4, responder_5, responder_6, responder_7, responder_8
- tag columns: **5**

| tag | true | false |
| --- | --- | --- |
| tag_0 | 3 | 6 |
| tag_1 | 3 | 6 |
| tag_2 | 3 | 6 |
| tag_3 | 3 | 6 |
| tag_4 | 3 | 6 |

- `responder_6` is exactly one metadata row: yes.

**Conclusion:** `responders.csv` is *metadata* describing the nine anonymized responders and their boolean tags. The actual responder *values* (including the `responder_6` target) live in `train.parquet`; this file is metadata only.

## 7. Competition submission status

Verified with the Kaggle CLI on 2026-06-24 (`kaggle competitions list -s jane-street-real-time` and `kaggle competitions submissions -c jane-street-real-time-market-data-forecasting`):

- Kaggle-reported competition **deadline: 2025-07-12** (end of the forecasting/evaluation phase).
- The competition's *final submission deadline* for the initial phase was **Jan 13, 2025**; the forecasting phase then scored submitted models against live market data until ~mid-2025 (deadline 2025-07-12). The leaderboard's last scored submissions are dated 2025-06-16.
- Audit date: **2026-06-24** — well past the deadline.
- The configured account *has entered* the competition (`userHasEntered=True`) but has **0 submissions** (`No submissions found`).
- **Empirical probe:** a CLI submit (`kaggle competitions submit -c jane-street-real-time-market-data-forecasting -f sample_submission.csv`) was attempted and the server **rejected it with `400 Bad Request` on `CreateSubmission`**; `submissions` still shows `No submissions found` afterwards. This is also a *code* competition (notebook-only submission), so a CSV file submit could not create a scored entry regardless. Submission is empirically closed.

**Answers:**

- Can we still train locally? **Yes.**
- Can we still run the evaluation API smoke locally? **Yes.**
- Can we still get a new *official* leaderboard score? **No** — the deadline (2025-07-12) has passed, so official scoring is closed. Even if Kaggle accepts a late submission, it must not be assumed to affect the official leaderboard or any private rescore.

## 8. Implications for validation protocol

- Local validation **must** come from `train.parquet` (the only file with labels).
- A random split is **invalid**: `date_id`/`time_id` are chronological, so random folds leak future information into the past.
- `test.parquet` is **not** local validation (no label).
- Lag features require careful leakage control: only D-1 (and earlier) responders may inform date D.
- The next stage should define a **split-protocol registry**:
  - A. `recent700_v200_g0` — recent 700 days, 200-day valid, gap 0
  - B. repo-style 2-fold CV
  - C. 200-day gap test
  - D. test-API smoke
