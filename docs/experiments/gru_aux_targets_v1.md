# GRU auxiliary-target sets — experiment log

Does giving the day-batch GRU **more auxiliary supervision** improve the shared
representation it learns? The public solution trains 4 auxiliary responder heads
(`responder_10/9/8/7`, "base4") alongside the target (`responder_6`), then a final
linear head combines the aux predictions into the target. This experiment makes that
aux set a config knob (`aux_target_set`, see `src/js2024/modeling/gru.py`
`GRU_AUX_TARGET_SETS`) and A/Bs four sets:

| set | responder heads | n |
| --- | --- | ---: |
| `base4` (default) | responder_10, 9, 8, 7 | 4 |
| `target_family` | responder_6, 7, 8, 9, 10 | 5 |
| `all9` | responder_0 .. 8 (all real) | 9 |
| `all11` | responder_0 .. 10 (real + synthetic 9/10) | 11 |

`responder_9`/`responder_10` are synthetic (shifted sums of responder_8/6; see
`add_gru_aux_targets`), always generated regardless of the set. The aux targets are
**training targets, not model inputs** — feature columns are identical across sets, so
this isolates "more aux supervision" from any input change. The online inference
update is unchanged: it optimizes the `responder_6` target loss only, for every set.

## Method

Decisive metric is the **paired** online-incremental delta vs `base4` at the same seed:

    delta(set, seed) = gru_incremental(set) − gru_incremental(base4)

Engineered market/rolling features are **off** (so this is orthogonal to
`market_features_v1.md`). recent700 window, shared last-200 test block. The
`js2024-run-aux-sweep` runner loads one maximal (`all11`) frame and reuses it across
every (set, seed) cell so the pairing is exact.

Configs: `configs/gru_aux_{base4,target_family,all9,all11}.yaml`. Heavy per-run
artifacts are gitignored; this file is the committed record.

## Status / verdict

**More *real*-responder supervision helps; adding the target itself as a head hurts.**
Both `all9` and `all11` beat `base4` on the paired delta with **3/3 seeds positive**
(small but sign-stable, same ~+2–3% magnitude as the market features). `all11` (adds
the synthetic responder_9/10 heads on top of `all9`) is best by a hair, but the
+0.000034 over `all9` is well inside the seed std — not a real difference. The synthetic
heads add ~20% wall-clock (11 vs 9 heads) for no measurable gain. `target_family`
(adding `responder_6`, the target, as an aux head) is the only loser: 0/3 seeds, a clear
−0.00034. Paired sweep, seeds 42/43/44, recent700, test block `[1499, 1698]`, engineered
features off.

| aux_set | n seeds | mean Δ vs base4 | std Δ | +seeds | mean set R² | mean base4 R² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base4 (ref) | 3 | — | — | — | 0.011128 | 0.011128 |
| **all11** | 3 | **+0.000303** | 0.000135 | **3/3** | 0.011430 | 0.011128 |
| **all9** | 3 | **+0.000269** | 0.000244 | **3/3** | 0.011397 | 0.011128 |
| target_family | 3 | −0.000339 | 0.000170 | 0/3 | 0.010789 | 0.011128 |

- **verdict:** adopt **`all9`** — it captures essentially all of the gain (+~2.4%) at 9
  heads, while `all11`'s synthetic heads cost ~20% more compute for a within-noise
  +0.000034. Keep `base4` as the documented default (bit-equivalent to prior runs);
  `all9` is the recommended upgrade. **Avoid `target_family`** — putting the target on
  an aux head is actively harmful.
- base4's 3-seed mean (0.011128) reproduces the historical baseline exactly — confirms
  the refactor is behaviour-preserving.

### Production adoption

`configs/gru_marketroll_v1.yaml` (the SOTA config: market-avg + per-symbol rolling) now
sets `aux_target_set: all9`, stacking this win on top of the market features. **Caveat:**
this sweep measured `all9` with engineered features *off*, so the all9 × market-features
combination is a best-bet, not a paired A/B yet — verify it before trusting the stacked
number. `GRUConfig`'s default stays `base4` (bit-equivalent to prior runs); only the
production config opts in.

## Reproduce

```bash
# single configs (one seed each):
uv run js2024-run-experiment --config configs/gru_aux_base4.yaml         --variants incremental
uv run js2024-run-experiment --config configs/gru_aux_target_family.yaml --variants incremental
uv run js2024-run-experiment --config configs/gru_aux_all9.yaml          --variants incremental
uv run js2024-run-experiment --config configs/gru_aux_all11.yaml         --variants incremental

# preferred — paired sweep across seeds (decisive):
uv run js2024-run-aux-sweep --aux-sets base4,target_family,all9,all11 --seeds 42,43,44
```
