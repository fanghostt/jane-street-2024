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

**TODO** — fill after the sweep completes.

| aux_set | n seeds | mean Δ vs base4 | std Δ | +seeds | mean set R² | mean base4 R² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base4 (ref) | — | — | — | — | TODO | TODO |
| target_family | TODO | TODO | TODO | TODO | TODO | TODO |
| all9 | TODO | TODO | TODO | TODO | TODO | TODO |
| all11 | TODO | TODO | TODO | TODO | TODO | TODO |

- **verdict:** TODO (keep base4 / adopt larger set?).

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
