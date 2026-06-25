# Experiments — layout & retention policy

## `experiments/` is disposable scratch

The `experiments/` tree is **gitignored and disposable**. Nothing in it is the
record of an experiment — it is just where runs dump heavy artifacts (models, OOF
frames, per-run `summary.csv` / `manifest.json` / `report.md`). You can `rm -rf` any
of it at any time without losing knowledge.

The committed record lives in **`docs/experiments/*.md`** (e.g. `lgbm_v0.md`,
`gru_v0.md`). A run is one of two things:

1. **Valuable** → promote it: add the numbers + conclusion to the relevant
   `docs/experiments/*.md`. Once promoted, the raw run folder is disposable.
2. **Just an experiment** → leave it in `experiments/` and let it be pruned.

This makes "valuable vs throwaway" an explicit promote step rather than a folder that
grows forever.

## Folder convention

The config-driven runner (`js2024-run-experiment`) writes to a **dated project
folder** so `ls experiments/` is sortable by date:

```
experiments/<YYYYMMDD>_<model>/<HHMMSS>/
  summary.csv      # tiny — the per-variant R² table
  manifest.json    # run metadata (config, variants, split)
  report.md        # generated markdown report
```

Same-day runs of one model group under one dated folder; old folders are obvious by
their date prefix and easy to delete.

## Avoiding bloat

- The walk-forward runner only writes KB-sized `summary`/`manifest`/`report` — cheap.
- The heavy directories are the LightGBM savers (`train_lgbm` / split sweeps), which
  persist full `model.txt` + `oof.parquet` per run (hundreds of MB for a sweep).
  **Follow-up:** gate those behind a default-off `save_models` switch; until then,
  prune `experiments/<old>_*/` after promoting any keepers to docs.
- Prune by date, e.g.: `find experiments -maxdepth 1 -type d -name '2026*' -mtime +30`.
