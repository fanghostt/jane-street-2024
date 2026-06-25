"""Tests for the data-semantics audit CLI.

These build tiny fake raw files under ``tmp_path`` and never touch the real
(47M-row) Kaggle data or the Kaggle API.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from js2024.data.data_semantics_audit import (
    audit_features_csv,
    audit_parquet,
    audit_responders_csv,
    main,
    run_audit,
)

FEATURES = [f"feature_{i:02d}" for i in range(79)]
RESPONDERS = [f"responder_{i}" for i in range(9)]
LAGS = [f"responder_{i}_lag_1" for i in range(9)]


def _make_raw_dir(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    n = 6

    # train: ids + weight + features + all responders (incl. responder_6).
    train = {
        "date_id": [0, 0, 1, 1, 2, 2],
        "time_id": [0, 1, 0, 1, 0, 1],
        "symbol_id": [0, 1, 0, 1, 0, 1],
        "weight": [1.0, 2.0, 1.5, 2.5, 1.2, 2.2],
    }
    for f in FEATURES:
        train[f] = [0.1 * i for i in range(n)]
    for r in RESPONDERS:
        # responder values within the clipped [-5, 5] range.
        train[r] = [-1.0, 0.5, 1.0, -0.5, 2.0, -2.0]
    pl.DataFrame(train).write_parquet(raw / "train.parquet")

    # test: row_id + ids + weight + is_scored + features, but NO responder_6.
    test = {
        "row_id": list(range(n)),
        "date_id": [0] * n,
        "time_id": [0] * n,
        "symbol_id": [0, 1, 2, 3, 4, 5],
        "weight": [1.0] * n,
        "is_scored": [False] * n,
    }
    for f in FEATURES:
        test[f] = [0.2 * i for i in range(n)]
    pl.DataFrame(test).write_parquet(raw / "test.parquet")

    # lags: ids + responder_*_lag_1.
    lags = {
        "date_id": [0] * n,
        "time_id": [0] * n,
        "symbol_id": [0, 1, 2, 3, 4, 5],
    }
    for lcol in LAGS:
        lags[lcol] = [-0.3 * i for i in range(n)]
    pl.DataFrame(lags).write_parquet(raw / "lags.parquet")

    # features.csv: feature + a few boolean tags.
    feat_rows = []
    for i, f in enumerate(FEATURES):
        feat_rows.append(
            {
                "feature": f,
                "tag_0": i % 2 == 0,
                "tag_1": i % 3 == 0,
                "tag_2": f == "feature_61",
            }
        )
    pl.DataFrame(feat_rows).write_csv(raw / "features.csv")

    # responders.csv: responder + boolean tags.
    resp_rows = []
    for i, r in enumerate(RESPONDERS):
        resp_rows.append(
            {"responder": r, "tag_0": i % 2 == 0, "tag_1": i == 6}
        )
    pl.DataFrame(resp_rows).write_csv(raw / "responders.csv")

    # sample_submission.csv
    pl.DataFrame(
        {"row_id": list(range(n)), "responder_6": [0.0] * n}
    ).write_csv(raw / "sample_submission.csv")

    return raw


def test_audit_parquet_train_has_responder_6(tmp_path):
    raw = _make_raw_dir(tmp_path)
    a = audit_parquet(raw / "train.parquet")
    assert a["exists"]
    assert a["presence"]["has_responder_6"] is True
    assert a["presence"]["has_all_features"] is True
    assert a["presence"]["n_responder_columns"] == 9
    assert a["responders_clipped_-5_5"] is True
    assert a["row_count"] == 6


def test_audit_parquet_test_has_no_responder_6(tmp_path):
    raw = _make_raw_dir(tmp_path)
    a = audit_parquet(raw / "test.parquet")
    assert a["exists"]
    assert a["presence"]["has_responder_6"] is False
    assert a["presence"]["is_scored"] is True
    assert a["presence"]["row_id"] is True


def test_audit_parquet_lags_columns(tmp_path):
    raw = _make_raw_dir(tmp_path)
    a = audit_parquet(raw / "lags.parquet")
    assert a["exists"]
    assert a["presence"]["has_all_lags"] is True
    assert a["presence"]["n_lag_columns"] == 9


def test_audit_features_tags(tmp_path):
    raw = _make_raw_dir(tmp_path)
    a = audit_features_csv(raw / "features.csv")
    assert a["exists"]
    assert a["row_count"] == 79
    assert a["n_tag_columns"] == 3
    # tag_2 is only set on feature_61.
    assert a["tag_summary"]["tag_2"]["true"] == 1
    assert a["groups"]["tag_2"] == ["feature_61"]


def test_audit_responders_tags(tmp_path):
    raw = _make_raw_dir(tmp_path)
    a = audit_responders_csv(raw / "responders.csv")
    assert a["exists"]
    assert a["row_count"] == 9
    assert "responder_6" in a["names"]


def test_run_audit_collects_all(tmp_path):
    raw = _make_raw_dir(tmp_path)
    results = run_audit(raw)
    for key in (
        "train",
        "test",
        "lags",
        "features",
        "responders",
        "sample_submission",
    ):
        assert results[key]["exists"], key


def test_cli_writes_docs_and_artifacts(tmp_path, capsys):
    raw = _make_raw_dir(tmp_path)
    out_dir = tmp_path / "outputs" / "audit"
    docs_out = tmp_path / "docs" / "data_semantics_audit.md"

    rc = main(
        [
            "--raw-dir",
            str(raw),
            "--out-dir",
            str(out_dir),
            "--docs-out",
            str(docs_out),
        ]
    )
    assert rc == 0

    text = docs_out.read_text(encoding="utf-8")
    # test has no label; train does.
    assert "contains responder_6: no" in text
    assert "responder_6 exists: yes" in text
    # lags semantics documented.
    assert "responder_*_lag_1" in text or "responder_0_lag_1" in text
    assert "first `time_id`" in text
    # features / responders tag summaries present.
    assert "## 5. features.csv audit" in text
    assert "## 6. responders.csv audit" in text
    assert "tag_0" in text
    # competition status answered.
    assert "official" in text.lower()

    # git-ignored artifacts.
    assert (out_dir / "train_schema.csv").exists()
    assert (out_dir / "test_schema.csv").exists()
    assert (out_dir / "lags_schema.csv").exists()
    assert (out_dir / "features_metadata.csv").exists()
    assert (out_dir / "responders_metadata.csv").exists()
    assert (out_dir / "file_summary.json").exists()


def test_missing_files_do_not_crash(tmp_path, capsys):
    empty = tmp_path / "empty_raw"
    empty.mkdir()
    out_dir = tmp_path / "out"
    docs_out = tmp_path / "doc.md"
    rc = main(
        ["--raw-dir", str(empty), "--out-dir", str(out_dir), "--docs-out", str(docs_out)]
    )
    assert rc == 0
    text = docs_out.read_text(encoding="utf-8")
    assert "not found in raw dir" in text
