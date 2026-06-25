from js2024.data.data_profile import main, profile_train_data, write_profile


def test_profile_stats(write_train):
    train = write_train()
    profile = profile_train_data(train)
    stats = profile["stats"]
    assert stats["row_count"] == 12
    assert stats["date_min"] == 0
    assert stats["date_max"] == 3
    assert stats["symbol_n_unique"] == 2
    assert "target_mean" in stats
    assert "weight_mean" in stats
    # feature_00 has nulls -> should appear at the top of the missing-ratio list.
    top_features = [name for name, _ in profile["missing_ratio"]]
    assert "feature_00" in top_features
    assert profile["missing_ratio"][0][0] == "feature_00"
    assert profile["missing_ratio"][0][1] > 0.0


def test_write_profile_markdown(tmp_path, write_train):
    train = write_train()
    out = tmp_path / "reports" / "data_profile.md"
    written = write_profile(train, out, start_date_id=0, end_date_id=3)
    assert written == out
    text = out.read_text(encoding="utf-8")
    assert "# Train data profile" in text
    assert "row count" in text
    assert "date_id min/max" in text
    assert "responder_6" in text
    assert "weight" in text
    assert "missing ratio" in text.lower()


def test_cli_writes_report(tmp_path, write_train, capsys):
    train = write_train()
    out = tmp_path / "data_profile.md"
    rc = main(["--train-path", str(train), "--out", str(out)])
    assert rc == 0
    assert out.exists()
    assert "Wrote data profile" in capsys.readouterr().out
