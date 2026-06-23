from js2024.data_check import main


def test_cli_success(raw_dir_factory, capsys):
    raw = raw_dir_factory()
    rc = main(["--raw-dir", str(raw)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Contract OK" in out
    assert "Required files" in out


def test_cli_missing_file_fails(raw_dir_factory, capsys):
    raw = raw_dir_factory(missing_files=["lags.parquet"])
    rc = main(["--raw-dir", str(raw)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "NOT satisfied" in out


def test_cli_missing_columns_fails(raw_dir_factory, capsys):
    raw = raw_dir_factory(drop_train_columns=["responder_6"])
    rc = main(["--raw-dir", str(raw)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "responder_6" in out


def test_cli_nonexistent_dir_fails(tmp_path, capsys):
    rc = main(["--raw-dir", str(tmp_path / "nope")])
    assert rc == 1


def test_cli_missing_train_reports_not_checked(raw_dir_factory, capsys):
    # raw dir exists but train.parquet is absent -> schema can't be checked.
    raw = raw_dir_factory(missing_files=["train.parquet"])
    rc = main(["--raw-dir", str(raw)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "not checked" in out
    assert "missing columns:** none" not in out
