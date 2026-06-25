import polars as pl
import pytest

from js2024.data.make_smoke_data import main, make_smoke_data


def test_make_smoke_date_range(tmp_path, write_train):
    train = write_train()  # date_ids 0..3
    out = tmp_path / "interim" / "train_smoke.parquet"
    stats = make_smoke_data(train, out, start_date_id=1, end_date_id=2)
    assert out.exists()
    assert stats["date_min"] == 1
    assert stats["date_max"] == 2
    df = pl.read_parquet(out)
    assert df.height == stats["rows"] == 6
    assert set(df.get_column("date_id").to_list()) == {1, 2}


def test_make_smoke_refuses_overwrite(tmp_path, write_train):
    train = write_train()
    out = tmp_path / "train_smoke.parquet"
    make_smoke_data(train, out, start_date_id=0, end_date_id=3)
    with pytest.raises(FileExistsError):
        make_smoke_data(train, out, start_date_id=0, end_date_id=3)
    # force overwrites without error.
    make_smoke_data(train, out, start_date_id=0, end_date_id=3, force=True)


def test_cli_overwrite_returns_error(tmp_path, write_train, capsys):
    train = write_train()
    out = tmp_path / "train_smoke.parquet"
    rc = main(["--train-path", str(train), "--out-path", str(out)])
    assert rc == 0
    rc2 = main(["--train-path", str(train), "--out-path", str(out)])
    err = capsys.readouterr().err
    assert rc2 == 1
    assert "already exists" in err
