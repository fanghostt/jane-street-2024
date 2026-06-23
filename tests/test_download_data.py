"""Tests for the Kaggle download CLI. These never touch the network or Kaggle.

The kaggle subprocess call is mocked; credential detection is steered via env
vars / KAGGLE_CONFIG_DIR.
"""

from __future__ import annotations

import subprocess
import types
import zipfile
from pathlib import Path

import pytest

from js2024 import download_data
from js2024.download_data import (
    CredentialsError,
    download_competition,
    ensure_kaggle_credentials,
    extract_zip,
    have_kaggle_credentials,
    main,
)


def _clear_creds(monkeypatch, tmp_path):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    # Point the config dir at an empty tmp dir so no kaggle.json is found.
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path / "empty_kaggle"))


def _set_env_creds(monkeypatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "user")
    monkeypatch.setenv("KAGGLE_KEY", "key")


def _make_zip(path: Path, files: dict[str, str]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return path


# --- credentials -----------------------------------------------------------


def test_credentials_missing_message(monkeypatch, tmp_path, capsys):
    _clear_creds(monkeypatch, tmp_path)
    assert have_kaggle_credentials() is False
    with pytest.raises(CredentialsError):
        ensure_kaggle_credentials()

    rc = main(["--competition", "x", "--out-dir", str(tmp_path / "out")])
    err = capsys.readouterr().err
    assert rc == 1
    assert "kaggle.json" in err
    assert "KAGGLE_USERNAME" in err
    assert "accept the competition rules".lower() in err.lower()


def test_credentials_env_detected(monkeypatch, tmp_path):
    _clear_creds(monkeypatch, tmp_path)
    _set_env_creds(monkeypatch)
    assert have_kaggle_credentials() is True
    ensure_kaggle_credentials()  # should not raise


def test_credentials_kaggle_json_detected(monkeypatch, tmp_path):
    _clear_creds(monkeypatch, tmp_path)
    cfg = tmp_path / "kaggle_home"
    cfg.mkdir()
    (cfg / "kaggle.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(cfg))
    assert have_kaggle_credentials() is True


def test_credentials_access_token_detected(monkeypatch, tmp_path):
    # Newer kaggle CLI stores a ~/.kaggle/access_token instead of kaggle.json.
    _clear_creds(monkeypatch, tmp_path)
    cfg = tmp_path / "kaggle_home"
    cfg.mkdir()
    (cfg / "access_token").write_text("KGAT_xxx", encoding="utf-8")
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(cfg))
    assert have_kaggle_credentials() is True
    ensure_kaggle_credentials()  # should not raise


# --- extract / force -------------------------------------------------------


def test_extract_zip_skips_existing_without_force(tmp_path):
    zip_path = _make_zip(tmp_path / "d.zip", {"a.txt": "new", "b.txt": "b"})
    out = tmp_path / "out"
    out.mkdir()
    (out / "a.txt").write_text("original", encoding="utf-8")

    written = extract_zip(zip_path, out, force=False)
    assert "b.txt" in written
    assert "a.txt" not in written
    assert (out / "a.txt").read_text() == "original"  # not overwritten


def test_extract_zip_force_overwrites(tmp_path):
    zip_path = _make_zip(tmp_path / "d.zip", {"a.txt": "new"})
    out = tmp_path / "out"
    out.mkdir()
    (out / "a.txt").write_text("original", encoding="utf-8")

    written = extract_zip(zip_path, out, force=True)
    assert "a.txt" in written
    assert (out / "a.txt").read_text() == "new"


# --- download (subprocess mocked) ------------------------------------------


def _mock_subprocess_run(files: dict[str, str]):
    """Return a fake subprocess.run that drops a zip into the ``-p`` directory."""

    def _run(cmd, capture_output=False, text=False):
        dest = Path(cmd[cmd.index("-p") + 1])
        dest.mkdir(parents=True, exist_ok=True)
        _make_zip(dest / "download.zip", files)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    return _run


def test_download_competition_extracts(monkeypatch, tmp_path):
    monkeypatch.setattr(
        download_data.subprocess,
        "run",
        _mock_subprocess_run({"features.csv": "x", "train.parquet": "p"}),
    )
    out = tmp_path / "raw"
    written = download_competition("comp", out, force=False)
    assert set(written) == {"features.csv", "train.parquet"}
    assert (out / "features.csv").exists()


def test_download_competition_failure_raises(monkeypatch, tmp_path):
    def _fail(cmd, capture_output=False, text=False):
        return types.SimpleNamespace(returncode=1, stdout="", stderr="403 Forbidden")

    monkeypatch.setattr(download_data.subprocess, "run", _fail)
    with pytest.raises(RuntimeError) as exc:
        download_competition("comp", tmp_path / "raw")
    assert "403" in str(exc.value)


def test_main_force_overwrites_existing(monkeypatch, tmp_path):
    _clear_creds(monkeypatch, tmp_path)
    _set_env_creds(monkeypatch)
    monkeypatch.setattr(
        download_data.subprocess,
        "run",
        _mock_subprocess_run({"train.parquet": "new-content"}),
    )
    out = tmp_path / "raw"
    out.mkdir()
    (out / "train.parquet").write_text("old-content", encoding="utf-8")

    # Without force: existing file preserved.
    rc = main(["--competition", "c", "--out-dir", str(out), "--no-check"])
    assert rc == 0
    assert (out / "train.parquet").read_text() == "old-content"

    # With force: overwritten.
    rc = main(["--competition", "c", "--out-dir", str(out), "--no-check", "--force"])
    assert rc == 0
    assert (out / "train.parquet").read_text() == "new-content"
