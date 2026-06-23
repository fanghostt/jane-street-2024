"""CLI: download + extract the Kaggle competition data into ``data/raw``.

Usage
-----
    uv run js2024-download-data \\
        --competition jane-street-real-time-market-data-forecasting \\
        --out-dir data/raw

This wraps the official ``kaggle`` CLI (``kaggle competitions download``). It
relies entirely on the user's *local* Kaggle credentials and never writes,
prints, or commits any secret. The downloaded zip and extracted data are written
under ``--out-dir`` (gitignored) and are never committed.

After a successful extract it runs the raw-data contract check so a bad/partial
download fails loudly.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from .data_check import render_report
from .data_contract import check_raw_data_contract, contract_ok

DEFAULT_COMPETITION = "jane-street-real-time-market-data-forecasting"

_CREDENTIALS_HELP = (
    "Kaggle credentials were not found. To download competition data you must:\n"
    "  1. Create an API token on https://www.kaggle.com/settings (Account -> "
    "Create New Token) and save it to ~/.kaggle/kaggle.json,\n"
    "     OR export KAGGLE_USERNAME and KAGGLE_KEY in your environment.\n"
    "  2. Accept the competition rules on the Kaggle website for "
    f"'{DEFAULT_COMPETITION}'.\n"
    "This tool never reads or stores your credentials itself — it defers to the "
    "kaggle CLI."
)


class CredentialsError(RuntimeError):
    """Raised when Kaggle credentials are not configured locally."""


def _kaggle_config_dir() -> Path:
    """Directory the kaggle CLI reads ``kaggle.json`` from."""
    env = os.environ.get("KAGGLE_CONFIG_DIR")
    if env:
        return Path(env)
    return Path.home() / ".kaggle"


def have_kaggle_credentials() -> bool:
    """True if a kaggle.json file or KAGGLE_USERNAME/KAGGLE_KEY env vars are present."""
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return (_kaggle_config_dir() / "kaggle.json").is_file()


def ensure_kaggle_credentials() -> None:
    """Raise :class:`CredentialsError` with actionable help if creds are missing."""
    if not have_kaggle_credentials():
        raise CredentialsError(_CREDENTIALS_HELP)


def extract_zip(zip_path: str | Path, out_dir: str | Path, force: bool = False) -> list[str]:
    """Extract ``zip_path`` into ``out_dir``; skip members that already exist.

    Returns the list of member names actually written. Existing files are left
    untouched unless ``force`` is True.
    """
    zip_path = Path(zip_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if member.endswith("/"):
                continue
            target = out_dir / member
            if target.exists() and not force:
                print(f"[js2024] skip existing {member} (use --force to overwrite)")
                continue
            zf.extract(member, out_dir)
            written.append(member)
    return written


def download_competition(
    competition: str,
    out_dir: str | Path,
    force: bool = False,
) -> list[str]:
    """Download a competition zip via the kaggle CLI and extract it into ``out_dir``.

    Returns the list of extracted member names. Raises :class:`RuntimeError` if
    the kaggle CLI fails or produces no zip.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        cmd = ["kaggle", "competitions", "download", "-c", competition, "-p", td]
        print(f"[js2024] Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "The 'kaggle' CLI was not found on PATH. Install it (it is a "
                "project dependency: `uv sync`) and ensure your venv is active."
            ) from exc

        if result.returncode != 0:
            raise RuntimeError(
                "kaggle download failed (exit "
                f"{result.returncode}).\n--- stderr ---\n{result.stderr.strip()}"
            )

        zips = sorted(Path(td).glob("*.zip"))
        if not zips:
            raise RuntimeError(
                f"kaggle reported success but no .zip was found in {td}."
            )

        written: list[str] = []
        for z in zips:
            print(f"[js2024] Extracting {z.name} -> {out_dir}")
            written.extend(extract_zip(z, out_dir, force=force))
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download the Kaggle competition data into data/raw."
    )
    parser.add_argument(
        "--competition",
        default=DEFAULT_COMPETITION,
        help=f"Kaggle competition slug (default: {DEFAULT_COMPETITION}).",
    )
    parser.add_argument(
        "--out-dir",
        default="data/raw",
        help="Where to extract the data (default: data/raw).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite files that already exist in --out-dir.",
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Skip the raw-data contract check after download.",
    )
    args = parser.parse_args(argv)

    try:
        ensure_kaggle_credentials()
    except CredentialsError as exc:
        print(f"[js2024] ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        written = download_competition(args.competition, args.out_dir, force=args.force)
    except RuntimeError as exc:
        print(f"[js2024] ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"[js2024] Extracted {len(written)} file(s) into {args.out_dir}.")

    if args.no_check:
        return 0

    print("\n[js2024] Running contract check ...")
    report = check_raw_data_contract(args.out_dir)
    print(render_report(report))
    if not contract_ok(report):
        print("\n[js2024] ERROR: downloaded data failed the contract check.", file=sys.stderr)
        return 1
    print("\n[js2024] Contract OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
