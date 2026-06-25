from datetime import datetime, timezone

from js2024.modeling.experiments import (
    dated_run_dir,
    timestamp_slug,
    timestamped_run_dir,
)


def test_timestamp_slug_is_filesystem_friendly():
    dt = datetime(2026, 6, 25, 15, 8, 35, tzinfo=timezone.utc)
    assert timestamp_slug(dt) == "20260625_150835"


def test_timestamped_run_dir_appends_optional_label():
    dt = timestamped_run_dir("experiments/demo", label="gpu")
    assert str(dt.parent).endswith("experiments/demo")
    assert dt.name.endswith("_gpu")


def test_dated_run_dir_dates_the_project_folder():
    dt = datetime(2026, 6, 25, 17, 6, 28, tzinfo=timezone.utc)
    run = dated_run_dir("experiments", "gru_evgeniavolkova", now=dt)
    assert run.parent.name == "20260625_gru_evgeniavolkova"
    assert run.name == "170628"
    assert str(run) == "experiments/20260625_gru_evgeniavolkova/170628"

