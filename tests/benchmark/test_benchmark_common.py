"""Fast tests for benchmark_common profiling helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from benchmark.benchmark_common import (
    apply_ga_benchmark_overrides,
    format_ga_profile_lines,
    get_benchmark_params,
    load_latest_ga_profile,
)
from scgo.utils.timing_report import TIMING_JSON_FILENAME


@pytest.mark.requires_mace
def test_apply_ga_benchmark_overrides_enables_timing():
    params = get_benchmark_params(42, backend="mace")
    updated = apply_ga_benchmark_overrides(
        params,
        niter=5,
        population_size=10,
    )
    ga = updated["optimizer_params"]["ga"]
    assert ga["write_timing_json"] is True
    assert ga["detailed_timing"] is True
    assert ga["niter"] == 5
    assert ga["population_size"] == 10


def test_format_ga_profile_lines_torchsim():
    profile = {
        "backend": "torchsim_ga",
        "timings_s": {
            "total_wall_s": 100.0,
            "initial_relax_batch_s": 40.0,
            "relax_batch_s": 30.0,
            "db_read_s": 2.0,
        },
    }
    lines = format_ga_profile_lines(profile, detailed=False)
    assert len(lines) == 1
    assert "Profiling (torchsim_ga)" in lines[0]
    assert "relax=70.0s" in lines[0]
    assert "db_io=2.0s" in lines[0]


def test_format_ga_profile_lines_ase_local_relaxation():
    profile = {
        "backend": "ga",
        "timings_s": {
            "total_wall_s": 50.0,
            "local_relaxation_s": 35.0,
        },
    }
    lines = format_ga_profile_lines(profile, detailed=False)
    assert "relax=35.0s" in lines[0]


def test_load_latest_ga_profile_reads_latest_run(tmp_path: Path):
    searches = tmp_path / "Pt5_searches"
    run_old = searches / "run_001"
    run_new = searches / "run_002"
    run_old.mkdir(parents=True)
    run_new.mkdir(parents=True)
    (run_old / TIMING_JSON_FILENAME).write_text(
        json.dumps({"backend": "ga", "timings_s": {"total_wall_s": 1.0}}),
        encoding="utf-8",
    )
    (run_new / TIMING_JSON_FILENAME).write_text(
        json.dumps({"backend": "ga", "timings_s": {"total_wall_s": 9.0}}),
        encoding="utf-8",
    )

    profile = load_latest_ga_profile(tmp_path, "Pt5")
    assert profile is not None
    assert profile["timings_s"]["total_wall_s"] == 9.0


def test_load_latest_ga_profile_warns_when_no_runs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.WARNING)
    assert load_latest_ga_profile(tmp_path, "Pt5") is None
    assert "No run_* directories" in caplog.text


def test_load_latest_ga_profile_warns_when_missing_timing_json(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    searches = tmp_path / "Pt5_searches" / "run_001"
    searches.mkdir(parents=True)
    caplog.set_level(logging.WARNING)
    assert load_latest_ga_profile(tmp_path, "Pt5") is None
    assert TIMING_JSON_FILENAME in caplog.text
