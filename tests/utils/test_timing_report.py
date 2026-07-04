"""Tests for timing report helpers."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from scgo.utils.logging import get_logger
from scgo.utils.timing_report import (
    build_run_timing_document,
    cpu_non_relax_seconds_from_timings,
    flatten_run_timing_payload,
    ga_relax_seconds_from_timings,
    load_run_timing_payload,
    log_timing_summary,
    read_timing_file,
    relax_seconds_from_timings,
    sum_neb_seconds_from_ts_results,
    write_run_timing_file,
)


def test_ga_relax_includes_initial_and_offspring_batches():
    timings = {
        "initial_relax_batch_s": 100.0,
        "relax_batch_s": 50.0,
        "total_wall_s": 200.0,
    }
    assert ga_relax_seconds_from_timings(timings) == 150.0
    assert relax_seconds_from_timings(timings) == 150.0
    assert cpu_non_relax_seconds_from_timings(timings) == 50.0


def test_build_run_timing_document_attaches_run_id():
    payload = {"backend": "torchsim_ga", "timings_s": {"total_wall_s": 1.0}}
    doc = build_run_timing_document(run_id="run_test", payload=payload)
    assert doc["run_id"] == "run_test"
    assert doc["timings_s"]["total_wall_s"] == 1.0


def test_load_run_timing_payload(tmp_path: Path):
    run_dir = tmp_path / "run_001"
    run_dir.mkdir(parents=True)

    assert load_run_timing_payload(str(run_dir)) is None

    write_run_timing_file(
        str(run_dir), {"backend": "x", "timings_s": {"total_wall_s": 9.0}}
    )
    loaded = load_run_timing_payload(str(run_dir))
    assert loaded is not None
    assert loaded["timings_s"]["total_wall_s"] == 9.0
    assert flatten_run_timing_payload(loaded)["timings_s"]["total_wall_s"] == 9.0


def test_read_timing_file_warns_on_corrupt_json(tmp_path: Path, caplog):
    path = tmp_path / "timing.json"
    path.write_text("{not valid json", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        assert read_timing_file(str(path)) is None
    assert "Failed to read timing file" in caplog.text


@pytest.mark.parametrize(
    ("timings", "expected"),
    [
        ({"go_phase_s": 10.0, "ts_neb_sum_s": 5.0}, 15.0),
        ({"neb_optimization_s": 7.0}, 7.0),
        ({"local_relaxation_s": 3.0}, 3.0),
        (
            {"initial_local_relaxation_s": 2.0, "offspring_local_relaxation_s": 4.0},
            6.0,
        ),
    ],
)
def test_relax_seconds_from_timings_branches(timings: dict, expected: float):
    assert relax_seconds_from_timings(timings) == expected


def test_cpu_non_relax_uses_precomputed_when_present():
    timings = {"cpu_non_relax_s": 8.0, "total_wall_s": 20.0}
    assert cpu_non_relax_seconds_from_timings(timings) == 8.0


def test_cpu_non_relax_infers_from_total_minus_relax():
    timings = {
        "total_wall_s": 100.0,
        "initial_relax_batch_s": 60.0,
        "relax_batch_s": 10.0,
    }
    assert cpu_non_relax_seconds_from_timings(timings) == 30.0


def test_flatten_run_timing_payload_rejects_multi_trial():
    doc = {
        "schema_version": 1,
        "trials": [
            {"backend": "ga", "timings_s": {"total_wall_s": 1.0}},
        ],
    }
    with pytest.raises(ValueError, match="Multi-trial timing"):
        flatten_run_timing_payload(doc)


def test_read_timing_file_invalid_json(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    bad = tmp_path / "timing.json"
    bad.write_text("{not json", encoding="utf-8")
    caplog.set_level(logging.DEBUG)
    assert read_timing_file(str(bad)) is None
    assert "Failed to read timing file" in caplog.text


def test_log_timing_summary_respects_verbosity(caplog: pytest.LogCaptureFixture):
    logger = get_logger("test_timing_summary")
    timings = {"total_wall_s": 10.0, "local_relaxation_s": 4.0}
    caplog.set_level(logging.INFO)
    log_timing_summary(logger, "ga", timings, verbosity=0)
    assert "Timing (ga)" not in caplog.text
    log_timing_summary(logger, "ga", timings, verbosity=1)
    assert "Timing (ga)" in caplog.text


def test_sum_neb_seconds_from_ts_results():
    results = [
        {"timings_s": {"neb_optimization_s": 1.5}},
        {"timings_s": {"neb_optimization_s": 2.5}},
        {},
    ]
    assert sum_neb_seconds_from_ts_results(results) == pytest.approx(4.0)
