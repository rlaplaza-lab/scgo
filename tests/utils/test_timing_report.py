"""Tests for timing report helpers."""

from __future__ import annotations

import json
from pathlib import Path

from scgo.utils.timing_report import (
    build_run_timing_document,
    cpu_non_relax_seconds_from_timings,
    flatten_run_timing_payload,
    ga_relax_seconds_from_timings,
    load_run_timing_payload,
    relax_seconds_from_timings,
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


def test_multi_trial_timing_document(tmp_path: Path):
    payloads = [
        {"backend": "torchsim_ga", "timings_s": {"total_wall_s": 1.0}},
        {"backend": "torchsim_ga", "timings_s": {"total_wall_s": 2.0}},
    ]
    doc = build_run_timing_document(run_id="run_test", trial_payloads=payloads)
    assert doc["schema_version"] == 1
    assert doc["n_trials"] == 2
    assert len(doc["trials"]) == 2

    single = build_run_timing_document(run_id="run_test", trial_payloads=[payloads[0]])
    assert single == payloads[0]


def test_load_run_timing_fallback(tmp_path: Path):
    run_dir = tmp_path / "run_001"
    trial_dir = run_dir / "trial_1"
    trial_dir.mkdir(parents=True)
    legacy = {"backend": "torchsim_ga", "timings_s": {"total_wall_s": 3.0}}
    (trial_dir / "timing.json").write_text(json.dumps(legacy), encoding="utf-8")

    loaded = load_run_timing_payload(str(run_dir))
    assert loaded is not None
    assert flatten_run_timing_payload(loaded)["timings_s"]["total_wall_s"] == 3.0

    write_run_timing_file(
        str(run_dir), {"backend": "x", "timings_s": {"total_wall_s": 9.0}}
    )
    loaded_run = load_run_timing_payload(str(run_dir))
    assert loaded_run is not None
    assert loaded_run["timings_s"]["total_wall_s"] == 9.0
