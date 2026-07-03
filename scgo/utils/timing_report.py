"""Timing summary logging and ``timing.json`` for GO, basin hopping, NEB/TS, and GO+TS.

GA/BH: set ``write_timing_json`` and ``detailed_timing`` in
``optimizer_params['ga']`` (or ``bh``) inside ``params``/``go_params``.
TS: set ``write_timing_json`` in ``ts_params``.

GO timing is written at **run** level: ``{run_dir}/timing.json`` (alongside
``metadata.json``). Trial directories hold DB artifacts only.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

TIMING_JSON_FILENAME = "timing.json"
RUN_TIMING_SCHEMA_VERSION = 1

_DB_IO_SUM_KEYS: tuple[str, ...] = (
    "db_read_s",
    "db_write_s",
    "offspring_db_io_s",
    "initial_unrelaxed_insert_s",
    "initial_relaxed_write_s",
    "offspring_unrelaxed_insert_s",
    "offspring_relaxed_write_s",
    "unrelaxed_insert_s",
    "relaxed_write_s",
)


def ga_relax_seconds_from_timings(timings: dict[str, float]) -> float:
    """Total MLIP relax wall time for TorchSim GA (initial + offspring batches)."""
    if "local_relaxation_s" in timings and "relax_batch_s" not in timings:
        return float(timings.get("local_relaxation_s", 0.0))
    return float(timings.get("initial_relax_batch_s", 0.0)) + float(
        timings.get("relax_batch_s", 0.0)
    )


def relax_seconds_from_timings(timings: dict[str, float]) -> float:
    if "go_phase_s" in timings or "ts_neb_sum_s" in timings:
        return float(timings.get("go_phase_s", 0.0)) + float(
            timings.get("ts_neb_sum_s", 0.0)
        )
    if "neb_optimization_s" in timings:
        return float(timings.get("neb_optimization_s", 0.0))
    if "initial_relax_batch_s" in timings or (
        "relax_batch_s" in timings and "initial_local_relaxation_s" not in timings
    ):
        return ga_relax_seconds_from_timings(timings)
    if "local_relaxation_s" in timings and "relax_batch_s" not in timings:
        return float(timings.get("local_relaxation_s", 0.0))
    if "relax_batch_s" in timings:
        return float(timings.get("relax_batch_s", 0.0))
    return float(timings.get("initial_local_relaxation_s", 0.0)) + float(
        timings.get("offspring_local_relaxation_s", 0.0)
    )


def cpu_non_relax_seconds_from_timings(timings: dict[str, float]) -> float:
    total = float(timings.get("total_wall_s", 0.0))
    if "cpu_non_relax_s" in timings and "initial_relax_batch_s" not in timings:
        return float(timings["cpu_non_relax_s"])
    return max(0.0, total - relax_seconds_from_timings(timings))


def log_timing_summary(
    logger: logging.Logger,
    backend: str,
    timings_s: dict[str, float],
    *,
    verbosity: int,
) -> None:
    if verbosity < 1:
        return
    total = float(timings_s.get("total_wall_s", 0.0))
    relax = relax_seconds_from_timings(timings_s)
    cpu = cpu_non_relax_seconds_from_timings(timings_s)
    db_io = sum(float(timings_s.get(k, 0.0)) for k in _DB_IO_SUM_KEYS)
    logger.info(
        "Timing (%s): total=%.1fs, relax=%.1fs, non_relax=%.1fs, db_io=%.1fs",
        backend,
        total,
        relax,
        cpu,
        db_io,
    )


def write_timing_file(
    output_dir: str,
    payload: dict[str, Any],
    *,
    filename: str | None = None,
) -> str:
    name = filename if filename is not None else TIMING_JSON_FILENAME
    path = os.path.join(output_dir, name)
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def read_timing_file(path: str) -> dict[str, Any] | None:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def resolve_run_timing_path(run_dir: str) -> str:
    return os.path.join(run_dir, TIMING_JSON_FILENAME)


def load_run_timing_payload(run_dir: str) -> dict[str, Any] | None:
    """Load ``timing.json`` from a run directory."""
    return read_timing_file(resolve_run_timing_path(run_dir))


def flatten_run_timing_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a single-trial payload for consumers expecting flat GA timing."""
    if "trials" in payload and isinstance(payload["trials"], list):
        if len(payload["trials"]) == 1:
            return dict(payload["trials"][0])
        if payload["trials"]:
            return dict(payload["trials"][-1])
    return payload


def build_run_timing_document(
    *,
    run_id: str,
    trial_payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(trial_payloads) == 1:
        return trial_payloads[0]
    return {
        "schema_version": RUN_TIMING_SCHEMA_VERSION,
        "run_id": run_id,
        "n_trials": len(trial_payloads),
        "trials": trial_payloads,
    }


def write_run_timing_file(
    run_dir: str,
    payload: dict[str, Any],
) -> str:
    return write_timing_file(run_dir, payload)


def sum_neb_seconds_from_ts_results(
    ts_results: list[dict[str, Any]],
) -> float:
    return sum(
        float((r.get("timings_s") or {}).get("neb_optimization_s", 0.0))
        for r in ts_results
    )
