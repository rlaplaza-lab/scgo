"""Timing summary logging and ``timing.json`` for GO, basin hopping, NEB/TS, and GO+TS.

GA/BH: set ``write_timing_json`` and ``detailed_timing`` in
``optimizer_params['ga']`` (or ``bh``) inside ``params``/``go_params``.
TS: set ``write_timing_json`` in ``ts_params``.

GO timing is written at **run** level: ``{run_dir}/timing.json`` (alongside
``metadata.json`` and the optimizer database).

GO+TS pipeline rollup timing is written at the campaign root as
``go_ts_timing.json`` when ``write_timing_json=True`` in ``go_params`` and/or
``ts_params``.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from scgo.exceptions import (
    SCGOValidationError,
)
from scgo.utils.logging import get_logger
from scgo.utils.ts_provenance import ts_output_provenance

_logger = get_logger(__name__)

TIMING_JSON_FILENAME = "timing.json"
GO_TS_TIMING_JSON_FILENAME = "go_ts_timing.json"
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
    """Return total relax/NEB wall time inferred from a timing payload."""
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
    """Return non-relax CPU wall time (precomputed or total minus relax)."""
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
    """Log a one-line timing summary when ``verbosity >= 1``."""
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
    """Load a timing JSON file; return ``None`` if missing or unreadable."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        _logger.warning("Failed to read timing file %s: %s", path, exc)
        return None


def resolve_run_timing_path(run_dir: str) -> str:
    return os.path.join(run_dir, TIMING_JSON_FILENAME)


def load_run_timing_payload(run_dir: str) -> dict[str, Any] | None:
    """Load ``timing.json`` from a run directory."""
    return read_timing_file(resolve_run_timing_path(run_dir))


def flatten_run_timing_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a flat timing payload (legacy multi-trial documents are rejected)."""
    if "trials" in payload:
        raise SCGOValidationError(
            "Multi-trial timing documents are no longer supported; "
            "expected a flat timing.json at run root."
        )
    return payload


def build_timing_payload(
    *,
    backend: str,
    timings_s: dict[str, float],
    run_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a structured timing document with provenance header and schema version."""
    payload: dict[str, Any] = {
        **ts_output_provenance(),
        "timing_schema_version": RUN_TIMING_SCHEMA_VERSION,
        "backend": backend,
        "timings_s": timings_s,
    }
    if run_id is not None:
        payload["run_id"] = run_id
    if extra:
        payload.update(extra)
    return payload


def build_run_timing_document(
    *,
    run_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Attach run_id to a single-run timing payload."""
    out = dict(payload)
    out.setdefault("run_id", run_id)
    return out


def write_run_timing_file(
    run_dir: str,
    payload: dict[str, Any],
    *,
    run_id: str | None = None,
) -> str:
    if run_id is not None:
        payload = build_run_timing_document(run_id=run_id, payload=payload)
    return write_timing_file(run_dir, payload)


def sum_neb_seconds_from_ts_results(
    ts_results: list[dict[str, Any]],
) -> float:
    """Sum per-pair ``neb_optimization_s`` values from TS result dicts."""
    return sum(
        float((r.get("timings_s") or {}).get("neb_optimization_s", 0.0))
        for r in ts_results
    )
