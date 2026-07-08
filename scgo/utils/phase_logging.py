"""Phase-oriented logging helpers for SCGO runs (headers, summaries, collectors)."""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from logging import Logger


def infer_verbosity(logger: Logger, explicit: int | None = None) -> int:
    """Map an explicit verbosity or infer from the configured logger level."""
    if explicit is not None:
        return explicit
    return 2 if logger.isEnabledFor(logging.DEBUG) else 1


def log_phase_header(
    logger: Logger,
    title: str,
    *,
    verbosity: int,
    level: int = 1,
) -> None:
    """Emit a visible phase banner when ``verbosity >= level``."""
    if verbosity < level:
        return
    line = "=" * 60
    logger.info(line)
    logger.info(title)
    logger.info(line)


def log_phase_subheader(
    logger: Logger,
    title: str,
    *,
    verbosity: int,
    level: int = 1,
) -> None:
    """Emit a lighter sub-phase banner (e.g. per generation)."""
    if verbosity < level:
        return
    logger.info("--- %s ---", title)


def format_count_summary(counts: Mapping[str, int]) -> str:
    """Format outcome counts as ``label×N, ...``."""
    parts = [
        f"{label}×{count}"
        for label, count in sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        if count > 0
    ]
    return ", ".join(parts) if parts else ""


class InitDiagnosticsCollector:
    """Thread-safe accumulator for initialization fallbacks and placement failures."""

    _lock = threading.Lock()
    _fallback_records: list[tuple[str, str]] = []
    _placement_failures: list[tuple[str, str]] = []

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._fallback_records.clear()
            cls._placement_failures.clear()

    @classmethod
    def record_fallback(cls, used_strategy: str, from_strategy: str) -> None:
        with cls._lock:
            cls._fallback_records.append((used_strategy, from_strategy))

    @classmethod
    def record_placement_failure(cls, compact_line: str, detail_msg: str) -> None:
        with cls._lock:
            cls._placement_failures.append((compact_line, detail_msg))

    @classmethod
    def emit_summary(
        cls,
        logger: Logger,
        *,
        verbosity: int,
        n_structures: int,
        prefix: str = "Population initialization",
        extra: str = "",
    ) -> None:
        """Emit one INFO summary at v1+ and per-record DEBUG detail at v2+."""
        if verbosity < 1:
            return

        with cls._lock:
            fallbacks = list(cls._fallback_records)
            placement_failures = list(cls._placement_failures)

        template_to_random = sum(
            1 for used, fb in fallbacks if used == "random_spherical" and fb == "template"
        )
        seed_to_random = sum(
            1
            for used, fb in fallbacks
            if used == "random_spherical" and fb == "seed+growth"
        )

        parts: list[str] = [f"built {n_structures}/{n_structures} candidates"]
        if template_to_random or seed_to_random:
            parts.append(
                f"fallbacks template→random×{template_to_random}, "
                f"seed→random×{seed_to_random}"
            )
        if placement_failures:
            parts.append(f"placement failures×{len(placement_failures)}")
        if extra:
            parts.append(extra)

        logger.info("%s: %s", prefix, "; ".join(parts))

        if verbosity < 2:
            return

        for used, fb in fallbacks:
            logger.debug("Init fallback: %s→%s", fb, used)
        for compact, detail in placement_failures:
            logger.debug("Placement failure: %s", compact)
            if detail != compact:
                logger.debug("%s", detail)


def log_generation_offspring_summaries(
    logger: Logger,
    *,
    verbosity: int,
    job_results: list[Mapping[str, Any]],
    total_jobs: int,
    created: int,
    n_offspring: int,
    attempts: int,
) -> None:
    """Log v1 generation crossover/mutation/offspring summaries and v2 per-job detail."""
    if verbosity >= 1:
        failures: dict[str, int] = {}
        mutation_applied = 0
        for result in job_results:
            reason = result.get("failure_reason")
            if reason:
                failures[str(reason)] = failures.get(str(reason), 0) + 1
            if result.get("mutation_applied"):
                mutation_applied += 1

        if total_jobs > 0:
            succeeded = total_jobs - sum(failures.values())
            detail = format_count_summary(failures)
            crossover_msg = f"Crossover: {succeeded}/{total_jobs} succeeded"
            if detail:
                crossover_msg = f"{crossover_msg} ({detail})"
            logger.info(crossover_msg)
            logger.info(
                "Mutation: applied to %d/%d offspring",
                mutation_applied,
                total_jobs,
            )
        logger.info(
            "Offspring: created %d/%d (attempts=%d)",
            created,
            n_offspring,
            attempts,
        )

    if verbosity < 2:
        return

    for result in job_results:
        logger.debug(
            "%s",
            format_offspring_outcome_line(
                int(result["index"]) + 1,
                failure_reason=result.get("failure_reason"),
                desc=result.get("desc"),
                mutation_applied=bool(result.get("mutation_applied")),
                validation_error=result.get("validation_error"),
            ),
        )


def format_offspring_outcome_line(
    index: int,
    *,
    failure_reason: str | None,
    desc: str | None,
    mutation_applied: bool,
    validation_error: str | None,
) -> str:
    """One-line DEBUG summary for a single offspring build attempt."""
    if failure_reason == "pairing_failed":
        return f"Offspring {index}: crossover failed"
    if failure_reason == "too_close_prefilter":
        return f"Offspring {index}: rejected (atoms too close prefilter)"
    if failure_reason == "validation_failed":
        err = (validation_error or "validation failed").splitlines()[0]
        return f"Offspring {index}: validation_failed — {err}"
    if desc and "mutation:" in desc:
        op = desc.split("mutation:", 1)[1].strip().split()[0]
        mutation = f"mutation={op}"
    else:
        mutation = "mutation applied" if mutation_applied else "no mutation"
    return f"Offspring {index}: crossover ok, {mutation}, eligible"
