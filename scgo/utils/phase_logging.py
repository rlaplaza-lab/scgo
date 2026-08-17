"""Phase-oriented logging helpers for SCGO runs (headers, summaries, collectors)."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from logging import Logger


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
    """Format outcome counts as ``label x N, ...``."""
    parts = [
        f"{label}x{count}"
        for label, count in sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        if count > 0
    ]
    return ", ".join(parts) if parts else ""


def compact_neb_pair_reason(message: str) -> str:
    """Short label for grouping NEB skip/fail reasons (strip wrappers and numbers)."""
    text = str(message).strip()
    for prefix in (
        "Initial NEB path rejected (energy profile): ",
        "Initial NEB path rejected (clashing/discontinuous interpolation): ",
        "Initial NEB path rejected: ",
    ):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
            break

    lower = text.lower()
    if "idpp barrier" in lower and "exceeds" in lower:
        return "IDPP barrier exceeds limit"
    if "aligned product energy drifted" in lower:
        return "aligned product energy drifted"
    if "aligned reactant energy drifted" in lower:
        return "aligned reactant energy drifted"
    if "highest-energy image is an endpoint" in lower:
        return "highest-energy image is an endpoint"
    if "neb barrier" in lower and "exceeds" in lower:
        return "NEB barrier too high"
    if "non-finite image energies" in lower:
        return "non-finite image energies"
    if "interior max prominence" in lower:
        return "interior max prominence too low"
    if "min mobile distance" in lower:
        return "clashing interior image"
    if "aligned endpoint mobile max displacement" in lower:
        return "endpoint displacement too large"

    text = text.split(" (", 1)[0].strip().rstrip(".")
    return text or "unknown"


def compact_ga_ineligible_reason(message: str) -> str:
    """Short label for grouping GA post-relax ineligibility reasons."""
    text = str(message).strip()
    lower = text.lower()
    if "not connected" in lower or "connected components" in lower:
        return "disconnected"
    if "clash" in lower or "too close" in lower:
        return "clash"
    return text.split(" (", 1)[0].strip().rstrip(".") or "unknown"


def log_neb_search_summaries(
    logger: Logger,
    ts_results: list[Mapping[str, Any]],
    *,
    verbosity: int,
    run_dir: str,
) -> None:
    """Emit v1 NEB outcome/artifact summaries; per-pair skip/fail detail at v2+."""
    if verbosity < 1:
        return

    n_total = len(ts_results)
    n_success = 0
    skipped_reasons: dict[str, int] = {}
    failed_reasons: dict[str, int] = {}
    n_ts = 0
    n_reactant = 0
    n_product = 0

    for result in ts_results:
        status = str(result.get("status") or "")
        if status == "success":
            n_success += 1
            if result.get("transition_state") is not None:
                n_ts += 1
        else:
            reason = compact_neb_pair_reason(str(result.get("error") or "unknown"))
            bucket = skipped_reasons if status == "skipped" else failed_reasons
            bucket[reason] = bucket.get(reason, 0) + 1

        if result.get("reactant_structure") is not None:
            n_reactant += 1
        if result.get("product_structure") is not None:
            n_product += 1

    n_skipped = sum(skipped_reasons.values())
    n_failed = sum(failed_reasons.values())
    parts = [f"{n_success}/{n_total} succeeded"]
    if n_skipped:
        detail = format_count_summary(skipped_reasons)
        parts.append(f"skippedx{n_skipped}" + (f" ({detail})" if detail else ""))
    if n_failed:
        detail = format_count_summary(failed_reasons)
        parts.append(f"failedx{n_failed}" + (f" ({detail})" if detail else ""))
    logger.info("NEB search: %s", "; ".join(parts))
    logger.info(
        "Saved NEB artifacts under %s: %s",
        run_dir,
        format_count_summary(
            {
                "TS": n_ts,
                "reactant": n_reactant,
                "product": n_product,
                "metadata": n_total,
            }
        )
        or "none",
    )

    if verbosity < 2:
        return

    for result in ts_results:
        if str(result.get("status") or "") == "success":
            continue
        logger.debug(
            "Skipping pair %s: %s",
            result.get("pair_id", "?"),
            result.get("error") or result.get("status"),
        )


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
        """Emit one INFO summary at v1+ and per-record DEBUG detail at v2+.

        Clears accumulated records after copying.
        """
        with cls._lock:
            fallbacks = list(cls._fallback_records)
            placement_failures = list(cls._placement_failures)
            cls._fallback_records.clear()
            cls._placement_failures.clear()

        if verbosity < 1:
            return

        template_to_random = sum(
            1
            for used, fb in fallbacks
            if used == "random_spherical" and fb == "template"
        )
        seed_to_random = sum(
            1
            for used, fb in fallbacks
            if used == "random_spherical" and fb == "seed+growth"
        )

        parts: list[str] = [f"built {n_structures}/{n_structures} candidates"]
        if template_to_random or seed_to_random:
            parts.append(
                f"fallbacks template→randomx{template_to_random}, "
                f"seed→randomx{seed_to_random}"
            )
        if placement_failures:
            parts.append(f"placement failuresx{len(placement_failures)}")
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
