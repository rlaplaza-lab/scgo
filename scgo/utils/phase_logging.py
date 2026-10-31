"""Phase-oriented logging helpers for SCGO runs (headers, summaries, collectors)."""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from scgo.utils.logging import TRACE
from scgo.utils.mutation_weights import operator_credit_weight_factor

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
    """Format outcome counts as ``N label, ...``.

    Highest counts come first. Underscores in a label are written as spaces.
    """
    parts = [
        f"{count} {str(label).replace('_', ' ')}"
        for label, count in sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        if count > 0
    ]
    return ", ".join(parts) if parts else ""


def format_count_of(
    count: int,
    total: int,
    *,
    before: str = "",
    after: str = "",
) -> str:
    """Return ``8 succeeded of 10`` or ``0 of 12 pairs``."""
    left = f"{count} {before}".rstrip()
    right = f"{total} {after}".rstrip()
    return f"{left} of {right}"


def format_counted(count: int, singular: str, plural: str | None = None) -> str:
    """Return ``1 seed`` or ``2 seeds``."""
    word = (
        singular if count == 1 else (plural if plural is not None else f"{singular}s")
    )
    return f"{count} {word}"


def with_detail(text: str, detail: str) -> str:
    """Append `` (detail)`` when ``detail`` is non-empty."""
    if not detail:
        return text
    return f"{text} ({detail})"


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
    parts = [format_count_of(n_success, n_total, before="succeeded")]
    if n_skipped:
        parts.append(
            with_detail(f"{n_skipped} skipped", format_count_summary(skipped_reasons))
        )
    if n_failed:
        parts.append(
            with_detail(f"{n_failed} failed", format_count_summary(failed_reasons))
        )
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

        extras: list[str] = []
        if template_to_random:
            extras.append(
                f"{format_counted(template_to_random, 'template')} fell back to random"
            )
        if seed_to_random:
            extras.append(
                f"{format_counted(seed_to_random, 'seed')} fell back to random"
            )
        if placement_failures:
            extras.append(format_counted(len(placement_failures), "placement failure"))
        if extra:
            extras.append(extra)

        summary = with_detail(
            f"built {format_count_of(n_structures, n_structures, after='candidates')}",
            ", ".join(extras),
        )
        logger.info("%s: %s", prefix, summary)

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
            logger.info(
                "Crossover: %s",
                with_detail(
                    format_count_of(succeeded, total_jobs, before="succeeded"),
                    format_count_summary(failures),
                ),
            )
            logger.info(
                "Mutation: %s",
                format_count_of(
                    mutation_applied,
                    total_jobs,
                    before="applied",
                    after="offspring",
                ),
            )
        logger.info(
            "Offspring: %s (%d attempts)",
            format_count_of(created, n_offspring, before="created"),
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


def log_initial_population_summary(
    logger: Logger,
    *,
    verbosity: int,
    population_size: int,
    eligible: int,
    discarded: int,
    ineligible: int,
    reasons: Mapping[str, int],
) -> None:
    """Log the relaxed initial population as one info line."""
    if verbosity < 1:
        return
    relaxed = eligible + ineligible
    text = (
        f"Initial population: {format_counted(population_size, 'structure')}, "
        f"{format_count_of(eligible, relaxed, before='eligible', after='relaxed')}, "
        f"{discarded} discarded before relaxation, "
        f"{ineligible} ineligible after relaxation"
    )
    logger.info("%s", with_detail(text, format_count_summary(reasons)))


def log_relaxation_summary(
    logger: Logger,
    *,
    verbosity: int,
    eligible: int,
    ineligible: int,
    reasons: Mapping[str, int],
) -> None:
    """Log how many relaxed offspring stayed eligible for the GA."""
    if verbosity < 1 or eligible + ineligible <= 0:
        return
    total = eligible + ineligible
    text = (
        f"{format_count_of(eligible, total, before='eligible')}, "
        f"{ineligible} ineligible"
    )
    logger.info(
        "Relaxation: %s",
        with_detail(text, format_count_summary(reasons)),
    )


def _mutation_yield_detail(
    name: str,
    unique: int,
    trials: int,
    *,
    include_selection: bool,
) -> str:
    """One operator line: counts, percent new, and optional selection effect."""
    percent = round(100.0 * unique / trials)
    line = (
        f"{name}: {format_count_of(unique, trials, before='new', after='relaxed')} "
        f"({percent}% new)"
    )
    if not include_selection:
        return line
    factor = operator_credit_weight_factor(unique, trials)
    if factor is None:
        note = "too few relaxations to change how often it is picked"
    elif factor < 1.0:
        note = f"picked less often (weight x{factor:.2f})"
    elif factor > 1.0:
        note = f"picked more often (weight x{factor:.2f})"
    else:
        note = f"picked about as often (weight x{factor:.2f})"
    return f"{line}; {note}"


def log_mutation_structure_yield(
    logger: Logger,
    stats: Mapping[str, Mapping[str, int]],
    *,
    verbosity: int,
) -> None:
    """Log how many relaxed mutants of each operator were new structures.

    Verbosity 1 prints one summary. Verbosity 2 adds each operator's share
    of new structures. Verbosity 3 also reports whether that share changes
    how often the operator is picked.
    """
    if verbosity < 1 or not stats:
        return
    active = [
        (name, int(record.get("unique", 0)), int(record.get("trials", 0)))
        for name, record in sorted(stats.items())
        if int(record.get("trials", 0))
    ]
    if not active:
        return
    summary = "New structures by mutation: " + ", ".join(
        f"{name} {format_count_of(unique, trials, before='new', after='relaxed')}"
        for name, unique, trials in active
    )
    logger.info("%s", summary)
    if verbosity < 2:
        return
    include_selection = verbosity >= 3
    level = TRACE if include_selection else logging.DEBUG
    for name, unique, trials in active:
        logger.log(
            level,
            "%s",
            _mutation_yield_detail(
                name,
                unique,
                trials,
                include_selection=include_selection,
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
        return f"Offspring {index}: rejected (atoms too close before relaxation)"
    if failure_reason == "validation_failed":
        err = (validation_error or "validation failed").splitlines()[0]
        return f"Offspring {index}: validation failed ({err})"
    if desc and "mutation:" in desc:
        op = desc.split("mutation:", 1)[1].strip().split()[0]
        mutation = f"mutation {op}"
    else:
        mutation = "mutation applied" if mutation_applied else "no mutation"
    return f"Offspring {index}: crossover succeeded, {mutation}, eligible"
