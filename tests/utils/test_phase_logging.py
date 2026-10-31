"""Tests for phase-oriented logging helpers."""

import logging

from scgo.utils.logging import TRACE, get_logger
from scgo.utils.phase_logging import (
    InitDiagnosticsCollector,
    compact_ga_ineligible_reason,
    compact_neb_pair_reason,
    format_count_of,
    format_count_summary,
    format_counted,
    format_offspring_outcome_line,
    log_mutation_structure_yield,
    log_neb_search_summaries,
    log_phase_header,
    with_detail,
)


def test_format_count_summary():
    assert format_count_summary({}) == ""
    assert format_count_summary({"b": 2, "a": 3}) == "3 a, 2 b"
    assert format_count_summary({"pairing_failed": 2}) == "2 pairing failed"


def test_count_phrase_helpers():
    assert format_count_of(8, 10, before="succeeded") == "8 succeeded of 10"
    assert format_count_of(0, 12, after="pairs") == "0 of 12 pairs"
    assert format_count_of(6, 10, before="applied", after="offspring") == (
        "6 applied of 10 offspring"
    )
    assert format_counted(1, "database") == "1 database"
    assert format_counted(2, "placement failure") == "2 placement failures"
    assert with_detail("8 succeeded of 10", "") == "8 succeeded of 10"
    assert with_detail("8 succeeded of 10", "2 pairing failed") == (
        "8 succeeded of 10 (2 pairing failed)"
    )


def test_compact_neb_pair_reason():
    assert (
        compact_neb_pair_reason(
            "Initial NEB path rejected (energy profile): "
            "IDPP barrier 47.438 eV exceeds "
            "neb_max_spurious_barrier=8.000 eV (likely discontinuous)"
        )
        == "IDPP barrier exceeds limit"
    )
    assert (
        compact_neb_pair_reason(
            "Initial NEB path rejected (energy profile): "
            "aligned product energy drifted by 1.553 eV (limit 0.500 eV)"
        )
        == "aligned product energy drifted"
    )
    assert (
        compact_neb_pair_reason(
            "Highest-energy image is an endpoint (image 6); no interior saddle found"
        )
        == "highest-energy image is an endpoint"
    )
    assert (
        compact_neb_pair_reason(
            "NEB barrier 12.000 eV exceeds "
            "neb_max_spurious_barrier=8.000 eV (likely discontinuous path)"
        )
        == "NEB barrier too high"
    )


def test_compact_ga_ineligible_reason():
    assert (
        compact_ga_ineligible_reason(
            "Structure is not connected (found 2 connected components)"
        )
        == "disconnected"
    )
    assert compact_ga_ineligible_reason("Atoms too close (clash)") == "clash"


def test_log_neb_search_summaries_verbosity(caplog, tmp_path):
    logger = get_logger("test.phase_logging.neb")
    results = [
        {
            "pair_id": "1_2",
            "status": "success",
            "error": None,
            "transition_state": object(),
            "reactant_structure": object(),
            "product_structure": object(),
        },
        {
            "pair_id": "3_4",
            "status": "skipped",
            "error": (
                "Initial NEB path rejected (energy profile): "
                "IDPP barrier 47.438 eV exceeds "
                "neb_max_spurious_barrier=8.000 eV (likely discontinuous)"
            ),
        },
        {
            "pair_id": "5_6",
            "status": "skipped",
            "error": (
                "Initial NEB path rejected (energy profile): "
                "aligned product energy drifted by 1.553 eV (limit 0.500 eV)"
            ),
        },
        {
            "pair_id": "7_8",
            "status": "failed",
            "error": (
                "Highest-energy image is an endpoint (image 6); "
                "no interior saddle found"
            ),
            "reactant_structure": object(),
            "product_structure": object(),
        },
    ]

    with caplog.at_level(logging.INFO):
        log_neb_search_summaries(logger, results, verbosity=1, run_dir=str(tmp_path))
    assert "NEB search: 1 succeeded of 4" in caplog.text
    assert "2 skipped" in caplog.text
    assert "1 IDPP barrier exceeds limit" in caplog.text
    assert "1 aligned product energy drifted" in caplog.text
    assert "1 failed" in caplog.text
    assert "1 highest-energy image is an endpoint" in caplog.text
    assert f"Saved NEB artifacts under {tmp_path}" in caplog.text
    assert "1 TS" in caplog.text
    assert "2 reactant" in caplog.text
    assert "2 product" in caplog.text
    assert "4 metadata" in caplog.text
    assert "Skipping pair" not in caplog.text

    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        log_neb_search_summaries(logger, results, verbosity=2, run_dir=str(tmp_path))
    assert "Skipping pair 3_4:" in caplog.text
    assert "Skipping pair 5_6:" in caplog.text
    assert "Skipping pair 7_8:" in caplog.text
    assert "Skipping pair 1_2:" not in caplog.text


def test_init_diagnostics_collector_emit_summary(caplog):
    caplog.set_level(logging.DEBUG)
    logger = get_logger("test.phase_logging")
    InitDiagnosticsCollector.reset()
    InitDiagnosticsCollector.record_fallback("random_spherical", "template")
    InitDiagnosticsCollector.record_placement_failure(
        "Could not place atom Pt (3/4 placed)",
        "Could not place atom Pt (3/4 placed)\n  parameters: ...",
    )
    InitDiagnosticsCollector.emit_summary(
        logger, verbosity=2, n_structures=5, prefix="Test init"
    )
    assert caplog.text.count("Test init:") == 1
    assert "1 placement failure" in caplog.text
    assert "Init fallback: template→random_spherical" in caplog.text
    assert "Placement failure: Could not place atom Pt" in caplog.text

    # emit_summary clears; a later window must not inherit prior records.
    caplog.clear()
    InitDiagnosticsCollector.record_fallback("random_spherical", "seed+growth")
    InitDiagnosticsCollector.emit_summary(
        logger, verbosity=1, n_structures=2, prefix="Window2"
    )
    assert "1 seed fell back to random" in caplog.text
    assert "placement failure" not in caplog.text


def test_log_phase_header_respects_verbosity(caplog):
    caplog.set_level(logging.INFO)
    logger = get_logger("test.phase_logging.header")
    log_phase_header(logger, "Population initialization", verbosity=0)
    assert caplog.text == ""
    log_phase_header(logger, "Population initialization", verbosity=1)
    assert "Population initialization" in caplog.text


def test_log_mutation_structure_yield_scales_with_verbosity(caplog):
    logger = get_logger("test.phase_logging.mutation_yield")
    stats = {
        "breathing": {"unique": 0, "trials": 2},
        "idle": {"unique": 0, "trials": 0},
        "rattle": {"unique": 0, "trials": 11},
        "rotational": {"unique": 8, "trials": 10},
        "matched": {"unique": 11, "trials": 21},
    }
    summary = (
        "New structures by mutation: breathing 0 new of 2 relaxed, "
        "matched 11 new of 21 relaxed, rattle 0 new of 11 relaxed, "
        "rotational 8 new of 10 relaxed"
    )

    with caplog.at_level(logging.INFO):
        log_mutation_structure_yield(logger, {}, verbosity=1)
        log_mutation_structure_yield(logger, stats, verbosity=0)
    assert caplog.text == ""

    caplog.clear()
    with caplog.at_level(logging.INFO):
        log_mutation_structure_yield(logger, stats, verbosity=1)
    assert summary in caplog.text
    assert "idle" not in caplog.text
    assert "% new" not in caplog.text
    assert "picked " not in caplog.text

    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        log_mutation_structure_yield(logger, stats, verbosity=2)
    assert summary in caplog.text
    assert "rattle: 0 new of 11 relaxed (0% new)" in caplog.text
    assert "rotational: 8 new of 10 relaxed (80% new)" in caplog.text
    assert "picked " not in caplog.text

    caplog.clear()
    with caplog.at_level(TRACE):
        log_mutation_structure_yield(logger, stats, verbosity=3)
    assert summary in caplog.text
    assert (
        "breathing: 0 new of 2 relaxed (0% new); "
        "too few relaxations to change how often it is picked"
    ) in caplog.text
    assert "rattle: 0 new of 11 relaxed (0% new); picked less often (weight x" in (
        caplog.text
    )
    assert (
        "rotational: 8 new of 10 relaxed (80% new); picked more often (weight x"
    ) in caplog.text
    assert (
        "matched: 11 new of 21 relaxed (52% new); picked about as often (weight x1.00)"
    ) in caplog.text


def test_format_offspring_outcome_line():
    line = format_offspring_outcome_line(
        3,
        failure_reason="validation_failed",
        desc="pairing: 1 2",
        mutation_applied=True,
        validation_error="Cluster is not connected",
    )
    assert "Offspring 3: validation failed" in line
    assert "not connected" in line

    ok = format_offspring_outcome_line(
        1,
        failure_reason=None,
        desc="mutation: rattle",
        mutation_applied=True,
        validation_error=None,
    )
    assert "crossover succeeded" in ok
    assert "mutation rattle" in ok
