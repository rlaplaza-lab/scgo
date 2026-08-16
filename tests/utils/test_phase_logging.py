"""Tests for phase-oriented logging helpers."""

import logging

from scgo.utils.logging import get_logger
from scgo.utils.phase_logging import (
    InitDiagnosticsCollector,
    compact_neb_pair_reason,
    format_count_summary,
    format_offspring_outcome_line,
    log_neb_search_summaries,
    log_phase_header,
)


def test_format_count_summary():
    assert format_count_summary({}) == ""
    assert format_count_summary({"b": 2, "a": 3}) == "ax3, bx2"


def test_compact_neb_pair_reason():
    assert (
        compact_neb_pair_reason(
            "Initial NEB path rejected (energy profile): "
            "IDPP barrier 47.438 eV exceeds 8.000 eV (likely discontinuous)"
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
            "NEB barrier 12.000 eV exceeds 8.000 eV (likely discontinuous path)"
        )
        == "NEB barrier too high"
    )


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
                "IDPP barrier 47.438 eV exceeds 8.000 eV (likely discontinuous)"
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
    assert "NEB search: 1/4 succeeded" in caplog.text
    assert "skippedx2" in caplog.text
    assert "IDPP barrier exceeds limitx1" in caplog.text
    assert "aligned product energy driftedx1" in caplog.text
    assert "failedx1" in caplog.text
    assert "highest-energy image is an endpointx1" in caplog.text
    assert f"Saved NEB artifacts under {tmp_path}" in caplog.text
    assert "TSx1" in caplog.text
    assert "reactantx2" in caplog.text
    assert "productx2" in caplog.text
    assert "metadatax4" in caplog.text
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
    assert "placement failuresx1" in caplog.text
    assert "Init fallback: template→random_spherical" in caplog.text
    assert "Placement failure: Could not place atom Pt" in caplog.text

    # emit_summary clears; a later window must not inherit prior records.
    caplog.clear()
    InitDiagnosticsCollector.record_fallback("random_spherical", "seed+growth")
    InitDiagnosticsCollector.emit_summary(
        logger, verbosity=1, n_structures=2, prefix="Window2"
    )
    assert "seed→randomx1" in caplog.text
    assert "placement failures" not in caplog.text


def test_log_phase_header_respects_verbosity(caplog):
    caplog.set_level(logging.INFO)
    logger = get_logger("test.phase_logging.header")
    log_phase_header(logger, "Population initialization", verbosity=0)
    assert caplog.text == ""
    log_phase_header(logger, "Population initialization", verbosity=1)
    assert "Population initialization" in caplog.text


def test_format_offspring_outcome_line():
    line = format_offspring_outcome_line(
        3,
        failure_reason="validation_failed",
        desc="pairing: 1 2",
        mutation_applied=True,
        validation_error="Cluster is not connected",
    )
    assert "Offspring 3: validation_failed" in line
    assert "not connected" in line

    ok = format_offspring_outcome_line(
        1,
        failure_reason=None,
        desc="mutation: rattle",
        mutation_applied=True,
        validation_error=None,
    )
    assert "mutation=rattle" in ok
