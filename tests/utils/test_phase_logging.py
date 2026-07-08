"""Tests for phase-oriented logging helpers."""

import logging

from scgo.utils.logging import get_logger
from scgo.utils.phase_logging import (
    InitDiagnosticsCollector,
    format_count_summary,
    format_offspring_outcome_line,
    log_phase_header,
)


def test_format_count_summary():
    assert format_count_summary({}) == ""
    assert format_count_summary({"b": 2, "a": 3}) == "a×3, b×2"


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
    assert "placement failures×1" in caplog.text
    assert "Init fallback: template→random_spherical" in caplog.text
    assert "Placement failure: Could not place atom Pt" in caplog.text


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
