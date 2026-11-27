"""Tests for shared user-facing message templates."""

from __future__ import annotations

from scgo.utils import messages


def test_validation_context_prefix() -> None:
    assert messages.validation_context_prefix(None) == ""
    assert messages.validation_context_prefix("") == ""
    assert messages.validation_context_prefix("run_go") == "run_go: "


def test_message_constants() -> None:
    assert (
        messages.NO_OUTPUT_DIR_CAMPAIGN_ROOT
        == "No output_dir provided; using default campaign root %s"
    )
    assert (
        messages.SKIPPING_DATABASE_NO_RUN_ID
        == "Skipping database %s: could not resolve run_id from path layout"
    )
    assert (
        messages.COULD_NOT_RESOLVE_RUN_ID_UNSET
        == "Could not resolve run_id from path %s; leaving run_id unset"
    )
    assert (
        messages.COULD_NOT_RESOLVE_RUN_ID_FOR_DATABASE
        == "Could not resolve run_id for database %s under %s"
    )
    assert (
        messages.TS_NEED_AT_LEAST_TWO_MINIMA_TO_PAIR
        == "Only %d minima, need at least 2 to pair"
    )
    assert (
        messages.TS_NEED_AT_LEAST_TWO_MINIMA_TO_FIND_TS
        == "Only %d minima found, need at least 2 to find TS"
    )
