"""Shared user-facing log / error format strings.

Use this module for wording reused across modules (logger templates and
tiny format helpers). Keep one-off and domain-local messages at the call site.
Parameterized validation rules belong in ``scgo.utils.validation``.
"""

from __future__ import annotations


def validation_context_prefix(context: str | None) -> str:
    """Return ``"{context}: "`` when context is set, else ``""``."""
    return f"{context}: " if context else ""


NO_OUTPUT_DIR_CAMPAIGN_ROOT = "No output_dir provided; using default campaign root %s"
SKIPPING_DATABASE_NO_RUN_ID = (
    "Skipping database %s: could not resolve run_id from path layout"
)
COULD_NOT_RESOLVE_RUN_ID_UNSET = (
    "Could not resolve run_id from path %s; leaving run_id unset"
)
COULD_NOT_RESOLVE_RUN_ID_FOR_DATABASE = (
    "Could not resolve run_id for database %s under %s"
)
TS_NEED_AT_LEAST_TWO_MINIMA_TO_PAIR = "Only %d minima, need at least 2 to pair"
TS_NEED_AT_LEAST_TWO_MINIMA_TO_FIND_TS = (
    "Only %d minima found, need at least 2 to find TS"
)
