"""Regression tests for SCGO database/file error handling in the GO runners.

``SCGODatabaseError`` and ``SCGOFileError`` subclass ``SCGOError`` (not
``sqlite3.Error``/``OSError``), so they used to escape the ``except`` tuples in
``scgo.minima_search.core.run_trials`` and
``scgo.runner_go._run_go_campaign_compositions``.
"""

from __future__ import annotations

import logging

import pytest

from scgo.exceptions import SCGODatabaseError, SCGOFileError
from scgo.runner_api import run_go, run_go_campaign

CORE_LOGGER_NAME = "scgo.minima_search.core"
DB_TAG_TARGET = "scgo.minima_search.core.mark_final_minima_in_db"


class _RecordingHandler(logging.Handler):
    """Collect every record that reaches it, regardless of level."""

    def __init__(self) -> None:
        super().__init__(level=logging.NOTSET)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _raise_factory(exc_type):
    def _raise(*args, **kwargs):
        raise exc_type("simulated database failure")

    return _raise


@pytest.mark.parametrize("exc_type", [SCGODatabaseError, SCGOFileError])
def test_run_trials_logs_scgo_db_errors_from_final_tagging(
    monkeypatch, tmp_path, exc_type
):
    """``run_trials`` must catch and log SCGO DB/file errors before re-raising."""
    monkeypatch.setattr(DB_TAG_TARGET, _raise_factory(exc_type))

    handler = _RecordingHandler()
    core_logger = logging.getLogger(CORE_LOGGER_NAME)
    core_logger.addHandler(handler)
    try:
        with pytest.raises(exc_type):
            run_go(
                ["Pt", "Pt"],
                params={"calculator": "EMT"},
                system_type="gas_cluster",
                verbosity=0,
                output_dir=tmp_path,
                clean=True,
            )
    finally:
        core_logger.removeHandler(handler)

    assert any(
        "Failed to tag final minima in DB" in record.getMessage()
        for record in handler.records
    ), "SCGO DB errors escaped the run_trials handler without being logged"


@pytest.mark.parametrize("exc_type", [SCGODatabaseError, SCGOFileError])
def test_run_go_campaign_survives_scgo_db_errors(monkeypatch, tmp_path, exc_type):
    """A DB tagging failure must degrade to an empty result, not abort the campaign."""
    monkeypatch.setattr(DB_TAG_TARGET, _raise_factory(exc_type))

    results = run_go_campaign(
        [["Pt", "Pt"]],
        params={"calculator": "EMT"},
        system_type="gas_cluster",
        verbosity=0,
        output_dir=tmp_path,
        clean=True,
    )

    assert results == {"Pt2": []}


@pytest.mark.parametrize("exc_type", [SCGODatabaseError, SCGOFileError])
def test_run_go_campaign_continues_after_failed_composition(
    monkeypatch, tmp_path, exc_type
):
    """SCGO DB/file errors from one composition must not stop the campaign."""

    def _fake_trials(composition, system_type, params, **kwargs):
        if list(composition) == ["Pt", "Pt"]:
            raise exc_type("simulated database failure")
        return []

    monkeypatch.setattr("scgo.runner_go._run_go_trials", _fake_trials)

    results = run_go_campaign(
        [["Pt", "Pt"], ["Au", "Au"]],
        params={"calculator": "EMT"},
        system_type="gas_cluster",
        verbosity=0,
        output_dir=tmp_path,
        clean=True,
    )

    assert results == {"Pt2": [], "Au2": []}
