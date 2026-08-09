"""Regression tests: GO runners must forward ``verbosity`` to ``run_trials``.

The single-run path used to drop ``verbosity``, so ``run_trials`` (and every
algorithm below it) fell back to the default ``verbosity=1`` even when the
caller asked for a quiet run.
"""

from __future__ import annotations

import logging

import pytest
from ase.calculators.emt import EMT

from scgo.runner_api import run_go, run_go_campaign
from scgo.utils.logging import log_info_v

CORE_LOGGER_NAME = "scgo.minima_search.core"


class _RecordingHandler(logging.Handler):
    """Collect every record that reaches it, regardless of level."""

    def __init__(self) -> None:
        super().__init__(level=logging.NOTSET)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def captured_run_trials(monkeypatch):
    """Replace ``run_trials`` with a stub that records its keyword arguments."""
    captured: dict[str, object] = {}

    def _fake_run_trials(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr("scgo.runner_go.run_trials", _fake_run_trials)
    monkeypatch.setattr("scgo.runner_go.get_calculator_class", lambda name: EMT)
    return captured


@pytest.mark.parametrize("verbosity", [0, 1, 2])
def test_run_go_forwards_verbosity_to_run_trials(
    captured_run_trials, tmp_path, verbosity
):
    run_go(
        ["Pt", "Pt"],
        params={"calculator": "EMT"},
        system_type="gas_cluster",
        verbosity=verbosity,
        output_dir=tmp_path,
    )

    assert captured_run_trials["verbosity"] == verbosity


def test_run_go_campaign_forwards_verbosity_to_run_trials(
    captured_run_trials, tmp_path
):
    run_go_campaign(
        [["Pt", "Pt"]],
        params={"calculator": "EMT"},
        system_type="gas_cluster",
        verbosity=0,
        output_dir=tmp_path,
    )

    assert captured_run_trials["verbosity"] == 0


@pytest.mark.parametrize(
    "verbosity,expect_info",
    [(0, False), (1, True)],
)
def test_run_go_verbosity_gates_run_trials_info_logs(
    monkeypatch, tmp_path, verbosity, expect_info
):
    """``verbosity=0`` must silence INFO logging emitted from ``run_trials``."""
    handler = _RecordingHandler()
    core_logger = logging.getLogger(CORE_LOGGER_NAME)

    def _fake_run_trials(**kwargs):
        # ``configure_logging`` already ran inside the runner; re-enable INFO on
        # the root logger so this assertion isolates the verbosity gate itself
        # instead of the global log level.
        root = logging.getLogger()
        previous_level = root.level
        root.setLevel(logging.INFO)
        core_logger.addHandler(handler)
        try:
            log_info_v(
                core_logger,
                "run_trials info message",
                verbosity=kwargs.get("verbosity", 1),
            )
        finally:
            core_logger.removeHandler(handler)
            root.setLevel(previous_level)
        return []

    monkeypatch.setattr("scgo.runner_go.run_trials", _fake_run_trials)
    monkeypatch.setattr("scgo.runner_go.get_calculator_class", lambda name: EMT)

    run_go(
        ["Pt", "Pt"],
        params={"calculator": "EMT"},
        system_type="gas_cluster",
        verbosity=verbosity,
        output_dir=tmp_path,
    )

    info_messages = [
        record.getMessage()
        for record in handler.records
        if record.levelno >= logging.INFO
    ]
    assert bool(info_messages) is expect_info
