"""``simple_go`` honours ``verbosity``/``rng`` and no longer drops ``**kwargs``."""

from __future__ import annotations

import logging
import os

import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT

from scgo.algorithms.simple_go import simple_go

_SIMPLE_GO_LOGGER = "scgo.algorithms.simple_go"


def _pt2_with_calc() -> Atoms:
    atoms = Atoms(
        "Pt2",
        positions=[[0.0, 0.0, 0.0], [2.6, 0.0, 0.0]],
        cell=[20.0, 20.0, 20.0],
        pbc=False,
    )
    atoms.calc = EMT()
    return atoms


def _records(caplog, level: int) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.name == _SIMPLE_GO_LOGGER and record.levelno == level
    ]


def test_simple_go_verbosity_zero_emits_no_info_logs(tmp_path, caplog) -> None:
    with caplog.at_level(logging.DEBUG, logger=_SIMPLE_GO_LOGGER):
        minima = simple_go(
            _pt2_with_calc(),
            str(tmp_path / "quiet"),
            rng=np.random.default_rng(0),
            niter_local_relaxation=5,
            verbosity=0,
        )

    assert isinstance(minima, list)
    assert not _records(caplog, logging.INFO)
    assert not _records(caplog, logging.DEBUG)


def test_simple_go_verbosity_one_emits_info_logs(tmp_path, caplog) -> None:
    with caplog.at_level(logging.DEBUG, logger=_SIMPLE_GO_LOGGER):
        simple_go(
            _pt2_with_calc(),
            str(tmp_path / "normal"),
            rng=np.random.default_rng(0),
            niter_local_relaxation=5,
            verbosity=1,
        )

    messages = [record.getMessage() for record in _records(caplog, logging.INFO)]
    assert any("simple optimization" in msg for msg in messages)


def test_simple_go_forwards_known_kwargs_and_reports_unknown(tmp_path, caplog) -> None:
    logfile = tmp_path / "relax.log"
    with caplog.at_level(logging.DEBUG, logger=_SIMPLE_GO_LOGGER):
        simple_go(
            _pt2_with_calc(),
            str(tmp_path / "kwargs"),
            rng=np.random.default_rng(0),
            niter_local_relaxation=5,
            verbosity=2,
            logfile=str(logfile),
            system_type="gas_cluster",
        )

    # ``logfile`` is forwarded to the local relaxation instead of being dropped.
    assert os.path.exists(logfile)
    debug_messages = [record.getMessage() for record in _records(caplog, logging.DEBUG)]
    assert any("system_type" in msg for msg in debug_messages)


def test_simple_go_uses_rng_to_break_coincident_atoms(tmp_path) -> None:
    """Degenerate input is separated reproducibly using the supplied rng."""
    results = []
    for run in range(2):
        atoms = Atoms(
            "Pt2",
            positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            cell=[20.0, 20.0, 20.0],
            pbc=False,
        )
        atoms.calc = EMT()
        minima = simple_go(
            atoms,
            str(tmp_path / f"degenerate_{run}"),
            rng=np.random.default_rng(7),
            niter_local_relaxation=5,
            verbosity=0,
        )
        assert minima
        results.append(minima[0][1].get_positions())

    distance = float(np.linalg.norm(results[0][1] - results[0][0]))
    assert distance > 1e-3
    np.testing.assert_allclose(results[0], results[1])
