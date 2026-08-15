"""A single invalid BH trial must not abort the whole run (regression)."""

from __future__ import annotations

import logging

import pytest
from ase import Atoms

from scgo.algorithms import basinhopping_go
from scgo.algorithms.basinhopping_go import bh_go
from scgo.exceptions import SCGOValidationError
from tests.helpers import _pt3_with_calc


def _patch_validation_failure(
    monkeypatch: pytest.MonkeyPatch,
    fail_on_call: int,
) -> dict[str, int]:
    """Make ``validate_minimum_structure`` raise once, on ``fail_on_call``."""
    calls = {"n": 0}
    real = basinhopping_go.validate_minimum_structure

    def _flaky(atoms: Atoms, **kwargs: object) -> None:
        calls["n"] += 1
        if calls["n"] == fail_on_call:
            raise SCGOValidationError("injected invalid trial structure")
        real(atoms, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(basinhopping_go, "validate_minimum_structure", _flaky)
    return calls


def test_bh_run_completes_when_one_trial_is_invalid(
    tmp_path, monkeypatch, caplog, rng
) -> None:
    """The invalid trial is rejected and the remaining iterations still run."""
    # Call 1 validates the initial structure; call 2 is the first trial.
    calls = _patch_validation_failure(monkeypatch, fail_on_call=2)

    with caplog.at_level(logging.WARNING, logger="scgo.algorithms.basinhopping_go"):
        minima = bh_go(
            atoms=_pt3_with_calc()[0],
            output_dir=str(tmp_path / "bh_invalid"),
            niter=3,
            temperature=0.0,
            dr=0.3,
            niter_local_relaxation=3,
            verbosity=0,
            rng=rng,
        )

    # Initial structure + all three trials were validated: no early abort.
    assert calls["n"] == 4
    assert isinstance(minima, list)
    assert len(minima) >= 1
    messages = [record.getMessage() for record in caplog.records]
    assert any("rejecting invalid trial structure" in msg for msg in messages)


def test_bh_run_completes_when_last_trial_is_invalid(
    tmp_path, monkeypatch, rng
) -> None:
    """A rejected final trial still yields the minima collected so far."""
    calls = _patch_validation_failure(monkeypatch, fail_on_call=3)

    minima = bh_go(
        atoms=_pt3_with_calc()[0],
        output_dir=str(tmp_path / "bh_invalid_last"),
        niter=2,
        temperature=0.0,
        dr=0.3,
        niter_local_relaxation=3,
        verbosity=0,
        rng=rng,
    )

    assert calls["n"] == 3
    assert isinstance(minima, list)
    assert len(minima) >= 1
