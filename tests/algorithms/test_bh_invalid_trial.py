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
    # Call 1 validates the initial structure.
    # Each trial validates once before relax and once after; fail the first
    # trial's post-relax check (call 3 = initial + pre + post).
    calls = _patch_validation_failure(monkeypatch, fail_on_call=3)

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

    # Initial + 3 trials × (pre + post), with one post-relax rejection:
    # initial(1) + trial0 pre(2) post-fail(3) + trial1 pre(4) post(5)
    # + trial2 pre(6) post(7) = 7.
    assert calls["n"] == 7
    assert isinstance(minima, list)
    assert len(minima) >= 1
    messages = [record.getMessage() for record in caplog.records]
    assert any("rejecting invalid trial structure" in msg for msg in messages)


def test_bh_run_completes_when_last_trial_is_invalid(
    tmp_path, monkeypatch, rng
) -> None:
    """A rejected final trial still yields the minima collected so far."""
    # niter=2: initial(1), t0 pre(2) post(3), t1 pre(4) post-fail(5).
    calls = _patch_validation_failure(monkeypatch, fail_on_call=5)

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

    assert calls["n"] == 5
    assert isinstance(minima, list)
    assert len(minima) >= 1


def test_bh_invalid_initial_seed_does_not_poison_later_trials(
    tmp_path, monkeypatch, rng
) -> None:
    """An invalid initial seed stays as walk start but later trials remain returnable."""
    from scgo.metadata.atoms import get_tag

    calls = _patch_validation_failure(monkeypatch, fail_on_call=1)

    minima = bh_go(
        atoms=_pt3_with_calc()[0],
        output_dir=str(tmp_path / "bh_bad_seed"),
        niter=2,
        temperature=0.0,
        dr=0.3,
        niter_local_relaxation=3,
        verbosity=0,
        rng=rng,
        deduplicate=False,
    )

    assert calls["n"] >= 2
    assert len(minima) >= 1
    assert all(
        bool(get_tag(atoms, "ga_eligible", default=False)) for _, atoms in minima
    )
