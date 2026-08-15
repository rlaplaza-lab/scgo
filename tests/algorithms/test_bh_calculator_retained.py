"""Accepted BH structures keep their calculator after the copy (regression).

``ase.Atoms.copy()`` does not carry ``calc`` over, so the accepted structure of
the previous iteration used to enter the next move without a calculator. Any
force-based move (or MLIP relaxation) then failed with
``RuntimeError: Atoms object has no calculator``.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

from scgo.algorithms import basinhopping_go
from scgo.algorithms.basinhopping_go import bh_go
from tests.helpers import _pt3_with_calc

# Effectively always accept the trial in the Metropolis step.
_ALWAYS_ACCEPT_TEMPERATURE = 1e6


def test_accepted_structure_keeps_calculator_for_next_move(
    tmp_path, monkeypatch
) -> None:
    atoms, calc = _pt3_with_calc()
    seen_calcs: list[object] = []
    real_move = basinhopping_go._move_atoms

    def _spy_move(atoms_in: Atoms, *args: object, **kwargs: object):
        seen_calcs.append(atoms_in.calc)
        return real_move(atoms_in, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(basinhopping_go, "_move_atoms", _spy_move)

    minima = bh_go(
        atoms=atoms,
        output_dir=str(tmp_path / "bh_calc"),
        niter=3,
        temperature=_ALWAYS_ACCEPT_TEMPERATURE,
        dr=0.3,
        niter_local_relaxation=3,
        verbosity=0,
        rng=np.random.default_rng(0),
    )

    assert isinstance(minima, list)
    assert len(seen_calcs) == 3
    assert all(c is not None for c in seen_calcs)
    # Every iteration after the first starts from an accepted copy, which must
    # carry the run calculator (not None, not a stale SinglePointCalculator).
    assert all(c is calc for c in seen_calcs[1:])


def test_force_based_move_survives_accepted_copy(tmp_path) -> None:
    """``highest_force`` needs a live calculator on the accepted structure."""
    atoms, _calc = _pt3_with_calc()

    minima = bh_go(
        atoms=atoms,
        output_dir=str(tmp_path / "bh_forces"),
        niter=3,
        temperature=_ALWAYS_ACCEPT_TEMPERATURE,
        dr=0.3,
        move_strategy="highest_force",
        niter_local_relaxation=3,
        verbosity=0,
        rng=np.random.default_rng(1),
    )

    assert isinstance(minima, list)
    assert len(minima) >= 1
