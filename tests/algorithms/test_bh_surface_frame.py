"""BH surface move/relax frame coherence."""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.build import fcc111
from ase.calculators.emt import EMT

from scgo.algorithms.basinhopping_go import _move_atoms
from scgo.utils.helpers import perform_local_relaxation


def test_move_atoms_skips_com_recenter_on_surface() -> None:
    rng = np.random.default_rng(0)
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    slab.pbc = [True, True, False]
    z0 = float(slab.get_positions()[:, 2].max() + 1.5)
    atoms = slab.copy() + Atoms("Pt2", positions=[[0.5, 0.5, z0], [1.5, 0.6, z0]])
    movable = [len(slab), len(slab) + 1]
    slab_pos_before = atoms.get_positions()[: len(slab)].copy()

    moved, _desc = _move_atoms(
        atoms,
        dr=0.8,
        move_fraction=1.0,
        rng=rng,
        movable_indices=movable,
        recenter_com=False,
    )
    # Mobile atoms moved; slab stays put (no full-structure COM recenter).
    assert (
        np.linalg.norm(moved.get_positions()[movable] - atoms.get_positions()[movable])
        > 1e-6
    )
    np.testing.assert_allclose(moved.get_positions()[: len(slab)], slab_pos_before)


def test_move_atoms_recenters_com_for_gas() -> None:
    rng = np.random.default_rng(1)
    atoms = Atoms(
        "Pt3",
        positions=[[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [1.2, 2.1, 0.0]],
        cell=[20, 20, 20],
        pbc=False,
    )
    com_before = atoms.get_center_of_mass().copy()
    moved, _desc = _move_atoms(
        atoms,
        dr=0.5,
        move_fraction=1.0,
        rng=rng,
        recenter_com=True,
    )
    np.testing.assert_allclose(moved.get_center_of_mass(), com_before, atol=1e-8)


def test_bh_surface_relax_keeps_slab_frame() -> None:
    """BH-style surface relax (center_after_relax=False, surface_mode=True)."""

    class _NoOpOpt:
        def __init__(self, atoms, **kwargs):
            self.atoms = atoms

        def run(self, fmax=0.05, steps=1):
            return True

    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    slab.pbc = [True, True, False]
    n_slab = len(slab)
    z0 = float(slab.get_positions()[:, 2].max() + 1.5)
    atoms = slab.copy() + Atoms("Pt", positions=[[1.0, 1.0, z0]])
    slab_before = atoms.get_positions()[:n_slab].copy()
    perform_local_relaxation(
        atoms,
        EMT(),
        _NoOpOpt,
        fmax=1.0,
        steps=1,
        center_after_relax=False,
        surface_mode=True,
        n_slab=n_slab,
    )
    np.testing.assert_allclose(atoms.get_positions()[:n_slab], slab_before, atol=1e-8)
