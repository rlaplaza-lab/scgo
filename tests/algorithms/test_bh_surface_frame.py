"""BH surface move/relax frame coherence."""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.build import fcc111
from ase.calculators.emt import EMT

from scgo.algorithms.basinhopping_go import _move_atoms
from scgo.system_types import get_system_policy
from scgo.utils.helpers import perform_local_relaxation


def test_move_atoms_skips_com_recenter_on_surface(rng) -> None:
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


def test_move_atoms_recenters_com_for_gas(rng) -> None:
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


def test_move_atoms_single_tag_group_displaces_rigidly(rng) -> None:
    atoms = Atoms(
        "OHH",
        positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
        cell=[20, 20, 20],
        pbc=False,
    )
    atoms.set_tags([1, 1, 1])
    before = atoms.get_positions().copy()
    moved, desc = _move_atoms(
        atoms,
        dr=0.5,
        move_fraction=1.0,
        rng=rng,
        movable_indices=[0, 1, 2],
        move_by_tag_groups=True,
        recenter_com=False,
    )
    delta = moved.get_positions() - before
    assert np.linalg.norm(delta) > 1e-6
    np.testing.assert_allclose(delta[0], delta[1], atol=1e-12)
    np.testing.assert_allclose(delta[0], delta[2], atol=1e-12)
    assert desc.startswith("Moved_atoms:")
    assert "none" not in desc


def test_move_atoms_zero_movable_honest_description(rng) -> None:
    atoms = Atoms("Pt2", positions=[[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
    before = atoms.get_positions().copy()
    moved, desc = _move_atoms(
        atoms,
        dr=0.5,
        move_fraction=1.0,
        rng=rng,
        movable_indices=[],
    )
    np.testing.assert_allclose(moved.get_positions(), before)
    assert desc == "Moved_atoms: none"


def test_move_atoms_single_movable_atom_displaces(rng) -> None:
    atoms = Atoms(
        "Pt3",
        positions=[[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [1.2, 2.1, 0.0]],
        cell=[20, 20, 20],
        pbc=False,
    )
    before = atoms.get_positions().copy()
    moved, desc = _move_atoms(
        atoms,
        dr=0.8,
        move_fraction=1.0,
        rng=rng,
        movable_indices=[1],
        recenter_com=False,
    )
    assert np.linalg.norm(moved.get_positions()[1] - before[1]) > 1e-6
    np.testing.assert_allclose(moved.get_positions()[0], before[0])
    np.testing.assert_allclose(moved.get_positions()[2], before[2])
    assert desc == "Moved_atoms: [2]"


def test_move_atoms_adsorbate_scale_does_not_throttle_core() -> None:
    """Mixed core+ads: ads displacements respect scale; core may exceed it."""
    policy = get_system_policy("surface_cluster_adsorbate")
    scale = policy.adsorbate_move_scale
    assert policy.constrain_adsorbate_moves
    assert scale < 1.0

    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    slab.pbc = [True, True, False]
    n_slab = len(slab)
    z0 = float(slab.get_positions()[:, 2].max() + 1.5)
    core = Atoms(
        "Pt2",
        positions=[[0.5, 0.5, z0], [2.0, 0.5, z0]],
    )
    ads = Atoms(
        "OH",
        positions=[[1.2, 1.5, z0 + 1.2], [1.2, 1.5, z0 + 2.16]],
    )
    atoms = slab.copy() + core + ads
    tags = np.zeros(len(atoms), dtype=int)
    tags[n_slab : n_slab + 2] = 0
    tags[n_slab + 2 :] = 1
    atoms.set_tags(tags)

    movable = list(range(n_slab, len(atoms)))
    core_idx = [n_slab, n_slab + 1]
    ads_idx = [n_slab + 2, n_slab + 3]
    dr = 1.0

    core_mags: list[float] = []
    ads_mags: list[float] = []
    for seed in range(40):
        moved, _desc = _move_atoms(
            atoms,
            dr=dr,
            move_fraction=1.0,
            rng=np.random.default_rng(seed),
            movable_indices=movable,
            recenter_com=False,
            adsorbate_movable_indices=ads_idx,
            adsorbate_dr=dr * scale,
            adsorbate_move_fraction=1.0,
        )
        delta = moved.get_positions() - atoms.get_positions()
        for i in core_idx:
            mag = float(np.linalg.norm(delta[i]))
            if mag > 1e-12:
                core_mags.append(mag)
        for i in ads_idx:
            mag = float(np.linalg.norm(delta[i]))
            if mag > 1e-12:
                ads_mags.append(mag)

    assert core_mags and ads_mags
    ads_bound = dr * scale * np.sqrt(3.0) + 1e-9
    assert max(ads_mags) <= ads_bound
    assert max(core_mags) > dr * scale


def test_move_atoms_adsorbate_only_respects_global_scale(rng) -> None:
    policy = get_system_policy("gas_cluster_adsorbate")
    scale = policy.adsorbate_move_scale
    atoms = Atoms(
        "OH",
        positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0]],
        cell=[20, 20, 20],
        pbc=False,
    )
    atoms.set_tags([1, 1])
    before = atoms.get_positions().copy()
    dr = 1.0
    moved, _desc = _move_atoms(
        atoms,
        dr=dr * scale,
        move_fraction=1.0,
        rng=rng,
        movable_indices=[0, 1],
        recenter_com=False,
    )
    delta = moved.get_positions() - before
    mags = np.linalg.norm(delta, axis=1)
    assert np.all(mags <= dr * scale * np.sqrt(3.0) + 1e-9)
    assert np.any(mags > 1e-6)


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
