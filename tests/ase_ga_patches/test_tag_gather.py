"""Tests for constraint-safe tagged-atom gathering."""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.build import fcc111
from ase.constraints import FixBondLengths
from ase_ga.utilities import get_all_atom_types

from scgo.ase_ga_patches._tag_gather import (
    gather_atoms_by_tag,
    periodic_sheet_tag_to_skip,
)
from scgo.ase_ga_patches.cutandsplicepairing import CutAndSplicePairing
from scgo.initialization.atomic_radii import build_blmin_from_zs
from scgo.surface.partition import prepare_slab_search_surface_config
from scgo.surface.presets import make_n_doped_graphite_surface_config


def test_gather_skips_tag_zero_and_does_not_raise_with_fix_bond_lengths() -> None:
    atoms = Atoms(
        "CCCHH",
        positions=[
            [0.0, 0.0, 0.0],
            [1.4, 0.0, 0.0],
            [2.8, 0.0, 0.0],
            [0.2, 0.9, 0.0],
            [0.2, 0.9, 0.96],
        ],
        cell=[12.0, 12.0, 12.0],
        pbc=[True, True, False],
        tags=[0, 0, 0, 1, 1],
    )
    atoms.set_constraint(FixBondLengths([(3, 4)]))
    before = atoms.get_positions().copy()

    gather_atoms_by_tag(atoms, skip_tag=0)

    assert np.allclose(atoms.positions[:3], before[:3])
    assert np.isclose(
        np.linalg.norm(atoms.positions[4] - atoms.positions[3]),
        np.linalg.norm(before[4] - before[3]),
        atol=1e-8,
    )


def test_gather_unwraps_finite_core_with_tag_zero() -> None:
    """surface_cluster tag 0 is a molecule: MIC gather must unwrap it."""
    atoms = Atoms(
        "Pt2",
        positions=[[0.2, 0.0, 0.0], [9.8, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
        tags=[0, 0],
    )
    gather_atoms_by_tag(atoms, skip_tag=periodic_sheet_tag_to_skip("surface_cluster"))
    assert float(np.linalg.norm(atoms.positions[1] - atoms.positions[0])) < 2.0


def test_periodic_sheet_tag_skipped_only_for_slab_search() -> None:
    assert periodic_sheet_tag_to_skip("surface_adsorbate") == 0
    assert periodic_sheet_tag_to_skip("surface") == 0
    assert periodic_sheet_tag_to_skip("surface_cluster") is None
    assert periodic_sheet_tag_to_skip("gas_cluster_adsorbate") is None


def test_gather_unwraps_tagged_fragment_across_pbc() -> None:
    atoms = Atoms(
        "OH",
        positions=[[0.2, 0.0, 0.0], [9.8, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
        tags=[1, 1],
    )
    atoms.set_constraint(FixBondLengths([(0, 1)]))
    gather_atoms_by_tag(atoms, skip_tag=0)
    assert float(np.linalg.norm(atoms.positions[1] - atoms.positions[0])) < 2.0


def test_gather_does_not_fold_fcc_surface_layer() -> None:
    slab = fcc111("Pt", size=(2, 2, 2), vacuum=8.0, orthogonal=True)
    slab.pbc = [True, True, False]
    top = slab[-4:]
    top.set_tags(np.zeros(len(top), dtype=int))
    before = top.get_positions().copy()
    gather_atoms_by_tag(top, skip_tag=0)
    assert np.allclose(top.get_positions(), before)


def test_surface_pairing_with_frozen_oh_does_not_raise() -> None:
    cfg, part = prepare_slab_search_surface_config(
        make_n_doped_graphite_surface_config(
            slab_layers=3, slab_repeat_xy=2, n_dopants=1, seed=0
        )
    )
    n_fixed = int(part.n_fixed)
    z_top = float(np.max(cfg.slab.positions[:, 2]))
    xy = np.mean(cfg.slab.positions[n_fixed:, :2], axis=0)
    oh = Atoms(
        "OH",
        positions=[[xy[0], xy[1], z_top + 1.5], [xy[0], xy[1], z_top + 2.46]],
        cell=cfg.slab.cell,
        pbc=cfg.slab.pbc,
    )
    combined = cfg.slab.copy() + oh
    tags = np.zeros(len(combined), dtype=int)
    tags[-2:] = 1
    combined.set_tags(tags)
    combined.set_constraint(FixBondLengths([(len(combined) - 2, len(combined) - 1)]))
    # Wrap H across the in-plane cell so a constraint-applying reset after
    # MIC gather would formerly raise FixBondLengths "Did not converge".
    combined.positions[-1, 0] += float(combined.cell[0, 0]) * 0.8
    parent1 = combined.copy()
    parent2 = combined.copy()
    parent1.info["confid"] = "p1"
    parent2.info["confid"] = "p2"
    n_top = len(combined) - n_fixed
    slab = combined[:n_fixed]
    top_z = list({int(z) for z in combined[n_fixed:].numbers})
    blmin = build_blmin_from_zs(get_all_atom_types(combined, top_z), ratio=0.7)
    op = CutAndSplicePairing(
        slab,
        n_top,
        blmin,
        system_type="surface_adsorbate",
        use_tags=True,
        target_tags=[0],
        rng=np.random.default_rng(0),
    )
    child = op.cross(parent1, parent2)
    assert child is None or len(child) == len(combined)
