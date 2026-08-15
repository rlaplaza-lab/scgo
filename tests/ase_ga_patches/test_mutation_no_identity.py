"""Unit tests for identity-free mutation operators and connectivity retries."""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.build import fcc111
from ase_ga.utilities import (
    atoms_too_close,
    closest_distances_generator,
    get_all_atom_types,
)

from scgo.ase_ga_patches.mutations.flattening import FlatteningMutation
from scgo.ase_ga_patches.mutations.overlap_relief import OverlapReliefMutation
from scgo.ase_ga_patches.mutations.rattle import RattleMutation
from scgo.initialization.geometry_helpers import validate_cluster_structure
from scgo.initialization.initialization_config import (
    CONNECTIVITY_FACTOR,
    MIN_DISTANCE_FACTOR_DEFAULT,
)


def _pt4_valid(rng: np.random.Generator) -> tuple[Atoms, dict]:
    atoms = Atoms(
        "Pt4",
        positions=[
            [0.0, 0.0, 0.0],
            [2.6, 0.0, 0.0],
            [1.3, 2.25, 0.0],
            [1.3, 0.75, 2.12],
        ],
    )
    atoms.center(vacuum=8.0)
    blmin = closest_distances_generator(get_all_atom_types(atoms, range(4)), 0.7)
    assert not atoms_too_close(atoms, blmin)
    _ = rng
    return atoms, blmin


def test_overlap_relief_does_not_return_unchanged_valid_parent(rng) -> None:
    atoms, blmin = _pt4_valid(rng)
    mut = OverlapReliefMutation(
        blmin,
        len(atoms),
        n_sweeps=4,
        jitter=0.02,
        test_dist_to_slab=False,
        system_type="gas_cluster",
        rng=rng,
    )
    out = mut.mutate(atoms.copy())
    if out is None:
        return
    assert not np.allclose(out.get_positions(), atoms.get_positions(), atol=1e-8)
    assert not atoms_too_close(out, blmin)


def test_flattening_skips_identity_and_changes_geometry(rng) -> None:
    atoms, blmin = _pt4_valid(rng)
    mut = FlatteningMutation(
        blmin,
        len(atoms),
        system_type="gas_cluster",
        thickness_factor=0.5,
        test_dist_to_slab=False,
        rng=rng,
        max_inner_attempts=12,
    )
    out = mut.mutate(atoms.copy())
    assert out is not None
    rms = np.linalg.norm(out.get_positions() - atoms.get_positions()) / 2.0
    assert rms > 1e-6
    assert not atoms_too_close(out, blmin)


def test_surface_rattle_returns_connected_mobile(rng) -> None:
    slab = fcc111("Pt", size=(3, 4, 2), vacuum=8.0, orthogonal=True)
    n_slab = len(slab)
    z_top = float(np.max(slab.positions[:, 2]))
    center = np.mean(slab.positions[:, :2], axis=0)
    cluster = Atoms(
        "Pt4",
        positions=[
            [center[0], center[1], z_top + 2.4],
            [center[0] + 2.6, center[1], z_top + 2.4],
            [center[0] + 1.3, center[1] + 2.25, z_top + 2.4],
            [center[0] + 1.3, center[1] + 0.75, z_top + 4.5],
        ],
        cell=slab.get_cell(),
        pbc=slab.get_pbc(),
    )
    full = slab + cluster
    blmin = closest_distances_generator(get_all_atom_types(full, range(4)), 0.7)
    mut = RattleMutation(
        blmin,
        n_top=4,
        system_type="surface_cluster",
        rattle_strength=0.8,
        rattle_prop=0.4,
        rng=rng,
    )
    out = mut.mutate(full.copy())
    assert out is not None
    mobile = out[n_slab:]
    ok, msg = validate_cluster_structure(
        mobile,
        MIN_DISTANCE_FACTOR_DEFAULT,
        CONNECTIVITY_FACTOR,
        check_clashes=False,
        check_connectivity=True,
        use_mic=True,
    )
    assert ok, msg
