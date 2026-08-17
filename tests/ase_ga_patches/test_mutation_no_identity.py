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

import scgo.ase_ga_patches.mutations.mirror as mirror_mod
from scgo.ase_ga_patches.mutations._common import _pin_subset_contact_atom
from scgo.ase_ga_patches.mutations.flattening import FlatteningMutation
from scgo.ase_ga_patches.mutations.mirror import MirrorMutation
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


def _tight_pt3_oh_on_slab() -> tuple[Atoms, dict, int]:
    """Pt3-OH with Pt-O just inside the default connectivity cutoff."""
    slab = fcc111("Pt", size=(4, 4, 2), vacuum=6.0, orthogonal=True)
    n_slab = len(slab)
    z_top = float(np.max(slab.positions[:, 2]))
    xy = np.mean(slab.positions[:, :2], axis=0)
    r_pt_o = 2.82
    cluster = Atoms(
        symbols=["Pt", "Pt", "Pt", "O", "H"],
        positions=[
            [xy[0], xy[1], z_top + 2.2],
            [xy[0] + 2.6, xy[1], z_top + 2.2],
            [xy[0] + 1.3, xy[1] + 2.25, z_top + 2.2],
            [xy[0] + 1.3, xy[1] + 2.25, z_top + 2.2 + r_pt_o],
            [xy[0] + 1.3, xy[1] + 2.25, z_top + 2.2 + r_pt_o + 0.96],
        ],
        cell=slab.get_cell(),
        pbc=slab.get_pbc(),
    )
    cluster.set_tags([0, 0, 0, 1, 1])
    full = slab + cluster
    blmin = closest_distances_generator(get_all_atom_types(full, range(5)), 0.7)
    return full, blmin, n_slab


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


def test_pin_subset_contact_atom_holds_binding_atom() -> None:
    parent = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    new = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [2.4, 0.0, 0.0]], dtype=float)
    out = _pin_subset_contact_atom(parent, new, np.array([False, True, True]))
    np.testing.assert_allclose(out[0], parent[0])
    np.testing.assert_allclose(out[1], parent[1])
    np.testing.assert_allclose(out[2], [2.2, 0.0, 0.0])


def test_flattening_ads_keeps_barely_connected_core_contact(rng) -> None:
    full, blmin, n_slab = _tight_pt3_oh_on_slab()
    mut = FlatteningMutation(
        blmin,
        n_top=5,
        system_type="surface_cluster_adsorbate",
        thickness_factor=0.5,
        target_tags=[1],
        rng=rng,
        max_inner_attempts=12,
    )
    out = mut.mutate(full.copy())
    assert out is not None
    ok, msg = validate_cluster_structure(
        out[n_slab:],
        MIN_DISTANCE_FACTOR_DEFAULT,
        CONNECTIVITY_FACTOR,
        check_clashes=False,
        check_connectivity=True,
        use_mic=True,
    )
    assert ok, msg
    assert not np.allclose(
        out.get_positions()[n_slab + 3 :],
        full.get_positions()[n_slab + 3 :],
        atol=1e-8,
    )


def test_mirror_rescue_reanchors_to_slab(monkeypatch, rng) -> None:
    full, blmin, n_slab = _tight_pt3_oh_on_slab()
    parent_low = float(np.min(full[n_slab:].get_positions()[:, 2]))
    mut_kwargs = {
        "blmin": blmin,
        "n_top": 5,
        "system_type": "surface_cluster_adsorbate",
        "target_tags": [0],
        "rng": rng,
        "max_tries": 12,
    }

    monkeypatch.setattr(
        mirror_mod,
        "_preserves_mobile_connectivity",
        lambda *args, **kwargs: False,
    )
    probe = MirrorMutation(**mut_kwargs)
    assert probe.mutate(full.copy()) is None
    main_limit = probe.last_attempt_count

    attempt = {"n": 0}

    def _preserves_in_rescue(parent, mutant, **kwargs):
        attempt["n"] += 1
        return attempt["n"] > main_limit

    monkeypatch.setattr(
        mirror_mod,
        "_preserves_mobile_connectivity",
        _preserves_in_rescue,
    )
    monkeypatch.setattr(mirror_mod, "atoms_too_close", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        mirror_mod,
        "atoms_too_close_two_sets",
        lambda *args, **kwargs: False,
    )

    out = MirrorMutation(**mut_kwargs).mutate(full.copy())
    assert out is not None
    out_low = float(np.min(out[n_slab:].get_positions()[:, 2]))
    assert abs(out_low - parent_low) < 1e-6


def test_mirror_core_keeps_adsorbate_contact(rng) -> None:
    full, blmin, n_slab = _tight_pt3_oh_on_slab()
    mut = MirrorMutation(
        blmin,
        n_top=5,
        system_type="surface_cluster_adsorbate",
        target_tags=[0],
        rng=rng,
        max_tries=12,
    )
    out = mut.mutate(full.copy())
    assert out is not None
    ok, msg = validate_cluster_structure(
        out[n_slab:],
        MIN_DISTANCE_FACTOR_DEFAULT,
        CONNECTIVITY_FACTOR,
        check_clashes=False,
        check_connectivity=True,
        use_mic=True,
    )
    assert ok, msg
    assert not np.allclose(
        out.get_positions()[n_slab : n_slab + 3],
        full.get_positions()[n_slab : n_slab + 3],
        atol=1e-8,
    )
