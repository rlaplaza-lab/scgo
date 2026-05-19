"""Tests for supported-cluster deposit validation (surface contact + connectivity)."""

from __future__ import annotations

import pytest
from ase import Atoms
from ase.build import fcc111

from scgo.surface.deposition import combine_slab_adsorbate, slab_surface_extreme
from scgo.surface.validation import validate_supported_cluster_deposit
from scgo.system_types import validate_structure_for_system_type


@pytest.fixture
def pt_slab() -> Atoms:
    return fcc111("Pt", size=(2, 2, 2), vacuum=6.0, orthogonal=True)


def _combined_with_mobile(pt_slab: Atoms, mobile_positions: list[list[float]]) -> Atoms:
    ads = Atoms(
        symbols=["Pt"] * len(mobile_positions),
        positions=mobile_positions,
        cell=pt_slab.cell,
        pbc=pt_slab.pbc,
    )
    return combine_slab_adsorbate(pt_slab, ads)


def test_validate_supported_cluster_deposit_accepts_typical_deposit(
    pt_slab: Atoms,
) -> None:
    n_slab = len(pt_slab)
    z_top = slab_surface_extreme(pt_slab, 2, upper=True)
    ads = Atoms(
        "Pt",
        positions=[[0.0, 0.0, z_top + 2.0]],
        cell=pt_slab.cell,
        pbc=pt_slab.pbc,
    )
    combined = combine_slab_adsorbate(pt_slab, ads)
    ok, msg = validate_supported_cluster_deposit(
        combined,
        n_slab,
        surface_normal_axis=2,
        use_mic=False,
    )
    assert ok, msg


def test_validate_supported_cluster_deposit_rejects_no_surface_contact(
    pt_slab: Atoms,
) -> None:
    n_slab = len(pt_slab)
    z_top = slab_surface_extreme(pt_slab, 2, upper=True)
    ads = Atoms(
        "Pt",
        positions=[[0.0, 0.0, z_top + 12.0]],
        cell=pt_slab.cell,
        pbc=pt_slab.pbc,
    )
    combined = combine_slab_adsorbate(pt_slab, ads)
    ok, msg = validate_supported_cluster_deposit(
        combined,
        n_slab,
        surface_normal_axis=2,
        use_mic=False,
    )
    assert not ok
    assert "No adsorbate–slab pair" in msg


def test_validate_supported_cluster_deposit_rejects_disconnected_adsorbate(
    pt_slab: Atoms,
) -> None:
    n_slab = len(pt_slab)
    z_top = slab_surface_extreme(pt_slab, 2, upper=True)
    combined = _combined_with_mobile(
        pt_slab,
        [
            [0.0, 0.0, z_top + 2.0],
            [0.0, 0.0, z_top + 4.0],
            [5.0, 5.0, z_top + 2.0],
        ],
    )
    ok, msg = validate_supported_cluster_deposit(
        combined,
        n_slab,
        surface_normal_axis=2,
        use_mic=False,
    )
    assert not ok
    assert "Adsorbate validation failed" in msg


def test_validate_supported_cluster_deposit_rejects_penetration(pt_slab: Atoms) -> None:
    n_slab = len(pt_slab)
    z_top = slab_surface_extreme(pt_slab, 2, upper=True)
    ads = Atoms(
        "Pt",
        positions=[[0.0, 0.0, z_top - 1.0]],
        cell=pt_slab.cell,
        pbc=pt_slab.pbc,
    )
    combined = combine_slab_adsorbate(pt_slab, ads)
    ok, msg = validate_supported_cluster_deposit(
        combined,
        n_slab,
        surface_normal_axis=2,
        use_mic=False,
    )
    assert not ok
    assert "penetrates" in msg


def test_dissociative_accepts_two_slab_bound_subgroups(pt_slab: Atoms) -> None:
    """Two mobile subgroups, each connected and slab-bound, are allowed in dissociative mode."""
    n_slab = len(pt_slab)
    z_top = slab_surface_extreme(pt_slab, 2, upper=True)
    combined = _combined_with_mobile(
        pt_slab,
        [
            [0.0, 0.0, z_top + 2.0],
            [0.0, 0.0, z_top + 4.0],
            [5.0, 5.0, z_top + 2.0],
            [5.0, 5.0, z_top + 4.0],
        ],
    )
    ok, msg = validate_supported_cluster_deposit(
        combined,
        n_slab,
        surface_normal_axis=2,
        use_mic=False,
        allow_dissociative_adsorption=True,
    )
    assert ok, msg


def test_dissociative_rejects_subgroup_not_touching_slab(pt_slab: Atoms) -> None:
    n_slab = len(pt_slab)
    z_top = slab_surface_extreme(pt_slab, 2, upper=True)
    combined = _combined_with_mobile(
        pt_slab,
        [
            [0.0, 0.0, z_top + 2.0],
            [0.0, 0.0, z_top + 4.0],
            [5.0, 5.0, z_top + 12.0],
            [5.0, 5.0, z_top + 14.0],
        ],
    )
    ok, msg = validate_supported_cluster_deposit(
        combined,
        n_slab,
        surface_normal_axis=2,
        use_mic=False,
        allow_dissociative_adsorption=True,
    )
    assert not ok
    assert "Dissociative adsorption requires every mobile subgroup" in msg


def test_validate_structure_for_system_type_respects_dissociative_flag(
    pt_slab: Atoms,
) -> None:
    from scgo.runner_surface import make_surface_config

    n_slab = len(pt_slab)
    z_top = slab_surface_extreme(pt_slab, 2, upper=True)
    combined = _combined_with_mobile(
        pt_slab,
        [
            [0.0, 0.0, z_top + 2.0],
            [0.0, 0.0, z_top + 4.0],
            [5.0, 5.0, z_top + 2.0],
            [5.0, 5.0, z_top + 4.0],
        ],
    )
    # In-plane separation must be judged without MIC (default config uses MIC).
    surface_config = make_surface_config(pt_slab, comparator_use_mic=False)

    with pytest.raises(ValueError, match="Adsorbate validation failed"):
        validate_structure_for_system_type(
            combined,
            system_type="surface_cluster",
            surface_config=surface_config,
            n_slab=n_slab,
            allow_dissociative_adsorption=False,
        )

    validate_structure_for_system_type(
        combined,
        system_type="surface_cluster",
        surface_config=surface_config,
        n_slab=n_slab,
        allow_dissociative_adsorption=True,
    )
