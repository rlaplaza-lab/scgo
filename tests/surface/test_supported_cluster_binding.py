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


def _combined_core_ads_mobile(
    pt_slab: Atoms,
    core_positions: list[list[float]],
    ads_positions: list[list[float]],
    *,
    core_symbols: list[str] | None = None,
    ads_symbols: list[str] | None = None,
) -> Atoms:
    core_syms = core_symbols or ["Pt"] * len(core_positions)
    ads_syms = ads_symbols or ["O"] * len(ads_positions)
    mobile = Atoms(
        symbols=core_syms + ads_syms,
        positions=core_positions + ads_positions,
        cell=pt_slab.cell,
        pbc=pt_slab.pbc,
    )
    return combine_slab_adsorbate(pt_slab, mobile)


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


def test_cluster_fragmentation_accepts_two_slab_bound_core_subgroups(
    pt_slab: Atoms,
) -> None:
    """Two mobile core subgroups, each connected and slab-bound, are allowed."""
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
        allow_cluster_fragmentation=True,
    )
    assert ok, msg


def test_cluster_fragmentation_rejects_adsorbate_only_subgroup(pt_slab: Atoms) -> None:
    """Fragmentation without detachment does not allow adsorbate-only subgroups."""
    n_slab = len(pt_slab)
    z_top = slab_surface_extreme(pt_slab, 2, upper=True)
    combined = _combined_core_ads_mobile(
        pt_slab,
        [[0.0, 0.0, z_top + 2.0], [0.0, 0.0, z_top + 4.0]],
        [[5.0, 5.0, z_top + 2.0], [5.0, 5.0, z_top + 4.0]],
        core_symbols=["Pt", "Pt"],
        ads_symbols=["O", "O"],
    )
    ok, msg = validate_supported_cluster_deposit(
        combined,
        n_slab,
        surface_normal_axis=2,
        use_mic=False,
        n_core_mobile=2,
        allow_cluster_fragmentation=True,
        allow_adsorbate_surface_detachment=False,
    )
    assert not ok
    assert "adsorbate-only" in msg.lower()


def test_both_relaxations_reject_subgroup_not_touching_slab(pt_slab: Atoms) -> None:
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
        allow_cluster_fragmentation=True,
        allow_adsorbate_surface_detachment=True,
    )
    assert not ok
    assert "Every mobile subgroup must touch the slab" in msg


def test_adsorbate_surface_detachment_accepts_detached_ads_on_slab(
    pt_slab: Atoms,
) -> None:
    """One core subgroup plus a slab-bound adsorbate-only subgroup is allowed."""
    n_slab = len(pt_slab)
    z_top = slab_surface_extreme(pt_slab, 2, upper=True)
    combined = _combined_core_ads_mobile(
        pt_slab,
        [[0.0, 0.0, z_top + 2.0], [0.0, 0.0, z_top + 4.0]],
        [[4.5, 4.5, z_top + 1.5], [5.0, 5.0, z_top + 1.5]],
        core_symbols=["Pt", "Pt"],
        ads_symbols=["O", "O"],
    )
    ok, msg = validate_supported_cluster_deposit(
        combined,
        n_slab,
        surface_normal_axis=2,
        use_mic=False,
        n_core_mobile=2,
        allow_cluster_fragmentation=False,
        allow_adsorbate_surface_detachment=True,
    )
    assert ok, msg


def test_adsorbate_surface_detachment_rejects_multiple_core_subgroups(
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
            [5.0, 5.0, z_top + 4.0],
        ],
    )
    ok, msg = validate_supported_cluster_deposit(
        combined,
        n_slab,
        surface_normal_axis=2,
        use_mic=False,
        allow_cluster_fragmentation=False,
        allow_adsorbate_surface_detachment=True,
    )
    assert not ok
    assert "Exactly one core-connected mobile component" in msg


def test_validate_structure_for_system_type_respects_connectivity_flags(
    pt_slab: Atoms,
) -> None:
    from scgo.surface import make_surface_config

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
    surface_config = make_surface_config(pt_slab, comparator_use_mic=False)

    with pytest.raises(ValueError, match="Adsorbate validation failed"):
        validate_structure_for_system_type(
            combined,
            system_type="surface_cluster",
            surface_config=surface_config,
            n_slab=n_slab,
            allow_cluster_fragmentation=False,
            allow_adsorbate_surface_detachment=False,
        )

    validate_structure_for_system_type(
        combined,
        system_type="surface_cluster",
        surface_config=surface_config,
        n_slab=n_slab,
        allow_cluster_fragmentation=True,
        allow_adsorbate_surface_detachment=False,
    )
