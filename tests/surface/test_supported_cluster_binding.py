"""Tests for supported-cluster deposit validation (surface contact + connectivity)."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111

from scgo.exceptions import SCGOValidationError
from scgo.initialization.geometry_helpers import (
    get_covalent_radius,
    pairwise_distances,
)
from scgo.surface.deposition import combine_slab_adsorbate, slab_surface_extreme
from scgo.surface.validation import (
    _mobile_indices_touch_slab,
    validate_supported_cluster_deposit,
)
from scgo.system_types import validate_structure_for_system_type


@pytest.fixture
def pt_slab() -> Atoms:
    return fcc111("Pt", size=(2, 2, 2), vacuum=6.0, orthogonal=True)


def _z(pt_slab: Atoms) -> float:
    return slab_surface_extreme(pt_slab, 2, upper=True)


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


def _build_single(pt_slab: Atoms, offsets: list[tuple[float, float, float]]) -> Atoms:
    z = _z(pt_slab)
    positions = [[x, y, z + dz] for (x, y, dz) in offsets]
    return _combined_with_mobile(pt_slab, positions)


def _build_core_ads(
    pt_slab: Atoms,
    core_offsets: list[tuple[float, float, float]],
    ads_offsets: list[tuple[float, float, float]],
    core_symbols: list[str],
    ads_symbols: list[str],
) -> Atoms:
    z = _z(pt_slab)
    core = [[x, y, z + dz] for (x, y, dz) in core_offsets]
    ads = [[x, y, z + dz] for (x, y, dz) in ads_offsets]
    return _combined_core_ads_mobile(
        pt_slab, core, ads, core_symbols=core_symbols, ads_symbols=ads_symbols
    )


# (id, builder, flags, expected_ok, expected_message_substr_or_None)
FLAG_MATRIX = [
    (
        "typical_deposit",
        lambda s: _build_single(s, [(0.0, 0.0, 2.0)]),
        {},
        True,
        None,
    ),
    (
        "rejects_no_surface_contact",
        lambda s: _build_single(s, [(0.0, 0.0, 12.0)]),
        {},
        False,
        "No adsorbate-slab pair",
    ),
    (
        "rejects_disconnected_adsorbate",
        lambda s: _build_single(s, [(0.0, 0.0, 2.0), (0.0, 0.0, 4.0), (5.0, 5.0, 2.0)]),
        {},
        False,
        "Adsorbate validation failed",
    ),
    (
        "rejects_penetration",
        lambda s: _build_single(s, [(0.0, 0.0, -1.0)]),
        {},
        False,
        "penetrates",
    ),
    (
        "fragmentation_accepts_two_slab_bound_core_subgroups",
        lambda s: _build_single(
            s,
            [(0.0, 0.0, 2.0), (0.0, 0.0, 4.0), (5.0, 5.0, 2.0), (5.0, 5.0, 4.0)],
        ),
        {"allow_cluster_fragmentation": True},
        True,
        None,
    ),
    (
        "fragmentation_rejects_adsorbate_only_subgroup",
        lambda s: _build_core_ads(
            s,
            [(0.0, 0.0, 2.0), (0.0, 0.0, 4.0)],
            [(5.0, 5.0, 2.0), (5.0, 5.0, 4.0)],
            ["Pt", "Pt"],
            ["O", "O"],
        ),
        {
            "n_core_mobile": 2,
            "allow_cluster_fragmentation": True,
            "allow_adsorbate_surface_detachment": False,
        },
        False,
        "adsorbate-only",
    ),
    (
        "both_relaxations_reject_subgroup_not_touching_slab",
        lambda s: _build_single(
            s,
            [(0.0, 0.0, 2.0), (0.0, 0.0, 4.0), (5.0, 5.0, 12.0), (5.0, 5.0, 14.0)],
        ),
        {
            "allow_cluster_fragmentation": True,
            "allow_adsorbate_surface_detachment": True,
        },
        False,
        "Every mobile subgroup must touch the slab",
    ),
    (
        "detachment_accepts_detached_ads_on_slab",
        lambda s: _build_core_ads(
            s,
            [(0.0, 0.0, 2.0), (0.0, 0.0, 4.0)],
            [(4.5, 4.5, 1.5), (5.0, 5.0, 1.5)],
            ["Pt", "Pt"],
            ["O", "O"],
        ),
        {
            "n_core_mobile": 2,
            "allow_cluster_fragmentation": False,
            "allow_adsorbate_surface_detachment": True,
        },
        True,
        None,
    ),
    (
        "detachment_rejects_multiple_core_subgroups",
        lambda s: _build_single(
            s,
            [(0.0, 0.0, 2.0), (0.0, 0.0, 4.0), (5.0, 5.0, 2.0), (5.0, 5.0, 4.0)],
        ),
        {
            "allow_cluster_fragmentation": False,
            "allow_adsorbate_surface_detachment": True,
        },
        False,
        "Exactly one core-connected mobile component",
    ),
]


@pytest.mark.parametrize("case", FLAG_MATRIX, ids=[c[0] for c in FLAG_MATRIX])
def test_validate_supported_cluster_deposit_flag_matrix(pt_slab: Atoms, case) -> None:
    """Parametrized matrix over mobile layouts and contact/connectivity flags.

    Each case reproduces the exact ``(ok, msg)`` expectation of the original
    single-purpose tests it replaces.
    """
    _name, build, flags, expect_ok, expect_msg = case
    n_slab = len(pt_slab)
    combined = build(pt_slab)
    ok, msg = validate_supported_cluster_deposit(
        combined,
        n_slab,
        surface_normal_axis=2,
        use_mic=False,
        **flags,
    )
    if expect_ok:
        assert ok, msg
    else:
        assert not ok
        if expect_msg is not None:
            assert expect_msg.lower() in msg.lower(), msg


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

    with pytest.raises(SCGOValidationError, match="Adsorbate validation failed"):
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


def _slab_with_buried_atom() -> Atoms:
    """Three-atom slab: two spread-out surface atoms plus one buried atom.

    A mobile atom placed over the middle is within bonding distance of the
    buried atom only; the surface (top-layer) atoms are too far away.
    """
    return Atoms(
        "Pt3",
        positions=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [5.0, 0.0, -3.0]],
        cell=[30.0, 30.0, 30.0],
        pbc=False,
    )


def test_contact_with_buried_slab_atom_is_rejected_in_both_paths() -> None:
    """Bonding a buried slab atom is not slab contact, split path included."""
    slab = _slab_with_buried_atom()
    mobile = Atoms(
        "Pt2",
        positions=[[5.0, 0.0, 0.5], [5.0, 0.0, 3.0]],
        cell=slab.cell,
        pbc=slab.pbc,
    )
    combined = combine_slab_adsorbate(slab, mobile)

    ok_strict, msg_strict = validate_supported_cluster_deposit(
        combined,
        len(slab),
        surface_normal_axis=2,
        use_mic=False,
    )
    assert not ok_strict
    assert "No adsorbate-slab pair" in msg_strict

    ok_split, msg_split = validate_supported_cluster_deposit(
        combined,
        len(slab),
        surface_normal_axis=2,
        use_mic=False,
        allow_cluster_fragmentation=True,
        allow_adsorbate_surface_detachment=True,
    )
    assert not ok_split
    assert "Every mobile subgroup must touch the slab" in msg_split


def test_two_and_three_atom_disconnected_mobile_share_one_message() -> None:
    """``n_ads == 2`` must use the same connectivity owner (and text) as ``n_ads == 3``."""
    slab = _slab_with_buried_atom()
    two = Atoms(
        "Pt2",
        positions=[[0.0, 0.0, 2.0], [10.0, 0.0, 2.0]],
        cell=slab.cell,
        pbc=slab.pbc,
    )
    three = Atoms(
        "Pt3",
        positions=[[0.0, 0.0, 2.0], [0.0, 0.0, 4.4], [10.0, 0.0, 2.0]],
        cell=slab.cell,
        pbc=slab.pbc,
    )

    messages = []
    for mobile in (two, three):
        ok, msg = validate_supported_cluster_deposit(
            combine_slab_adsorbate(slab, mobile),
            len(slab),
            surface_normal_axis=2,
            use_mic=False,
        )
        assert not ok
        messages.append(msg)

    assert all(m.startswith("Adsorbate validation failed: ") for m in messages)
    assert all("not connected" in m for m in messages)


def test_validate_supported_cluster_deposit_rejects_dissociated_adsorbate_fragment(
    pt_slab: Atoms,
) -> None:
    n_slab = len(pt_slab)
    z_top = slab_surface_extreme(pt_slab, 2, upper=True)
    combined = _combined_core_ads_mobile(
        pt_slab,
        [[0.0, 0.0, z_top + 2.0], [0.0, 0.0, z_top + 4.0]],
        [
            [2.2, 0.4, z_top + 1.5],  # O (fragment 0)
            [2.6, 0.4, z_top + 1.5],  # H (fragment 0) connected
            [3.0, 0.4, z_top + 1.5],  # O (fragment 1)
            [1.0, 0.4, z_top + 1.5],  # H (fragment 1) dissociated from O
        ],
        core_symbols=["Pt", "Pt"],
        ads_symbols=["O", "H", "O", "H"],
    )
    ok, msg = validate_supported_cluster_deposit(
        combined,
        n_slab,
        surface_normal_axis=2,
        use_mic=False,
        n_core_mobile=2,
        adsorbate_fragment_lengths=[2, 2],
        allow_cluster_fragmentation=True,
        allow_adsorbate_surface_detachment=True,
    )
    assert not ok
    assert "fragment integrity check failed" in msg


def test_validate_supported_cluster_deposit_allows_dissociation_when_integrity_disabled(
    pt_slab: Atoms,
) -> None:
    n_slab = len(pt_slab)
    z_top = slab_surface_extreme(pt_slab, 2, upper=True)
    combined = _combined_core_ads_mobile(
        pt_slab,
        [[0.0, 0.0, z_top + 2.0], [0.0, 0.0, z_top + 4.0]],
        [
            [2.2, 0.4, z_top + 1.5],
            [2.6, 0.4, z_top + 1.5],
            [3.0, 0.4, z_top + 1.5],
            [1.0, 0.4, z_top + 1.5],
        ],
        core_symbols=["Pt", "Pt"],
        ads_symbols=["O", "H", "O", "H"],
    )
    ok, msg = validate_supported_cluster_deposit(
        combined,
        n_slab,
        surface_normal_axis=2,
        use_mic=False,
        n_core_mobile=2,
        adsorbate_fragment_lengths=[2, 2],
        allow_cluster_fragmentation=True,
        allow_adsorbate_surface_detachment=True,
        enforce_adsorbate_subgraph_integrity=False,
    )
    assert ok, msg


def test_validate_supported_cluster_deposit_rejects_cross_fragment_adsorbate_bonding(
    pt_slab: Atoms,
) -> None:
    n_slab = len(pt_slab)
    z_top = slab_surface_extreme(pt_slab, 2, upper=True)
    combined = _combined_core_ads_mobile(
        pt_slab,
        [[0.0, 0.0, z_top + 2.0], [0.0, 0.0, z_top + 4.0]],
        [
            [2.2, 0.4, z_top + 1.5],  # O (fragment 0)
            [2.2, 0.4, z_top + 2.46],  # H (fragment 0) connected
            [3.0, 0.4, z_top + 1.5],  # H (fragment 1) bonded to O (unwanted merge)
        ],
        core_symbols=["Pt", "Pt"],
        ads_symbols=["O", "H", "H"],
    )
    ok, msg = validate_supported_cluster_deposit(
        combined,
        n_slab,
        surface_normal_axis=2,
        use_mic=False,
        n_core_mobile=2,
        adsorbate_fragment_lengths=[2, 1],
        allow_cluster_fragmentation=True,
        allow_adsorbate_surface_detachment=True,
    )
    assert not ok
    assert "bonded to fragment" in msg


def test_mobile_touch_slab_kdtree_matches_bruteforce():
    slab = Atoms(
        "Cu8",
        positions=[
            [0.0, 0.0, 0.0],
            [1.8, 0.0, 0.0],
            [0.0, 1.8, 0.0],
            [1.8, 1.8, 0.0],
            [0.0, 0.0, 1.8],
            [1.8, 0.0, 1.8],
            [0.0, 1.8, 1.8],
            [1.8, 1.8, 1.8],
        ],
        cell=[3.6, 3.6, 3.6],
        pbc=[True, True, True],
    )
    n_slab = len(slab)
    mobile = Atoms("Cu", positions=[[0.9, 0.9, 2.3]])
    combined = slab + mobile
    combined.center(vacuum=0.0)

    cf = 1.4
    symbols = combined.get_chemical_symbols()
    mobile_global = [n_slab]
    slab_indices = list(range(n_slab))
    touches, min_dist = _mobile_indices_touch_slab(
        combined,
        n_slab,
        mobile_global,
        connectivity_factor=cf,
        use_mic=True,
        surface_normal_axis=2,
    )

    mob_pos = combined.get_positions()[mobile_global]
    slab_pos = combined.get_positions()[slab_indices]
    dist = pairwise_distances(mob_pos, slab_pos, combined, use_mic=True)
    r_mobile = np.array([get_covalent_radius(symbols[i]) for i in mobile_global])
    r_slab = np.array([get_covalent_radius(symbols[j]) for j in slab_indices])
    thresholds = (r_mobile[:, None] + r_slab[None, :]) * cf
    ref_touch = bool(np.any(dist <= thresholds))
    ref_min = float(np.min(dist))

    assert touches is ref_touch
    assert min_dist == pytest.approx(ref_min, abs=1e-9)
