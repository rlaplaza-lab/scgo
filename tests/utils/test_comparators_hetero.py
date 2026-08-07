"""Opt-in hetero-atomic discrimination for the pure distance comparator."""

from __future__ import annotations

import numpy as np
from ase import Atoms

from scgo.utils.comparators import (
    PureInteratomicDistanceComparator,
    get_sorted_dist_list,
    get_sorted_hetero_dist_list,
)


def _pt3_o(o_position: list[float]) -> Atoms:
    """Equilateral Pt3 triangle plus one O at ``o_position``."""
    side = 2.7
    pt_positions = [
        [0.0, 0.0, 0.0],
        [side, 0.0, 0.0],
        [side / 2.0, side * np.sqrt(3.0) / 2.0, 0.0],
    ]
    return Atoms("Pt3O", positions=[*pt_positions, o_position])


def test_hetero_pairs_separate_atop_from_bridge() -> None:
    """Atop vs bridge share every same-species distance but differ in Pt–O."""
    atop = _pt3_o([0.0, 0.0, 1.8])
    bridge = _pt3_o([1.35, 0.0, 1.5])

    # Same-species fingerprints are identical: ASE's comparator is blind here.
    fp_atop = get_sorted_dist_list(atop)
    fp_bridge = get_sorted_dist_list(bridge)
    assert set(fp_atop) == set(fp_bridge)
    for z in fp_atop:
        assert np.allclose(fp_atop[z], fp_bridge[z])

    # Hetero fingerprints differ.
    het_atop = get_sorted_hetero_dist_list(atop)
    het_bridge = get_sorted_hetero_dist_list(bridge)
    assert set(het_atop) == {(8, 78)}
    assert not np.allclose(het_atop[(8, 78)], het_bridge[(8, 78)])

    ase_compatible = PureInteratomicDistanceComparator(n_top=4)
    hetero_aware = PureInteratomicDistanceComparator(n_top=4, include_hetero_pairs=True)

    assert ase_compatible.looks_like(atop, bridge) is True
    assert hetero_aware.looks_like(atop, bridge) is False

    cum_off, max_off = ase_compatible.get_differences(atop, bridge)
    cum_on, max_on = hetero_aware.get_differences(atop, bridge)
    assert (cum_off, max_off) == (0.0, 0.0)
    assert cum_on > ase_compatible.tol
    assert max_on > ase_compatible.pair_cor_max


def test_hetero_pairs_still_match_identical_structures() -> None:
    """Enabling hetero pairs must not make identical structures look different."""
    atoms = _pt3_o([0.0, 0.0, 1.8])
    same = atoms.copy()
    same.positions[3, 2] += 1e-5

    hetero_aware = PureInteratomicDistanceComparator(n_top=4, include_hetero_pairs=True)
    assert hetero_aware.looks_like(atoms, same) is True


def test_hetero_pairs_are_noop_for_single_element() -> None:
    """Single-element clusters have no hetero pairs, so results are unchanged."""
    a1 = Atoms("Pt3", positions=[[0, 0, 0], [2.5, 0, 0], [1.25, 2.1, 0]])
    a2 = a1.copy()
    a2.positions[2, 1] += 0.4

    assert get_sorted_hetero_dist_list(a1) == {}

    off = PureInteratomicDistanceComparator(n_top=3)
    on = PureInteratomicDistanceComparator(n_top=3, include_hetero_pairs=True)
    assert off.get_differences(a1, a2) == on.get_differences(a1, a2)


def test_hetero_pairs_default_off_keeps_ase_reference_behavior() -> None:
    """The constructor default must stay ASE-compatible (same-species only)."""
    assert PureInteratomicDistanceComparator().include_hetero_pairs is False


def test_hetero_pairs_mic_uses_minimum_image() -> None:
    """MIC folding applies to hetero distances as it does to same-species ones."""
    cell = [8.0, 8.0, 12.0]
    atoms = Atoms(
        "PtO",
        positions=[[0.1, 0.0, 0.0], [7.9, 0.0, 0.0]],
        cell=cell,
        pbc=[True, True, False],
    )
    no_mic = get_sorted_hetero_dist_list(atoms, mic=False)[(8, 78)]
    with_mic = get_sorted_hetero_dist_list(atoms, mic=True)[(8, 78)]
    assert float(no_mic[0]) > 7.0
    assert float(with_mic[0]) < 1.0


def test_filter_unique_minima_keeps_hetero_distinct_minima() -> None:
    """Run-level filtering opts into hetero discrimination by default."""
    from scgo.utils.helpers import filter_unique_minima

    atop = _pt3_o([0.0, 0.0, 1.8])
    bridge = _pt3_o([1.35, 0.0, 1.5])

    minima = [(-1.0, atop), (-1.0, bridge)]

    kept = filter_unique_minima(minima, n_top=4)
    assert len(kept) == 2

    collapsed = filter_unique_minima(
        [(-1.0, atop.copy()), (-1.0, bridge.copy())],
        n_top=4,
        comparator_include_hetero_pairs=False,
    )
    assert len(collapsed) == 1
