"""Placement heuristics: ranked candidates and progressive relaxation."""

from __future__ import annotations

from ase import Atoms
from numpy.random import default_rng

from scgo.cluster_adsorbate.config import ClusterAdsorbateConfig
from scgo.cluster_adsorbate.placement import (
    place_fragment_on_cluster,
    radii_derived_height_bounds,
)


def _pt3_tetrahedron() -> Atoms:
    return Atoms(
        "Pt3",
        positions=[
            [0.0, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [1.25, 2.165, 0.0],
        ],
        pbc=False,
    )


def _oh_template() -> Atoms:
    return Atoms("OH", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.96]], pbc=False)


def test_radii_derived_height_bounds_positive() -> None:
    core = _pt3_tetrahedron()
    frag = _oh_template()
    h_min, h_max = radii_derived_height_bounds(frag, core, anchor_index=0)
    assert h_min > 0.0
    assert h_max > h_min


def test_ranked_placement_succeeds_with_few_attempts() -> None:
    core = _pt3_tetrahedron()
    frag = _oh_template()
    cfg = ClusterAdsorbateConfig(max_placement_attempts=20)
    rng = default_rng(42)
    placed = place_fragment_on_cluster(core, frag, rng, cfg, bond_axis=(0, 1))
    assert placed is not None
    assert len(placed) == 2


def test_multi_fragment_placement_with_relaxation() -> None:
    core = _pt3_tetrahedron()
    oh1 = _oh_template()
    oh2 = _oh_template()
    cfg = ClusterAdsorbateConfig(max_placement_attempts=120)
    rng = default_rng(7)
    first = place_fragment_on_cluster(
        core, oh1, rng, cfg, bond_axis=(0, 1), anchor_index=0
    )
    assert first is not None
    combined = core + first
    second = place_fragment_on_cluster(
        core,
        oh2,
        rng,
        cfg,
        bond_axis=(0, 1),
        anchor_index=0,
        site_core=core,
        clash_atoms=combined,
    )
    assert second is not None
