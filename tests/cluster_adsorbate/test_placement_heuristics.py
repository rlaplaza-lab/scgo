"""Placement heuristics: ranked candidates and progressive relaxation."""

from __future__ import annotations

import logging

import numpy as np
from ase import Atoms
from ase.build import molecule
from numpy.random import default_rng

from scgo.cluster_adsorbate.config import ClusterAdsorbateConfig
from scgo.cluster_adsorbate.placement import (
    infer_fragment_bond_axis,
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


def _ooh_template() -> Atoms:
    # Binding O, distal O along x, H on the distal O only.
    return Atoms(
        "OOH",
        positions=[[0.0, 0.0, 0.0], [1.45, 0.0, 0.0], [1.45, 0.96, 0.0]],
        pbc=False,
    )


def _hydrogen_points_outward(core: Atoms, placed: Atoms) -> bool:
    com = np.mean(core.get_positions(), axis=0)
    oxygen = placed.positions[0]
    hydrogen = placed.positions[1]
    return float(np.dot(hydrogen - oxygen, oxygen - com)) > 0.0


def test_infer_bond_axis_for_single_neighbor_anchors() -> None:
    assert infer_fragment_bond_axis(_oh_template(), 0) == (0, 1)
    assert infer_fragment_bond_axis(_ooh_template(), 0) == (0, 1)
    assert infer_fragment_bond_axis(Atoms("O", positions=[[0.0, 0.0, 0.0]]), 0) is None
    assert infer_fragment_bond_axis(molecule("H2O"), 0) is None


def test_omitted_bond_axis_points_oh_hydrogen_outward() -> None:
    core = _pt3_tetrahedron()
    placed = place_fragment_on_cluster(
        core,
        _oh_template(),
        default_rng(3),
        ClusterAdsorbateConfig(max_placement_attempts=20),
    )
    assert placed is not None
    assert _hydrogen_points_outward(core, placed)


def test_explicit_bond_axis_is_not_replaced() -> None:
    core = _pt3_tetrahedron()
    placed = place_fragment_on_cluster(
        core,
        _oh_template(),
        default_rng(4),
        ClusterAdsorbateConfig(max_placement_attempts=30),
        bond_axis=(1, 0),
    )
    assert placed is not None
    assert not _hydrogen_points_outward(core, placed)


def test_exhausted_placement_logs_only_at_debug(caplog, monkeypatch) -> None:
    monkeypatch.setattr(
        "scgo.cluster_adsorbate.placement._generate_placement_trials",
        lambda *_args, **_kwargs: [],
    )
    logger_name = "scgo.cluster_adsorbate.placement"
    with caplog.at_level(logging.INFO, logger=logger_name):
        placed = place_fragment_on_cluster(
            _pt3_tetrahedron(),
            _oh_template(),
            default_rng(5),
            ClusterAdsorbateConfig(max_placement_attempts=2),
        )
    assert placed is None
    assert "max_placement_attempts" not in caplog.text

    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=logger_name):
        place_fragment_on_cluster(
            _pt3_tetrahedron(),
            _oh_template(),
            default_rng(5),
            ClusterAdsorbateConfig(max_placement_attempts=2),
        )
    assert "max_placement_attempts=2" in caplog.text
    assert "bond_axis=(0, 1)" in caplog.text


def test_surface_site_cache_reuses_hull_for_identical_core() -> None:
    from unittest.mock import patch

    from scgo.cluster_adsorbate import sites as sites_mod

    sites_mod.clear_surface_site_cache()
    core = Atoms(
        "Pt4",
        positions=[
            [0.0, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [1.25, 2.165, 0.0],
            [1.25, 0.722, 2.0],
        ],
        pbc=False,
    )
    with patch.object(
        sites_mod,
        "compute_surface_site_candidates",
        wraps=sites_mod.compute_surface_site_candidates,
    ) as mocked:
        first = sites_mod.get_or_compute_surface_site_candidates(core)
        second = sites_mod.get_or_compute_surface_site_candidates(core)
        assert first is second
        assert mocked.call_count == 1
    sites_mod.clear_surface_site_cache()
