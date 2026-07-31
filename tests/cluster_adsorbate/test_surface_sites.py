"""Tests for convex-hull and planar-layer adsorption site discovery."""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.build import graphene

from scgo.cluster_adsorbate.sites import (
    compute_surface_site_candidates,
    count_site_candidates,
    planar_layer_site_candidates,
)


def test_planar_layer_site_candidates_for_graphene() -> None:
    layer = graphene(size=(2, 2, 1), vacuum=8.0)
    sites = planar_layer_site_candidates(layer, surface_normal_axis=2)
    assert count_site_candidates(sites) > 0
    assert len(sites["vertex"]) == len(layer)
    assert all(np.allclose(s.normal, [0.0, 0.0, 1.0]) for s in sites["vertex"])
    assert all(np.allclose(s.normal, [0.0, 0.0, 1.0]) for s in sites["edge"])


def test_compute_surface_site_candidates_empty_for_planar_layer() -> None:
    """Perfectly planar layers have no 3D hull; callers must use planar fallback."""
    layer = graphene(size=(2, 2, 1), vacuum=8.0)
    # Flatten numerically (graphene is already planar).
    pos = layer.get_positions()
    pos[:, 2] = pos[0, 2]
    layer.set_positions(pos)
    sites = compute_surface_site_candidates(layer)
    assert count_site_candidates(sites) == 0


def test_planar_layer_site_candidates_empty_layer() -> None:
    empty = Atoms()
    sites = planar_layer_site_candidates(empty)
    assert count_site_candidates(sites) == 0
