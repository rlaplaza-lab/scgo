"""Tests for convex-hull and planar-layer adsorption site discovery."""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.build import fcc111, graphene

from scgo.cluster_adsorbate.sites import (
    compute_surface_site_candidates,
    count_site_candidates,
    filter_sites_to_outward,
    planar_layer_site_candidates,
)


def _unique_in_plane_nn_pairs(layer: Atoms, axis: int = 2) -> set[tuple[int, int]]:
    """Unordered nearest-neighbour pairs of an in-plane projection."""
    pos = np.asarray(layer.get_positions(), dtype=float)
    in_plane = [i for i in (0, 1, 2) if i != axis]
    pairs: set[tuple[int, int]] = set()
    for i, pi in enumerate(pos):
        deltas = pos - pi
        d2 = deltas[:, in_plane[0]] ** 2 + deltas[:, in_plane[1]] ** 2
        d2[i] = np.inf
        j = int(np.argmin(d2))
        pairs.add((min(i, j), max(i, j)))
    return pairs


def test_planar_layer_site_candidates_for_graphene() -> None:
    layer = graphene(size=(2, 2, 1), vacuum=8.0)
    sites = planar_layer_site_candidates(layer, surface_normal_axis=2)
    assert count_site_candidates(sites) > 0
    assert len(sites["vertex"]) == len(layer)
    assert all(np.allclose(s.normal, [0.0, 0.0, 1.0]) for s in sites["vertex"])
    assert all(np.allclose(s.normal, [0.0, 0.0, 1.0]) for s in sites["edge"])

    # Every unique nearest-neighbour pair yields exactly one bridge site; the
    # nearest-neighbour relation is asymmetric, so pairs (i, j) with j < i must
    # not be discarded.
    expected_pairs = _unique_in_plane_nn_pairs(layer)
    assert len(sites["edge"]) == len(expected_pairs)
    anchors = [tuple(np.round(s.anchor, 6)) for s in sites["edge"]]
    assert len(set(anchors)) == len(anchors)


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


def test_filter_sites_to_outward_drops_downward_normals() -> None:
    """A slab-slice hull yields downward normals; the filter must remove them."""
    slab = fcc111("Pt", size=(2, 2, 3), vacuum=6.0, orthogonal=True)
    pos = slab.get_positions()
    z_top = float(np.max(pos[:, 2]))
    slice_indices = [i for i, p in enumerate(pos) if p[2] >= z_top - 2.5]
    top_slice = slab[slice_indices].copy()

    unfiltered = compute_surface_site_candidates(top_slice)
    assert count_site_candidates(unfiltered) > 0
    all_unfiltered = [c for entries in unfiltered.values() for c in entries]
    assert any(float(c.normal[2]) <= 0.0 for c in all_unfiltered)

    top_layer_z_min = float(np.min(top_slice.get_positions()[:, 2]))
    filtered = filter_sites_to_outward(
        unfiltered, axis=2, top_layer_z_min=top_layer_z_min
    )
    assert count_site_candidates(filtered) > 0
    all_filtered = [c for entries in filtered.values() for c in entries]
    assert all(float(c.normal[2]) > 0.0 for c in all_filtered)
    assert all(float(c.anchor[2]) >= top_layer_z_min for c in all_filtered)

    # The cached input dict must never be mutated in place.
    assert count_site_candidates(unfiltered) > count_site_candidates(filtered)
