"""Site diversity tests for gas-cluster adsorbate placement."""

from __future__ import annotations

from collections import Counter

import numpy as np
from ase import Atoms
from numpy.random import default_rng

from scgo.cluster_adsorbate import ClusterAdsorbateConfig, place_fragment_on_cluster
from scgo.cluster_adsorbate.placement import (
    _compute_surface_site_candidates,
    _select_site_type,
)


def _pt_tetrahedron() -> Atoms:
    return Atoms(
        "Pt4",
        positions=[
            [0.0, 0.0, 0.0],
            [2.3, 0.0, 0.0],
            [1.15, 1.98, 0.0],
            [1.15, 0.66, 1.86],
        ],
        cell=[20.0, 20.0, 20.0],
        pbc=False,
    )


def test_surface_site_candidate_counts_for_tetrahedron() -> None:
    core = _pt_tetrahedron()
    candidates = _compute_surface_site_candidates(core)
    assert len(candidates["vertex"]) == 4
    assert len(candidates["edge"]) == 6
    assert len(candidates["facet"]) == 4


def test_site_type_selector_prefers_underrepresented_types() -> None:
    rng = default_rng(12)
    local_counts = {"vertex": 4, "edge": 0, "facet": 1}
    batch_counts = {"vertex": 15, "edge": 0, "facet": 3}
    picks = [
        _select_site_type(["vertex", "edge", "facet"], rng, local_counts, batch_counts)
        for _ in range(200)
    ]
    counts = Counter(picks)
    assert counts["edge"] > counts["facet"] > counts["vertex"]


def test_fragment_placement_spans_vertex_edge_and_facet_sites() -> None:
    core = _pt_tetrahedron()
    o_tmpl = Atoms("O", positions=[[0.0, 0.0, 0.0]], cell=core.get_cell(), pbc=False)
    cfg = ClusterAdsorbateConfig(max_placement_attempts=400)
    rng = default_rng(2026)
    batch_counts: dict[str, int] = {"vertex": 0, "edge": 0, "facet": 0}
    observed: set[str] = set()
    for _ in range(40):
        metadata: dict[str, str] = {}
        frag = place_fragment_on_cluster(
            core,
            o_tmpl,
            rng,
            cfg,
            anchor_index=0,
            within_structure_site_counts={},
            batch_site_counts=batch_counts,
            placement_metadata=metadata,
        )
        assert frag is not None
        site_type = metadata.get("site_type")
        assert site_type is not None
        if site_type in batch_counts:
            batch_counts[site_type] += 1
            observed.add(site_type)
    assert observed == {"vertex", "edge", "facet"}
    arr = np.array(
        [batch_counts["vertex"], batch_counts["edge"], batch_counts["facet"]]
    )
    assert int(arr.max() - arr.min()) <= 8
