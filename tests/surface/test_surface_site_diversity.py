"""Surface hull-site diversity for adsorbate placement on slab-supported clusters."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase_ga.utilities import closest_distances_generator
from numpy.random import default_rng

from scgo.cluster_adsorbate import ClusterAdsorbateConfig, place_fragment_on_cluster
from scgo.surface.deposition import create_deposited_cluster
from tests.cluster_adsorbate.test_site_diversity import _pt_tetrahedron


def _o_template() -> Atoms:
    return Atoms("O", positions=[[0.0, 0.0, 0.0]], pbc=False)


@pytest.mark.slow
def test_surface_fragment_placement_spans_vertex_edge_and_facet_sites(
    surface_config_pt111,
) -> None:
    """Deposit a Pt cluster on a slab, then verify hull-site diversity for O placement."""
    cfg = surface_config_pt111
    slab = cfg.slab
    blmin = closest_distances_generator(
        list({int(z) for z in slab.numbers} | {78, 8}),
        ratio_of_covalent_radii=0.7,
    )
    rng = default_rng(2026)
    deposited = create_deposited_cluster(
        ["Pt", "Pt", "Pt", "Pt"],
        slab,
        blmin,
        rng,
        cfg,
    )
    assert deposited is not None

    # Use a tetrahedral Pt4 core (same geometry as gas site-diversity reference).
    core = _pt_tetrahedron()
    ads_cfg = ClusterAdsorbateConfig(max_placement_attempts=400)
    batch_counts: dict[str, int] = {"vertex": 0, "edge": 0, "facet": 0}
    observed: set[str] = set()

    for _ in range(40):
        metadata: dict[str, str] = {}
        frag = place_fragment_on_cluster(
            core,
            _o_template(),
            rng,
            ads_cfg,
            anchor_index=0,
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
