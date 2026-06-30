"""Multi-fragment placement records per-fragment site metadata."""

from __future__ import annotations

import json

from ase import Atoms
from numpy.random import default_rng

from scgo.cluster_adsorbate.hierarchical import build_hierarchical_core_fragment_cluster


def _oh() -> Atoms:
    return Atoms(
        symbols=["O", "H"],
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.96]],
        pbc=False,
    )


def test_two_oh_fragments_record_site_types_json() -> None:
    mobile = ["Pt", "Pt", "Pt", "O", "H", "O", "H"]
    ads_def = {
        "core_symbols": ["Pt", "Pt", "Pt"],
        "adsorbate_symbols": ["O", "H", "O", "H"],
        "adsorbate_fragment_lengths": [2, 2],
    }
    out = build_hierarchical_core_fragment_cluster(
        mobile,
        ads_def,
        default_rng(21),
        "**/*.db",
        [_oh(), _oh()],
        None,
        cluster_init_vacuum=8.0,
        init_mode="random_spherical",
        max_placement_attempts=600,
    )
    assert out is not None
    raw = out.info.get("adsorbate_site_types_json")
    assert raw is not None
    site_types = json.loads(raw)
    assert len(site_types) == 2
    assert all(isinstance(x, str) for x in site_types)
