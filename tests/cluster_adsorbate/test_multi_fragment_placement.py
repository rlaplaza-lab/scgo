"""Multi-fragment adsorbate placement and feasibility validation."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from numpy.random import default_rng

from scgo.cluster_adsorbate.feasibility import validate_adsorbate_placement_feasibility
from scgo.cluster_adsorbate.hierarchical import build_hierarchical_core_fragment_cluster
from scgo.cluster_adsorbate.validation import validate_combined_cluster_structure
from scgo.exceptions import SCGOValidationError
from scgo.metadata.atoms import get_tag
from scgo.system_types import (
    AdsorbateDefinition,
    build_adsorbate_definition_from_inputs,
    resolve_adsorbate_fragments,
)

_SITE_TYPE_VOCABULARY = {"vertex", "edge", "facet", "directional_fallback"}


def _oh_template(offset_x: float = 0.0) -> Atoms:
    return Atoms(
        symbols=["O", "H"],
        positions=[[offset_x, 0.0, 0.0], [offset_x, 0.0, 0.96]],
        pbc=False,
    )


def test_resolve_rejects_combined_template_for_multiple_fragments() -> None:
    combined = _oh_template(0.0) + _oh_template(2.2)
    ads_def = AdsorbateDefinition(
        core_symbols=["Pt", "Pt", "Pt"],
        adsorbate_symbols=["O", "H", "O", "H"],
        adsorbate_fragment_lengths=[2, 2],
    )
    with pytest.raises(SCGOValidationError, match="one combined adsorbate template"):
        resolve_adsorbate_fragments(combined, ads_def)


def test_gas_hierarchical_places_two_oh_separately() -> None:
    mobile = ["Pt", "Pt", "Pt", "O", "H", "O", "H"]
    ads_def = AdsorbateDefinition(
        core_symbols=["Pt", "Pt", "Pt"],
        adsorbate_symbols=["O", "H", "O", "H"],
        adsorbate_fragment_lengths=[2, 2],
    )
    rng = default_rng(11)
    out = build_hierarchical_core_fragment_cluster(
        ads_def,
        rng,
        "**/*.db",
        [_oh_template(), _oh_template()],
        None,
        cluster_init_vacuum=8.0,
        init_mode="random_spherical",
        max_placement_attempts=500,
    )
    assert out is not None
    assert out.get_chemical_symbols() == mobile
    o_indices = [i for i, s in enumerate(out.get_chemical_symbols()) if s == "O"]
    o_positions = out.get_positions()[o_indices]
    assert len(o_positions) == 2
    assert float(np.linalg.norm(o_positions[0] - o_positions[1])) > 1.0
    ok, err = validate_combined_cluster_structure(out)
    assert ok, err
    # Per-fragment site metadata must be recorded and drawn from the known
    # site-type vocabulary (folded from the deleted test_multi_fragment_site_metadata).
    site_types = get_tag(out, "adsorbate_site_types_json")
    assert site_types is not None
    assert len(site_types) == 2
    assert all(isinstance(t, str) and t in _SITE_TYPE_VOCABULARY for t in site_types)


def test_hierarchical_site_core_stays_metal_core(monkeypatch) -> None:
    """Later fragments must place on the original core, not the growing combined system."""
    from scgo.cluster_adsorbate import hierarchical as hier

    site_core_lens: list[int] = []
    real_place = hier.place_fragment_on_cluster

    def _tracking_place(*args, **kwargs):
        site_core = kwargs.get("site_core")
        assert site_core is not None
        site_core_lens.append(len(site_core))
        return real_place(*args, **kwargs)

    monkeypatch.setattr(hier, "place_fragment_on_cluster", _tracking_place)

    ads_def = AdsorbateDefinition(
        core_symbols=["Pt", "Pt", "Pt"],
        adsorbate_symbols=["O", "H", "O", "H"],
        adsorbate_fragment_lengths=[2, 2],
    )
    out = build_hierarchical_core_fragment_cluster(
        ads_def,
        default_rng(11),
        "**/*.db",
        [_oh_template(), _oh_template()],
        None,
        cluster_init_vacuum=8.0,
        init_mode="random_spherical",
        max_placement_attempts=500,
    )
    assert out is not None
    assert site_core_lens
    assert all(n == 3 for n in site_core_lens)


def test_feasibility_rejects_too_many_fragments_on_tiny_core() -> None:
    with pytest.raises(SCGOValidationError, match="heuristic site capacity"):
        validate_adsorbate_placement_feasibility(
            ["Pt"],
            [1, 1, 1],
            [_oh_template(), _oh_template(), _oh_template()],
        )


def test_build_adsorbate_definition_runs_feasibility() -> None:
    with pytest.raises(SCGOValidationError, match="heuristic site capacity"):
        build_adsorbate_definition_from_inputs(
            system_type="gas_cluster_adsorbate",
            composition=["Pt"],
            adsorbates=[_oh_template(), _oh_template(), _oh_template()],
            context="test",
        )
