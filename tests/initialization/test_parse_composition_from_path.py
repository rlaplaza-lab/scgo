"""Tests for *_searches path composition parsing."""

from __future__ import annotations

from scgo.initialization.candidate_discovery import _parse_composition_from_path


def test_parse_packed_adsorbate_metal_formula_accepted():
    """Packed ASE formulas parse as concatenated elemental symbols."""
    assert _parse_composition_from_path("/tmp/H2O2Pt5_searches/run_001") == (
        ["H", "H", "O", "O"] + ["Pt"] * 5
    )


def test_parse_component_path_key_skips_surface_name():
    assert _parse_composition_from_path(
        "/tmp/campaign/Pt5_OH_OH_graphite_searches/run_001"
    ) == ["Pt"] * 5 + ["O", "H", "O", "H"]


def test_parse_component_path_key_skips_default_slab_name():
    assert _parse_composition_from_path("Pt5_slab_searches") == ["Pt"] * 5


def test_parse_bimetallic_cluster_formula():
    assert _parse_composition_from_path("Au2Pt3_searches") == [
        "Au",
        "Au",
        "Pt",
        "Pt",
        "Pt",
    ]


def test_parse_unparseable_returns_none():
    assert _parse_composition_from_path("/tmp/results/run_001") is None
