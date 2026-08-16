"""Unit tests for float / dict connectivity-factor specs."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from scgo.cluster_adsorbate import ClusterAdsorbateConfig
from scgo.exceptions import SCGOValidationError
from scgo.initialization.atomic_radii import get_covalent_radius
from scgo.initialization.geometry_helpers import _bonded_pairs, is_cluster_connected
from scgo.initialization.initialization_config import CONNECTIVITY_FACTOR
from scgo.surface import make_surface_config
from scgo.system_types import resolve_connectivity_factor
from scgo.system_types.connectivity_factor import (
    bond_threshold_matrix,
    connectivity_factor_for_json,
    format_connectivity_factor,
    max_connectivity_scale,
    min_connectivity_scale,
    normalize_connectivity_factor,
    pair_bond_threshold,
)


def test_normalize_float() -> None:
    n = normalize_connectivity_factor(1.8)
    assert n.global_factor == 1.8
    assert n.element_items == ()
    assert n.pair_items == ()
    assert format_connectivity_factor(n) == "1.80"


def test_normalize_element_dict() -> None:
    n = normalize_connectivity_factor({"Pt": 1.8, "C": 1.4})
    assert n.global_factor is None
    assert n.element_factors == {"C": 1.4, "Pt": 1.8}
    r_pt, r_c = get_covalent_radius("Pt"), get_covalent_radius("C")
    assert pair_bond_threshold(r_pt, r_c, "Pt", "C", n) == pytest.approx(
        r_pt * 1.8 + r_c * 1.4
    )
    # Missing symbol falls back to CONNECTIVITY_FACTOR.
    assert pair_bond_threshold(r_pt, r_pt, "Pt", "Au", n) == pytest.approx(
        r_pt * 1.8 + get_covalent_radius("Au") * CONNECTIVITY_FACTOR
    )


def test_normalize_pair_tuple_and_string() -> None:
    r_pt, r_c = get_covalent_radius("Pt"), get_covalent_radius("C")
    for key in (("Pt", "C"), ("C", "Pt"), "Pt-C", "C-Pt"):
        n = normalize_connectivity_factor({key: 2.0})
        assert pair_bond_threshold(r_pt, r_c, "Pt", "C", n) == pytest.approx(
            (r_pt + r_c) * 2.0
        )


def test_mixed_pair_overrides_element() -> None:
    r_pt, r_c = get_covalent_radius("Pt"), get_covalent_radius("C")
    n = normalize_connectivity_factor({"Pt": 1.4, "C": 1.4, "Pt-C": 2.0})
    assert pair_bond_threshold(r_pt, r_c, "Pt", "C", n) == pytest.approx(
        (r_pt + r_c) * 2.0
    )
    assert pair_bond_threshold(r_pt, r_pt, "Pt", "Pt", n) == pytest.approx(
        r_pt * 1.4 + r_pt * 1.4
    )
    mat = bond_threshold_matrix(np.array([r_pt, r_c, r_pt]), ["Pt", "C", "Pt"], n)
    assert mat[0, 1] == pytest.approx((r_pt + r_c) * 2.0)
    assert mat[0, 2] == pytest.approx(r_pt * 1.4 + r_pt * 1.4)


def test_normalize_rejects_invalid() -> None:
    with pytest.raises(SCGOValidationError):
        normalize_connectivity_factor({})
    with pytest.raises(SCGOValidationError):
        normalize_connectivity_factor({"Pt": -1.0})
    with pytest.raises(SCGOValidationError):
        normalize_connectivity_factor({"Pt-C-O": 1.5})
    with pytest.raises(SCGOValidationError):
        normalize_connectivity_factor("1.4")  # type: ignore[arg-type]


def test_min_max_scale() -> None:
    n = normalize_connectivity_factor({"Pt": 1.2, "C": 1.8, "Pt-C": 2.5})
    assert min_connectivity_scale(n) == pytest.approx(1.2)
    assert max_connectivity_scale(n) == pytest.approx(2.5)


def test_json_roundtrip_pair_keys() -> None:
    raw = {"Pt": 1.4, ("C", "Pt"): 1.8}
    n = normalize_connectivity_factor(raw)
    as_json = connectivity_factor_for_json(n)
    assert as_json == {"C-Pt": 1.8, "Pt": 1.4}
    assert normalize_connectivity_factor(as_json) == n


def test_resolve_prefers_explicit_dict_over_config() -> None:
    from ase.build import fcc111

    ca = ClusterAdsorbateConfig(structure_connectivity_factor=2.5)
    surf = make_surface_config(
        fcc111("Pt", size=(2, 2, 2), vacuum=6.0, orthogonal=True)
    )
    explicit = {"Pt": 1.1}
    resolved = resolve_connectivity_factor(
        explicit, cluster_adsorbate_config=ca, surface_config=surf
    )
    assert resolved == normalize_connectivity_factor(explicit)


def test_bonded_pairs_honor_pair_override() -> None:
    """Pt–C distance bonded only under a pair override, not under default 1.4."""
    r_pt, r_c = get_covalent_radius("Pt"), get_covalent_radius("C")
    # Midway between 1.4× and 1.8× thresholds.
    d = (r_pt + r_c) * 1.6
    atoms = Atoms(
        "PtC",
        positions=[[0.0, 0.0, 0.0], [d, 0.0, 0.0]],
        cell=[30.0, 30.0, 30.0],
        pbc=False,
    )
    assert not is_cluster_connected(atoms, CONNECTIVITY_FACTOR)
    assert is_cluster_connected(atoms, {"Pt-C": 1.8})
    i_idx, j_idx = _bonded_pairs(atoms, {"Pt-C": 1.8}, use_mic=False)
    assert len(i_idx) == 1
