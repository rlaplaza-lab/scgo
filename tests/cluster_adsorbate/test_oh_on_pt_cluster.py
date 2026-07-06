"""Tests for OH placement and relaxation on small Pt clusters (EMT)."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT
from numpy.random import default_rng

from scgo.cluster_adsorbate import (
    ClusterAdsorbateConfig,
    attach_fix_bond_lengths,
    combine_core_adsorbate,
    place_fragment_on_cluster,
    relax_metal_cluster_with_adsorbate,
)
from scgo.utils.ts_provenance import CLUSTER_ADSORBATE_OUTPUT_SCHEMA_VERSION
from tests.test_utils import assert_pt_o_distance_reasonable

_OH_BOND = 0.96


def _oh_template(bond_length: float = _OH_BOND) -> Atoms:
    return Atoms(
        symbols=["O", "H"],
        positions=np.array([[0.0, 0.0, 0.0], [bond_length, 0.0, 0.0]], dtype=float),
    )


def _pt_linear_dimer() -> Atoms:
    return Atoms(
        "Pt2",
        positions=[[0.0, 0.0, 0.0], [2.3, 0.0, 0.0]],
        cell=[18.0, 18.0, 18.0],
        pbc=False,
    )


def _pt_triangle() -> Atoms:
    return Atoms(
        "Pt3",
        positions=[
            [0.0, 0.0, 0.0],
            [2.3, 0.0, 0.0],
            [1.15, 2.0, 0.0],
        ],
        cell=[20.0, 20.0, 20.0],
        pbc=False,
    )


def test_place_oh_succeeds_pt3_fixed_seed() -> None:
    core = _pt_triangle()
    rng = default_rng(42)
    cfg = ClusterAdsorbateConfig(max_placement_attempts=200)
    oh = place_fragment_on_cluster(
        core, _oh_template(), rng, cfg, anchor_index=0, bond_axis=(0, 1)
    )
    assert oh is not None
    assert oh.get_chemical_symbols() == ["O", "H"]
    combined = combine_core_adsorbate(core, oh)
    assert len(combined) == 5


def test_attach_fix_bond_lengths_on_oh() -> None:
    a = Atoms("OH", positions=[[0, 0, 0], [0.96, 0, 0]], cell=[10, 10, 10], pbc=False)
    attach_fix_bond_lengths(a, [(0, 1)])
    with pytest.raises(ValueError, match="Invalid"):
        attach_fix_bond_lengths(a, [(0, 5)])


def test_oh_relax_reports_connected_structure_emt() -> None:
    core = _pt_linear_dimer()
    d0 = _OH_BOND
    n = np.array([0.0, 0.0, 1.0])
    o_pos = np.array([1.15, 0.0, 1.55])
    h_pos = o_pos + d0 * n
    pre = Atoms(
        "OH", positions=np.vstack([o_pos, h_pos]), cell=core.get_cell(), pbc=False
    )

    relaxed, info = relax_metal_cluster_with_adsorbate(
        core,
        EMT(),
        _oh_template(),
        preplaced=pre,
        anchor_index=0,
        bond_axis=(0, 1),
        fix_core=True,
        fmax=0.15,
        steps=120,
        config=ClusterAdsorbateConfig(cell_margin=8.0),
    )
    assert np.isfinite(info["final_energy"])
    assert info["structure_ok_initial"] is True
    assert info["structure_ok_final"] is True
    assert "oh_distance" in info
    prov = info["provenance"]
    assert (
        prov["cluster_adsorbate_schema_version"]
        == CLUSTER_ADSORBATE_OUTPUT_SCHEMA_VERSION
    )
    assert prov["calculator_class"] == "EMT"
    assert prov["n_frag"] == 2
    assert_pt_o_distance_reasonable(relaxed, pt_idx=0, o_idx=2)


def test_relax_metal_cluster_with_adsorbate_oh_placement_emt() -> None:
    core = _pt_triangle()
    rng = default_rng(123)
    cfg = ClusterAdsorbateConfig(
        max_placement_attempts=300, height_min=0.85, height_max=2.0
    )
    relaxed, info = relax_metal_cluster_with_adsorbate(
        core,
        EMT(),
        _oh_template(),
        rng=rng,
        config=cfg,
        anchor_index=0,
        bond_axis=(0, 1),
        fix_core=True,
        fmax=0.2,
        steps=100,
    )
    assert len(relaxed) == 5
    assert np.isfinite(info["final_energy"])
    assert info["structure_ok_initial"] is True
    assert info["structure_ok_final"] is True
    assert_pt_o_distance_reasonable(relaxed, pt_idx=0, o_idx=3)


def test_preplaced_wrong_length_raises() -> None:
    core = _pt_linear_dimer()
    bad = Atoms("O", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=False)
    with pytest.raises(ValueError, match="fragment_template"):
        relax_metal_cluster_with_adsorbate(
            core, EMT(), _oh_template(), preplaced=bad, anchor_index=0, bond_axis=(0, 1)
        )
