"""Hierarchical (core + fragment) surface deposition and validation."""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import fcc111
from ase_ga.utilities import closest_distances_generator

from scgo.cluster_adsorbate.hierarchical import build_hierarchical_core_fragment_cluster
from scgo.cluster_adsorbate.validation import validate_combined_cluster_structure
from scgo.surface.config import SurfaceSystemConfig, describe_surface_config
from scgo.surface.deposition import create_deposited_cluster
from scgo.surface.fragment_templates import build_default_fragment_template
from scgo.system_types import validate_adsorbate_definition
from tests.test_utils import assert_supported_cluster_binding


def _small_slab() -> SurfaceSystemConfig:
    slab = fcc111("Pt", size=(2, 2, 2), vacuum=8.0, orthogonal=True)
    slab.pbc = [True, True, True]
    return SurfaceSystemConfig(
        slab=slab,
        adsorption_height_min=1.5,
        adsorption_height_max=3.0,
        fix_all_slab_atoms=True,
        max_placement_attempts=400,
    )


def test_build_default_fragment_template_oh_dimer():
    frag = build_default_fragment_template(["O", "H", "O", "H"])
    assert frag is not None
    assert frag.get_chemical_symbols() == ["O", "H", "O", "H"]


def test_describe_surface_config_smoke():
    cfg = _small_slab()
    s = describe_surface_config(cfg)
    assert "adsorption_height" in s
    assert "n_slab=" in s


def test_validate_partition_core_adsorbate():
    validate_adsorbate_definition(
        system_type="surface_cluster_adsorbate",
        composition=["Pt", "Pt", "Pt", "Pt", "Pt", "O", "H", "O", "H"],
        adsorbate_definition={
            "adsorbate_symbols": ["O", "H", "O", "H"],
            "core_symbols": ["Pt", "Pt", "Pt", "Pt", "Pt"],
            "adsorbate_fragment_lengths": [2, 2],
        },
        context="test",
    )


def test_validate_rejects_bad_partition():
    with pytest.raises(ValueError, match="composition|partition|adsorbate"):
        validate_adsorbate_definition(
            system_type="surface_cluster_adsorbate",
            composition=["Pt", "Pt", "Pt", "O"],
            adsorbate_definition={
                "adsorbate_symbols": ["O", "H"],
                "core_symbols": ["Pt", "Pt", "Pt"],
                "adsorbate_fragment_lengths": [2],
            },
            context="test",
        )


def test_validate_accepts_wrong_list_order_with_matching_multiset():
    validate_adsorbate_definition(
        system_type="gas_cluster_adsorbate",
        composition=["O", "H", "Pt", "Pt", "Pt"],
        adsorbate_definition={
            "core_symbols": ["Pt", "Pt", "Pt"],
            "adsorbate_symbols": ["O", "H"],
            "adsorbate_fragment_lengths": [2],
        },
        context="test",
    )


def test_hierarchical_deposition_ordering_and_slab_prefix():
    cfg = _small_slab()
    slab = cfg.slab
    n_slab = len(slab)
    mobile = ["Pt", "Pt", "Pt", "O", "H", "O", "H"]
    ads_def = {
        "adsorbate_symbols": ["O", "H", "O", "H"],
        "core_symbols": ["Pt", "Pt", "Pt"],
        "adsorbate_fragment_lengths": [2, 2],
    }
    rng = np.random.default_rng(2026)
    blmin = closest_distances_generator(
        list({int(z) for z in slab.numbers} | {78, 8, 1}),
        ratio_of_covalent_radii=0.7,
    )
    oh = build_default_fragment_template(["O", "H"])
    assert oh is not None
    out = create_deposited_cluster(
        mobile,
        slab,
        blmin,
        rng,
        cfg,
        adsorbate_definition=ads_def,
        adsorbate_fragment_template=[oh, oh.copy()],
    )
    assert out is not None
    sym = out.get_chemical_symbols()
    assert sym[:n_slab] == list(slab.get_chemical_symbols())
    assert sym[n_slab:] == mobile
    assert_supported_cluster_binding(
        out,
        cfg,
        n_core_mobile=len(ads_def["core_symbols"]),
    )


def test_surface_deposition_accepts_empty_core_symbols():
    slab = fcc111("Pt", size=(2, 2, 2), vacuum=8.0, orthogonal=True)
    slab.pbc = [True, True, True]
    cfg = SurfaceSystemConfig(
        slab=slab,
        adsorption_height_min=3.0,
        adsorption_height_max=4.5,
        fix_all_slab_atoms=True,
        max_placement_attempts=1000,
    )
    slab = cfg.slab
    n_slab = len(slab)
    mobile = ["O", "H", "O", "H"]
    ads_def = {
        "adsorbate_symbols": ["O", "H", "O", "H"],
        "core_symbols": [],
        "adsorbate_fragment_lengths": [2, 2],
    }
    rng = np.random.default_rng(2027)
    blmin = closest_distances_generator(
        list({int(z) for z in slab.numbers} | {8, 1}),
        ratio_of_covalent_radii=0.7,
    )
    oh = build_default_fragment_template(["O", "H"])
    assert oh is not None
    out = create_deposited_cluster(
        mobile,
        slab,
        blmin,
        rng,
        cfg,
        adsorbate_definition=ads_def,
        adsorbate_fragment_template=[oh, oh.copy()],
    )
    assert out is not None
    sym = out.get_chemical_symbols()
    assert sym[:n_slab] == list(slab.get_chemical_symbols())
    assert sym[n_slab:] == mobile


def test_gas_hierarchical_core_fragment_smoke():
    """Gas-phase hierarchical build matches core then fragment symbol order."""
    mobile = ["Pt", "Pt", "O", "H"]
    ads_def = {
        "core_symbols": ["Pt", "Pt"],
        "adsorbate_symbols": ["O", "H"],
        "adsorbate_fragment_lengths": [2],
    }
    rng = np.random.default_rng(2026)
    tmpl = build_default_fragment_template(["O", "H"])
    assert tmpl is not None
    out = build_hierarchical_core_fragment_cluster(
        mobile,
        ads_def,
        rng,
        "**/*.db",
        tmpl,
        None,
        cluster_init_vacuum=8.0,
        init_mode="random_spherical",
        max_placement_attempts=400,
    )
    assert out is not None
    assert out.get_chemical_symbols()[:2] == ["Pt", "Pt"]
    assert out.get_chemical_symbols()[2:] == ["O", "H"]
    ok, err = validate_combined_cluster_structure(out)
    assert ok, err
