"""Hierarchical (core + fragment) surface deposition and validation."""

from __future__ import annotations

import pytest
from ase.build import fcc111
from ase.db import connect
from ase_ga.utilities import closest_distances_generator

from scgo.cluster_adsorbate.hierarchical import build_hierarchical_core_fragment_cluster
from scgo.cluster_adsorbate.validation import validate_combined_cluster_structure
from scgo.exceptions import SCGOValidationError
from scgo.initialization import create_initial_cluster
from scgo.metadata.atoms import get_tag, set_tags
from scgo.metadata.db_stamp import stamp_db
from scgo.surface.config import SurfaceSystemConfig, describe_surface_config
from scgo.surface.deposition import create_deposited_cluster
from scgo.surface.fragment_templates import build_default_fragment_template
from scgo.surface.partition import prepare_slab_search_surface_config
from scgo.surface.presets import make_n_doped_graphite_surface_config
from scgo.system_types import (
    AdsorbateDefinition,
    resolve_mobile_composition,
    validate_adsorbate_definition,
)
from tests.helpers import assert_supported_cluster_binding


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
        adsorbate_definition=AdsorbateDefinition(
            adsorbate_symbols=["O", "H", "O", "H"],
            core_symbols=["Pt", "Pt", "Pt", "Pt", "Pt"],
            adsorbate_fragment_lengths=[2, 2],
        ),
        context="test",
    )


def test_validate_rejects_bad_partition():
    with pytest.raises(SCGOValidationError, match="composition|partition|adsorbate"):
        validate_adsorbate_definition(
            system_type="surface_cluster_adsorbate",
            composition=["Pt", "Pt", "Pt", "O"],
            adsorbate_definition=AdsorbateDefinition(
                adsorbate_symbols=["O", "H"],
                core_symbols=["Pt", "Pt", "Pt"],
                adsorbate_fragment_lengths=[2],
            ),
            context="test",
        )


def test_validate_accepts_wrong_list_order_with_matching_multiset():
    validate_adsorbate_definition(
        system_type="gas_cluster_adsorbate",
        composition=["O", "H", "Pt", "Pt", "Pt"],
        adsorbate_definition=AdsorbateDefinition(
            core_symbols=["Pt", "Pt", "Pt"],
            adsorbate_symbols=["O", "H"],
            adsorbate_fragment_lengths=[2],
        ),
        context="test",
    )


def test_validate_reconciles_wrong_preset_core_surface():
    """Surface adsorbate runs reconcile core_symbols from full mobile formulas.

    Reconciliation is non-mutating: callers must use the returned definition
    rather than relying on in-place mutation of the input dict.
    """
    from scgo.utils.helpers import get_composition_counts

    ads_def = AdsorbateDefinition(
        core_symbols=["Pt"] * 4 + ["O"],
        adsorbate_symbols=["O", "H"],
        adsorbate_fragment_lengths=[2],
    )
    composition = ["Pt"] * 5 + ["O", "O", "H"]
    validate_adsorbate_definition(
        system_type="surface_cluster_adsorbate",
        composition=composition,
        adsorbate_definition=ads_def,
        context="test",
    )
    _mobile, reconciled = resolve_mobile_composition(
        composition, ads_def, context="test"
    )
    assert get_composition_counts(reconciled.core_symbols) == get_composition_counts(
        ["Pt"] * 5 + ["O"]
    )
    # The original definition is left untouched (reconciliation is non-mutating).
    assert get_composition_counts(ads_def.core_symbols) == get_composition_counts(
        ["Pt"] * 4 + ["O"]
    )


def test_hierarchical_deposition_ordering_and_slab_prefix(rng):
    cfg = _small_slab()
    slab = cfg.slab
    n_slab = len(slab)
    mobile = ["Pt", "Pt", "Pt", "O", "H", "O", "H"]
    ads_def = AdsorbateDefinition(
        adsorbate_symbols=["O", "H", "O", "H"],
        core_symbols=["Pt", "Pt", "Pt"],
        adsorbate_fragment_lengths=[2, 2],
    )
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
        n_core_mobile=len(ads_def.core_symbols),
    )


def test_surface_deposition_accepts_empty_core_symbols(rng):
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
    ads_def = AdsorbateDefinition(
        adsorbate_symbols=["O", "H", "O", "H"],
        core_symbols=[],
        adsorbate_fragment_lengths=[2, 2],
    )
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


def test_surface_deposition_empty_core_on_graphite(rng):
    """Planar graphite top layers need planar site fallback (no 3D hull)."""
    cfg = make_n_doped_graphite_surface_config(
        slab_layers=3, slab_repeat_xy=2, n_dopants=1, seed=7
    )
    cfg, _part = prepare_slab_search_surface_config(cfg)
    cfg = SurfaceSystemConfig(
        slab=cfg.slab,
        name=cfg.name,
        fix_all_slab_atoms=False,
        n_relax_top_slab_layers=1,
        max_placement_attempts=40,
    )
    ads_def = AdsorbateDefinition(
        adsorbate_symbols=["O", "H"],
        core_symbols=[],
        adsorbate_fragment_lengths=[2],
    )
    blmin = closest_distances_generator(
        list({int(z) for z in cfg.slab.numbers} | {8, 1}),
        ratio_of_covalent_radii=0.7,
    )
    oh = build_default_fragment_template(["O", "H"])
    assert oh is not None
    out = create_deposited_cluster(
        ["O", "H"],
        cfg.slab,
        blmin,
        rng,
        cfg,
        adsorbate_definition=ads_def,
        adsorbate_fragment_template=[oh],
    )
    assert out is not None
    assert len(out) == len(cfg.slab) + 2
    assert out.get_chemical_symbols()[-2:] == ["O", "H"]


def test_gas_hierarchical_core_fragment_smoke(rng):
    """Gas-phase hierarchical build matches core then fragment symbol order."""
    ads_def = AdsorbateDefinition(
        core_symbols=["Pt", "Pt"],
        adsorbate_symbols=["O", "H"],
        adsorbate_fragment_lengths=[2],
    )
    tmpl = build_default_fragment_template(["O", "H"])
    assert tmpl is not None
    out = build_hierarchical_core_fragment_cluster(
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


def test_surface_deposition_reuses_exact_prior_minimum(rng, tmp_path):
    """A prior same-composition gas-phase minimum is reused above the slab.

    Regression guard for the exact-match tier: the reused seed must not leak the
    source ``final_unique_minimum`` tag into the new combined slab structure.
    """
    cfg = _small_slab()
    slab = cfg.slab
    n_slab = len(slab)
    mobile = ["Pt"] * 5

    pt5 = create_initial_cluster(mobile, mode="random_spherical", rng=rng)
    db_path = tmp_path / "Pt5_searches" / "run_001" / "cluster.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    set_tags(pt5, final_unique_minimum=True, raw_score=-15.0)
    with connect(db_path) as dbobj:
        dbobj.write(
            pt5,
            relaxed=True,
            gaid=1,
            key_value_pairs={"raw_score": -15.0, "final_unique_minimum": True},
        )
    stamp_db(db_path)
    glob = str(tmp_path / "**" / "*.db")

    blmin = closest_distances_generator(
        list({int(z) for z in slab.numbers}), ratio_of_covalent_radii=0.7
    )
    cfg = SurfaceSystemConfig(
        slab=slab,
        adsorption_height_min=1.5,
        adsorption_height_max=3.0,
        fix_all_slab_atoms=True,
        max_placement_attempts=200,
        init_mode="smart",
    )
    out = create_deposited_cluster(
        mobile,
        slab,
        blmin,
        rng,
        cfg,
        previous_search_glob=glob,
        allocation=("exact", None),
    )
    assert out is not None
    sym = out.get_chemical_symbols()
    assert sym[:n_slab] == list(slab.get_chemical_symbols())
    assert sym[n_slab:] == mobile
    assert get_tag(out, "final_unique_minimum") is None
