"""Per-system-type init must produce structures that pass validate_minimum_structure."""

from __future__ import annotations

import pytest
from ase import Atoms

from scgo.algorithms.ga_common import maybe_apply_mobile_core_ads_tags
from scgo.exceptions import SCGOValidationError
from scgo.initialization import create_initial_cluster
from scgo.minima_search.core import (
    _create_gas_cluster_adsorbate_initial_atoms,
    _create_surface_initialized_atoms,
)
from scgo.surface.config import SurfaceSystemConfig
from scgo.system_types import (
    AdsorbateDefinition,
    resolve_mobile_composition,
    validate_minimum_structure,
)

N_CANDIDATES = 4
MAX_GENERATION_ATTEMPTS = 60

ADS_DEF = AdsorbateDefinition(
    core_symbols=["Pt", "Pt"],
    adsorbate_symbols=["O", "H"],
    adsorbate_fragment_lengths=[2],
)
SURFACE_ADS_DEF = AdsorbateDefinition(
    core_symbols=[],
    adsorbate_symbols=["O", "H"],
    adsorbate_fragment_lengths=[2],
)
OH_TEMPLATE = Atoms("OH", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.96]])


@pytest.fixture
def slab_search_cfg(pt_slab_small):
    """Slab-as-target config that relaxes the top layer (not fully fixed)."""
    return SurfaceSystemConfig(
        slab=pt_slab_small,
        adsorption_height_min=1.6,
        adsorption_height_max=2.2,
        fix_all_slab_atoms=False,
        n_relax_top_slab_layers=1,
        comparator_use_mic=False,
        max_placement_attempts=400,
    )


def _build(system_type, *, rng, cluster_cfg, slab_cfg, db_glob):
    if system_type == "gas_cluster":
        return create_initial_cluster(
            ["Pt", "Pt", "Pt"], rng=rng, previous_search_glob=db_glob
        )
    if system_type == "gas_cluster_adsorbate":
        return _create_gas_cluster_adsorbate_initial_atoms(
            composition=["Pt", "Pt", "Pt"],
            rng=rng,
            adsorbate_definition=ADS_DEF,
            adsorbate_fragment_template=OH_TEMPLATE,
            previous_search_glob=db_glob,
            max_hierarchical_attempts=500,
        )
    if system_type == "surface_cluster":
        return _create_surface_initialized_atoms(
            composition=["Pt", "Pt", "Pt"],
            surface_config=cluster_cfg,
            rng=rng,
            system_type="surface_cluster",
        )
    if system_type == "surface_cluster_adsorbate":
        comp, ad = resolve_mobile_composition(["Pt", "Pt", "O", "H"], ADS_DEF)
        return _create_surface_initialized_atoms(
            composition=comp,
            surface_config=cluster_cfg,
            rng=rng,
            adsorbate_definition=ad,
            adsorbate_fragment_template=OH_TEMPLATE,
            system_type="surface_cluster_adsorbate",
        )
    if system_type == "surface":
        return _create_surface_initialized_atoms(
            composition=[],
            surface_config=slab_cfg,
            rng=rng,
            system_type="surface",
        )
    if system_type == "surface_adsorbate":
        return _create_surface_initialized_atoms(
            composition=["O", "H"],
            surface_config=slab_cfg,
            rng=rng,
            adsorbate_definition=SURFACE_ADS_DEF,
            adsorbate_fragment_template=OH_TEMPLATE,
            system_type="surface_adsorbate",
        )
    raise AssertionError(f"unknown system_type: {system_type!r}")


def _validation_kwargs(system_type, *, cluster_cfg, slab_cfg):
    if system_type == "gas_cluster":
        return {"system_type": "gas_cluster"}
    if system_type == "gas_cluster_adsorbate":
        return {"system_type": "gas_cluster_adsorbate", "adsorbate_definition": ADS_DEF}
    if system_type == "surface_cluster":
        return {"system_type": "surface_cluster", "surface_config": cluster_cfg}
    if system_type == "surface_cluster_adsorbate":
        return {
            "system_type": "surface_cluster_adsorbate",
            "surface_config": cluster_cfg,
            "adsorbate_definition": ADS_DEF,
        }
    if system_type == "surface":
        return {"system_type": "surface", "surface_config": slab_cfg}
    if system_type == "surface_adsorbate":
        return {
            "system_type": "surface_adsorbate",
            "surface_config": slab_cfg,
            "adsorbate_definition": SURFACE_ADS_DEF,
        }
    raise AssertionError(f"unknown system_type: {system_type!r}")


def _apply_runner_tags(atoms, system_type, *, cluster_cfg, slab_cfg):
    """Apply the same core/adsorbate tags ``run_trials`` sets before the gate."""
    ads_def = None
    composition = None
    n_slab = 0
    if system_type == "gas_cluster_adsorbate":
        ads_def = ADS_DEF
        composition = ["Pt", "Pt", "O", "H"]
    elif system_type == "surface_cluster_adsorbate":
        ads_def = ADS_DEF
        composition, _ = resolve_mobile_composition(["Pt", "Pt", "O", "H"], ADS_DEF)
        n_slab = len(cluster_cfg.slab)
    elif system_type == "surface_adsorbate":
        ads_def = SURFACE_ADS_DEF
        composition = ["O", "H"]
        n_slab = len(slab_cfg.slab)
    else:
        return
    maybe_apply_mobile_core_ads_tags(atoms, n_slab, composition, ads_def, system_type)


def _collect_valid(system_type, *, rng, cluster_cfg, slab_cfg, db_glob, n):
    valid = []
    attempts = 0
    while len(valid) < n and attempts < MAX_GENERATION_ATTEMPTS:
        attempts += 1
        atoms = _build(
            system_type,
            rng=rng,
            cluster_cfg=cluster_cfg,
            slab_cfg=slab_cfg,
            db_glob=db_glob,
        )
        assert atoms is not None
        _apply_runner_tags(
            atoms, system_type, cluster_cfg=cluster_cfg, slab_cfg=slab_cfg
        )
        kwargs = _validation_kwargs(
            system_type, cluster_cfg=cluster_cfg, slab_cfg=slab_cfg
        )
        try:
            validate_minimum_structure(atoms, **kwargs)
        except SCGOValidationError:
            continue
        valid.append(atoms)
    assert len(valid) == n, (
        f"{system_type} init flow failed to produce {n} gate-valid candidates "
        f"within {MAX_GENERATION_ATTEMPTS} attempts (init flow may be broken)."
    )
    return valid


@pytest.mark.parametrize(
    "system_type",
    [
        "gas_cluster",
        "gas_cluster_adsorbate",
        "surface_cluster",
        "surface_cluster_adsorbate",
        "surface",
        "surface_adsorbate",
    ],
)
def test_initial_structure_passes_system_type_gate(
    system_type, rng, surface_config_pt111, slab_search_cfg, tmp_path
):
    db_glob = str(tmp_path / "*.db")
    _collect_valid(
        system_type,
        rng=rng,
        cluster_cfg=surface_config_pt111,
        slab_cfg=slab_search_cfg,
        db_glob=db_glob,
        n=N_CANDIDATES,
    )
