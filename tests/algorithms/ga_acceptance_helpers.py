"""Shared Pt3+OH parent builders for GA acceptance tests."""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.build import fcc111

from scgo.algorithms.ga_common import apply_mobile_core_ads_tags
from scgo.cluster_adsorbate.hierarchical import build_hierarchical_core_fragment_cluster
from scgo.initialization.atomic_radii import build_blmin, build_blmin_from_zs
from scgo.surface.config import SurfaceSystemConfig
from scgo.surface.deposition import create_deposited_cluster
from scgo.system_types import AdsorbateDefinition


def _oh_template() -> Atoms:
    return Atoms("OH", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.96]], pbc=False)


def _oh_adsorbate_definition(n_oh: int) -> AdsorbateDefinition:
    ads_symbols = ["O", "H"] * n_oh
    return AdsorbateDefinition(
        core_symbols=["Pt", "Pt", "Pt"],
        adsorbate_symbols=ads_symbols,
        adsorbate_fragment_lengths=[2] * n_oh,
        fragment_bond_axis=[0, 1],
    )


def _gas_pt3_oh_parent(
    *, seed: int = 101, n_oh: int = 1
) -> tuple[Atoms, list[str], dict, AdsorbateDefinition]:
    ads = _oh_adsorbate_definition(n_oh)
    templates = [_oh_template() for _ in range(n_oh)]
    built = build_hierarchical_core_fragment_cluster(
        ads,
        np.random.default_rng(seed),
        previous_search_glob=".__scgo_no_prior_runs__/**/*.db",
        fragment_templates=templates,
        cluster_adsorbate_config=None,
        max_placement_attempts=200 if n_oh == 1 else 500,
    )
    assert built is not None
    parent = built.copy()
    parent.set_cell([20, 20, 20])
    parent.set_pbc(False)
    apply_mobile_core_ads_tags(
        parent, n_slab=0, n_core=3, ads_fragment_lengths=[2] * n_oh
    )
    comp = list(ads.core_symbols) + list(ads.adsorbate_symbols)
    blmin = build_blmin_from_zs(parent.numbers, ratio=0.7)
    return parent, comp, blmin, ads


def _surface_pt3_oh_parent(
    *, seed: int = 3, n_oh: int = 1
) -> tuple[Atoms, list[str], dict, AdsorbateDefinition, int]:
    slab = fcc111("Pt", size=(4, 4, 2), vacuum=6.0, orthogonal=True)
    ads = _oh_adsorbate_definition(n_oh)
    comp = list(ads.core_symbols) + list(ads.adsorbate_symbols)
    templates = [_oh_template() for _ in range(n_oh)]
    cfg = SurfaceSystemConfig(
        slab=slab,
        adsorption_height_min=1.0,
        adsorption_height_max=3.0,
        max_placement_attempts=400 if n_oh == 1 else 600,
    )
    blmin = build_blmin(list(slab.get_chemical_symbols()) + comp, ratio=0.7)
    deposited = create_deposited_cluster(
        comp,
        slab,
        blmin,
        np.random.default_rng(seed),
        cfg,
        adsorbate_definition=ads,
        adsorbate_fragment_template=templates,
    )
    assert deposited is not None
    n_slab = len(slab)
    apply_mobile_core_ads_tags(
        deposited, n_slab=n_slab, n_core=3, ads_fragment_lengths=[2] * n_oh
    )
    return deposited, comp, blmin, ads, n_slab
