"""GA operator acceptance for gas and surface cluster+adsorbate systems."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111
from ase_ga.utilities import atoms_too_close, atoms_too_close_two_sets

from scgo.algorithms.ga_common import (
    apply_mobile_core_ads_tags,
    create_mutation_operators,
)
from scgo.cluster_adsorbate.hierarchical import build_hierarchical_core_fragment_cluster
from scgo.initialization.atomic_radii import build_blmin, build_blmin_from_zs
from scgo.surface.config import SurfaceSystemConfig
from scgo.surface.deposition import create_deposited_cluster

MAX_MUTATION_ATTEMPTS = 40

_GAS_ADSORBATE_OPS = (
    "rattle",
    "anisotropic_rattle",
    "overlap_relief",
    "rotational",
    "mirror",
    "fragment_reposition",
)

_SURFACE_ADSORBATE_OPS = (
    "rattle",
    "anisotropic_rattle",
    "overlap_relief",
    "rotational",
    "mirror",
    "in_plane_slide_core",
    "fragment_reposition",
)


def _oh_template() -> Atoms:
    return Atoms("OH", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.96]], pbc=False)


def _oh_bond_length(atoms: Atoms, o_index: int, h_index: int) -> float:
    return float(np.linalg.norm(atoms.positions[h_index] - atoms.positions[o_index]))


def _prepare_parent(atoms: Atoms, confid: int) -> Atoms:
    p = atoms.copy()
    p.info["confid"] = confid
    return p


def _gas_pt3_oh_parent() -> tuple[Atoms, list[str], dict, dict]:
    comp = ["Pt", "Pt", "Pt", "O", "H"]
    ads = {
        "core_symbols": ["Pt", "Pt", "Pt"],
        "adsorbate_symbols": ["O", "H"],
        "adsorbate_fragment_lengths": [2],
        "fragment_bond_axis": [0, 1],
    }
    oh = _oh_template()
    built = build_hierarchical_core_fragment_cluster(
        comp,
        ads,
        np.random.default_rng(101),
        previous_search_glob="**/*.db",
        fragment_templates=[oh],
        cluster_adsorbate_config=None,
        max_placement_attempts=200,
    )
    assert built is not None
    parent = built.copy()
    parent.set_cell([20, 20, 20])
    parent.set_pbc(False)
    apply_mobile_core_ads_tags(parent, n_slab=0, n_core=3, ads_fragment_lengths=[2])
    blmin = build_blmin_from_zs(parent.numbers, ratio=0.7)
    return parent, comp, blmin, ads


def _surface_pt3_oh_parent() -> tuple[Atoms, list[str], dict, dict, int]:
    slab = fcc111("Pt", size=(4, 4, 2), vacuum=6.0, orthogonal=True)
    comp = ["Pt", "Pt", "Pt", "O", "H"]
    ads = {
        "core_symbols": ["Pt", "Pt", "Pt"],
        "adsorbate_symbols": ["O", "H"],
        "adsorbate_fragment_lengths": [2],
        "fragment_bond_axis": [0, 1],
    }
    oh = _oh_template()
    cfg = SurfaceSystemConfig(
        slab=slab,
        adsorption_height_min=1.0,
        adsorption_height_max=3.0,
        max_placement_attempts=400,
    )
    blmin = build_blmin(list(slab.get_chemical_symbols()) + comp, ratio=0.7)
    deposited = create_deposited_cluster(
        comp,
        slab,
        blmin,
        np.random.default_rng(202),
        cfg,
        adsorbate_definition=ads,
        adsorbate_fragment_template=[oh],
    )
    assert deposited is not None
    n_slab = len(slab)
    apply_mobile_core_ads_tags(
        deposited, n_slab=n_slab, n_core=3, ads_fragment_lengths=[2]
    )
    return deposited, comp, blmin, ads, n_slab


def _assert_oh_bond_preserved(
    child: Atoms, parent: Atoms, o_idx: int, h_idx: int
) -> None:
    bond_parent = _oh_bond_length(parent, o_idx, h_idx)
    bond_child = _oh_bond_length(child, o_idx, h_idx)
    assert abs(bond_child - bond_parent) < 1e-5


def _mutation_succeeds(
    op_name: str,
    parent: Atoms,
    composition: list[str],
    blmin: dict,
    ads: dict,
    oh_template: Atoms,
    *,
    system_type: str,
    n_slab: int = 0,
    o_idx: int = 3,
    h_idx: int = 4,
) -> bool:
    n_opt = len(parent) - n_slab
    setup_rng = np.random.default_rng(77)
    ops, name_map = create_mutation_operators(
        composition,
        n_opt,
        blmin,
        rng=setup_rng,
        use_adaptive=True,
        system_type=system_type,
        n_slab=n_slab,
        adsorbate_definition=ads,
        adsorbate_fragment_template=[oh_template],
        flattening_max_inner_attempts=12,
        rotational_max_inner_attempts=24,
        breathing_max_inner_attempts=12,
    )
    if op_name not in name_map:
        return True
    for attempt in range(MAX_MUTATION_ATTEMPTS):
        op_rng = np.random.default_rng(80_000 + attempt * 31)
        op = ops[name_map[op_name]]
        if hasattr(op, "rng"):
            op.rng = op_rng
        cand, _desc = op.get_new_individual([parent])
        if cand is None:
            continue
        mobile = cand if n_slab == 0 else cand[n_slab:]
        use_tags = system_type.endswith("_adsorbate")
        if n_slab == 0:
            assert not atoms_too_close(cand, blmin, use_tags=use_tags)
        else:
            assert not atoms_too_close(mobile, blmin, use_tags=use_tags)
            assert not atoms_too_close_two_sets(cand[:n_slab], mobile, blmin)
        _assert_oh_bond_preserved(cand, parent, o_idx, h_idx)
        return True
    return False


@pytest.mark.parametrize("op_name", _GAS_ADSORBATE_OPS)
def test_gas_cluster_adsorbate_mutation_preserves_fragment(op_name: str) -> None:
    parent, comp, blmin, ads = _gas_pt3_oh_parent()
    parent = _prepare_parent(parent, 1)
    ok = _mutation_succeeds(
        op_name,
        parent,
        comp,
        blmin,
        ads,
        _oh_template(),
        system_type="gas_cluster_adsorbate",
    )
    assert ok, f"{op_name} failed for gas_cluster_adsorbate"


@pytest.mark.parametrize("op_name", _SURFACE_ADSORBATE_OPS)
def test_surface_cluster_adsorbate_mutation_preserves_fragment(op_name: str) -> None:
    parent, comp, blmin, ads, n_slab = _surface_pt3_oh_parent()
    parent = _prepare_parent(parent, 2)
    o_idx = n_slab + 3
    h_idx = n_slab + 4
    ok = _mutation_succeeds(
        op_name,
        parent,
        comp,
        blmin,
        ads,
        _oh_template(),
        system_type="surface_cluster_adsorbate",
        n_slab=n_slab,
        o_idx=o_idx,
        h_idx=h_idx,
    )
    assert ok, f"{op_name} failed for surface_cluster_adsorbate"
