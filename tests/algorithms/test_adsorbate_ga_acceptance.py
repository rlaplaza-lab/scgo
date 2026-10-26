"""GA operator acceptance for gas and surface cluster+adsorbate systems."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111
from ase_ga.utilities import atoms_too_close, atoms_too_close_two_sets

from scgo.algorithms.ga_common import (
    apply_mobile_core_ads_tags,
    create_ga_pairing,
    create_mutation_operators,
)
from scgo.cluster_adsorbate.hierarchical import build_hierarchical_core_fragment_cluster
from scgo.initialization.atomic_radii import build_blmin, build_blmin_from_zs
from scgo.surface.config import SurfaceSystemConfig
from scgo.surface.deposition import create_deposited_cluster
from scgo.system_types import AdsorbateDefinition

pytestmark = pytest.mark.slow

MAX_MUTATION_ATTEMPTS = 15
MAX_CROSSOVER_ATTEMPTS = 12

_GAS_ADSORBATE_OPS = (
    "rattle",
    "anisotropic_rattle",
    "rotational",
    "mirror",
    "fragment_reposition",
)

_SURFACE_ADSORBATE_OPS = (
    "rattle",
    "anisotropic_rattle",
    "rotational",
    "mirror",
    "in_plane_slide",
    "in_plane_rotate",
    "in_plane_slide_core",
    "fragment_reposition",
)

# Ops that preserve OH integrity on multi-fragment parents (non-frozen factory).
_GAS_MULTI_FRAGMENT_OPS = (
    "rattle",
    "fragment_reposition",
    "mirror",
    "flattening_core",
)

_SURFACE_MULTI_FRAGMENT_OPS = (
    "rattle",
    "fragment_reposition",
    "in_plane_slide_ads",
    "in_plane_slide_core",
)


def _oh_template() -> Atoms:
    return Atoms("OH", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.96]], pbc=False)


def _oh_bond_length(atoms: Atoms, o_index: int, h_index: int) -> float:
    return float(np.linalg.norm(atoms.positions[h_index] - atoms.positions[o_index]))


def _prepare_parent(atoms: Atoms, confid: int) -> Atoms:
    p = atoms.copy()
    p.info["confid"] = confid
    return p


def _oh_adsorbate_definition(n_oh: int) -> AdsorbateDefinition:
    ads_symbols = ["O", "H"] * n_oh
    return AdsorbateDefinition(
        core_symbols=["Pt", "Pt", "Pt"],
        adsorbate_symbols=ads_symbols,
        adsorbate_fragment_lengths=[2] * n_oh,
        fragment_bond_axis=[0, 1],
    )


def _oh_pairs(n_slab: int, n_core: int, n_oh: int) -> list[tuple[int, int]]:
    start = n_slab + n_core
    return [(start + 2 * i, start + 2 * i + 1) for i in range(n_oh)]


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


def _assert_oh_bonds_preserved(
    child: Atoms, parent: Atoms, pairs: list[tuple[int, int]]
) -> None:
    for o_idx, h_idx in pairs:
        assert (
            abs(
                _oh_bond_length(child, o_idx, h_idx)
                - _oh_bond_length(parent, o_idx, h_idx)
            )
            < 1e-5
        )


def _mutation_succeeds(
    op_name: str,
    parent: Atoms,
    composition: list[str],
    blmin: dict,
    ads: AdsorbateDefinition,
    *,
    system_type: str,
    n_slab: int = 0,
) -> bool:
    n_opt = len(parent) - n_slab
    n_oh = len(ads.adsorbate_fragment_lengths)
    oh_pairs = _oh_pairs(n_slab, len(ads.core_symbols), n_oh)
    templates = [_oh_template() for _ in range(n_oh)]
    ops, name_map = create_mutation_operators(
        composition,
        n_opt,
        blmin,
        rng=np.random.default_rng(77),
        use_adaptive=True,
        system_type=system_type,
        n_slab=n_slab,
        adsorbate_definition=ads,
        adsorbate_fragment_template=templates,
        flattening_max_inner_attempts=12,
        rotational_max_inner_attempts=24,
        breathing_max_inner_attempts=12,
        in_plane_slide_max_inner_attempts=12,
    )
    if op_name not in name_map:
        return True
    for attempt in range(MAX_MUTATION_ATTEMPTS):
        op = ops[name_map[op_name]]
        if hasattr(op, "rng"):
            op.rng = np.random.default_rng(80_000 + attempt * 31)
        cand, _desc = op.get_new_individual([parent])
        if cand is None:
            continue
        assert np.array_equal(cand.get_atomic_numbers(), parent.get_atomic_numbers())
        if len(parent.get_tags()) == len(parent):
            assert np.array_equal(cand.get_tags(), parent.get_tags())
        mobile = cand if n_slab == 0 else cand[n_slab:]
        use_tags = system_type.endswith("_adsorbate")
        if n_slab == 0:
            assert not atoms_too_close(cand, blmin, use_tags=use_tags)
        else:
            assert not atoms_too_close(mobile, blmin, use_tags=use_tags)
            assert not atoms_too_close_two_sets(cand[:n_slab], mobile, blmin)
        _assert_oh_bonds_preserved(cand, parent, oh_pairs)
        return True
    return False


def _crossover_child(
    p1: Atoms,
    p2: Atoms,
    *,
    composition: list[str],
    blmin: dict,
    ads: AdsorbateDefinition,
    system_type: str,
    n_slab: int,
    slab_atoms: Atoms | None,
    template_atoms: Atoms,
) -> Atoms | None:
    oh_pairs = _oh_pairs(
        n_slab, len(ads.core_symbols), len(ads.adsorbate_fragment_lengths)
    )
    for attempt in range(MAX_CROSSOVER_ATTEMPTS):
        pairing = create_ga_pairing(
            template_atoms,
            len(composition),
            rng=np.random.default_rng(80_000 + attempt),
            slab_atoms=slab_atoms,
            system_type=system_type,
            composition=composition,
            adsorbate_definition=ads,
            exploratory_crossover_probability=0.0,
        )
        cand, _desc = pairing.get_new_individual([p1, p2])
        if cand is None:
            continue
        assert np.array_equal(cand.get_atomic_numbers(), p1.get_atomic_numbers())
        mobile = cand if n_slab == 0 else cand[n_slab:]
        if n_slab == 0:
            assert not atoms_too_close(cand, blmin, use_tags=True)
        else:
            assert not atoms_too_close(mobile, blmin, use_tags=True)
            assert not atoms_too_close_two_sets(cand[:n_slab], mobile, blmin)
        _assert_oh_bonds_preserved(cand, p1, oh_pairs)
        return cand
    return None


@pytest.mark.parametrize("op_name", _GAS_ADSORBATE_OPS)
def test_gas_cluster_adsorbate_mutation_preserves_fragment(op_name: str) -> None:
    parent, comp, blmin, ads = _gas_pt3_oh_parent()
    ok = _mutation_succeeds(
        op_name,
        _prepare_parent(parent, 1),
        comp,
        blmin,
        ads,
        system_type="gas_cluster_adsorbate",
    )
    assert ok, f"{op_name} failed for gas_cluster_adsorbate"


@pytest.mark.parametrize("op_name", _SURFACE_ADSORBATE_OPS)
def test_surface_cluster_adsorbate_mutation_preserves_fragment(op_name: str) -> None:
    parent, comp, blmin, ads, n_slab = _surface_pt3_oh_parent()
    ok = _mutation_succeeds(
        op_name,
        _prepare_parent(parent, 2),
        comp,
        blmin,
        ads,
        system_type="surface_cluster_adsorbate",
        n_slab=n_slab,
    )
    assert ok, f"{op_name} failed for surface_cluster_adsorbate"


@pytest.mark.parametrize("op_name", _GAS_MULTI_FRAGMENT_OPS)
def test_gas_cluster_two_oh_mutation_preserves_fragments(op_name: str) -> None:
    parent, comp, blmin, ads = _gas_pt3_oh_parent(n_oh=2, seed=111)
    ok = _mutation_succeeds(
        op_name,
        _prepare_parent(parent, 1),
        comp,
        blmin,
        ads,
        system_type="gas_cluster_adsorbate",
    )
    assert ok, f"{op_name} failed for gas_cluster_adsorbate with two OH"


@pytest.mark.parametrize("op_name", _SURFACE_MULTI_FRAGMENT_OPS)
def test_surface_cluster_two_oh_mutation_preserves_fragments(op_name: str) -> None:
    parent, comp, blmin, ads, n_slab = _surface_pt3_oh_parent(n_oh=2, seed=17)
    ok = _mutation_succeeds(
        op_name,
        _prepare_parent(parent, 2),
        comp,
        blmin,
        ads,
        system_type="surface_cluster_adsorbate",
        n_slab=n_slab,
    )
    assert ok, f"{op_name} failed for surface_cluster_adsorbate with two OH"


def test_crossover_gas_pt3_oh_then_rattle() -> None:
    p1_raw, comp, blmin, ads = _gas_pt3_oh_parent(seed=101)
    p2_raw, _, _, _ = _gas_pt3_oh_parent(seed=202)
    p1 = _prepare_parent(p1_raw, 1)
    p2 = _prepare_parent(p2_raw, 2)
    child = _crossover_child(
        p1,
        p2,
        composition=comp,
        blmin=blmin,
        ads=ads,
        system_type="gas_cluster_adsorbate",
        n_slab=0,
        slab_atoms=None,
        template_atoms=p1.copy(),
    )
    assert child is not None, (
        f"gas adsorbate crossover failed after {MAX_CROSSOVER_ATTEMPTS} attempts"
    )
    ok = _mutation_succeeds(
        "rattle",
        _prepare_parent(child, 3),
        comp,
        blmin,
        ads,
        system_type="gas_cluster_adsorbate",
    )
    assert ok, "rattle after gas adsorbate crossover failed"


def test_crossover_surface_pt3_oh_then_rattle() -> None:
    p1_raw, comp, blmin, ads, n_slab = _surface_pt3_oh_parent(seed=3)
    p2_raw, _, _, _, _ = _surface_pt3_oh_parent(seed=9)
    p1 = _prepare_parent(p1_raw, 1)
    p2 = _prepare_parent(p2_raw, 2)
    child = _crossover_child(
        p1,
        p2,
        composition=comp,
        blmin=blmin,
        ads=ads,
        system_type="surface_cluster_adsorbate",
        n_slab=n_slab,
        slab_atoms=p1[:n_slab],
        template_atoms=p1.copy(),
    )
    assert child is not None, (
        f"surface adsorbate crossover failed after {MAX_CROSSOVER_ATTEMPTS} attempts"
    )
    ok = _mutation_succeeds(
        "rattle",
        _prepare_parent(child, 3),
        comp,
        blmin,
        ads,
        system_type="surface_cluster_adsorbate",
        n_slab=n_slab,
    )
    assert ok, "rattle after surface adsorbate crossover failed"
