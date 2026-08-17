"""Acceptance tests for GA mutation operators and crossover at production-like sizes.

Uses ``create_mutation_operators`` and ``create_ga_pairing`` from ga_common.
Geometry acceptance mirrors production: ``atoms_too_close`` on mobile atoms and
``atoms_too_close_two_sets`` between slab and adsorbate when ``n_slab > 0``.
Successful mutants must also change internal coordinates (gas) or mobile-slab
distances (surface), so isometric no-ops cannot pass.

Two gas-phase 55-atom setups cover icosahedral template vs random spherical
geometries. Untagged gas-phase clusters omit ``mirror`` (full-cluster reflection
is an isometry) and decline ``rotational`` (whole-cluster rotation is a no-op).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111
from ase_ga.utilities import (
    atoms_too_close,
    atoms_too_close_two_sets,
    closest_distances_generator,
    get_all_atom_types,
)
from scipy.spatial.distance import pdist

from scgo.algorithms.ga_common import (
    create_ga_pairing,
    create_mutation_operators,
)
from scgo.exceptions import SCGOValidationError
from scgo.initialization import create_initial_cluster
from scgo.surface.config import SurfaceSystemConfig
from scgo.surface.deposition import create_deposited_cluster
from scgo.system_types import validate_minimum_structure
from tests.algorithms.test_adsorbate_ga_acceptance import (
    _gas_pt3_oh_parent,
    _surface_pt3_oh_parent,
)

# The active mirror and bounded cut selection make crossover acceptance fast
# enough that large legacy outer retry caps are unnecessary here.
MAX_MUTATION_ATTEMPTS = 20
MAX_CROSSOVER_ATTEMPTS = 12

# Flattening now succeeds at the production default thickness in these
# acceptance cases; keep the tests aligned with the real operator setting.
_ACCEPTANCE_FLATTENING_THICKNESS = 0.5
# Cap inner trials so surface + gas acceptance stays CI-bounded while matching
# the bounded candidate sets used by the operators.
_ACCEPTANCE_FLATTEN_MAX_INNER = 12
_ACCEPTANCE_ROT_MAX_INNER = 24
_ACCEPTANCE_MIRROR_TRIES = 12
_ACCEPTANCE_BREATHING_MAX_INNER = 5
_ACCEPTANCE_SLIDE_MAX_INNER = 12

_GAS_PT55_RANDOM_SPHERICAL_SEED = 1234

_TEMPLATE_CORE_OPERATOR_NAMES = ("rattle", "rotational", "anisotropic_rattle")

# An untagged gas-phase cluster forms a single tag group spanning every mobile
# atom, so RotationalMutation could only produce a rigid rotation of the whole
# (re-centred) cluster, i.e. an energy-identical duplicate. The operator now
# declines those moves instead of burning a GA evaluation.
_GAS_PHASE_NOOP_OPERATOR_NAMES = frozenset({"rotational"})


def _assert_mutation_expectation(op_name: str, ok: bool, *, n_slab: int) -> None:
    if n_slab == 0 and op_name in _GAS_PHASE_NOOP_OPERATOR_NAMES:
        assert not ok, (
            f"mutation {op_name!r} must decline no-op rotations of an untagged "
            "gas-phase cluster"
        )
        return
    assert ok, f"mutation {op_name!r} failed after {MAX_MUTATION_ATTEMPTS} attempts"


def _plain_atoms(a: Atoms) -> Atoms:
    """Coerce to plain Atoms so CutAndSplicePairing.copy() never hits Cluster bugs."""
    return Atoms(
        numbers=a.get_atomic_numbers(),
        positions=a.get_positions(),
        cell=a.get_cell(),
        pbc=a.get_pbc(),
    )


def _internal_fingerprints_differ(a: Atoms, b: Atoms, n_slab: int) -> bool:
    """True when mobile all-pair distances differ (tag-aware rigid moves count)."""
    pos_a = np.asarray(a.positions[n_slab:], dtype=float)
    pos_b = np.asarray(b.positions[n_slab:], dtype=float)
    if pos_a.shape != pos_b.shape:
        return True
    if len(pos_a) < 2:
        return not np.allclose(pos_a, pos_b, atol=1e-6, rtol=0.0)
    return not np.allclose(
        np.sort(pdist(pos_a)), np.sort(pdist(pos_b)), atol=1e-6, rtol=0.0
    )


def _slab_fingerprints_differ(a: Atoms, b: Atoms, n_slab: int) -> bool:
    delta_a = a.positions[n_slab:, None, :] - a.positions[None, :n_slab, :]
    delta_b = b.positions[n_slab:, None, :] - b.positions[None, :n_slab, :]
    da = np.sort(np.linalg.norm(delta_a, axis=2).ravel())
    db = np.sort(np.linalg.norm(delta_b, axis=2).ravel())
    if da.shape != db.shape:
        return True
    return not np.allclose(da, db, atol=1e-6, rtol=0.0)


def _assert_mutant_is_new(cand: Atoms, parent: Atoms, n_slab: int) -> None:
    if n_slab == 0:
        assert _internal_fingerprints_differ(cand, parent, 0), (
            "gas-phase mutant did not change internal coordinates"
        )
        return
    assert _internal_fingerprints_differ(
        cand, parent, n_slab
    ) or _slab_fingerprints_differ(cand, parent, n_slab), (
        "surface mutant changed neither internals nor mobile-slab distances"
    )


def _assert_accepted_geometry(
    atoms: Atoms,
    n_slab: int,
    blmin: dict,
    parent: Atoms,
    *,
    adsorbate_use_tags: bool = False,
    preserve_tags: bool = False,
) -> None:
    assert len(atoms) == len(parent)
    assert np.array_equal(atoms.get_atomic_numbers(), parent.get_atomic_numbers())
    if preserve_tags and len(parent.get_tags()) == len(parent):
        assert np.array_equal(atoms.get_tags(), parent.get_tags())
    assert np.allclose(atoms.get_cell(), parent.get_cell())
    assert np.all(atoms.get_pbc() == parent.get_pbc())
    if n_slab == 0:
        assert not atoms_too_close(atoms, blmin, use_tags=adsorbate_use_tags)
    else:
        slab_part = atoms[:n_slab]
        ads = atoms[n_slab:]
        assert not atoms_too_close(ads, blmin, use_tags=adsorbate_use_tags)
        assert not atoms_too_close_two_sets(slab_part, ads, blmin)


def _gas_pt55_template_parent() -> tuple[Atoms, list[str], dict]:
    composition = ["Pt"] * 55
    raw = create_initial_cluster(
        composition,
        rng=np.random.default_rng(2025),
        mode="template",
        vacuum=10.0,
    )
    parent = _plain_atoms(raw)
    blmin = closest_distances_generator(
        get_all_atom_types(parent, range(55)),
        ratio_of_covalent_radii=0.7,
    )
    return parent, composition, blmin


def _gas_pt55_random_spherical_parent(seed: int) -> tuple[Atoms, list[str], dict]:
    composition = ["Pt"] * 55
    raw = create_initial_cluster(
        composition,
        rng=np.random.default_rng(seed),
        mode="random_spherical",
        vacuum=10.0,
    )
    parent = _plain_atoms(raw)
    blmin = closest_distances_generator(
        get_all_atom_types(parent, range(55)),
        ratio_of_covalent_radii=0.7,
    )
    return parent, composition, blmin


def _surface_pt20_system(
    rng_a: np.random.Generator,
    rng_b: np.random.Generator,
) -> tuple[Atoms, Atoms, Atoms, Atoms, list[str], dict, int]:
    # Larger in-plane cell improves odds of placing a 20-atom cluster without clashes.
    slab = fcc111("Pt", size=(4, 4, 2), vacuum=6.0, orthogonal=True)
    composition = ["Pt"] * 20
    n_slab = len(slab)
    n_top = len(composition)

    dummy = np.vstack([slab.get_positions(), np.zeros((n_top, 3))])
    tmpl = Atoms(
        symbols=list(slab.get_chemical_symbols()) + composition,
        positions=dummy,
        cell=slab.cell,
        pbc=slab.pbc,
    )
    idx_top = range(n_slab, n_slab + n_top)
    blmin = closest_distances_generator(
        get_all_atom_types(tmpl, idx_top),
        ratio_of_covalent_radii=0.7,
    )

    cfg = SurfaceSystemConfig(
        slab=slab,
        adsorption_height_min=1.0,
        adsorption_height_max=3.2,
        max_placement_attempts=3500,
        cluster_init_vacuum=10.0,
        init_mode="random_spherical",
    )

    p1_raw = create_deposited_cluster(composition, slab, blmin, rng_a, cfg)
    p2_raw = create_deposited_cluster(composition, slab, blmin, rng_b, cfg)
    assert p1_raw is not None and p2_raw is not None

    return (
        slab,
        tmpl,
        _plain_atoms(p1_raw),
        _plain_atoms(p2_raw),
        composition,
        blmin,
        n_slab,
    )


def _prepare_ga_parent(atoms: Atoms, confid: int) -> Atoms:
    p = atoms.copy()
    p.info["confid"] = confid
    return p


def _mutation_operator_succeeds(
    op_name: str,
    name_map: dict[str, int],
    parent: Atoms,
    composition: list[str],
    n_opt: int,
    blmin: dict,
    n_slab: int,
    *,
    flattening_thickness_factor: float,
    surface_normal_axis: int = 2,
    flattening_max_inner_attempts: int = _ACCEPTANCE_FLATTEN_MAX_INNER,
    rotational_max_inner_attempts: int = _ACCEPTANCE_ROT_MAX_INNER,
    mirror_max_tries: int = _ACCEPTANCE_MIRROR_TRIES,
    breathing_max_inner_attempts: int = _ACCEPTANCE_BREATHING_MAX_INNER,
    in_plane_slide_max_inner_attempts: int = _ACCEPTANCE_SLIDE_MAX_INNER,
    breathing_scale_min: float = 0.82,
    breathing_scale_max: float = 1.22,
) -> bool:
    adsorbate_use_tags = op_name == "rotational" and n_slab > 0
    for attempt in range(MAX_MUTATION_ATTEMPTS):
        op_rng = np.random.default_rng(50_000 + attempt * 97 + name_map[op_name] * 13)
        ops, nm = create_mutation_operators(
            composition,
            n_opt,
            blmin,
            rng=op_rng,
            use_adaptive=True,
            system_type="surface_cluster" if n_slab > 0 else "gas_cluster",
            n_slab=n_slab,
            surface_normal_axis=surface_normal_axis,
            flattening_thickness_factor=flattening_thickness_factor,
            flattening_max_inner_attempts=flattening_max_inner_attempts,
            rotational_max_inner_attempts=rotational_max_inner_attempts,
            mirror_max_tries=mirror_max_tries,
            breathing_max_inner_attempts=breathing_max_inner_attempts,
            in_plane_slide_max_inner_attempts=in_plane_slide_max_inner_attempts,
            breathing_scale_min=breathing_scale_min,
            breathing_scale_max=breathing_scale_max,
        )
        assert nm == name_map
        op = ops[nm[op_name]]
        cand, _desc = op.get_new_individual([parent])
        if cand is None:
            continue
        _assert_accepted_geometry(
            cand,
            n_slab,
            blmin,
            parent,
            adsorbate_use_tags=adsorbate_use_tags,
            preserve_tags=adsorbate_use_tags,
        )
        _assert_mutant_is_new(cand, parent, n_slab)
        return True
    return False


@pytest.mark.slow
def test_mutations_gas_pt55_icosahedral_template_core_operators() -> None:
    parent0, composition, blmin = _gas_pt55_template_parent()
    parent = _prepare_ga_parent(parent0, confid=1)

    _, name_map = create_mutation_operators(
        composition,
        55,
        blmin,
        rng=np.random.default_rng(0),
        use_adaptive=True,
    )
    assert "permutation" not in name_map
    assert "mirror" not in name_map

    for op_name in _TEMPLATE_CORE_OPERATOR_NAMES:
        assert op_name in name_map
        ok = _mutation_operator_succeeds(
            op_name,
            name_map,
            parent,
            composition,
            55,
            blmin,
            0,
            flattening_thickness_factor=0.5,
            flattening_max_inner_attempts=12,
            rotational_max_inner_attempts=24,
            mirror_max_tries=12,
        )
        _assert_mutation_expectation(op_name, ok, n_slab=0)


@pytest.mark.slow
def test_mutations_gas_pt55_random_spherical_all_factory_operators() -> None:
    parent0, composition, blmin = _gas_pt55_random_spherical_parent(
        _GAS_PT55_RANDOM_SPHERICAL_SEED
    )
    parent = _prepare_ga_parent(parent0, confid=1)
    assert not atoms_too_close(parent, blmin, use_tags=False), (
        "random_spherical parent must satisfy GA blmin for rigid rotational mutation"
    )

    _, name_map = create_mutation_operators(
        composition,
        55,
        blmin,
        rng=np.random.default_rng(0),
        use_adaptive=True,
        flattening_thickness_factor=_ACCEPTANCE_FLATTENING_THICKNESS,
    )
    assert "permutation" not in name_map
    assert "mirror" not in name_map

    for op_name in sorted(name_map.keys(), key=lambda k: name_map[k]):
        ok = _mutation_operator_succeeds(
            op_name,
            name_map,
            parent,
            composition,
            55,
            blmin,
            0,
            flattening_thickness_factor=_ACCEPTANCE_FLATTENING_THICKNESS,
        )
        _assert_mutation_expectation(op_name, ok, n_slab=0)


@pytest.mark.slow
def test_mutations_surface_pt20_all_factory_operators() -> None:
    _slab, _tmpl, raw_p1, _raw_p2, composition, blmin, n_slab = _surface_pt20_system(
        np.random.default_rng(101),
        np.random.default_rng(202),
    )
    parent = _prepare_ga_parent(raw_p1, confid=1)

    _, name_map = create_mutation_operators(
        composition,
        len(composition),
        blmin,
        rng=np.random.default_rng(0),
        use_adaptive=True,
        system_type="surface_cluster",
        n_slab=n_slab,
        surface_normal_axis=2,
        flattening_thickness_factor=_ACCEPTANCE_FLATTENING_THICKNESS,
    )
    assert "permutation" not in name_map
    assert "in_plane_slide" in name_map
    assert "mirror" in name_map

    for op_name in sorted(name_map.keys(), key=lambda k: name_map[k]):
        # RotationalMutation now re-anchors the adsorbate to the slab and applies
        # an in-plane rescue, so it passes the per-call accepted-geometry check.
        ok = _mutation_operator_succeeds(
            op_name,
            name_map,
            parent,
            composition,
            len(composition),
            blmin,
            n_slab,
            flattening_thickness_factor=_ACCEPTANCE_FLATTENING_THICKNESS,
        )
        assert ok, f"mutation {op_name!r} failed after {MAX_MUTATION_ATTEMPTS} attempts"


def _crossover_child(
    p1: Atoms,
    p2: Atoms,
    n_top: int,
    blmin: dict,
    n_slab: int,
    slab_atoms: Atoms | None,
    template_atoms: Atoms,
    **pairing_kwargs: Any,
) -> Atoms | None:
    for attempt in range(MAX_CROSSOVER_ATTEMPTS):
        prng = np.random.default_rng(80_000 + attempt)
        pairing = create_ga_pairing(
            template_atoms,
            n_top,
            rng=prng,
            slab_atoms=slab_atoms,
            system_type="surface_cluster" if n_slab > 0 else "gas_cluster",
            **pairing_kwargs,
        )
        cand, _desc = pairing.get_new_individual([p1, p2])
        if cand is None:
            continue
        _assert_accepted_geometry(
            cand,
            n_slab,
            blmin,
            p1,
            adsorbate_use_tags=False,
        )
        return cand
    return None


@pytest.mark.slow
def test_crossover_gas_pt55_random_spherical_then_rattle_mutate() -> None:
    p1_raw, composition, blmin = _gas_pt55_random_spherical_parent(11)
    p2_raw, _, _ = _gas_pt55_random_spherical_parent(22)
    p1 = _prepare_ga_parent(p1_raw, confid=1)
    p2 = _prepare_ga_parent(p2_raw, confid=2)
    rng_pert = np.random.default_rng(5)
    for p, _subseed in ((p1, 1), (p2, 2)):
        for _trial in range(30):
            delta = rng_pert.normal(0, 0.06, size=p.positions.shape)
            trial_pos = p.positions + delta
            probe = p.copy()
            probe.positions[:] = trial_pos
            if not atoms_too_close(probe, blmin):
                p.positions[:] = trial_pos
                break
        else:
            pytest.fail(
                f"could not apply small diversity perturbation (confid={p.info['confid']})"
            )

    child = _crossover_child(p1, p2, 55, blmin, 0, None, p1.copy())
    assert child is not None, (
        f"crossover failed after {MAX_CROSSOVER_ATTEMPTS} attempts"
    )

    mut_ok = False
    for attempt in range(MAX_MUTATION_ATTEMPTS):
        op_rng = np.random.default_rng(90_000 + attempt)
        ops, nm = create_mutation_operators(
            composition,
            55,
            blmin,
            rng=op_rng,
            use_adaptive=True,
            flattening_thickness_factor=_ACCEPTANCE_FLATTENING_THICKNESS,
            flattening_max_inner_attempts=_ACCEPTANCE_FLATTEN_MAX_INNER,
            rotational_max_inner_attempts=_ACCEPTANCE_ROT_MAX_INNER,
            mirror_max_tries=_ACCEPTANCE_MIRROR_TRIES,
        )
        rattle = ops[nm["rattle"]]
        mutated = rattle.mutate(child)
        if mutated is None:
            continue
        _assert_accepted_geometry(mutated, 0, blmin, p1, adsorbate_use_tags=False)
        mut_ok = True
        break
    assert mut_ok, "rattle.mutate after crossover did not yield accepted geometry"


@pytest.mark.slow
def test_crossover_surface_pt20_then_rattle_mutate() -> None:
    slab, tmpl, raw_p1, raw_p2, composition, blmin, n_slab = _surface_pt20_system(
        np.random.default_rng(303),
        np.random.default_rng(404),
    )
    p1 = _prepare_ga_parent(raw_p1, confid=1)
    p2 = _prepare_ga_parent(raw_p2, confid=2)

    child = _crossover_child(p1, p2, len(composition), blmin, n_slab, slab, tmpl)
    assert child is not None, (
        f"surface crossover failed after {MAX_CROSSOVER_ATTEMPTS} attempts"
    )

    mut_ok = False
    for attempt in range(MAX_MUTATION_ATTEMPTS):
        op_rng = np.random.default_rng(110_000 + attempt)
        ops, nm = create_mutation_operators(
            composition,
            len(composition),
            blmin,
            rng=op_rng,
            use_adaptive=True,
            flattening_thickness_factor=_ACCEPTANCE_FLATTENING_THICKNESS,
            flattening_max_inner_attempts=_ACCEPTANCE_FLATTEN_MAX_INNER,
            rotational_max_inner_attempts=_ACCEPTANCE_ROT_MAX_INNER,
            mirror_max_tries=_ACCEPTANCE_MIRROR_TRIES,
        )
        rattle = ops[nm["rattle"]]
        mutated = rattle.mutate(child)
        if mutated is None:
            continue
        _assert_accepted_geometry(mutated, n_slab, blmin, p1, adsorbate_use_tags=False)
        mut_ok = True
        break
    assert mut_ok, (
        "rattle.mutate after surface crossover did not yield accepted geometry"
    )


# ---------------------------------------------------------------------------
# Full-gate (validate_minimum_structure) 10/10 acceptance.
#
# Each operator is called 10 times on a fixed deterministic per-run RNG and must
# ALWAYS return a structure that passes the complete system-type gate (blmin +
# connectivity + slab contact + adsorbate fragment integrity). A ``None`` return
# or any gate error is a hard failure. Seeds are fixed and never tuned to pass:
# failures indicate genuine operator robustness gaps that must be fixed.
# ---------------------------------------------------------------------------

_GAS_CLUSTER_ADSORBATE_COMMON = (
    "rattle",
    "anisotropic_rattle",
    "overlap_relief",
    "rotational",
    "mirror",
    "fragment_reposition",
)

_SURFACE_CLUSTER_ADSORBATE_COMMON = (
    "rattle",
    "anisotropic_rattle",
    "overlap_relief",
    "rotational",
    "mirror",
    "in_plane_slide",
    "in_plane_rotate",
    "in_plane_slide_core",
    "in_plane_slide_ads",
    "fragment_reposition",
)


def _oh_template() -> Atoms:
    return Atoms("OH", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.96]], pbc=False)


def _surface_config_from_slab(
    slab: Atoms, *, height_min: float, height_max: float
) -> SurfaceSystemConfig:
    return SurfaceSystemConfig(
        slab=slab,
        adsorption_height_min=height_min,
        adsorption_height_max=height_max,
        comparator_use_mic=True,
        surface_normal_axis=2,
    )


def _gas_cluster_setup():
    # Use the compact icosahedral template parent (not the loose random-spherical
    # cloud): per-atom rattle/breathing on a sparse cloud fragments the connectivity
    # gate, whereas a dense metal cluster preserves connectivity under the operators'
    # translate/scale/rotate moves. Composition is unchanged (55 Pt atoms).
    parent0, composition, blmin = _gas_pt55_template_parent()
    parent = _prepare_ga_parent(parent0, confid=1)
    return (
        parent,
        composition,
        len(composition),
        blmin,
        "gas_cluster",
        0,
        None,
        None,
    )


def _surface_cluster_setup():
    slab, _tmpl, raw_p1, _raw_p2, composition, blmin, n_slab = _surface_pt20_system(
        np.random.default_rng(101),
        np.random.default_rng(202),
    )
    surface_config = _surface_config_from_slab(slab, height_min=1.0, height_max=3.2)
    parent = _prepare_ga_parent(raw_p1, confid=1)
    return (
        parent,
        composition,
        len(composition),
        blmin,
        "surface_cluster",
        n_slab,
        surface_config,
        None,
    )


def _gas_cluster_adsorbate_setup():
    parent0, composition, blmin, ads = _gas_pt3_oh_parent()
    parent = _prepare_ga_parent(parent0, confid=1)
    return (
        parent,
        composition,
        len(composition),
        blmin,
        "gas_cluster_adsorbate",
        0,
        None,
        ads,
    )


def _surface_cluster_adsorbate_setup():
    deposited, composition, blmin, ads, n_slab = _surface_pt3_oh_parent()
    slab = deposited[:n_slab]
    surface_config = _surface_config_from_slab(slab, height_min=1.0, height_max=3.0)
    parent = _prepare_ga_parent(deposited, confid=2)
    return (
        parent,
        composition,
        len(composition),
        blmin,
        "surface_cluster_adsorbate",
        n_slab,
        surface_config,
        ads,
    )


def _mutation_ten_of_ten(
    op_name: str,
    name_map: dict[str, int],
    parent: Atoms,
    composition: list[str],
    n_opt: int,
    blmin: dict,
    system_type: str,
    *,
    n_slab: int = 0,
    surface_config: SurfaceSystemConfig | None = None,
    adsorbate_definition: dict | None = None,
    connectivity_factor: float | None = None,
    allow_cluster_fragmentation: bool = False,
    allow_adsorbate_surface_detachment: bool = False,
    enforce_adsorbate_subgraph_integrity: bool = True,
    n_runs: int = 10,
) -> None:
    """Assert an operator passes the full gate on 10/10 fixed-seed calls."""
    oh = _oh_template() if adsorbate_definition is not None else None
    failures: list[str] = []
    for run in range(n_runs):
        op_rng = np.random.default_rng(1_000 + run * 100 + name_map[op_name])
        ops, nm = create_mutation_operators(
            composition,
            n_opt,
            blmin,
            rng=op_rng,
            use_adaptive=True,
            system_type=system_type,
            n_slab=n_slab,
            surface_normal_axis=2,
            adsorbate_definition=adsorbate_definition,
            adsorbate_fragment_template=[oh] if oh is not None else None,
            flattening_thickness_factor=0.5,
            flattening_max_inner_attempts=12,
            rotational_max_inner_attempts=48,
            mirror_max_tries=12,
            breathing_max_inner_attempts=12,
            in_plane_slide_max_inner_attempts=12,
        )
        assert nm == name_map
        op = ops[nm[op_name]]
        cand, _desc = op.get_new_individual([parent])
        if cand is None:
            failures.append(f"run {run}: operator returned None")
            continue
        try:
            validate_minimum_structure(
                cand,
                system_type=system_type,
                surface_config=surface_config,
                n_slab=n_slab,
                adsorbate_definition=adsorbate_definition,
                connectivity_factor=connectivity_factor,
                allow_cluster_fragmentation=allow_cluster_fragmentation,
                allow_adsorbate_surface_detachment=allow_adsorbate_surface_detachment,
                enforce_adsorbate_subgraph_integrity=enforce_adsorbate_subgraph_integrity,
            )
        except (ValueError, SCGOValidationError) as exc:
            failures.append(f"run {run}: {type(exc).__name__}: {exc}")
            continue
        try:
            _assert_mutant_is_new(cand, parent, n_slab)
        except AssertionError as exc:
            failures.append(f"run {run}: isometric no-op: {exc}")

    assert not failures, (
        f"{op_name!r} ({system_type}) failed {len(failures)}/{n_runs} full-gate runs:\n"
        + "\n".join(failures[:3])
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "setup, excluded",
    [
        (_gas_cluster_setup, frozenset({"rotational"})),
        (_surface_cluster_setup, frozenset()),
        (_gas_cluster_adsorbate_setup, frozenset()),
        (_surface_cluster_adsorbate_setup, frozenset()),
    ],
)
def test_mutation_operators_pass_full_gate_ten_of_ten(setup, excluded) -> None:
    parent, composition, n_opt, blmin, system_type, n_slab, surface_config, ads = (
        setup()
    )
    _, name_map = create_mutation_operators(
        composition,
        n_opt,
        blmin,
        rng=np.random.default_rng(0),
        use_adaptive=True,
        system_type=system_type,
        n_slab=n_slab,
        surface_normal_axis=2,
        adsorbate_definition=ads,
        flattening_thickness_factor=0.5,
    )
    assert "permutation" not in name_map
    if system_type == "gas_cluster":
        assert "mirror" not in name_map
    else:
        assert "mirror" in name_map

    for op_name in sorted(name_map.keys(), key=lambda k: name_map[k]):
        if op_name in excluded:
            continue
        _mutation_ten_of_ten(
            op_name,
            name_map,
            parent,
            composition,
            n_opt,
            blmin,
            system_type,
            n_slab=n_slab,
            surface_config=surface_config,
            adsorbate_definition=ads,
        )
