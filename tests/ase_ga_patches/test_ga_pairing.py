from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from ase import Atoms
from ase_ga.utilities import closest_distances_generator, get_all_atom_types

from scgo.ase_ga_patches.cutandsplicepairing import (
    CutAndSplicePairing,
    DualCutAndSplicePairing,
    _assert_offspring_integrity,
    _pop_closest_to_plane,
)
from scgo.ase_ga_patches.mutations._finalize import _finalize_mutant
from scgo.exceptions import SCGOValidationError
from tests.helpers import create_paired_rngs


def test_cut_and_splice_preserves_stoichiometry_and_is_deterministic(au2pt2_atoms, rng):
    # Prepare two parent structures
    p1 = au2pt2_atoms.copy()
    p2 = au2pt2_atoms.copy()
    p1.info["confid"] = "p1"
    p2.info["confid"] = "p2"

    # Use identical seeds to test determinism across different operator instances
    rng1, rng2 = create_paired_rngs(123)

    # minimal bond-length dict to avoid KeyError in atoms_too_close
    pt = 78
    au = 79
    blmin = {(pt, pt): 0.1, (pt, au): 0.1, (au, au): 0.1}
    op1 = CutAndSplicePairing(
        slab=Atoms(), n_top=4, blmin=blmin, system_type="gas_cluster", rng=rng1
    )
    op2 = CutAndSplicePairing(
        slab=Atoms(), n_top=4, blmin=blmin, system_type="gas_cluster", rng=rng2
    )

    child1 = op1.cross(p1, p2)
    child2 = op2.cross(p1, p2)

    assert child1 is not None
    assert child2 is not None

    # Stoichiometry (element counts) should be preserved
    assert sorted(child1.get_chemical_symbols()) == sorted(p1.get_chemical_symbols())
    assert sorted(child2.get_chemical_symbols()) == sorted(p1.get_chemical_symbols())

    # Deterministic for identical seeds
    assert np.allclose(child1.get_positions(), child2.get_positions())


def test_dual_cut_and_splice_returns_offspring(pt3_atoms, rng):
    n_top = len(pt3_atoms)
    blmin = closest_distances_generator(
        get_all_atom_types(pt3_atoms, range(n_top)),
        ratio_of_covalent_radii=0.7,
    )
    slab = Atoms(cell=pt3_atoms.get_cell(), pbc=pt3_atoms.get_pbc())
    primary = CutAndSplicePairing(
        slab,
        n_top,
        blmin,
        minfrac=0.3,
        system_type="gas_cluster",
        rng=rng,
    )
    exploratory = CutAndSplicePairing(
        slab,
        n_top,
        blmin,
        minfrac=0.15,
        system_type="gas_cluster",
        rng=rng,
    )
    dual = DualCutAndSplicePairing(
        primary,
        exploratory,
        0.5,
        rng=rng,
    )
    p1 = pt3_atoms.copy()
    p2 = pt3_atoms.copy()
    p1.info["confid"] = "a"
    p2.info["confid"] = "b"
    child, _desc = dual.get_new_individual([p1, p2])
    assert child is not None
    assert len(child) == n_top
    assert child.get_chemical_symbols() == pt3_atoms.get_chemical_symbols()


def test_dual_pairing_falls_back_only_when_the_first_operator_fails():
    parent = Atoms("Pt2", positions=[[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
    child = parent.copy()
    child.positions[0, 0] += 0.4

    class _Operator:
        def __init__(self, result):
            self.result = result
            self.calls = 0
            self.last_attempt_count = 4

        def get_new_individual(self, parents):
            self.calls += 1
            if self.result is None:
                return None, "pairing failed"
            return self.result.copy(), "pairing ok"

    failing = _Operator(None)
    succeeding = _Operator(child)
    # Probability 1 selects the exploratory operator first.
    dual = DualCutAndSplicePairing(
        succeeding,
        failing,
        1.0,
        rng=np.random.default_rng(0),
    )
    indi, _desc = dual.get_new_individual([parent, parent.copy()])
    assert indi is not None
    assert failing.calls == 1
    assert succeeding.calls == 1
    assert dual.last_attempt_count == 8

    primary = _Operator(child)
    exploratory = _Operator(child)
    dual = DualCutAndSplicePairing(
        primary,
        exploratory,
        0.0,
        rng=np.random.default_rng(0),
    )
    indi, _desc = dual.get_new_individual([parent, parent.copy()])
    assert indi is not None
    assert primary.calls == 1
    assert exploratory.calls == 0
    assert dual.last_attempt_count == 4


def test_create_ga_pairing_returns_single_operator_when_explore_probability_zero(
    pt3_atoms,
):
    from numpy.random import default_rng

    from scgo.algorithms.ga_common import create_ga_pairing

    pairing = create_ga_pairing(
        pt3_atoms,
        len(pt3_atoms),
        default_rng(0),
        exploratory_crossover_probability=0.0,
    )
    assert isinstance(pairing, CutAndSplicePairing)


def test_cut_and_splice_constructor_rejects_legacy_randomstate():
    import numpy as _np

    with pytest.raises(
        SCGOValidationError, match="rng must be an instance of numpy.random.Generator"
    ):
        # Legacy RandomState should be rejected after enforcing Generator-only policy
        CutAndSplicePairing(
            slab=Atoms(),
            n_top=2,
            blmin={},
            system_type="gas_cluster",
            rng=_np.random.RandomState(42),
        )


def test_cut_and_splice_gas_cluster_uses_lower_pairing_attempt_cap(rng):
    pairing = CutAndSplicePairing(
        slab=Atoms(),
        n_top=5,
        blmin={},
        system_type="gas_cluster",
        rng=rng,
    )
    assert pairing.max_pairing_attempts == 150


def test_cut_and_splice_surface_keeps_high_pairing_attempt_cap(rng):
    pairing = CutAndSplicePairing(
        slab=Atoms("C", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True),
        n_top=5,
        blmin={},
        system_type="surface_cluster",
        rng=rng,
    )
    assert pairing.max_pairing_attempts == 1000


def test_cut_and_splice_target_tags_keeps_non_target_groups(rng):
    p1 = Atoms(
        symbols=["Co", "Co", "Co", "O"],
        positions=[
            [0.0, 0.0, 0.0],
            [2.2, 0.0, 0.0],
            [1.1, 1.9, 0.0],
            [1.1, 0.7, 2.0],
        ],
        cell=[12.0, 12.0, 12.0],
        pbc=False,
    )
    p2 = p1.copy()
    p2.positions += np.array(
        [
            [0.10, 0.00, 0.00],
            [0.00, 0.12, 0.00],
            [-0.08, 0.00, 0.10],
            [0.00, 0.00, -0.10],
        ]
    )
    p1.set_tags([0, 0, 0, 1])
    p2.set_tags([0, 0, 0, 1])

    n_top = len(p1)
    # Keep geometric filters permissive; this test targets tag-handling logic.
    co, o = 27, 8
    blmin = {(co, co): 0.01, (co, o): 0.01, (o, co): 0.01, (o, o): 0.01}
    pairing = CutAndSplicePairing(
        slab=Atoms(cell=p1.get_cell(), pbc=p1.get_pbc()),
        n_top=n_top,
        blmin=blmin,
        minfrac=0.5,
        use_tags=True,
        target_tags=[0],
        system_type="gas_cluster_adsorbate",
        rng=rng,
    )

    child = pairing.cross(p1, p2)
    assert child is not None
    assert len(child) == n_top
    assert sorted(child.get_chemical_symbols()) == sorted(p1.get_chemical_symbols())


def _tagged_parents_with_distinct_adsorbates():
    """Two parents sharing a core tag group but with different adsorbates."""
    core = [
        [0.0, 0.0, 0.0],
        [2.2, 0.0, 0.0],
        [1.1, 1.9, 0.0],
    ]
    p1 = Atoms(
        symbols=["Co", "Co", "Co", "O", "H"],
        # Adsorbate O-H along z, 1.0 Angstrom.
        positions=[*core, [1.1, 0.7, 2.4], [1.1, 0.7, 3.4]],
        cell=[14.0, 14.0, 14.0],
        pbc=False,
    )
    p2 = Atoms(
        symbols=["Co", "Co", "Co", "O", "H"],
        # Shifted core and an adsorbate O-H along x, 1.4 Angstrom.
        positions=[
            [0.1, 0.0, 0.0],
            [2.2, 0.15, 0.0],
            [1.0, 1.9, 0.1],
            [1.1, 0.7, 2.4],
            [2.5, 0.7, 2.4],
        ],
        cell=[14.0, 14.0, 14.0],
        pbc=False,
    )
    p1.set_tags([0, 0, 0, 1, 1])
    p2.set_tags([0, 0, 0, 1, 1])
    return p1, p2


def test_cut_and_splice_non_target_groups_always_come_from_first_parent():
    """G2: non-target tag groups must not be pruned at random between parents."""
    p1, p2 = _tagged_parents_with_distinct_adsorbates()
    co, o, h = 27, 8, 1
    blmin = {(z1, z2): 0.01 for z1 in (co, o, h) for z2 in (co, o, h)}
    parent_vector = p1.get_positions()[4] - p1.get_positions()[3]

    for seed in range(20):
        pairing = CutAndSplicePairing(
            slab=Atoms(cell=p1.get_cell(), pbc=p1.get_pbc()),
            n_top=len(p1),
            blmin=blmin,
            minfrac=0.5,
            use_tags=True,
            target_tags=[0],
            system_type="gas_cluster_adsorbate",
            rng=np.random.default_rng(seed),
        )
        child = pairing.cross(p1, p2)
        assert child is not None, f"seed {seed} produced no offspring"
        child_vector = child.get_positions()[4] - child.get_positions()[3]
        assert np.allclose(child_vector, parent_vector, atol=1e-8), (
            f"seed {seed}: adsorbate came from the wrong parent"
        )


def _assert_preserves_atom_identity(child: Atoms, parent: Atoms) -> None:
    assert len(child) == len(parent)
    assert np.array_equal(child.get_atomic_numbers(), parent.get_atomic_numbers())
    if len(parent) > 0:
        assert np.array_equal(child.get_tags(), parent.get_tags())


def test_cut_and_splice_all_moieties_can_mix_adsorbate_from_second_parent():
    """When target_tags is None, adsorbate fragments may inherit parent-1 geometry."""
    p1, p2 = _tagged_parents_with_distinct_adsorbates()
    co, o, h = 27, 8, 1
    blmin = {(z1, z2): 0.01 for z1 in (co, o, h) for z2 in (co, o, h)}
    parent1_length = float(p1.get_distance(3, 4))
    parent2_length = float(p2.get_distance(3, 4))

    saw_parent2_ads = False
    for seed in range(40):
        pairing = CutAndSplicePairing(
            slab=Atoms(cell=p1.get_cell(), pbc=p1.get_pbc()),
            n_top=len(p1),
            blmin=blmin,
            minfrac=0.3,
            use_tags=True,
            target_tags=None,
            system_type="gas_cluster_adsorbate",
            rng=np.random.default_rng(seed),
        )
        child = pairing.cross(p1, p2)
        if child is None:
            continue
        _assert_preserves_atom_identity(child, p1)
        # Alignment rotates the inherited fragment, so the internal bond
        # length identifies the parent rather than the world-frame vector.
        child_length = float(child.get_distance(3, 4))
        if abs(child_length - parent2_length) < 1e-6:
            assert abs(child_length - parent1_length) > 0.2
            saw_parent2_ads = True
            break
    assert saw_parent2_ads


def test_create_ga_pairing_cluster_adsorbate_mixes_all_moieties():
    from scgo.algorithms.ga_common import create_ga_pairing
    from scgo.system_types import AdsorbateDefinition

    p1, _p2 = _tagged_parents_with_distinct_adsorbates()
    comp = p1.get_chemical_symbols()
    ads = AdsorbateDefinition(
        core_symbols=["Co", "Co", "Co"],
        adsorbate_symbols=["O", "H"],
        adsorbate_fragment_lengths=[2],
    )
    pairing = create_ga_pairing(
        p1,
        len(p1),
        rng=np.random.default_rng(0),
        system_type="gas_cluster_adsorbate",
        composition=comp,
        adsorbate_definition=ads,
        exploratory_crossover_probability=0.0,
    )
    assert pairing.use_tags is True
    assert pairing.target_tags is None


class _StubCreator:
    descriptor = "StubMutation"

    def initialize_individual(self, parent, mutant):
        return mutant

    def finalize_individual(self, indi):
        return indi


def test_assert_offspring_integrity_rejects_dropped_atom() -> None:
    parent = Atoms("Au4", positions=np.zeros((4, 3)))
    child = Atoms("Au3", positions=np.zeros((3, 3)))
    with pytest.raises(SCGOValidationError, match="produced 3 atoms"):
        _assert_offspring_integrity(child, parent)


def test_assert_offspring_integrity_rejects_changed_stoichiometry() -> None:
    parent = Atoms("Au4", positions=np.zeros((4, 3)))
    child = Atoms("Au3Pt", positions=np.zeros((4, 3)))
    with pytest.raises(SCGOValidationError, match="stoichiometry"):
        _assert_offspring_integrity(child, parent)


def test_finalize_mutant_rejects_dropped_atom() -> None:
    parent = Atoms("Au4", positions=np.zeros((4, 3)))
    mutant = Atoms("Au3", positions=np.zeros((3, 3)))
    with pytest.raises(SCGOValidationError, match="atom count"):
        _finalize_mutant(_StubCreator(), parent, mutant, "mut")


def test_finalize_mutant_rejects_changed_stoichiometry() -> None:
    parent = Atoms("Au4", positions=np.zeros((4, 3)))
    mutant = Atoms("Au3Pt", positions=np.zeros((4, 3)))
    with pytest.raises(SCGOValidationError, match="stoichiometry"):
        _finalize_mutant(_StubCreator(), parent, mutant, "mut")


def _pt4_parent(seed: int) -> Atoms:
    rng = np.random.default_rng(seed)
    pos = rng.normal(scale=1.0, size=(4, 3))
    atoms = Atoms("Pt4", positions=pos)
    atoms.center(vacuum=4.0)
    atoms.set_cell([10.0, 10.0, 10.0])
    atoms.set_pbc(False)
    atoms.info["confid"] = f"p{seed}"
    return atoms


def _pt_blmin() -> dict:
    return {(78, 78): 0.1}


def test_cross_caches_cell_and_cut_configs_once():
    p1, p2 = _pt4_parent(1), _pt4_parent(2)
    op = CutAndSplicePairing(
        slab=Atoms(),
        n_top=4,
        blmin=_pt_blmin(),
        system_type="gas_cluster",
        rng=np.random.default_rng(7),
    )
    with (
        patch.object(op, "generate_unit_cell", wraps=op.generate_unit_cell) as m_cell,
        patch.object(
            op, "_candidate_cut_configurations", wraps=op._candidate_cut_configurations
        ) as m_cuts,
    ):
        child = op.cross(p1, p2)
    assert child is not None
    assert len(child) == 4
    assert m_cell.call_count == 1
    assert m_cuts.call_count == 1


def test_cross_is_deterministic_with_caching():
    p1, p2 = _pt4_parent(3), _pt4_parent(4)
    op1 = CutAndSplicePairing(
        slab=Atoms(),
        n_top=4,
        blmin=_pt_blmin(),
        system_type="gas_cluster",
        rng=np.random.default_rng(11),
    )
    op2 = CutAndSplicePairing(
        slab=Atoms(),
        n_top=4,
        blmin=_pt_blmin(),
        system_type="gas_cluster",
        rng=np.random.default_rng(11),
    )
    c1 = op1.cross(p1, p2)
    c2 = op2.cross(p1, p2)
    assert c1 is not None and c2 is not None
    assert np.allclose(c1.get_positions(), c2.get_positions())


def _score_cut(prefer_unbalanced: bool, z1, z2, *, minfrac: float | None = None):
    op = CutAndSplicePairing(
        Atoms(cell=[20, 20, 20]),
        n_top=len(z1),
        blmin={(78, 78): 0.5},
        system_type="gas_cluster",
        rng=np.random.default_rng(0),
        prefer_unbalanced=prefer_unbalanced,
        minfrac=minfrac,
    )

    def centers(zs):
        return np.array([[0.0, 0.0, z] for z in zs])

    return op._estimate_cut_balance(
        centers(z1),
        centers(z2),
        np.ones(len(z1)),
        np.zeros(3),
        np.array([0.0, 0.0, 1.0]),
    )


def test_exploratory_pairing_prefers_unbalanced_cuts():
    even, uneven = ([-1.0, 1.0], [-1.0, 1.0]), ([1.0, 2.0], [-2.0, -1.0])
    assert _score_cut(False, *even) < _score_cut(False, *uneven)
    assert _score_cut(True, *uneven) < _score_cut(True, *even)


def test_unbalanced_preference_still_rejects_an_illegal_cut():
    # Legal but lopsided: parent 2 contributes only one atom.
    legal = ([1.0, 2.0, 3.0, 4.0], [-1.0, 1.0, 2.0, 3.0])
    # Parent 2 contributes nothing below the plane, so minfrac is missed.
    illegal = ([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])
    assert _score_cut(True, *legal, minfrac=0.25) < _score_cut(
        True, *illegal, minfrac=0.25
    )


def test_stoichiometry_repair_drops_group_nearest_the_plane():
    far, near, other = (
        SimpleNamespace(distance=2.5),
        SimpleNamespace(distance=-0.2),
        SimpleNamespace(distance=1.1),
    )
    items = [far, near, other]
    assert _pop_closest_to_plane(items) is near
    assert items == [far, other]


def test_second_pass_jitters_the_aligned_cuts():
    atoms = Atoms(
        "Pt4",
        positions=[[0, 0, 0], [2.5, 0, 0], [1.2, 2.2, 0.3], [1.2, 0.8, 2.1]],
        cell=[20, 20, 20],
        pbc=False,
    )
    atoms.center(vacuum=6.0)
    other = atoms.copy()
    op = CutAndSplicePairing(
        Atoms(cell=atoms.cell),
        n_top=4,
        blmin={(78, 78): 0.5},
        system_type="gas_cluster",
        rng=np.random.default_rng(1),
        fixed_cell_max_passes=2,
    )
    jitter_calls = {"n": 0}
    original = op._jittered_cut_configurations

    def _counting_jitter(*args, **kwargs):
        jitter_calls["n"] += 1
        return original(*args, **kwargs)

    op._jittered_cut_configurations = _counting_jitter
    op._get_pairing = lambda *args, **kwargs: None
    assert op.cross(atoms, other) is None
    assert jitter_calls["n"] == 1


def test_parent_alignment_removes_a_rigid_offset():
    reference = Atoms(
        "Pt3",
        positions=[[0.0, 0.0, 0.0], [2.6, 0.0, 0.2], [1.2, 2.3, 0.4]],
        cell=[20, 20, 20],
        pbc=False,
    )
    moving = reference.copy()
    angle = np.deg2rad(70.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    moving.positions[:] = moving.positions @ rotation.T + np.array([4.0, -2.0, 1.0])
    op = CutAndSplicePairing(
        Atoms(cell=reference.cell),
        n_top=3,
        blmin={(78, 78): 0.5},
        system_type="gas_cluster",
        rng=np.random.default_rng(2),
    )
    op._align_moving_to_reference(reference, moving)
    assert np.linalg.norm(moving.positions - reference.positions) < 1e-6
