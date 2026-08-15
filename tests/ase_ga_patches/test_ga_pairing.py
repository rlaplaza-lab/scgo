from unittest.mock import patch

import numpy as np
import pytest
from ase import Atoms
from ase_ga.utilities import closest_distances_generator, get_all_atom_types

from scgo.ase_ga_patches.cutandsplicepairing import (
    CutAndSplicePairing,
    DualCutAndSplicePairing,
    _assert_offspring_integrity,
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
