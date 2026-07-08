"""RNG helper utilities tests."""

import numpy as np
import pytest

from scgo.ase_ga_patches.standardmutations import (
    CustomPermutationMutation,
    RattleMutation,
)
from scgo.exceptions import SCGOValidationError
from scgo.utils.rng_helpers import create_child_rng, ensure_rng, offspring_rng_triple


class TestEnsureRng:
    """Tests for ensure_rng function."""

    def test_with_seed(self):
        rng = ensure_rng(42)
        assert isinstance(rng, np.random.Generator)

    def test_with_none(self):
        rng = ensure_rng(None)
        assert isinstance(rng, np.random.Generator)

    def test_with_zero_seed(self):
        rng = ensure_rng(0)
        assert isinstance(rng, np.random.Generator)

    def test_reproducibility_same_seed(self):
        rng1 = ensure_rng(42)
        rng2 = ensure_rng(42)

        nums1 = [rng1.random() for _ in range(10)]
        nums2 = [rng2.random() for _ in range(10)]

        assert nums1 == nums2

    def test_reproducibility_different_seeds(self):
        rng1 = ensure_rng(42)
        rng2 = ensure_rng(123)

        nums1 = [rng1.random() for _ in range(10)]
        nums2 = [rng2.random() for _ in range(10)]

        assert nums1 != nums2

    def test_large_seed(self):
        rng = ensure_rng(2**63 - 1)
        assert isinstance(rng, np.random.Generator)
        # Should be able to generate numbers
        num = rng.random()
        assert 0.0 <= num < 1.0


class TestCreateChildRng:
    """create_child_rng tests."""

    def test_creates_generator(self):
        parent = np.random.default_rng(42)
        child = create_child_rng(parent)
        assert isinstance(child, np.random.Generator)

    def test_child_independence(self):
        parent = np.random.default_rng(42)

        child = create_child_rng(parent)

        # Generate numbers from parent and child
        parent_nums = [parent.random() for _ in range(5)]
        child_nums = [child.random() for _ in range(5)]

        # Sequences should be different
        assert parent_nums != child_nums

    def test_reproducibility_of_children(self):
        parent1 = np.random.default_rng(42)
        parent2 = np.random.default_rng(42)

        child1 = create_child_rng(parent1)
        child2 = create_child_rng(parent2)

        nums1 = [child1.random() for _ in range(10)]
        nums2 = [child2.random() for _ in range(10)]

        assert nums1 == nums2

    def test_multiple_children_different(self):
        parent = np.random.default_rng(42)

        child1 = create_child_rng(parent)
        child2 = create_child_rng(parent)

        nums1 = [child1.random() for _ in range(10)]
        nums2 = [child2.random() for _ in range(10)]

        # Children should have different sequences
        assert nums1 != nums2

    def test_parent_state_advances(self):
        parent = np.random.default_rng(42)

        # Get parent number before creating child
        num_before = parent.random()

        # Create child
        _child = create_child_rng(parent)

        # Get parent number after creating child
        num_after = parent.random()

        # Numbers should be different (parent state advanced)
        assert num_before != num_after

    def test_nested_children(self):
        parent = np.random.default_rng(42)
        child = create_child_rng(parent)
        grandchild = create_child_rng(child)

        assert isinstance(grandchild, np.random.Generator)

        # All three should produce different sequences
        parent_nums = [parent.random() for _ in range(5)]
        child_nums = [child.random() for _ in range(5)]
        grandchild_nums = [grandchild.random() for _ in range(5)]

        assert parent_nums != child_nums
        assert child_nums != grandchild_nums
        assert parent_nums != grandchild_nums


class TestOffspringRngTriple:
    """offspring_rng_triple tests."""

    def test_three_streams_differ(self):
        pairing_rng, operator_rng, decision_rng = offspring_rng_triple(42)
        draws = [
            [rng.random() for _ in range(10)]
            for rng in (pairing_rng, operator_rng, decision_rng)
        ]
        assert draws[0] != draws[1]
        assert draws[1] != draws[2]
        assert draws[0] != draws[2]

    def test_reproducible_from_task_seed(self):
        triple_a = offspring_rng_triple(999)
        triple_b = offspring_rng_triple(999)
        for rng_a, rng_b in zip(triple_a, triple_b, strict=True):
            assert [rng_a.random() for _ in range(5)] == [
                rng_b.random() for _ in range(5)
            ]


def test_rattle_mutation_deterministic_and_accepts_generator(pt3_atoms, rng):
    # Operator-level determinism: same seed -> same mutation
    rng1 = np.random.default_rng(2021)
    rng2 = np.random.default_rng(2021)

    # Minimal blmin to avoid KeyError checks
    pt = 78
    blmin = {(pt, pt): 0.1}

    mut1 = RattleMutation(
        blmin=blmin, n_top=3, system_type="gas_cluster", rattle_strength=0.1, rng=rng1
    )
    mut2 = RattleMutation(
        blmin=blmin, n_top=3, system_type="gas_cluster", rattle_strength=0.1, rng=rng2
    )

    a = pt3_atoms.copy()
    m1 = mut1.mutate(a)
    m2 = mut2.mutate(a)

    assert m1 is not None
    assert m2 is not None
    assert np.allclose(m1.get_positions(), m2.get_positions())


def test_mutation_constructors_reject_legacy_randomstate():
    import numpy as _np

    with pytest.raises(SCGOValidationError):
        RattleMutation(
            blmin={}, n_top=2, system_type="gas_cluster", rng=_np.random.RandomState(1)
        )

    with pytest.raises(SCGOValidationError):
        CustomPermutationMutation(
            n_top=2, system_type="gas_cluster", rng=_np.random.RandomState(1)
        )


def test_ga_go_rejects_legacy_randomstate():
    import numpy as _np
    from ase.calculators.emt import EMT

    from scgo.algorithms import ga_go

    with pytest.raises(SCGOValidationError):
        ga_go(
            composition=["Pt", "Pt"],
            output_dir=".",
            calculator=EMT(),
            niter=1,
            population_size=2,
            niter_local_relaxation=1,
            mutation_probability=0.1,
            rng=_np.random.RandomState(42),
            clean=True,
        )
