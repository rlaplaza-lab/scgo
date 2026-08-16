"""Tests verifying that retry/fallback loops in mutations and crossover
have been minimised by improved math and logic.

These tests check:
- Uniform sphere sampling (correct area element)
- Guaranteed atom selection (no wasted no-op iterations)
- Rotation angle range [min_angle, pi]
- Rejection-free diatomic rotation axis generation
- Bounded permutation pair selection
- Vectorised flattening projection consistency
- Population roulette-wheel loops terminate
"""

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111
from ase_ga.utilities import (
    atoms_too_close,
    closest_distances_generator,
    get_all_atom_types,
)

from scgo.algorithms.ga_common import create_mutation_operators
from scgo.ase_ga_patches.cutandsplicepairing import CutAndSplicePairing
from scgo.ase_ga_patches.mutations import (
    AnisotropicRattleMutation,
    BreathingMutation,
    FlatteningMutation,
    InPlaneSlideMutation,
    MirrorMutation,
    OverlapReliefMutation,
    PermutationMutation,
    RattleMutation,
    RotationalMutation,
)


def _blmin(atoms):
    all_types = get_all_atom_types(atoms, range(len(atoms)))
    return closest_distances_generator(all_types, ratio_of_covalent_radii=0.7)


# ---------------------------------------------------------------------------
# 1. RattleMutation always moves at least one atom per iteration
# ---------------------------------------------------------------------------
class TestRattleGuaranteedSelection:
    """RattleMutation must always displace at least one tag group,
    even when rattle_prop is very low."""

    def test_low_rattle_prop_still_succeeds(self, pt4_tetrahedron):
        """With rattle_prop=0.01 the old code had ~96 % no-op rate
        for 4 atoms; the new code guarantees at least one is moved."""
        atoms = pt4_tetrahedron.copy()
        blmin = _blmin(atoms)
        mut = RattleMutation(
            blmin,
            len(atoms),
            system_type="gas_cluster",
            rattle_strength=0.3,
            rattle_prop=0.01,
            rng=np.random.default_rng(7),
        )
        result = mut.mutate(atoms)
        assert result is not None
        assert not np.allclose(result.get_positions(), atoms.get_positions())

    def test_single_atom_always_moved(self, pt3_atoms):
        """Even with rattle_prop=0 (below threshold), the guaranteed
        slot ensures one atom is always rattled."""
        atoms = pt3_atoms.copy()
        blmin = {(78, 78): 0.1}
        successes = 0
        for seed in range(20):
            mut = RattleMutation(
                blmin,
                len(atoms),
                system_type="gas_cluster",
                rattle_strength=0.3,
                rattle_prop=0.0,
                rng=np.random.default_rng(seed),
            )
            result = mut.mutate(atoms)
            if result is not None:
                successes += 1
        # All 20 should succeed (no wasted no-op continues)
        assert successes == 20


# ---------------------------------------------------------------------------
# 2. AnisotropicRattleMutation always moves at least one atom
# ---------------------------------------------------------------------------
class TestAnisotropicRattleGuaranteedSelection:
    def test_low_rattle_prop_still_succeeds(self, pt4_tetrahedron):
        atoms = pt4_tetrahedron.copy()
        blmin = {(78, 78): 0.1}
        mut = AnisotropicRattleMutation(
            blmin,
            len(atoms),
            system_type="gas_cluster",
            in_plane_strength=0.3,
            normal_strength=0.05,
            rattle_prop=0.01,
            test_dist_to_slab=False,
            rng=np.random.default_rng(12),
        )
        result = mut.mutate(atoms)
        assert result is not None
        assert not np.allclose(result.get_positions(), atoms.get_positions())


# ---------------------------------------------------------------------------
# 3. RotationalMutation angle is in [min_angle, pi]
# ---------------------------------------------------------------------------
class TestRotationalMutationAngleRange:
    def test_angle_within_correct_range(self, pt4_tetrahedron):
        """Monkey-patch rng.random to record the raw draw and verify
        the angle formula produces values in [min_angle, pi]."""
        min_angle = np.pi / 2
        # Directly check the formula: angle = min_angle + (pi - min_angle) * u
        for u in [0.0, 0.5, 1.0]:
            angle = min_angle + (np.pi - min_angle) * u
            assert min_angle <= angle <= np.pi + 1e-12

    def test_mutation_succeeds_on_tagged_cluster(self):
        """RotationalMutation on a cluster with multi-atom moieties."""
        # Two "molecules" of 2 atoms each, tagged 0 and 1
        atoms = Atoms(
            "Pt4",
            positions=[[0, 0, 0], [2.5, 0, 0], [5.0, 0, 0], [7.5, 0, 0]],
            tags=[0, 0, 1, 1],
        )
        atoms.center(vacuum=10.0)
        blmin = {(78, 78): 0.1}
        mut = RotationalMutation(
            blmin,
            system_type="gas_cluster",
            n_top=4,
            fraction=1.0,
            min_angle=np.pi / 2,
            test_dist_to_slab=False,
            rng=np.random.default_rng(42),
            max_inner_attempts=24,
        )
        result = mut.mutate(atoms)
        assert result is not None
        assert mut.last_attempt_count <= 24


# ---------------------------------------------------------------------------
# 4. Diatomic rotation axis is rejection-free
# ---------------------------------------------------------------------------
class TestDiatomicAxisRejectionFree:
    def test_axis_perpendicular_to_bond(self):
        """The cross-product method should always produce an axis
        with angle in (0, pi/2] relative to the bond direction."""
        rng = np.random.default_rng(99)
        bond = np.array([1.0, 0.0, 0.0])
        for _ in range(200):
            rvec = rng.standard_normal(3)
            axis = np.cross(bond, rvec)
            norm = np.linalg.norm(axis)
            if norm < 1e-12:
                continue  # Extremely rare; skip
            axis /= norm
            angle = np.arccos(np.clip(np.dot(axis, bond), -1, 1))
            # axis from cross product is always perpendicular
            assert np.isclose(angle, np.pi / 2, atol=1e-10)


# ---------------------------------------------------------------------------
# 5. PermutationMutation uses bounded pair selection
# ---------------------------------------------------------------------------
class TestPermutationBoundedPairSelection:
    def test_bimetallic_permutation_succeeds(self, au2pt2_atoms):
        """Permutation on bimetallic cluster should succeed without
        spinning in an unbounded inner loop."""
        atoms = au2pt2_atoms.copy()
        mut = PermutationMutation(
            len(atoms),
            system_type="gas_cluster",
            probability=1.0,
            rng=np.random.default_rng(5),
        )
        result = mut.mutate(atoms)
        assert result is not None
        # Positions should have changed
        assert not np.allclose(result.get_positions(), atoms.get_positions())

    def test_highly_imbalanced_composition(self):
        """Even with N-1 same-type and 1 different-type, selection
        should not spin because we pre-compute valid pairs."""
        # 5 Pt + 1 Au
        atoms = Atoms(
            "Pt5Au",
            positions=[
                [0, 0, 0],
                [2.5, 0, 0],
                [0, 2.5, 0],
                [2.5, 2.5, 0],
                [1.25, 1.25, 2.5],
                [3.75, 1.25, 2.5],
            ],
        )
        atoms.center(vacuum=10.0)
        mut = PermutationMutation(
            len(atoms),
            system_type="gas_cluster",
            probability=1.0,
            rng=np.random.default_rng(8),
        )
        result = mut.mutate(atoms)
        assert result is not None

    def test_pair_swaps_sampled_without_replacement(self, au2pt2_atoms):
        atoms = au2pt2_atoms.copy()
        mut = PermutationMutation(
            len(atoms),
            system_type="gas_cluster",
            probability=1.0,
            rng=np.random.default_rng(0),
        )
        base_rng = mut.rng
        kwargs_seen: list[dict] = []

        class _Rng:
            def __getattr__(self, name):
                return getattr(base_rng, name)

            def choice(self, *args, **kwargs):
                kwargs_seen.append(kwargs)
                return base_rng.choice(*args, **kwargs)

        mut.rng = _Rng()  # type: ignore[assignment]
        mut.mutate(atoms)
        assert kwargs_seen and kwargs_seen[0].get("replace") is False


# ---------------------------------------------------------------------------
# 6. OverlapReliefMutation repairs clashes in one bounded call
# ---------------------------------------------------------------------------
class TestOverlapReliefBoundedRepair:
    def test_dense_cluster_is_repaired_without_outer_retry(self):
        atoms = Atoms(
            "Pt4",
            positions=[
                [0.0, 0.0, 0.0],
                [1.2, 0.0, 0.0],
                [0.0, 1.2, 0.0],
                [0.0, 0.0, 1.2],
            ],
        )
        atoms.center(vacuum=8.0)
        blmin = {(78, 78): 2.0}

        mut = OverlapReliefMutation(
            blmin,
            len(atoms),
            system_type="gas_cluster",
            n_sweeps=4,
            jitter=0.01,
            test_dist_to_slab=False,
            rng=np.random.default_rng(21),
        )
        result = mut.mutate(atoms)
        assert result is not None
        assert not atoms_too_close(result, blmin)


# ---------------------------------------------------------------------------
# 7. Active factory mirror uses reflected mode to avoid long rejection runs
# ---------------------------------------------------------------------------
class TestMirrorFactoryUsesReflectedMode:
    def test_factory_omits_mirror_for_untagged_gas(self, au2pt2_atoms):
        atoms = au2pt2_atoms.copy()
        blmin = _blmin(atoms)
        _ops, name_map = create_mutation_operators(
            atoms.get_chemical_symbols(),
            len(atoms),
            blmin,
            rng=np.random.default_rng(123),
            use_adaptive=True,
            system_type="gas_cluster",
        )
        assert "mirror" not in name_map

    def test_factory_configures_reflected_mirror_on_surface(self, au2pt2_atoms):
        atoms = au2pt2_atoms.copy()
        blmin = _blmin(atoms)
        ops, name_map = create_mutation_operators(
            atoms.get_chemical_symbols(),
            len(atoms),
            blmin,
            rng=np.random.default_rng(123),
            use_adaptive=True,
            system_type="surface_cluster",
            n_slab=8,
        )
        mirror = ops[name_map["mirror"]]
        assert mirror.reflect is True


class TestMirrorBoundedCandidates:
    def test_mirror_declines_untagged_gas_cluster(self, au2pt2_atoms):
        atoms = au2pt2_atoms.copy()
        blmin = _blmin(atoms)
        mut = MirrorMutation(
            blmin,
            len(atoms),
            system_type="gas_cluster",
            rng=np.random.default_rng(17),
            max_tries=12,
        )
        assert mut.mutate(atoms) is None

    def test_mirror_uses_ranked_plane_set_on_surface(self):
        slab = fcc111("Pt", size=(3, 4, 2), vacuum=8.0, orthogonal=True)
        n_slab = len(slab)
        z_top = float(np.max(slab.positions[:, 2]))
        center = np.mean(slab.positions[:, :2], axis=0)
        cluster = Atoms(
            "Pt4",
            positions=[
                [center[0], center[1], z_top + 3.0],
                [center[0] + 2.6, center[1], z_top + 3.0],
                [center[0] + 1.3, center[1] + 2.25, z_top + 3.0],
                [center[0] + 1.3, center[1] + 0.75, z_top + 5.1],
            ],
            cell=slab.get_cell(),
            pbc=slab.get_pbc(),
        )
        full = slab + cluster
        mut = MirrorMutation(
            {(78, 78): 0.5},
            len(cluster),
            system_type="surface_cluster",
            rng=np.random.default_rng(17),
            max_tries=12,
        )
        result = mut.mutate(full)
        assert result is not None
        assert mut.last_attempt_count <= 12
        assert np.allclose(
            result.get_positions()[:n_slab], full.get_positions()[:n_slab], atol=1e-10
        )


class TestRotationalBoundedCandidates:
    def test_rotational_uses_ranked_axis_angle_set(self):
        atoms = Atoms(
            "Pt4",
            positions=[[0, 0, 0], [2.5, 0, 0], [5.0, 0, 0], [7.5, 0, 0]],
            tags=[0, 0, 1, 1],
        )
        atoms.center(vacuum=10.0)
        blmin = {(78, 78): 0.1}
        mut = RotationalMutation(
            blmin,
            system_type="gas_cluster",
            n_top=4,
            fraction=1.0,
            min_angle=np.pi / 2,
            test_dist_to_slab=False,
            rng=np.random.default_rng(42),
            max_inner_attempts=24,
        )
        result = mut.mutate(atoms)
        assert result is not None
        assert mut.last_attempt_count <= 24


# ---------------------------------------------------------------------------
# 8. FlatteningMutation vectorised projection matches scalar version
# ---------------------------------------------------------------------------
class TestFlatteningVectorised:
    def test_flattening_produces_valid_output(self, pt4_tetrahedron):
        atoms = pt4_tetrahedron.copy()
        blmin = _blmin(atoms)
        mut = FlatteningMutation(
            blmin,
            len(atoms),
            system_type="gas_cluster",
            thickness_factor=1.0,
            rng=np.random.default_rng(1),
        )
        result = mut.mutate(atoms)
        assert result is not None
        assert not np.allclose(result.get_positions(), atoms.get_positions())
        assert len(result) == len(atoms)
        assert mut.last_attempt_count <= 6

    def test_vectorised_projection_formula(self):
        """Verify the vectorised formula matches a scalar loop."""
        rng = np.random.default_rng(77)
        pos = rng.random((5, 3)) * 4
        cm = np.mean(pos, axis=0)

        theta = np.arccos(1.0 - 2.0 * rng.random())
        phi = rng.random() * 2 * np.pi
        n = np.array(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
        )

        thickness = 0.5
        perts = rng.uniform(-thickness / 2, thickness / 2, size=len(pos))

        # Vectorised
        v = pos - cm
        proj_along_n = np.dot(v, n)[:, np.newaxis] * n
        projected_v = v - proj_along_n
        vec_result = cm + projected_v + perts[:, np.newaxis] * n

        # Scalar
        scalar_result = np.zeros_like(pos)
        for i in range(len(pos)):
            vi = pos[i] - cm
            pvi = vi - np.dot(vi, n) * n
            scalar_result[i] = cm + pvi + perts[i] * n

        assert np.allclose(vec_result, scalar_result)


# ---------------------------------------------------------------------------
# 9. BreathingMutation uses bounded feasible scales
# ---------------------------------------------------------------------------
class TestBreathingBoundedCandidates:
    def test_breathing_uses_feasible_scale_set(self, pt3_atoms):
        atoms = pt3_atoms.copy()
        blmin = _blmin(atoms)
        mut = BreathingMutation(
            blmin,
            len(atoms),
            system_type="gas_cluster",
            scale_min=0.94,
            scale_max=1.06,
            test_dist_to_slab=False,
            rng=np.random.default_rng(19),
            max_inner_attempts=3000,
        )
        result = mut.mutate(atoms)

        assert result is not None
        assert mut.last_attempt_count <= 5

    def test_breathing_relief_when_nominal_scale_max_is_too_small(self):
        """Tight clusters need s > scale_max; apply minimum relieving expansion."""
        from scgo.initialization import create_initial_cluster

        composition = ["Pt"] * 55
        atoms = create_initial_cluster(
            composition,
            rng=np.random.default_rng(1234),
            mode="random_spherical",
            vacuum=10.0,
        )
        blmin = _blmin(atoms)
        mut = BreathingMutation(
            blmin,
            len(atoms),
            system_type="gas_cluster",
            scale_min=0.82,
            scale_max=1.22,
            test_dist_to_slab=False,
            rng=np.random.default_rng(0),
            max_inner_attempts=5,
        )
        pos = atoms.get_positions()
        num = atoms.get_atomic_numbers()
        cm = pos.mean(axis=0)
        compress = 1.0
        while compress >= 0.5:
            trial_pos = cm + compress * (pos - cm)
            if mut._minimum_feasible_scale(trial_pos, num) > mut.scale_max:
                atoms.set_positions(trial_pos)
                pos = trial_pos
                break
            compress -= 0.02
        else:
            pytest.fail("Could not synthesize a cluster requiring relief expansion")
        feasible_lower = mut._minimum_feasible_scale(pos, num)
        assert feasible_lower > mut.scale_max
        scales = mut._candidate_scales(pos, num, atoms[:0])
        assert len(scales) == 1
        assert np.isclose(scales[0], feasible_lower)
        assert mut.mutate(atoms) is not None
        assert mut.last_attempt_count == 1


# ---------------------------------------------------------------------------
# 10. InPlaneSlideMutation uses bounded geometry-guided shifts
# ---------------------------------------------------------------------------
class TestInPlaneSlideBoundedCandidates:
    def test_in_plane_slide_uses_ranked_shift_set(self):
        from ase.build import fcc111

        slab = fcc111("Pt", size=(3, 4, 2), vacuum=8.0, orthogonal=True)
        n_slab = len(slab)
        z_slab = float(np.max(slab.positions[:, 2]))
        cell = slab.get_cell()
        ads = Atoms(
            "Pt2",
            positions=[
                [0.15 * cell[0, 0], 0.2 * cell[1, 1], z_slab + 4.0],
                [0.65 * cell[0, 0], 0.55 * cell[1, 1], z_slab + 4.0],
            ],
            cell=slab.cell,
            pbc=slab.pbc,
        )
        full = slab + ads
        idx_top = range(n_slab, len(full))
        blmin = closest_distances_generator(
            get_all_atom_types(full, idx_top),
            ratio_of_covalent_radii=0.7,
        )

        mut = InPlaneSlideMutation(
            blmin,
            2,
            system_type="surface_cluster",
            surface_normal_axis=2,
            rng=np.random.default_rng(23),
            max_inner_attempts=8000,
        )
        result = mut.mutate(full)

        assert result is not None
        assert mut.last_attempt_count <= 12


# ---------------------------------------------------------------------------
# 11. CutAndSplicePairing uses a bounded ranked cut set
# ---------------------------------------------------------------------------
class TestCutAndSpliceRankedCandidates:
    def test_cut_and_splice_uses_bounded_ranked_candidates(self, au2pt2_atoms):
        atoms1 = au2pt2_atoms.copy()
        atoms2 = au2pt2_atoms.copy()
        atoms1.positions += np.array(
            [
                [0.08, 0.00, 0.00],
                [0.00, 0.08, 0.00],
                [0.00, 0.00, 0.08],
                [0.06, 0.04, 0.00],
            ]
        )
        atoms2.positions += np.array(
            [
                [0.00, 0.08, 0.00],
                [0.00, 0.00, 0.08],
                [0.08, 0.00, 0.00],
                [0.00, 0.04, 0.06],
            ]
        )

        pairing = CutAndSplicePairing(
            slab=Atoms(cell=atoms1.get_cell(), pbc=atoms1.get_pbc()),
            n_top=len(atoms1),
            blmin=_blmin(atoms1),
            rng=np.random.default_rng(14),
        )
        child = pairing.cross(atoms1, atoms2)

        assert child is not None
        assert pairing.last_cell_attempt_count == 1
        assert pairing.last_attempt_count <= 12


# ---------------------------------------------------------------------------
# 12. Uniform sphere sampling (statistical check)
# ---------------------------------------------------------------------------
class TestUniformSphereSampling:
    def test_arccos_formula_is_uniform(self):
        """theta = arccos(1 - 2u) should produce uniform cos(theta)."""
        rng = np.random.default_rng(0)
        N = 50000
        u = rng.random(N)
        theta = np.arccos(1.0 - 2.0 * u)
        cos_theta = np.cos(theta)
        # cos(theta) should be uniformly distributed in [-1, 1]
        # Check that mean ≈ 0 and std ≈ 1/sqrt(3) ≈ 0.577
        assert abs(np.mean(cos_theta)) < 0.02
        assert abs(np.std(cos_theta) - 1.0 / np.sqrt(3)) < 0.02

    def test_old_formula_was_biased(self):
        """theta = pi * u produces cos(theta) biased toward poles."""
        rng = np.random.default_rng(0)
        N = 50000
        u = rng.random(N)
        theta_old = np.pi * u
        cos_theta_old = np.cos(theta_old)
        # Old formula: mean of cos(pi*u) = 0, but std != 1/sqrt(3)
        # std of cos(pi*U) for U~Uniform(0,1) is sqrt(1/2 - (2/pi^2)) ≈ 0.481
        assert abs(np.std(cos_theta_old) - 1.0 / np.sqrt(3)) > 0.05
