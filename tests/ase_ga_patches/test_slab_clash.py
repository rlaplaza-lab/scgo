"""Tests for _slab_clash: cached slab clash helpers for CutAndSplicePairing.

Coverage:
- SlabClashChecker.is_too_close is boolean-equivalent to
  ase_ga.utilities.atoms_too_close_two_sets on 2D-PBC graphite-like slabs.
- A mobile atom that penetrates a *buried* graphite layer is rejected and would
  be missed by a hypothetical top-layer-only approach.
- mobile_too_close_no_copy reproduces the non-tag branch of atoms_too_close.
- CutAndSplicePairing produces valid, no-clash children with the cached helpers.
- Mobile-only payload round-trip: strip → reconstruct preserves atoms and info.
- Worker result includes pairing_attempt_count > 0 on a successful pairing.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111, graphene
from ase.calculators.emt import EMT
from ase_ga.utilities import (
    atoms_too_close,
    atoms_too_close_two_sets,
    closest_distances_generator,
    get_all_atom_types,
)

from scgo.algorithms.ga_common import create_mutation_operators
from scgo.algorithms.geneticalgorithm_go_torchsim import (
    _OFFSPRING_WORKER_STATE,
    OffspringBuildContext,
    _build_offspring_worker,
    _load_offspring_worker_state,
    _mobile_only_copy,
    _picklable_atoms_copy,
    _reconstruct_full_frame,
)
from scgo.ase_ga_patches._slab_clash import (
    SlabClashChecker,
    mobile_too_close_no_copy,
)
from scgo.ase_ga_patches.cutandsplicepairing import CutAndSplicePairing
from scgo.utils.mutation_weights import get_adaptive_mutation_config

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _graphite_slab(layers: int = 3, repeat_xy: int = 3) -> Atoms:
    """Multi-layer graphene slab with 2D PBC (xy periodic, z open)."""
    layer = graphene(formula="C2", vacuum=0.0)
    layer = layer.repeat((repeat_xy, repeat_xy, 1))
    all_pos = layer.get_positions().copy()
    all_sym = list(layer.get_chemical_symbols())
    cell = layer.get_cell().copy()
    interlayer = 3.35
    for li in range(1, layers):
        shift = np.array([0.0, 0.0, li * interlayer])
        if li % 2 == 1:
            shift = shift + (cell[0] + cell[1]) / 3.0  # Bernal AB stacking
        all_pos = np.vstack([all_pos, layer.get_positions() + shift])
        all_sym += list(layer.get_chemical_symbols())
    cell[2, 2] = (layers - 1) * interlayer + 20.0
    return Atoms(symbols=all_sym, positions=all_pos, cell=cell, pbc=[True, True, False])


def _blmin_for(slab: Atoms, mobile: Atoms) -> dict:
    combined = slab + mobile
    types = get_all_atom_types(combined, list(range(len(slab), len(combined))))
    return closest_distances_generator(types, ratio_of_covalent_radii=0.7)


def _make_checker(slab: Atoms, bl: dict) -> SlabClashChecker:
    return SlabClashChecker(
        slab_numbers=slab.get_atomic_numbers(),
        slab_positions=slab.get_positions(),
        cell=slab.get_cell(),
        pbc=slab.get_pbc(),
        blmin=bl,
    )


# ---------------------------------------------------------------------------
# 1. SlabClashChecker equivalence vs atoms_too_close_two_sets
# ---------------------------------------------------------------------------


class TestSlabCheckerEquivalence:
    """is_too_close must match atoms_too_close_two_sets exactly; tests also
    assert the *direction* of the result so equivalence checks aren't vacuous."""

    def _assert_equiv(
        self, slab: Atoms, mobile: Atoms, bl: dict, *, expect: bool
    ) -> None:
        ase_result = atoms_too_close_two_sets(slab, mobile, bl)
        checker_result = _make_checker(slab, bl).is_too_close(
            mobile.get_atomic_numbers(), mobile.get_positions()
        )
        assert ase_result == expect, (
            f"ASE-GA returned {ase_result} but expected {expect} - test geometry is wrong"
        )
        assert checker_result == expect, (
            f"SlabClashChecker returned {checker_result}, ASE-GA returned {ase_result}"
        )

    def test_clear_above_graphite(self):
        """Mobile cluster 3+ Å above slab – no clash."""
        slab = _graphite_slab(layers=3, repeat_xy=3)
        slab_top = slab.positions[:, 2].max()
        mobile = Atoms(
            "Pt3",
            positions=[
                [3.0, 3.0, slab_top + 3.0],
                [5.0, 3.0, slab_top + 3.5],
                [4.0, 5.0, slab_top + 4.0],
            ],
            cell=slab.cell,
            pbc=slab.pbc,
        )
        bl = _blmin_for(slab, mobile)
        self._assert_equiv(slab, mobile, bl, expect=False)

    def test_clash_top_layer_graphite(self):
        """Mobile Pt placed 0.5 Å above top C – within blmin, clash guaranteed."""
        slab = _graphite_slab(layers=3, repeat_xy=3)
        slab_top = slab.positions[:, 2].max()
        top_c = slab.positions[slab.positions[:, 2] > slab_top - 0.1][0]
        # C–Pt blmin(0.7) ≈ 1.51 Å; 0.5 Å offset < threshold → clash.
        mobile = Atoms(
            "Pt",
            positions=[[top_c[0], top_c[1], top_c[2] + 0.5]],
            cell=slab.cell,
            pbc=slab.pbc,
        )
        bl = _blmin_for(slab, mobile)
        self._assert_equiv(slab, mobile, bl, expect=True)

    def test_clash_buried_layer_graphite(self):
        """Mobile Pt overlapping an interior graphite layer – clash expected."""
        slab = _graphite_slab(layers=3, repeat_xy=3)
        z_layers = np.sort(np.unique(np.round(slab.positions[:, 2], 2)))
        buried_z = z_layers[1]  # middle layer of 3
        buried_c = slab.positions[np.abs(slab.positions[:, 2] - buried_z) < 0.1][0]
        mobile = Atoms(
            "Pt",
            positions=[[buried_c[0] + 0.3, buried_c[1], buried_c[2] + 0.5]],
            cell=slab.cell,
            pbc=slab.pbc,
        )
        bl = _blmin_for(slab, mobile)
        self._assert_equiv(slab, mobile, bl, expect=True)

    def test_pbc_clash_across_cell_boundary(self):
        """Clash detectable only via a PBC image; mobile sits at opposite edge from slab atom."""
        slab = _graphite_slab(layers=1, repeat_xy=2)
        slab_top = slab.positions[:, 2].max()
        # Find a slab atom near x=0 and place mobile near x=cell_x (image clash).
        near_origin = slab.positions[slab.positions[:, 0] < 0.5]
        assert len(near_origin) > 0, "No slab atom near x=0 in test geometry"
        c = near_origin[0]
        cell_x = slab.cell[0, 0]
        # Place Pt 0.5 Å above c, but shifted by ~cell_x so it only clashes via PBC.
        mobile = Atoms(
            "Pt",
            positions=[[c[0] + cell_x - 0.3, c[1], slab_top + 0.5]],
            cell=slab.cell,
            pbc=slab.pbc,
        )
        bl = _blmin_for(slab, mobile)
        # Verify the PBC collapse actually makes this a clash.
        self._assert_equiv(slab, mobile, bl, expect=True)

    @pytest.mark.parametrize("seed", range(10))
    def test_random_placements_graphite(self, seed):
        """Random placements spanning clear (far) and clash (close) regimes."""
        rng = np.random.default_rng(seed)
        slab = _graphite_slab(layers=2, repeat_xy=2)
        slab_top = slab.positions[:, 2].max()
        # Alternate between clear (z > 3.0 Å above slab) and near (z = 0.5–1.2 Å).
        if seed % 2 == 0:
            z = slab_top + rng.uniform(3.0, 5.0)  # clearly clear
        else:
            z = slab_top + rng.uniform(0.3, 0.8)  # likely clash
        mobile = Atoms(
            "Pt",
            positions=[
                [rng.uniform(0, slab.cell[0, 0]), rng.uniform(0, slab.cell[1, 1]), z]
            ],
            cell=slab.cell,
            pbc=slab.pbc,
        )
        bl = _blmin_for(slab, mobile)
        ase_result = atoms_too_close_two_sets(slab, mobile, bl)
        checker_result = _make_checker(slab, bl).is_too_close(
            mobile.get_atomic_numbers(), mobile.get_positions()
        )
        assert checker_result == ase_result

    def test_empty_slab_always_false(self):
        """Empty slab has no atoms to clash against."""
        slab = Atoms(cell=[[10, 0, 0], [0, 10, 0], [0, 0, 20]], pbc=[True, True, False])
        checker = SlabClashChecker(
            slab_numbers=np.array([], dtype=int),
            slab_positions=np.zeros((0, 3)),
            cell=slab.get_cell(),
            pbc=slab.get_pbc(),
            blmin={(78, 6): 1.5},
        )
        assert (
            checker.is_too_close(np.array([78]), np.array([[5.0, 5.0, 5.0]])) is False
        )

    def test_fcc111_slab_3d_pbc(self):
        """Equivalence on FCC(111) slab with full 3D PBC."""
        slab = fcc111("Pt", size=(2, 2, 3), vacuum=8.0, orthogonal=True)
        slab.pbc = True
        slab_top = slab.positions[:, 2].max()
        # Clear case.
        mobile_clear = Atoms(
            "Pt2",
            positions=[[3.0, 3.0, slab_top + 2.5], [5.0, 3.0, slab_top + 3.0]],
            cell=slab.cell,
            pbc=slab.pbc,
        )
        bl = _blmin_for(slab, mobile_clear)
        self._assert_equiv(slab, mobile_clear, bl, expect=False)
        # Clash case: place right on top of a surface atom.
        top_pt = slab.positions[slab.positions[:, 2] > slab_top - 0.1][0]
        mobile_clash = Atoms(
            "Pt",
            positions=[[top_pt[0], top_pt[1], top_pt[2] + 0.5]],
            cell=slab.cell,
            pbc=slab.pbc,
        )
        self._assert_equiv(slab, mobile_clash, bl, expect=True)


# ---------------------------------------------------------------------------
# 2. Buried-layer rejection: contrast with hypothetical top-layer-only check
# ---------------------------------------------------------------------------


def test_buried_layer_missed_by_top_layer_only():
    """A mobile atom inside a buried graphite layer clashes with the full slab
    but would pass a top-layer-only check – verifying SlabClashChecker is necessary.
    """
    slab = _graphite_slab(layers=3, repeat_xy=3)
    z_layers = np.sort(np.unique(np.round(slab.positions[:, 2], 2)))
    top_z = z_layers[-1]
    buried_z = z_layers[1]  # middle layer

    buried_c = slab.positions[np.abs(slab.positions[:, 2] - buried_z) < 0.1][0]
    # Place Pt just 0.4 Å above a buried-layer C.
    pt_pos = np.array([[buried_c[0], buried_c[1], buried_c[2] + 0.4]])
    bl = _blmin_for(slab, Atoms("Pt", positions=pt_pos, cell=slab.cell, pbc=slab.pbc))

    # Full slab: clash detected.
    full_checker = _make_checker(slab, bl)
    assert full_checker.is_too_close(np.array([78]), pt_pos), (
        "Expected clash with buried layer but checker returned False"
    )

    # Top-layer-only slab: the same position does NOT clash.
    top_only_mask = slab.positions[:, 2] > top_z - 0.1
    top_slab = Atoms(
        symbols=[
            s
            for s, m in zip(slab.get_chemical_symbols(), top_only_mask, strict=True)
            if m
        ],
        positions=slab.positions[top_only_mask],
        cell=slab.cell,
        pbc=slab.pbc,
    )
    top_checker = _make_checker(top_slab, bl)
    assert not top_checker.is_too_close(np.array([78]), pt_pos), (
        "Top-layer-only checker should NOT detect buried-layer clash "
        "(that is the whole point of using the full slab)"
    )


# ---------------------------------------------------------------------------
# 3. mobile_too_close_no_copy equivalence and both return values
# ---------------------------------------------------------------------------


class TestMobileTooCloseNoCopy:
    """mobile_too_close_no_copy must return the same bool as
    atoms_too_close(use_tags=False), and each case asserts the expected direction."""

    def _assert_equiv(self, atoms: Atoms, bl: dict, *, expect: bool) -> None:
        ase_result = atoms_too_close(atoms, bl, use_tags=False)
        got = mobile_too_close_no_copy(
            atoms.get_atomic_numbers(),
            atoms.get_positions(),
            atoms.get_pbc(),
            atoms.get_cell(),
            bl,
        )
        assert ase_result == expect, (
            f"ASE-GA returned {ase_result} but expected {expect} - test geometry wrong"
        )
        assert got == expect, (
            f"mobile_too_close_no_copy returned {got}, ASE-GA returned {ase_result}"
        )

    def test_single_atom_clear(self):
        a = Atoms("Pt", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=False)
        self._assert_equiv(a, {(78, 78): 2.5}, expect=False)

    def test_two_atoms_clash_non_pbc(self):
        a = Atoms(
            "Pt2", positions=[[0, 0, 0], [0.5, 0, 0]], cell=[10, 10, 10], pbc=False
        )
        self._assert_equiv(a, {(78, 78): 2.5}, expect=True)

    def test_two_atoms_clear_non_pbc(self):
        a = Atoms(
            "Pt2", positions=[[0, 0, 0], [3.0, 0, 0]], cell=[10, 10, 10], pbc=False
        )
        self._assert_equiv(a, {(78, 78): 2.5}, expect=False)

    def test_pbc_clash_via_image(self):
        """Clash only via PBC wrap; non-PBC version of same geometry would be clear."""
        a = Atoms(
            "Pt2", positions=[[0.1, 5, 5], [9.9, 5, 5]], cell=[10, 10, 10], pbc=True
        )
        bl = {(78, 78): 2.5}
        # Confirm ASE-GA actually sees this as a clash before asserting equivalence.
        self._assert_equiv(a, bl, expect=True)

    def test_pbc_off_same_geometry_clear(self):
        """Same positions without PBC – no clash, tests the branch separately."""
        a = Atoms(
            "Pt2", positions=[[0.1, 5, 5], [9.9, 5, 5]], cell=[10, 10, 10], pbc=False
        )
        self._assert_equiv(a, {(78, 78): 2.5}, expect=False)

    def test_mixed_species_complete_blmin(self):
        """Multi-species cluster: all type-pair thresholds present, clear arrangement."""
        a = Atoms(
            "PtCu", positions=[[0, 0, 0], [3.0, 0, 0]], cell=[10, 10, 10], pbc=False
        )
        bl = {(78, 78): 2.5, (29, 29): 2.0, (29, 78): 2.2, (78, 29): 2.2}
        self._assert_equiv(a, bl, expect=False)

    @pytest.mark.parametrize("seed", range(6))
    def test_random_cluster_both_outcomes(self, seed):
        """Dense half → clash; sparse half → clear; both outcomes covered."""
        rng = np.random.default_rng(seed)
        n = rng.integers(2, 5)
        bl = {(78, 78): 2.5}
        if seed % 2 == 0:
            # Dense: atoms < 1 Å apart → guaranteed clash.
            pos = np.zeros((n, 3))
            pos[:, 0] = np.arange(n) * 0.4
            expect = True
        else:
            # Sparse: atoms >= 3 Å apart → guaranteed clear.
            pos = np.zeros((n, 3))
            pos[:, 0] = np.arange(n) * 3.5
            expect = False
        a = Atoms("Pt" * n, positions=pos, cell=[max(20, n * 4), 10, 10], pbc=False)
        self._assert_equiv(a, bl, expect=expect)


# ---------------------------------------------------------------------------
# 4. CutAndSplicePairing with cached helpers – children pass both clash checks
# ---------------------------------------------------------------------------


def test_cut_and_splice_surface_children_pass_clash_check():
    """Children from the new pairing path must not clash with the slab (verified
    independently using atoms_too_close_two_sets)."""
    slab = fcc111("Pt", size=(2, 2, 2), vacuum=8.0, orthogonal=True)
    slab.pbc = True
    n_slab = len(slab)
    slab_top = slab.positions[:, 2].max()

    n_top = 4
    ads_sym = "Cu" * n_top
    dummy_ads = Atoms(
        ads_sym, positions=[[0, 0, 0]] * n_top, cell=slab.cell, pbc=slab.pbc
    )
    all_types = get_all_atom_types(
        slab + dummy_ads, list(range(n_slab, n_slab + n_top))
    )
    blmin = closest_distances_generator(all_types, ratio_of_covalent_radii=0.5)

    def _make_parent(seed: int) -> Atoms:
        r = np.random.default_rng(seed)
        pos = [
            [
                r.uniform(1, slab.cell[0, 0] - 1),
                r.uniform(1, slab.cell[1, 1] - 1),
                slab_top + 2.5 + i * 2.5,
            ]
            for i in range(n_top)
        ]
        combined = slab + Atoms(ads_sym, positions=pos, cell=slab.cell, pbc=slab.pbc)
        combined.info["confid"] = seed
        return combined

    pairing = CutAndSplicePairing(
        slab,
        n_top,
        blmin,
        test_dist_to_slab=True,
        system_type="surface_cluster",
        rng=np.random.default_rng(0),
    )

    successes = 0
    for p1_seed, p2_seed in [(10, 20), (30, 40), (50, 60), (70, 80)]:
        child, _ = pairing.get_new_individual(
            [_make_parent(p1_seed), _make_parent(p2_seed)]
        )
        if child is None:
            continue
        successes += 1
        mobile = child[n_slab:]
        assert not atoms_too_close_two_sets(slab, mobile, blmin), (
            "Pairing returned a child whose mobile atoms clash with the slab"
        )
        assert len(child) == n_slab + n_top, "Child has wrong atom count"
        assert np.allclose(child.positions[:n_slab], slab.positions), (
            "Slab positions modified by pairing"
        )

    assert successes >= 2, (
        f"Only {successes}/4 pairings succeeded; test geometry may be too constrained"
    )


# ---------------------------------------------------------------------------
# 5. Mobile-only payload round-trip
# ---------------------------------------------------------------------------


def test_mobile_only_copy_and_reconstruct():
    """_mobile_only_copy strips the frozen prefix; _reconstruct_full_frame restores it
    exactly - correct atom count, positions, and info at both ends."""
    slab = fcc111("Pt", size=(2, 2, 2), vacuum=8.0, orthogonal=True)
    slab.pbc = True
    n_slab = len(slab)
    slab_top = slab.positions[:, 2].max()
    n_top = 3
    ads_pos = [[3.0, 3.0, slab_top + 2.5 + i * 2.5] for i in range(n_top)]
    ads = Atoms("Cu" * n_top, positions=ads_pos, cell=slab.cell, pbc=slab.pbc)
    full = slab + ads
    full.info["confid"] = 42
    full.info["key_value_pairs"] = {"extinct": 0}

    mobile = _mobile_only_copy(full, n_slab)

    assert len(mobile) == n_top, "Mobile-only frame has wrong length"
    assert list(mobile.get_chemical_symbols()) == ["Cu"] * n_top, (
        "Wrong species in mobile"
    )
    assert mobile.info["confid"] == 42, "confid not carried through"
    assert mobile.calc is None, "Calculator must be cleared for pickling"
    assert np.allclose(mobile.positions, ads_pos), "Mobile positions differ from ads"

    reconstructed = _reconstruct_full_frame(mobile, slab)

    assert len(reconstructed) == n_slab + n_top, "Reconstructed frame has wrong length"
    assert reconstructed.info["confid"] == 42, "confid lost in reconstruction"
    assert np.allclose(reconstructed.positions[:n_slab], slab.positions), (
        "Slab positions not restored correctly"
    )
    assert np.allclose(reconstructed.positions[n_slab:], ads_pos), (
        "Mobile positions not restored correctly"
    )


# ---------------------------------------------------------------------------
# 6. Worker result includes a positive pairing_attempt_count on success
# ---------------------------------------------------------------------------


def test_pairing_attempt_count_in_worker_result():
    """_build_offspring_worker returns 'pairing_attempt_count' > 0 when pairing succeeds."""
    composition = ["Pt", "Pt", "Pt"]
    atoms_template = Atoms(
        symbols=composition,
        positions=[[i * 2.5, 0, 0] for i in range(3)],
        cell=[20, 20, 20],
        pbc=False,
    )
    atoms_template.calc = EMT()
    all_atom_types = get_all_atom_types(atoms_template, [78])
    blmin = closest_distances_generator(all_atom_types, ratio_of_covalent_radii=0.7)
    operators_list, name_map = create_mutation_operators(
        composition=composition,
        n_to_optimize=3,
        blmin=blmin,
        rng=np.random.default_rng(0),
        use_adaptive=False,
    )
    adaptive_config = get_adaptive_mutation_config(
        composition=composition,
        current_generation=0,
        total_generations=1,
        use_adaptive=False,
        generations_without_improvement=0,
    )
    ctx = OffspringBuildContext(
        atoms_template=_picklable_atoms_copy(atoms_template),
        n_to_optimize=3,
        composition=composition,
        blmin=blmin,
        system_type="gas_cluster",
        n_slab=0,
        n_frozen_prefix=0,
        slab_for_pairing=None,
        surface_normal_axis=2,
        adsorbate_definition=None,
        connectivity_factor=None,
        allow_cluster_fragmentation=False,
        allow_adsorbate_surface_detachment=False,
        enforce_adsorbate_subgraph_integrity=True,
        freeze_adsorbate_internal_geometry=False,
        adsorbate_fragment_templates=None,
        surface_config=None,
        adaptive_config=adaptive_config,
        current_mutation_probability=0.0,
        operators_list=operators_list,
        name_map=name_map,
        operators_epoch=0,
    )
    _OFFSPRING_WORKER_STATE.clear()
    _load_offspring_worker_state(ctx)

    p1 = atoms_template.copy()
    p1.calc = None
    p1.info = {"confid": 1, "key_value_pairs": {}, "data": {}}
    p2 = atoms_template.copy()
    p2.positions = p2.positions + np.array([0.5, 0.3, 0.1])
    p2.calc = None
    p2.info = {"confid": 2, "key_value_pairs": {}, "data": {}}

    # Use multiple seeds so we get at least one successful pairing.
    successful_results = []
    for seed in range(10):
        job = {
            "index": 0,
            "a1": p1,
            "a2": p2,
            "mobile_only": False,
            "task_seed": seed * 1000 + 1,
            "operators_epoch": 0,
            "adaptive_config": adaptive_config,
            "current_mutation_probability": 0.0,
        }
        result = _build_offspring_worker(job)
        assert "pairing_attempt_count" in result, "Result missing pairing_attempt_count"
        assert isinstance(result["pairing_attempt_count"], int)
        if result["child"] is not None:
            successful_results.append(result)

    _OFFSPRING_WORKER_STATE.clear()

    assert successful_results, (
        "No pairing succeeded across 10 seeds - check parent geometry"
    )
    for r in successful_results:
        assert r["pairing_attempt_count"] > 0, (
            f"Successful pairing should have attempt_count > 0, got {r['pairing_attempt_count']}"
        )
