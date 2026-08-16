"""Tests for helper utility functions.

This module tests various utility functions used throughout the SCGO package.
"""

import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms

from scgo.constants import PENALTY_ENERGY
from scgo.exceptions import SCGORuntimeError
from scgo.metadata.atoms import compute_final_id, ensure_final_id, get_tag, get_tags
from scgo.utils.helpers import (
    _assign_penalty_energy,
    auto_niter,
    auto_niter_local_relaxation,
    auto_niter_ts,
    auto_population_size,
    canonicalize_relaxed_for_storage,
    canonicalize_storage_frame,
    deep_merge_dicts,
    ensure_float64_forces,
    filter_unique_minima,
    get_cluster_formula,
    get_ordered_formula,
    get_system_path_key,
    perform_local_relaxation,
)
from tests.helpers import _make_minima_atoms


class TestDeepMergeDicts:
    def test_empty_override_returns_independent_copy(self):
        base = {"a": 1, "nested": {"b": 2}}
        merged = deep_merge_dicts(base, {})
        assert merged == base
        assert merged is not base
        assert merged["nested"] is not base["nested"]
        merged["nested"]["b"] = 99
        assert base["nested"]["b"] == 2

    def test_empty_override_copy_base_false_returns_same_object(self):
        base = {"a": 1}
        merged = deep_merge_dicts(base, {}, copy_base=False)
        assert merged is base

    def test_nested_override_wins(self):
        base = {"a": 1, "nested": {"b": 2, "c": 3}}
        merged = deep_merge_dicts(base, {"nested": {"b": 20}})
        assert merged == {"a": 1, "nested": {"b": 20, "c": 3}}
        assert base["nested"]["b"] == 2


def _reference_auto_scale(composition, *, base, scaling, min_val, max_val):
    """Compute the deterministic auto-scale reference from the live formula.

    Mirrors ``scgo.utils.helpers._auto_scale_parameter`` exactly so the tightened
    tests cannot silently drift if the source constants change.
    """
    n_atoms = max(len(composition), 1)
    return int(np.clip(base + scaling * np.log1p(n_atoms), min_val, max_val))


def test_get_tags_reads_key_value_pairs():
    atoms = Atoms("Pt", positions=[[0, 0, 0]])
    atoms.info = {"key_value_pairs": {"run_id": "run_test_abc"}}
    assert get_tags(atoms)["run_id"] == "run_test_abc"


class TestFilterUniqueMinima:
    """Tests for filter_unique_minima function."""

    def test_filter_unique_minima_empty(self):
        """Test filtering of empty minima list."""
        result = filter_unique_minima([], n_top=1)
        assert len(result) == 0

    def test_filter_unique_minima_single(self):
        """Test filtering of single minimum."""
        atoms = _make_minima_atoms(1.0, "run_test_1")

        result = filter_unique_minima([(1.0, atoms)], n_top=1)
        assert len(result) == 1

    def test_filter_unique_minima_basic(self):
        """Test basic filtering functionality."""
        # Create atoms with required metadata
        atoms1 = _make_minima_atoms(1.0, "run_test_1")

        atoms2 = Atoms("Pt", positions=[[1, 1, 1]])  # Different position
        atoms2.info = {
            "key_value_pairs": {"raw_score": 2.0, "run_id": "run_test_1"},
        }

        # Should return both since they have different positions
        result = filter_unique_minima([(1.0, atoms1), (2.0, atoms2)], n_top=1)
        assert len(result) == 2

    def test_filter_unique_minima_tags_only_returned_uniques(self):
        kept = Atoms("Pt", positions=[[0.0, 0.0, 0.0]])
        discarded = Atoms("Pt", positions=[[1e-4, 0.0, 0.0]])
        result = filter_unique_minima([(1.0, kept), (1.001, discarded)], n_top=1)
        assert result == [(1.0, kept)]
        assert get_tag(kept, "raw_score") == pytest.approx(-1.0)
        assert get_tag(discarded, "raw_score") is None

    def test_filter_unique_minima_ignores_fixed_slab_atom_differences(self):
        """Minima dedup should ignore changes confined to fixed slab atoms."""
        base = Atoms(
            "Pt4",
            positions=[
                [0.0, 0.0, 0.0],
                [1.2, 0.0, 0.0],
                [0.6, 1.0, 0.0],
                [0.6, 0.4, 1.8],
            ],
        )
        base.set_constraint(FixAtoms(indices=[0, 1, 2]))
        slab_shifted = base.copy()
        slab_shifted.set_constraint(FixAtoms(indices=[0, 1, 2]))
        p = slab_shifted.get_positions()
        p[:3, 2] += 0.3
        slab_shifted.set_positions(p)

        result = filter_unique_minima(
            [(0.0, base), (0.001, slab_shifted)], n_top=1, mic=False
        )
        assert len(result) == 1

    def test_filter_unique_minima_n_top_trailing_only(self):
        """When n_top=1, only the trailing atom is compared (GA-style surface tail)."""
        tail = [0.6, 0.4, 1.8]
        base = Atoms(
            "Pt4",
            positions=[
                [0.0, 0.0, 0.0],
                [1.2, 0.0, 0.0],
                [0.6, 1.0, 0.0],
                tail,
            ],
        )
        shifted = Atoms(
            "Pt4",
            positions=[
                [0.0, 0.0, 0.5],
                [1.2, 0.0, 0.5],
                [0.6, 1.0, 0.5],
                tail,
            ],
        )
        out = filter_unique_minima([(0.0, base), (0.001, shifted)], n_top=1, mic=False)
        assert len(out) == 1

    def test_filter_unique_minima_mic_keyword_smoke(self):
        """mic=True is accepted and runs for a small gas-phase duplicate pair."""
        atoms1 = Atoms("Cu2", positions=[[0, 0, 0], [2.5, 0, 0]])
        atoms1.center(vacuum=5.0)
        atoms2 = atoms1.copy()
        out = filter_unique_minima([(1.0, atoms1), (1.0001, atoms2)], n_top=2, mic=True)
        assert len(out) == 1

    def test_filter_unique_minima_preserves_energy_order(self):
        """Output is energy-ascending even when input order is non-monotonic.

        Locks the invariant relied on by T1: ``_find_unique_minima_with_binning``
        appends in the order of the already energy-sorted input, so the result is
        energy-ascending by construction (no trailing re-sort).
        """
        a0 = Atoms("Pt", positions=[[0.0, 0.0, 0.0]])
        a1 = Atoms("Pt", positions=[[1.0, 0.0, 0.0]])
        a2 = Atoms("Pt", positions=[[2.0, 0.0, 0.0]])
        out = filter_unique_minima([(2.0, a2), (-1.0, a0), (0.5, a1)], n_top=1)
        energies = [e for e, _ in out]
        assert energies == sorted(energies)
        assert energies == [-1.0, 0.5, 2.0]


class TestEnsureFinalId:
    def test_ensure_final_id_is_idempotent(self):
        atoms = Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
        fid1 = ensure_final_id(atoms, -1.0)
        fid2 = ensure_final_id(atoms, -2.0)
        assert fid1 == fid2
        assert atoms.info["key_value_pairs"]["final_id"] == fid1

    def test_ensure_final_id_matches_compute(self):
        atoms = Atoms("Pt2", positions=[[0, 0, 0], [2.6, 0, 0]])
        energy = -0.42
        assert ensure_final_id(atoms, energy) == compute_final_id(atoms, energy)


class TestAutoNiter:
    """Tests for auto_niter function."""

    @pytest.mark.parametrize(
        "composition,expected",
        [
            (["Pt"], 27),
            (["Pt", "Pt"], 41),
            (["Pt"] * 5, 65),
            (["Pt"] * 10, 86),
            (["Pt"] * 20, 109),
            (["Pt"] * 50, 140),
        ],
    )
    def test_auto_niter_scaling(self, composition, expected):
        """auto_niter must match the deterministic log1p scaling exactly."""
        result = auto_niter(composition)
        assert result == expected
        assert result == _reference_auto_scale(
            composition, base=3, scaling=35, min_val=3, max_val=1000
        )

    def test_auto_niter_empty_composition(self):
        """Empty composition yields n_atoms=1 (not 0), so the base offset applies."""
        assert auto_niter([]) == 27

    def test_auto_niter_monotonic(self):
        """Scaling must be non-decreasing with cluster size."""
        prev = -1
        for k in [1, 2, 5, 10, 20, 50]:
            val = auto_niter(["Pt"] * k)
            assert val >= prev
            prev = val

    def test_auto_niter_clipped(self):
        """Result stays within [min, max] for extreme compositions."""
        assert auto_niter(["Pt"]) >= 3
        assert auto_niter(["Pt"] * 5000) <= 1000

    @pytest.mark.reproducibility
    def test_auto_niter_reproducibility(self):
        """Test that auto_niter is reproducible."""
        composition = ["Pt"] * 5
        result1 = auto_niter(composition)
        result2 = auto_niter(composition)
        assert result1 == result2

    def test_auto_niter_different_elements(self):
        """Test auto_niter with different element types."""
        # Should work with any composition
        compositions = [
            ["H", "H"],
            ["Pt", "Au"],
            ["Pt", "Au", "Pd"],
            ["H", "He", "Li", "Be"],
        ]

        for comp in compositions:
            result = auto_niter(comp)
            assert isinstance(result, int)
            assert result > 0


class TestAutoPopulationSize:
    """Tests for auto_population_size function."""

    @pytest.mark.parametrize(
        "composition,expected",
        [
            (["Pt"], 27),
            (["Pt", "Pt"], 41),
            (["Pt"] * 5, 65),
            (["Pt"] * 10, 86),
            (["Pt"] * 20, 109),
            (["Pt"] * 50, 140),
        ],
    )
    def test_auto_population_size_scaling(self, composition, expected):
        """auto_population_size shares the same deterministic scaling as auto_niter."""
        result = auto_population_size(composition)
        assert result == expected
        assert result == _reference_auto_scale(
            composition, base=3, scaling=35, min_val=3, max_val=1000
        )

    def test_auto_population_size_monotonic(self):
        """Scaling must be non-decreasing with cluster size."""
        prev = -1
        for k in [1, 2, 5, 10, 20, 50]:
            val = auto_population_size(["Pt"] * k)
            assert val >= prev
            prev = val

    def test_auto_population_size_clipped(self):
        """Result stays within [min, max] for extreme compositions."""
        assert auto_population_size(["Pt"]) >= 3
        assert auto_population_size(["Pt"] * 5000) <= 1000

    @pytest.mark.reproducibility
    def test_auto_population_size_reproducibility(self):
        """Test that auto_population_size is reproducible."""
        composition = ["Pt"] * 5
        result1 = auto_population_size(composition)
        result2 = auto_population_size(composition)
        assert result1 == result2

    def test_auto_population_size_different_elements(self):
        """Test auto_population_size with different element types."""
        compositions = [
            ["H", "H"],
            ["Pt", "Au"],
            ["Pt", "Au", "Pd"],
            ["H", "He", "Li", "Be"],
        ]

        for comp in compositions:
            result = auto_population_size(comp)
            assert isinstance(result, int)
            assert result > 0


class TestAutoNiterLocalRelaxation:
    """Tests for auto_niter_local_relaxation function."""

    @pytest.mark.parametrize(
        "composition,expected",
        [
            (["Pt"], 84),
            (["Pt", "Pt"], 104),
            (["Pt"] * 5, 139),
            (["Pt"] * 10, 169),
            (["Pt"] * 20, 202),
            (["Pt"] * 50, 246),
        ],
    )
    def test_auto_niter_local_relaxation_scaling(self, composition, expected):
        """auto_niter_local_relaxation must match the deterministic scaling exactly."""
        result = auto_niter_local_relaxation(composition)
        assert result == expected
        assert result == _reference_auto_scale(
            composition, base=50, scaling=50, min_val=50, max_val=2000
        )

    @pytest.mark.reproducibility
    def test_auto_niter_local_relaxation_reproducibility(self):
        """Test that auto_niter_local_relaxation is reproducible."""
        composition = ["Pt"] * 5
        result1 = auto_niter_local_relaxation(composition)
        result2 = auto_niter_local_relaxation(composition)
        assert result1 == result2

    def test_auto_niter_local_relaxation_different_elements(self):
        """Test auto_niter_local_relaxation with different element types."""
        compositions = [
            ["H", "H"],
            ["Pt", "Au"],
            ["Pt", "Au", "Pd"],
            ["H", "He", "Li", "Be"],
        ]

        for comp in compositions:
            result = auto_niter_local_relaxation(comp)
            assert isinstance(result, int)
            assert result >= 50  # Minimum is 50

    def test_auto_niter_local_relaxation_increases_with_size(self):
        """Test that relaxation steps increase with cluster size."""
        sizes = [2, 5, 10, 20, 50]
        results = [auto_niter_local_relaxation(["Pt"] * n) for n in sizes]
        # Should generally increase (allowing for some non-monotonicity due to rounding)
        assert results[-1] >= results[0]  # Largest should be >= smallest

    def test_auto_niter_local_relaxation_monotonic(self):
        """Scaling must be non-decreasing with cluster size."""
        prev = -1
        for k in [1, 2, 5, 10, 20, 50]:
            val = auto_niter_local_relaxation(["Pt"] * k)
            assert val >= prev
            prev = val

    def test_auto_niter_local_relaxation_clipped(self):
        """Result stays within [min, max] for extreme compositions."""
        assert auto_niter_local_relaxation(["Pt"]) >= 50
        assert auto_niter_local_relaxation(["Pt"] * 5000) <= 2000


class TestAutoNiterTS:
    """Tests for `auto_niter_ts` (TS/NEB auto-scaling helper)."""

    @pytest.mark.parametrize(
        "composition,expected_range",
        [
            (["Pt"], (150, 220)),
            (["Pt", "Pt"], (220, 280)),
            (["Pt"] * 5, (350, 400)),
            (["Pt"] * 6, (380, 420)),
            (["Pt"] * 10, (430, 520)),
            (["Pt"] * 20, (520, 650)),
            (["Pt"] * 50, (700, 900)),
        ],
    )
    def test_auto_niter_ts_scaling(self, composition, expected_range):
        result = auto_niter_ts(composition)
        assert expected_range[0] <= result <= expected_range[1]

    @pytest.mark.parametrize(
        "composition,expected",
        [
            (["Pt"], 174),
            (["Pt"] * 5, 372),
            (["Pt"] * 10, 481),
            (["Pt"] * 50, 757),
        ],
    )
    def test_auto_niter_ts_exact(self, composition, expected):
        """auto_niter_ts must match the deterministic scaling exactly."""
        result = auto_niter_ts(composition)
        assert result == expected
        assert result == _reference_auto_scale(
            composition, base=50, scaling=180, min_val=150, max_val=5000
        )

    def test_auto_niter_ts_clipped(self):
        """Result stays within [min, max] for extreme compositions."""
        assert auto_niter_ts(["Pt"]) >= 150
        assert auto_niter_ts(["Pt"] * 5000) <= 5000

    @pytest.mark.reproducibility
    def test_auto_niter_ts_reproducibility(self):
        composition = ["Pt"] * 6
        result1 = auto_niter_ts(composition)
        result2 = auto_niter_ts(composition)
        assert result1 == result2

    def test_auto_niter_ts_different_elements(self):
        compositions = [
            ["H", "H"],
            ["Pt", "Au"],
            ["Pt", "Au", "Pd"],
            ["H", "He", "Li", "Be"],
        ]

        for comp in compositions:
            result = auto_niter_ts(comp)
            assert isinstance(result, int)
            assert result >= 150  # minimum is 150

    def test_auto_niter_ts_increases_with_size(self):
        sizes = [1, 6, 10, 20]
        results = [auto_niter_ts(["Pt"] * n) for n in sizes]
        assert results[-1] >= results[0]


class TestGetClusterFormula:
    """Tests for get_cluster_formula function."""

    @pytest.mark.parametrize(
        "composition,expected",
        [
            (["Pt"], "Pt"),
            (["Pt", "Pt"], "Pt2"),
            (["Pt", "Au"], "AuPt"),
            (["Pt", "Pt", "Au"], "AuPt2"),
            (["Pt", "Au", "Pd"], "AuPdPt"),
        ],
    )
    def test_get_cluster_formula(self, composition, expected):
        """Test get_cluster_formula with various compositions."""
        result = get_cluster_formula(composition)
        assert result == expected

    def test_get_cluster_formula_empty(self):
        """Test get_cluster_formula with empty composition."""
        result = get_cluster_formula([])
        assert result == ""

    def test_get_cluster_formula_performance(self):
        """Test get_cluster_formula performance with large clusters."""
        # Create a large composition
        composition = ["Pt"] * 50

        # Should complete quickly
        result = get_cluster_formula(composition)
        assert result == "Pt50"


class TestGetSystemPathKey:
    """Tests for component-aware path keys."""

    def test_ordered_formula_preserves_input_order(self):
        assert get_ordered_formula(["O", "H"]) == "OH"
        assert get_ordered_formula(["H", "O", "H"]) == "H2O"

    def test_gas_cluster_only(self):
        assert get_system_path_key(["Pt"] * 5) == "Pt5"

    def test_gas_with_adsorbates(self):
        ads_def = {
            "core_symbols": ["Pt"] * 5,
            "adsorbate_symbols": ["O", "H", "O", "H"],
            "adsorbate_fragment_lengths": [2, 2],
        }
        assert (
            get_system_path_key(
                ["Pt"] * 5 + ["O", "H", "O", "H"],
                adsorbate_definition=ads_def,
            )
            == "Pt5_OH_OH"
        )

    def test_surface_with_adsorbates(self):
        ads_def = {
            "core_symbols": ["Pt"] * 5,
            "adsorbate_symbols": ["O", "H", "O", "H"],
            "adsorbate_fragment_lengths": [2, 2],
        }
        assert (
            get_system_path_key(
                ["Pt"] * 5 + ["O", "H", "O", "H"],
                adsorbate_definition=ads_def,
                surface_name="graphite",
            )
            == "Pt5_OH_OH_graphite"
        )

    def test_surface_default_slab_name(self):
        assert get_system_path_key(["Pt"] * 5, surface_name="slab") == "Pt5_slab"

    def test_go_ts_alignment_matches_expected_keys(self):
        """GO resolve_run_path_key and TS get_system_path_key must agree."""
        from scgo.runner_params import resolve_run_path_key
        from scgo.surface.presets import make_graphite_surface_config

        ads_def = {
            "core_symbols": ["Pt"] * 5,
            "adsorbate_symbols": ["O", "H", "O", "H"],
            "adsorbate_fragment_lengths": [2, 2],
        }
        cfg = make_graphite_surface_config(slab_layers=2, slab_repeat_xy=1)
        full = ["Pt"] * 5 + ["O", "H", "O", "H"]
        go = resolve_run_path_key(
            full,
            system_type="surface_cluster_adsorbate",
            adsorbate_definition=ads_def,
            surface_config=cfg,
        )
        ts = get_system_path_key(
            full,
            adsorbate_definition=ads_def,
            surface_name=cfg.name,
        )
        assert go == ts == "Pt5_OH_OH_graphite"


class TestEnsureFloat64Forces:
    """Tests for ensure_float64_forces utility function."""

    def test_ensure_float64_forces_converts_float32(self):
        """Test that float32 forces are converted to float64."""
        from ase.calculators.emt import EMT

        atoms = Atoms("Pt2", positions=[[0, 0, 0], [1, 0, 0]])
        atoms.calc = EMT()

        # Simulate float32 forces
        forces_f32 = atoms.get_forces().astype(np.float32)
        atoms.arrays["forces"] = forces_f32

        # Apply conversion
        ensure_float64_forces(atoms)

        # Verify forces are now float64
        assert atoms.arrays["forces"].dtype == np.float64

    def test_ensure_float64_forces_updates_calc_results(self):
        """Test that calculator results are also updated."""
        from ase.calculators.emt import EMT

        atoms = Atoms("Pt2", positions=[[0, 0, 0], [1, 0, 0]])
        atoms.calc = EMT()

        # Get forces and convert to float32
        forces_f32 = atoms.get_forces().astype(np.float32)
        atoms.arrays["forces"] = forces_f32
        atoms.calc.results["forces"] = forces_f32

        # Apply conversion
        ensure_float64_forces(atoms)

        # Verify both locations updated
        assert atoms.arrays["forces"].dtype == np.float64
        assert atoms.calc.results["forces"].dtype == np.float64

    def test_ensure_float64_forces_no_forces(self):
        """Test handling when no forces are available."""
        atoms = Atoms("Pt2", positions=[[0, 0, 0], [1, 0, 0]])
        # No calculator attached

        # Should raise RuntimeError when no calculator is attached
        with pytest.raises(SCGORuntimeError, match="no calculator"):
            ensure_float64_forces(atoms)

    def test_ensure_float64_forces_preserves_values(self):
        """Test that force values are preserved during conversion."""
        from ase.calculators.emt import EMT

        atoms = Atoms("Pt2", positions=[[0, 0, 0], [1, 0, 0]])
        atoms.calc = EMT()

        # Get original forces
        original_forces = atoms.get_forces().copy()

        # Convert to float32
        atoms.arrays["forces"] = original_forces.astype(np.float32)

        # Apply conversion
        ensure_float64_forces(atoms)

        # Values should be approximately equal (allowing for float32 precision loss)
        np.testing.assert_allclose(atoms.arrays["forces"], original_forces, rtol=1e-5)


class TestCanonicalizeStorageFrame:
    """Tests for translation-only canonicalization before persistence."""

    def test_non_pbc_recenter_to_cell_midpoint(self):
        atoms = Atoms(
            "Pt2",
            positions=[[1.0, 2.0, 3.0], [2.5, 2.0, 3.0]],
            cell=[10.0, 10.0, 10.0],
            pbc=False,
        )
        canonicalize_storage_frame(atoms)
        com = atoms.get_center_of_mass()
        np.testing.assert_allclose(com, [5.0, 5.0, 5.0], atol=1e-8)

    def test_pbc_wraps_before_centering(self):
        atoms = Atoms(
            "Pt2",
            positions=[[-0.2, 0.1, 0.1], [9.8, 0.1, 0.1]],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        )
        canonicalize_storage_frame(atoms)
        positions = atoms.get_positions()
        assert np.all(positions >= -1e-8)
        assert np.all(positions <= 10.0 + 1e-8)

    def test_can_skip_centering_for_surface_frames(self):
        atoms = Atoms(
            "Pt2",
            positions=[[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]],
            cell=[10.0, 10.0, 10.0],
            pbc=False,
        )
        before = atoms.get_center_of_mass().copy()
        canonicalize_storage_frame(atoms, center=False)
        after = atoms.get_center_of_mass()
        np.testing.assert_allclose(after, before, atol=1e-8)


class TestCanonicalizeRelaxedForStorage:
    """Post-relaxation frame used by GA persistence and ASE local relaxation."""

    def test_gas_cluster_bbox_centered_in_cell(self):
        atoms = Atoms(
            "Pt3",
            positions=[[7.0, 7.2, 8.0], [8.5, 7.5, 7.3], [7.8, 8.1, 7.9]],
            cell=[15.58, 15.58, 15.58],
            pbc=False,
        )
        canonicalize_relaxed_for_storage(atoms)
        bbox_center = 0.5 * (
            atoms.get_positions().min(axis=0) + atoms.get_positions().max(axis=0)
        )
        np.testing.assert_allclose(
            bbox_center,
            np.diag(atoms.cell) / 2.0,
            atol=1e-8,
        )

    def test_slab_adsorbate_uses_primary_cell_shift_not_centering(self):
        cell = [[10.0, 0, 0], [0, 10.0, 0], [0, 0, 15.0]]
        atoms = Atoms(
            ["Pt", "Pt"],
            positions=[[1.0, 5.0, 2.0], [12.0, 5.0, 3.0]],
            cell=cell,
            pbc=[True, True, False],
        )
        slab_z_before = atoms.positions[0, 2]
        canonicalize_relaxed_for_storage(atoms, surface_mode=True, n_slab=1)
        scaled = atoms.get_scaled_positions(wrap=False)[1:]
        ads_com = np.average(scaled, axis=0, weights=atoms.get_masses()[1:])
        assert 0 <= ads_com[0] < 1
        assert 0 <= ads_com[1] < 1
        assert atoms.positions[0, 2] == pytest.approx(slab_z_before)


class TestCanonicalizeSlabAdsorbateFrame:
    """Rigid lattice shift for slab+adsorbate under in-plane PBC."""

    def test_adsorbate_com_fractional_in_unit_cell_after_canonicalize(self):
        cell = [[10.0, 0, 0], [0, 10.0, 0], [0, 0, 15.0]]
        pbc = [True, True, False]
        atoms = Atoms(
            ["Pt", "Pt"],
            positions=[[1.0, 5.0, 2.0], [12.0, 5.0, 3.0]],
            cell=cell,
            pbc=pbc,
        )
        canonicalize_storage_frame(atoms, pbc_aware=True, center=False, n_slab=1)
        scaled = atoms.get_scaled_positions(wrap=False)[1:]
        com = np.average(scaled, axis=0, weights=atoms.get_masses()[1:])
        assert 0 <= com[0] < 1
        assert 0 <= com[1] < 1

    def test_lattice_translated_pairs_match_after_canonicalize(self):
        cell = [[10.0, 0, 0], [0, 10.0, 0], [0, 0, 15.0]]
        pbc = [True, True, False]
        a = Atoms(
            ["Pt", "Pt"],
            positions=[[1.0, 5.0, 2.0], [12.0, 5.0, 3.0]],
            cell=cell,
            pbc=pbc,
        )
        b = a.copy()
        b.positions += np.array([-10.0, 0.0, 0.0])
        canonicalize_storage_frame(a, pbc_aware=True, center=False, n_slab=1)
        canonicalize_storage_frame(b, pbc_aware=True, center=False, n_slab=1)
        np.testing.assert_allclose(a.get_positions(), b.get_positions(), atol=1e-10)

    def test_n_slab_zero_still_wraps_and_centers_pbc_cluster(self):
        """Regression: gas-style PBC dimer path unchanged when n_slab is 0."""
        atoms = Atoms(
            "Pt2",
            positions=[[-0.2, 0.1, 0.1], [9.8, 0.1, 0.1]],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        )
        canonicalize_storage_frame(atoms, n_slab=0)
        positions = atoms.get_positions()
        assert np.all(positions >= -1e-8)
        assert np.all(positions <= 10.0 + 1e-8)


def test_perform_local_relaxation_surface_mode_skips_com_center():
    """Surface-mode post-relax canonicalize leaves slab COM unshifted."""
    from ase.build import fcc111
    from ase.calculators.emt import EMT

    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    slab.pbc = [True, True, False]
    n_slab = len(slab)
    z0 = float(slab.get_positions()[:, 2].max() + 1.5)
    atoms = slab.copy() + Atoms("Pt", positions=[[1.0, 1.0, z0]])
    slab_com_before = atoms.get_positions()[:n_slab].mean(axis=0).copy()

    class _NoOpOpt:
        def __init__(self, atoms, **kwargs):
            self.atoms = atoms

        def run(self, fmax=0.05, steps=1):
            return True

    perform_local_relaxation(
        atoms,
        EMT(),
        _NoOpOpt,
        fmax=1.0,
        steps=1,
        center_after_relax=False,
        surface_mode=True,
        n_slab=n_slab,
    )
    slab_com_after = atoms.get_positions()[:n_slab].mean(axis=0)
    np.testing.assert_allclose(slab_com_after, slab_com_before, atol=1e-8)


def test_assign_penalty_energy_attaches_single_point_calculator():
    """Penalty path must serve energy/forces via SPC without the old calc."""

    class _RaisingCalc:
        def get_potential_energy(self, atoms=None, force_consistent=False):
            raise RuntimeError("broken calculator")

        def get_forces(self, atoms=None):
            raise RuntimeError("broken calculator")

    atoms = Atoms("Pt2", positions=[[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
    atoms.calc = _RaisingCalc()
    with pytest.raises(RuntimeError, match="broken calculator"):
        atoms.get_potential_energy()

    energy = _assign_penalty_energy(atoms)
    assert energy == PENALTY_ENERGY
    assert atoms.get_potential_energy() == PENALTY_ENERGY
    forces = atoms.get_forces()
    assert forces.shape == (2, 3)
    np.testing.assert_allclose(forces, 0.0)
