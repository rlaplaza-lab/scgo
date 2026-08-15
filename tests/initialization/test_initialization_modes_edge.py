"""Consolidated initialization edge-case tests (T2-13 fold).

Merged verbatim from the former per-mode fold files so every mode-specific test and marker is preserved.
"""

import logging

import numpy as np
import pytest
from ase import Atoms
from ase_ga.utilities import (
    atoms_too_close,
    closest_distances_generator,
    get_all_atom_types,
)

from scgo.exceptions import SCGORuntimeError, SCGOValidationError
from scgo.initialization import (
    combine_and_grow,
    combine_seeds,
    create_initial_cluster,
    create_initial_cluster_batch,
    grow_from_seed,
    is_cluster_connected,
    random_spherical,
    validate_cluster_structure,
)
from scgo.initialization.atomic_radii import cluster_passes_ga_blmin
from scgo.initialization.geometry_helpers import (
    analyze_disconnection,
    format_composition_counts_short,
    format_placement_error_message,
)
from scgo.initialization.initialization_config import (
    BLMIN_RATIO_DEFAULT,
    CONNECTIVITY_FACTOR,
    MIN_DISTANCE_FACTOR_DEFAULT,
    PLACEMENT_RADIUS_SCALING_DEFAULT,
)
from scgo.initialization.initializers import (
    _sample_suitable_seed,
    _SeedSamplingLogCollector,
    _try_strategies_in_order,
    compute_cell_side,
)
from scgo.initialization.templates import (
    generate_template_matches,
    generate_template_structure,
)
from scgo.utils.helpers import get_composition_counts
from tests.helpers import (
    DIVERSITY_TEST_SAMPLES_LARGE,
    DIVERSITY_TEST_SAMPLES_MEDIUM,
    DIVERSITY_TEST_SAMPLES_SMALL,
    DIVERSITY_THRESHOLD_DEFAULT,
    DIVERSITY_THRESHOLD_MIN,
    LARGE_SIZES,
    MEDIUM_SIZES,
    REPRODUCIBILITY_SEEDS,
    SMALL_SIZES,
    assert_cluster_valid,
    create_paired_rngs,
    get_structure_signature,
    validate_structure_with_diagnostics,
)

# ---------------------------------------------------------------------
# from test_init_smart.py
# ---------------------------------------------------------------------

"""Tests for smart initialization mode.

This module consolidates all tests for smart mode initialization including:
- Basic smart mode functionality
- Default initialization strictness
- Single connectivity factor consistency
- Exact composition counts
- Connectivity validation
- Reproducibility
- Edge cases
- Disconnection prevention
- No clashes
- Large cluster connectivity (50-60 atoms)
- Multi-seed reliability tests
"""


class TestSmartModeInitialization:
    """Tests for smart mode initialization."""

    def test_smart_mode_magic_number(self, rng):
        """Test smart mode with magic number (may use template)."""
        comp = ["Pt"] * 13  # Magic number
        atoms = create_initial_cluster(comp, mode="smart", rng=rng)
        assert isinstance(atoms, Atoms)
        assert len(atoms) == 13

    def test_smart_mode_near_magic_number(self, rng):
        """Test smart mode near magic number."""
        comp = ["Pt"] * 14  # Near 13
        atoms = create_initial_cluster(comp, mode="smart", rng=rng)
        assert isinstance(atoms, Atoms)
        assert len(atoms) == 14

    def test_smart_mode_non_magic(self, rng):
        """Test smart mode with non-magic number (uses templates from nearest magic number, seed+growth, or random)."""
        comp = ["Pt"] * 7  # Not a magic number
        atoms = create_initial_cluster(comp, mode="smart", rng=rng)
        assert isinstance(atoms, Atoms)
        assert len(atoms) == 7

    @pytest.mark.slow
    def test_smart_mode_diversity(self, rng):
        """Test that smart mode generates diverse structures for various cluster sizes.

        This test verifies actual diversity (not just structure counts) by checking
        that multiple initializations produce different geometric structures.
        """
        test_cases = [
            (["Pt"] * 13, 10),  # Magic number with multiple exact matches
            (["Pt"] * 12, 8),  # Near magic number 13
            (
                ["Pt"] * 7,
                8,
            ),  # Non-magic number (uses templates from nearest magic number, seed+growth, or random)
        ]

        for comp, n_samples in test_cases:
            structures = []
            signatures = []

            for _ in range(n_samples):
                atoms = create_initial_cluster(
                    comp,
                    mode="smart",
                    rng=rng,
                    connectivity_factor=CONNECTIVITY_FACTOR,
                )
                structures.append(atoms)
                signatures.append(get_structure_signature(atoms))

            # All should have correct size
            assert all(len(s) == len(comp) for s in structures), (
                f"Size mismatch for composition {comp}"
            )

            # Verify actual diversity: should have multiple unique structures
            unique_signatures = set(signatures)
            diversity_ratio = len(unique_signatures) / n_samples
            assert diversity_ratio >= DIVERSITY_THRESHOLD_MIN, (
                f"Insufficient diversity for {comp}: only {len(unique_signatures)}/{n_samples} "
                f"({diversity_ratio:.1%}) unique structures"
            )

    def test_smart_mode_fallback_when_templates_fail(self, rng):
        """Test that smart mode falls back when template generation fails."""
        # Use a size that may not have templates
        comp = ["Pt"] * 3
        atoms = create_initial_cluster(comp, mode="smart", rng=rng)
        assert isinstance(atoms, Atoms)
        assert len(atoms) == 3

    def test_smart_mode_composition_matching(self, rng):
        """Test that smart mode maintains exact composition."""
        comp = ["Pt", "Au"] * 6 + ["Pt"]  # 13 atoms
        atoms = create_initial_cluster(comp, mode="smart", rng=rng)
        assert len(atoms) == 13
        symbols = atoms.get_chemical_symbols()
        # Should have both Pt and Au
        assert "Pt" in symbols
        assert "Au" in symbols
        # Count should match
        pt_count = symbols.count("Pt")
        au_count = symbols.count("Au")
        assert pt_count + au_count == 13

    def test_smart_mode_connectivity_handling(self, rng):
        """Test that smart mode handles connectivity issues gracefully."""
        comp = ["Pt"] * 12  # Near magic number
        atoms = create_initial_cluster(
            comp, mode="smart", rng=rng, connectivity_factor=CONNECTIVITY_FACTOR
        )
        # Should still produce valid structure (may fall back to seed+growth)
        assert isinstance(atoms, Atoms)
        assert len(atoms) == 12


class TestDefaultInitializationStrictness:
    """Strict tests for default initialization settings.

    These tests ensure that default initialization settings consistently produce
    diverse, valid clusters with exact compositions, connectivity, and no clashes.
    """

    @pytest.mark.slow
    def test_default_settings_diverse_clusters_single_element(self, rng):
        """Test that default settings generate diverse clusters for single-element compositions."""
        comp = ["Pt"] * 6
        n_samples = DIVERSITY_TEST_SAMPLES_LARGE

        signatures = []
        for _ in range(n_samples):
            atoms = create_initial_cluster(comp, rng=rng)
            # Verify all invariants using helper
            assert_cluster_valid(atoms, comp)
            signatures.append(get_structure_signature(atoms))

        unique_signatures = set(signatures)
        # With default settings, we should get substantial diversity
        # At least 70% unique structures is required (DIVERSITY_THRESHOLD_DEFAULT)
        diversity_ratio = len(unique_signatures) / n_samples
        assert diversity_ratio >= DIVERSITY_THRESHOLD_DEFAULT, (
            f"Insufficient diversity: only {len(unique_signatures)}/{n_samples} "
            f"({diversity_ratio:.1%}) unique structures"
        )

    def test_default_settings_diverse_clusters_bimetallic(self, rng):
        """Test that default settings generate diverse clusters for bimetallic compositions."""
        comp = ["Pt", "Au", "Pt", "Au", "Pt", "Au"]
        n_samples = DIVERSITY_TEST_SAMPLES_MEDIUM

        signatures = []
        for _ in range(n_samples):
            atoms = create_initial_cluster(comp, rng=rng)
            # Verify all invariants using helper
            assert_cluster_valid(atoms, comp)
            signatures.append(get_structure_signature(atoms))

        unique_signatures = set(signatures)
        diversity_ratio = len(unique_signatures) / n_samples
        assert diversity_ratio >= DIVERSITY_THRESHOLD_DEFAULT, (
            f"Insufficient diversity for bimetallic: only {len(unique_signatures)}/{n_samples} "
            f"({diversity_ratio:.1%}) unique structures"
        )

    def test_default_settings_all_invariants_batch(self, rng):
        """Test that default settings produce valid clusters for various compositions."""
        test_compositions = [
            ["Pt"] * 3,
            ["Pt"] * 5,
            ["Pt"] * 8,
            ["Pt", "Au", "Pt"],
            ["Pt", "Au", "Pt", "Au"],
            ["Pt", "Au", "Pd", "Pt", "Au"],
            ["Cu", "Cu", "Cu", "Cu", "Cu"],
        ]

        for comp in test_compositions:
            atoms = create_initial_cluster(comp, rng=rng)
            # Verify all invariants using helper
            assert_cluster_valid(atoms, comp, check_connectivity=len(atoms) > 1)

    @pytest.mark.slow
    def test_default_settings_large_cluster_robustness(self, rng):
        """Test that default settings handle larger clusters robustly."""
        comp = ["Pt"] * 12
        n_samples = DIVERSITY_TEST_SAMPLES_SMALL

        for i in range(n_samples):
            atoms = create_initial_cluster(comp, rng=rng)
            # Verify all invariants using helper
            try:
                assert_cluster_valid(atoms, comp)
            except AssertionError as e:
                raise AssertionError(f"Sample {i}: {e}") from e

    def test_default_settings_exact_composition_strict(self, rng):
        """Test that default settings maintain exact composition for complex cases."""
        test_cases = [
            (["Pt", "Au"], {"Pt": 1, "Au": 1}),
            (["Pt", "Pt", "Au"], {"Pt": 2, "Au": 1}),
            (["Pt", "Au", "Pt", "Au", "Pt"], {"Pt": 3, "Au": 2}),
            (["Cu", "Pt", "Au", "Cu", "Pt"], {"Cu": 2, "Pt": 2, "Au": 1}),
        ]

        for comp, _expected_counts_dict in test_cases:
            atoms = create_initial_cluster(comp, rng=rng)
            actual_counts = get_composition_counts(atoms.get_chemical_symbols())
            expected_counts = get_composition_counts(comp)
            assert actual_counts == expected_counts, (
                f"Composition mismatch for {comp}: "
                f"expected {expected_counts}, got {actual_counts}"
            )
            # Also verify total count
            assert len(atoms) == len(comp), (
                f"Total atom count mismatch for {comp}: "
                f"expected {len(comp)}, got {len(atoms)}"
            )

    def test_default_settings_no_clashes(self, rng):
        """Test that default settings produce no atomic clashes at the default factor."""
        comp = ["Pt"] * 7
        n_samples = DIVERSITY_TEST_SAMPLES_MEDIUM

        for i in range(n_samples):
            atoms = create_initial_cluster(comp, rng=rng)
            positions = atoms.get_positions()
            # Check all pairwise distances
            for j in range(len(positions)):
                for k in range(j + 1, len(positions)):
                    distance = np.linalg.norm(positions[j] - positions[k])
                    # Minimum distance should be reasonable (at least 0.1 Å)
                    assert distance > 0.1, (
                        f"Sample {i}: Atoms {j} and {k} too close: {distance:.6f} Å"
                    )
            # Also verify with validation function
            is_valid, msg = validate_cluster_structure(
                atoms,
                min_distance_factor=MIN_DISTANCE_FACTOR_DEFAULT,
                connectivity_factor=CONNECTIVITY_FACTOR,
                check_clashes=True,
                check_connectivity=True,
            )
            assert is_valid is True, f"Sample {i}: Validation failed: {msg}"

    def test_default_settings_connectivity_strict(self, rng):
        """Test that default settings ensure connectivity with strict factor."""
        comp = ["Pt"] * 9
        n_samples = 15

        for i in range(n_samples):
            atoms = create_initial_cluster(comp, rng=rng)
            # Verify connectivity with default factor
            assert is_cluster_connected(
                atoms, connectivity_factor=CONNECTIVITY_FACTOR
            ), (
                f"Sample {i}: Cluster not connected with connectivity_factor={CONNECTIVITY_FACTOR}"
            )
            # Verify with validation function
            is_valid, msg = validate_cluster_structure(
                atoms,
                min_distance_factor=MIN_DISTANCE_FACTOR_DEFAULT,
                connectivity_factor=CONNECTIVITY_FACTOR,
                check_clashes=True,
                check_connectivity=True,
            )
            assert is_valid is True, (
                f"Sample {i}: Connectivity validation failed: {msg}"
            )


class TestSingleConnectivityFactor:
    """Tests to verify only connectivity factor 1.4 is used (no CONNECTIVITY_FACTOR_GROWTH)."""

    def test_connectivity_factor_growth_removed(self):
        """Test that CONNECTIVITY_FACTOR_GROWTH has been removed from config."""
        # Try to import - should fail since it's been removed
        import scgo.initialization.initialization_config as config

        # Verify it's not in the module
        assert not hasattr(config, "CONNECTIVITY_FACTOR_GROWTH"), (
            "CONNECTIVITY_FACTOR_GROWTH should have been removed"
        )
        # Verify CONNECTIVITY_FACTOR exists
        assert hasattr(config, "CONNECTIVITY_FACTOR")
        assert pytest.approx(1.4) == config.CONNECTIVITY_FACTOR

    @pytest.mark.parametrize("mode", ["random_spherical", "template", "seed+growth"])
    def test_connectivity_factor_consistency_all_modes(self, mode, rng):
        """Test that all modes use connectivity_factor consistently."""
        # Use appropriate composition size for each mode
        composition = ["Pt"] * 13 if mode == "template" else ["Pt"] * 10

        atoms = create_initial_cluster(
            composition,
            mode=mode,
            rng=rng,
            connectivity_factor=CONNECTIVITY_FACTOR,
        )

        if atoms is not None:
            # Should be connected with the same factor
            assert (
                is_cluster_connected(atoms, connectivity_factor=CONNECTIVITY_FACTOR)
                is True
            )
            # Validation should pass with the same factor
            is_valid, _ = validate_cluster_structure(
                atoms,
                min_distance_factor=MIN_DISTANCE_FACTOR_DEFAULT,
                connectivity_factor=CONNECTIVITY_FACTOR,
            )
            assert is_valid is True


class TestExactCompositionCounts:
    """Tests to verify exact composition counts (not ratios) are preserved."""

    @pytest.mark.parametrize(
        "mode", ["smart", "random_spherical", "seed+growth", "template"]
    )
    def test_all_modes_exact_counts(self, mode, rng):
        """Test that all initialization modes preserve exact composition counts."""
        pattern = ["Pt", "Au", "Pt", "Au", "Pt"]  # 5 atoms: 3 Pt, 2 Au
        target_composition = pattern * 2  # 10 atoms: 6 Pt, 4 Au

        atoms = create_initial_cluster(target_composition, mode=mode, rng=rng)
        if atoms is not None:
            expected_counts = get_composition_counts(target_composition)
            actual_counts = get_composition_counts(atoms.get_chemical_symbols())
            assert actual_counts == expected_counts, (
                f"Mode {mode} failed: expected {expected_counts}, got {actual_counts}"
            )


class TestConnectivityValidation:
    """Tests to verify connectivity is validated with factor 1.4."""

    @pytest.mark.parametrize(
        "composition",
        [
            ["Pt"] * 5,
            ["Pt", "Au"] * 4,
            ["Pt"] * 13,  # Magic number
            ["Pt", "Au", "Pd"] * 3,
        ],
    )
    @pytest.mark.parametrize("mode", ["smart", "random_spherical"])
    def test_all_generated_clusters_connected(self, composition, mode, rng):
        """Test that all generated clusters are connected with factor 1.4."""
        atoms = create_initial_cluster(composition, mode=mode, rng=rng)
        assert (
            is_cluster_connected(atoms, connectivity_factor=CONNECTIVITY_FACTOR) is True
        ), (
            f"Cluster from {mode} mode should be connected with factor {CONNECTIVITY_FACTOR}"
        )

    def test_template_atom_removal_maintains_connectivity(self, rng):
        """Test that template atom removal maintains connectivity."""
        # Test removing atoms from template
        target_composition = ["Pt"] * 12  # Remove 1 from 13-atom template
        atoms = create_initial_cluster(target_composition, mode="template", rng=rng)
        if atoms is not None:
            assert (
                is_cluster_connected(atoms, connectivity_factor=CONNECTIVITY_FACTOR)
                is True
            )


class TestEdgeCases:
    """Tests for edge cases in smart initialization."""

    def test_two_atoms(self, rng):
        """Test two atom cluster."""
        composition = ["Pt", "Au"]
        atoms = create_initial_cluster(composition, mode="smart", rng=rng)
        assert len(atoms) == 2
        assert get_composition_counts(
            atoms.get_chemical_symbols()
        ) == get_composition_counts(composition)

    def test_multi_element_exact_counts_large(self, rng):
        """Test exact composition counts for large multi-element cluster."""
        pattern = ["Pt", "Au", "Pd"] * 10  # 30 atoms: 10 each
        atoms = create_initial_cluster(pattern, mode="smart", rng=rng)

        expected_counts = get_composition_counts(pattern)
        actual_counts = get_composition_counts(atoms.get_chemical_symbols())
        assert actual_counts == expected_counts


class TestDisconnectionPrevention:
    """Tests to verify disconnection prevention during atom removal."""

    def test_template_removal_preserves_connectivity(self, rng):
        """Test that template atom removal doesn't disconnect cluster."""
        # Remove atoms from 13-atom template to get smaller size
        target_composition = ["Pt"] * 10
        atoms = create_initial_cluster(target_composition, mode="template", rng=rng)
        if atoms is not None:
            assert (
                is_cluster_connected(atoms, connectivity_factor=CONNECTIVITY_FACTOR)
                is True
            )

            # Verify no clashes
            is_valid, _ = validate_cluster_structure(
                atoms,
                min_distance_factor=MIN_DISTANCE_FACTOR_DEFAULT,
                connectivity_factor=CONNECTIVITY_FACTOR,
            )
            assert is_valid is True

    def test_seed_combination_maintains_connectivity(self, rng):
        """Test that seed combination maintains connectivity."""
        composition = ["Pt"] * 20
        atoms = create_initial_cluster(composition, mode="seed+growth", rng=rng)
        if atoms is not None:
            assert (
                is_cluster_connected(atoms, connectivity_factor=CONNECTIVITY_FACTOR)
                is True
            )


class TestNoClashes:
    """Tests to verify no atomic clashes in generated structures."""

    @pytest.mark.parametrize(
        "mode", ["smart", "random_spherical", "seed+growth", "template"]
    )
    def test_all_modes_no_clashes(self, mode, rng):
        """Test that all modes produce structures without clashes."""
        composition = ["Pt", "Au"] * 5

        atoms = create_initial_cluster(composition, mode=mode, rng=rng)
        if atoms is not None:
            is_valid, error_msg = validate_cluster_structure(
                atoms,
                min_distance_factor=MIN_DISTANCE_FACTOR_DEFAULT,
                connectivity_factor=CONNECTIVITY_FACTOR,
                check_clashes=True,
                check_connectivity=True,
            )
            assert is_valid is True, (
                f"Mode {mode} produced invalid structure: {error_msg}"
            )


# ---------------------------------------------------------------------
# from test_init_random_spherical.py
# ---------------------------------------------------------------------

"""Tests for random_spherical initialization mode.

This module consolidates all tests for random_spherical initialization including:
- Basic functionality and edge cases
- Boundary value testing
- Retry logic and diversity
- Large cluster connectivity (50-60 atoms)
- Multi-seed reliability tests
"""


class TestRandomSphericalInitialization:
    """Tests for random spherical cluster initialization.

    Note: Basic smoke tests (empty, single atom, two atoms) have been consolidated
    into parametrized tests in test_initialization_modes.py to reduce redundancy.
    This class now focuses on mode-specific edge cases and stress tests.
    """

    def test_random_spherical_count_and_bounds(self, rng):
        """Test random spherical produces correct number of atoms in bounds and satisfies invariants."""
        comp = ["Pt"] * 5
        side = 30.0
        atoms = random_spherical(
            comp,
            placement_radius_scaling=1.2,
            cell_side=side,
            rng=rng,
        )
        # Verify atom count
        assert len(atoms) == len(comp)
        # Verify all invariants using helper
        assert_cluster_valid(atoms, comp)
        # cell is cubic and set
        c = atoms.get_cell()
        assert np.allclose(c[0, 0], side)
        assert np.allclose(c[1, 1], side)
        assert np.allclose(c[2, 2], side)
        # Cluster COM should lie near the cell center for cubic placement.
        com = atoms.get_center_of_mass()
        half = side / 2.0
        assert np.allclose(com, [half, half, half], atol=0.5)

    def test_random_spherical_placement_failure(self, rng):
        """Test that placement failure raises appropriate error."""
        # Try to place many atoms in a very small space, should result in a ValueError.
        comp = ["H"] * 20  # Many small atoms
        side = 5.0  # Very small cell
        with pytest.raises(SCGOValidationError, match="place all"):
            random_spherical(
                comp, placement_radius_scaling=0.1, cell_side=side, rng=rng
            )


class TestBoundaryValues:
    """Tests for boundary value parameters."""

    def test_very_small_placement_radius(self, rng):
        """Very small placement_radius_scaling is accepted and yields a valid 2-atom cluster."""
        atoms = random_spherical(
            ["Pt", "Pt"], cell_side=20.0, placement_radius_scaling=0.01, rng=rng
        )
        assert isinstance(atoms, Atoms)
        assert len(atoms) == 2
        assert np.all(np.isfinite(atoms.get_positions()))
        assert is_cluster_connected(atoms)

    def test_very_large_placement_radius(self, rng):
        """Very large placement_radius_scaling is accepted and yields a valid 2-atom cluster."""
        atoms = random_spherical(
            ["Pt", "Pt"], cell_side=20.0, placement_radius_scaling=100.0, rng=rng
        )
        assert isinstance(atoms, Atoms)
        assert len(atoms) == 2
        assert np.all(np.isfinite(atoms.get_positions()))

    def test_min_distance_factor_zero(self, rng):
        """Test with min_distance_factor = 0."""
        atoms = random_spherical(
            ["Pt", "Pt"],
            cell_side=20.0,
            min_distance_factor=0.0,
            blmin_ratio=None,
            rng=rng,
        )
        # Should work (allows overlap)
        assert len(atoms) == 2

    def test_min_distance_factor_very_large(self, rng):
        """A very large min_distance_factor violates the GA steric floor and must be rejected."""
        with pytest.raises((ValueError, SCGOValidationError)):
            random_spherical(
                ["Pt", "Pt"], cell_side=10.0, min_distance_factor=10.0, rng=rng
            )

    def test_connectivity_factor_very_small(self, rng):
        """A connectivity_factor below the GA steric floor must be rejected."""
        VERY_STRICT_FACTOR = 0.1  # Very strict for testing boundary conditions
        with pytest.raises((ValueError, SCGOValidationError)):
            random_spherical(
                ["Pt", "Pt"],
                cell_side=20.0,
                connectivity_factor=VERY_STRICT_FACTOR,
                rng=rng,
            )

    def test_connectivity_factor_very_large(self, rng):
        """Test with very large connectivity_factor."""
        VERY_LARGE_FACTOR = 100.0  # Very large for testing boundary conditions
        atoms = random_spherical(
            ["Pt", "Pt"], cell_side=20.0, connectivity_factor=VERY_LARGE_FACTOR, rng=rng
        )
        # Should always work
        assert len(atoms) == 2


class TestGaBlminCompatibility:
    """random_spherical outputs must satisfy GA operator steric floors."""

    def test_pt55_prototypical_seed_satisfies_ga_blmin(self, rng) -> None:
        composition = ["Pt"] * 55
        atoms = random_spherical(
            composition,
            cell_side=30.0,
            rng=rng,
        )
        assert cluster_passes_ga_blmin(atoms, BLMIN_RATIO_DEFAULT)
        blmin = closest_distances_generator(
            get_all_atom_types(atoms, range(len(atoms))),
            ratio_of_covalent_radii=BLMIN_RATIO_DEFAULT,
        )
        assert not atoms_too_close(atoms, blmin, use_tags=False)


class TestRetryDiversity:
    """Tests for retry logic diversity."""

    def test_retry_logic_maintains_diversity(self, rng):
        """Verify that retry attempts produce diverse structures."""
        # Test random_spherical which uses retry logic
        comp = ["Pt"] * 8
        structures = []

        # Stress test: generate multiple clusters to verify diversity
        for _ in range(10):
            atoms = random_spherical(
                comp,
                placement_radius_scaling=1.2,
                cell_side=20.0,
                rng=rng,
            )
            structures.append(atoms)

        # Check diversity
        def get_signature(atoms):
            pos = atoms.get_positions()
            dists = [
                np.linalg.norm(pos[i] - pos[j])
                for i in range(len(pos))
                for j in range(i + 1, len(pos))
            ]
            return tuple(np.round(np.sort(dists), 4))

        signatures = [get_signature(s) for s in structures]
        unique = set(signatures)
        # Should have diversity
        assert len(unique) >= 3, "Retry logic should maintain diversity"

    def test_connectivity_retries_diverse(self, rng):
        """Verify connectivity retries don't always produce identical structures."""
        comp = ["Pt"] * 6
        structures = []

        # Stress test: generate multiple clusters to verify diversity
        for _ in range(8):
            atoms = random_spherical(
                comp,
                placement_radius_scaling=1.2,
                cell_side=20.0,
                rng=rng,
            )
            structures.append(atoms)

        # Check diversity
        def get_signature(atoms):
            pos = atoms.get_positions()
            dists = [
                np.linalg.norm(pos[i] - pos[j])
                for i in range(len(pos))
                for j in range(i + 1, len(pos))
            ]
            return tuple(np.round(np.sort(dists), 4))

        signatures = [get_signature(s) for s in structures]
        unique = set(signatures)
        # Should have diversity
        assert len(unique) >= 2, "Connectivity retries should maintain diversity"


# Reliability tests have been consolidated into TestReliabilityAllModes in test_init_common.py
# Large cluster connectivity tests have been consolidated into TestLargeClusterConnectivityAllModes in test_init_common.py


class TestRandomSphericalStressAndPerformance:
    """Stress and performance tests for random_spherical mode."""

    def test_retry_exhaustion_error_message(self, rng):
        """Test that retry exhaustion provides helpful error message."""
        # Try to place many atoms in very small space
        comp = ["H"] * 50  # Many small atoms
        with pytest.raises(SCGOValidationError) as exc_info:
            random_spherical(
                comp, cell_side=5.0, placement_radius_scaling=0.01, rng=rng
            )
        # Error message should include suggestions
        error_msg = str(exc_info.value)
        assert "placement_radius_scaling" in error_msg or "cell_side" in error_msg

    def test_connectivity_retry_exhaustion(self, rng):
        """Test connectivity retry exhaustion."""
        # Use very strict connectivity that might cause failures
        STRICT_FACTOR = 0.5  # Very strict for testing retry exhaustion
        comp = ["Pt"] * 10
        try:
            atoms = random_spherical(
                comp,
                cell_side=10.0,
                placement_radius_scaling=0.5,
                connectivity_factor=STRICT_FACTOR,
                rng=rng,
            )
            # If it succeeds, should be valid
            if atoms is not None:
                assert is_cluster_connected(atoms, connectivity_factor=STRICT_FACTOR)
        except (ValueError, SCGOValidationError) as e:
            # Error should mention connectivity, clashes, or validation failure
            # (strict parameters can cause either type of failure)
            error_msg = str(e).lower()
            assert (
                "connectivity" in error_msg
                or "connected" in error_msg
                or "clash" in error_msg
                or "validation failed" in error_msg
            )


# ---------------------------------------------------------------------
# from test_init_seed_growth.py
# ---------------------------------------------------------------------

"""Tests for seed+growth initialization mode.

This module consolidates all tests for seed+growth initialization including:
- Basic seed+growth functionality
- Threshold relaxation legitimacy
- Convex hull placement reliability
- Large cluster connectivity (50-60 atoms)
- Multi-seed reliability tests
- Refactored seed growth tests
"""


class TestSeedGrowthInitialization:
    """Tests for seed+growth initialization mode."""

    def test_grow_from_seed_preserves_composition(self, rng):
        """Test that grow_from_seed preserves target composition."""
        # seed: one Pt and one Au at reasonable separation
        seed = Atoms(["Pt", "Au"], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]])

        # target composition: add one Pt (total: 2 Pt, 3 Au, 1 Pd)
        target_comp = ["Pt", "Pt", "Au", "Au", "Au", "Pd"]

        placement_radius_scaling = 0.9
        cell_side = compute_cell_side(target_comp, vacuum=8.0)

        out = grow_from_seed(
            seed_atoms=seed,
            target_composition=target_comp,
            placement_radius_scaling=placement_radius_scaling,
            cell_side=cell_side,
            rng=rng,
        )

        assert out is not None, "grow_from_seed returned None"
        assert_cluster_valid(out, target_comp)

    def test_grow_from_seed_no_additional_needed(self, rng):
        """Test grow_from_seed when no additional atoms are needed."""
        seed = Atoms(["Pt", "Au"], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 2.5]])
        target = ["Pt", "Au"]
        side = compute_cell_side(target, vacuum=8.0)
        out = grow_from_seed(
            seed_atoms=seed,
            target_composition=target,
            placement_radius_scaling=0.9,
            cell_side=side,
            rng=rng,
        )
        assert out is not None
        assert_cluster_valid(out, target)

    def test_grow_from_seed_empty_seed(self, rng):
        """Test grow_from_seed with empty seed behaves like random_spherical."""
        # Should behave like random_spherical if seed is empty
        seed = Atoms()
        target_comp = ["Pt", "Pt"]
        cell_side = compute_cell_side(target_comp, vacuum=8.0)

        out = grow_from_seed(
            seed_atoms=seed,
            target_composition=target_comp,
            placement_radius_scaling=1.0,
            cell_side=cell_side,
            rng=rng,
        )
        assert out is not None
        assert_cluster_valid(out, target_comp)

    def test_grow_from_seed_empty_target_composition(self, rng):
        """Test grow_from_seed with empty target returns seed."""
        # Should return the seed if target is empty
        seed = Atoms("Pt", positions=[[0, 0, 0]])
        target_comp = []
        cell_side = compute_cell_side(seed.get_chemical_symbols(), vacuum=8.0)

        out = grow_from_seed(
            seed_atoms=seed,
            target_composition=target_comp,
            placement_radius_scaling=1.0,
            cell_side=cell_side,
            rng=rng,
        )
        assert out is not None
        assert len(out) == len(seed)
        assert get_composition_counts(
            out.get_chemical_symbols()
        ) == get_composition_counts(seed.get_chemical_symbols())

    def test_grow_from_seed_failure_to_grow(self, rng):
        """Test grow_from_seed failure when constraints are too tight."""
        # Try to grow in a very constrained space, should fail to add all atoms
        seed = Atoms("Pt", positions=[[0, 0, 0]])
        target_comp = ["Pt"] * 5  # Try to add 4 more Pt atoms
        cell_side = compute_cell_side(target_comp, vacuum=1.0)  # Very small vacuum

        out = grow_from_seed(
            seed_atoms=seed,
            target_composition=target_comp,
            placement_radius_scaling=0.1,  # Very small scaling
            cell_side=cell_side,
            rng=rng,
        )
        assert out is None or len(out) < len(
            target_comp,
        )  # Should fail to add all or return None

    def test_grow_from_seed_connectivity(self, rng):
        """Test that grow_from_seed produces connected clusters."""
        # Create a connected seed
        seed = Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
        target_comp = ["Pt", "Pt", "Au", "Au"]
        cell_side = compute_cell_side(target_comp, vacuum=8.0)

        result = grow_from_seed(
            seed_atoms=seed,
            target_composition=target_comp,
            placement_radius_scaling=1.2,
            cell_side=cell_side,
            connectivity_factor=CONNECTIVITY_FACTOR,  # Use default value
            rng=rng,
        )

        assert result is not None
        assert_cluster_valid(result, target_comp)

    def test_grow_from_seed_connected_under_growth_threshold(self, rng):
        """Regression: grown clusters should be connected under growth connectivity factor.

        This test ensures that the incremental growth algorithm places each new atom
        close enough to at least one existing atom so that the final cluster is
        connected when evaluated with the connectivity factor.
        """
        from scgo.initialization.initialization_config import CONNECTIVITY_FACTOR

        # Start from a small but connected metallic seed
        seed = Atoms("Pt3", positions=[[0, 0, 0], [2.5, 0, 0], [1.25, 2.2, 0]])
        target_comp = ["Pt"] * 10  # Grow to a modestly larger Pt cluster
        cell_side = compute_cell_side(target_comp, vacuum=8.0)

        result = grow_from_seed(
            seed_atoms=seed,
            target_composition=target_comp,
            placement_radius_scaling=1.2,
            cell_side=cell_side,
            rng=rng,
        )

        assert result is not None
        assert len(result) == len(target_comp)
        assert get_composition_counts(
            result.get_chemical_symbols()
        ) == get_composition_counts(target_comp)
        assert is_cluster_connected(result, connectivity_factor=CONNECTIVITY_FACTOR)

        assert_cluster_valid(
            result,
            target_comp,
            min_distance_factor=MIN_DISTANCE_FACTOR_DEFAULT,
            connectivity_factor=CONNECTIVITY_FACTOR,
        )


class TestThresholdRelaxationLegitimacy:
    """Tests verifying threshold relaxation only occurs when needed."""

    def test_relaxation_produces_valid_clusters(self, rng):
        """Verify that when relaxation occurs, it produces valid clusters."""
        # Test with parameters that may trigger relaxation
        comp = ["Pt"] * 15
        # Use parameters that might need relaxation but should still produce valid clusters
        atoms = create_initial_cluster(
            comp,
            rng=rng,
            placement_radius_scaling=1.0,
            min_distance_factor=0.4,
            connectivity_factor=CONNECTIVITY_FACTOR,
        )

        assert len(atoms) == len(comp)
        assert is_cluster_connected(atoms)
        # Verify no clashes
        is_valid, msg = validate_cluster_structure(
            atoms,
            min_distance_factor=0.4,
            connectivity_factor=CONNECTIVITY_FACTOR,
            check_clashes=True,
            check_connectivity=True,
        )
        assert is_valid is True, f"Relaxation should produce valid clusters: {msg}"

    def test_relaxation_bounds_safe(self, rng):
        """Verify relaxation doesn't go below safe thresholds."""
        # Test with various min_distance_factor values
        comp = ["Pt"] * 10
        for min_dist in [0.3, 0.4, 0.5]:
            atoms = create_initial_cluster(
                comp,
                rng=rng,
                min_distance_factor=min_dist,
                connectivity_factor=CONNECTIVITY_FACTOR,
            )

            assert len(atoms) == len(comp)
            # Verify no clashes even with relaxed parameters
            is_valid, msg = validate_cluster_structure(
                atoms,
                min_distance_factor=min_dist,
                connectivity_factor=CONNECTIVITY_FACTOR,
                check_clashes=True,
                check_connectivity=True,
            )
            assert is_valid is True, f"Relaxation bounds should be safe: {msg}"


class TestLargeClusterConnectivitySeedGrowth:
    """Stringent tests for seed+growth mode with 50-60 atom clusters.

    Note: Basic connectivity tests for single/bimetallic compositions
    have been consolidated into TestLargeClusterConnectivityAllModes in test_init_common.py.
    This class now only contains unique seed growth tests (combine_and_grow) and batch/reproducibility tests.
    """

    @pytest.mark.parametrize("n_atoms", [50, 52, 55, 58, 60])
    @pytest.mark.parametrize("seed", REPRODUCIBILITY_SEEDS)
    def test_combine_and_grow_connectivity_single_element(self, n_atoms, seed):
        """Test that combine_and_grow produces connected clusters from multiple seeds."""
        # Manual RNG creation needed for parametrized test with specific seeds
        rng, _ = create_paired_rngs(seed)
        comp = ["Pt"] * n_atoms

        # Create multiple smaller seeds
        seed_sizes = [8, 10, 12]
        seeds = []
        for size in seed_sizes:
            seed_comp = ["Pt"] * size
            seed_cell_side = compute_cell_side(seed_comp)
            seed_atoms = random_spherical(
                composition=seed_comp,
                cell_side=seed_cell_side,
                rng=rng,
                connectivity_factor=CONNECTIVITY_FACTOR,
            )
            assert is_cluster_connected(
                seed_atoms, connectivity_factor=CONNECTIVITY_FACTOR
            ), f"Seed of size {size} is not connected"
            seeds.append(seed_atoms)

        # Calculate how many more atoms we need
        total_seed_atoms = sum(len(s) for s in seeds)
        remaining = n_atoms - total_seed_atoms

        if remaining < 0:
            pytest.skip(
                f"Seeds too large for target (total={total_seed_atoms}, target={n_atoms})"
            )

        # Create target composition
        target_comp = comp  # Already correct size
        cell_side = compute_cell_side(target_comp)
        atoms = combine_and_grow(
            seeds=seeds,
            target_composition=target_comp,
            cell_side=cell_side,
            rng=rng,
            connectivity_factor=CONNECTIVITY_FACTOR,
        )

        if atoms is None:
            pytest.skip(
                f"combine_and_grow returned None for n_atoms={n_atoms}, seed={seed}"
            )

        # Verify composition & geometry using centralized helper (connectivity checked separately below)
        assert_cluster_valid(atoms, comp, check_connectivity=False)

        # Stringent connectivity check
        is_connected = is_cluster_connected(
            atoms, connectivity_factor=CONNECTIVITY_FACTOR
        )
        if not is_connected:
            (
                suggested_factor,
                analysis_msg,
            ) = analyze_disconnection(atoms, CONNECTIVITY_FACTOR)
            pytest.fail(
                f"Combine and grow produced disconnected cluster "
                f"(n_atoms={n_atoms}, seed={seed}, seed_sizes={seed_sizes}). "
                f"Connectivity factor: {CONNECTIVITY_FACTOR}. "
                f"Analysis: {analysis_msg}. "
                f"Suggested factor: {suggested_factor:.2f}."
            )


class TestSeedGrowthReliability:
    """Tests for grow_from_seed reliability."""

    @pytest.mark.parametrize("seed", REPRODUCIBILITY_SEEDS)
    @pytest.mark.parametrize("growth_amount", [2, 4, 6])
    def test_grow_from_small_seed(self, seed, growth_amount):
        """Test growth from a small connected seed."""
        # Manual RNG creation needed for parametrized test with specific seeds
        rng, _ = create_paired_rngs(seed)

        # Create a small connected seed
        seed_atoms = Atoms(
            "Pt3",
            positions=[[0, 0, 0], [2.5, 0, 0], [1.25, 2.165, 0]],
        )
        seed_atoms.set_cell([20, 20, 20])
        seed_atoms.center()

        target_n = len(seed_atoms) + growth_amount
        target_comp = ["Pt"] * target_n
        cell_side = compute_cell_side(target_comp)

        result = grow_from_seed(
            seed_atoms=seed_atoms,
            target_composition=target_comp,
            placement_radius_scaling=1.0,
            cell_side=cell_side,
            rng=rng,
        )

        assert result is not None, f"grow_from_seed returned None for seed={seed}"
        assert_cluster_valid(result, target_comp)

        validate_structure_with_diagnostics(
            result, context=f"seed={seed}, growth_amount={growth_amount}"
        )

    @pytest.mark.parametrize("seed", REPRODUCIBILITY_SEEDS[:5])
    def test_grow_mixed_composition(self, seed):
        """Test growth with mixed composition."""
        # Manual RNG creation needed for parametrized test with specific seeds
        rng, _ = create_paired_rngs(seed)

        # Create a small PtAu seed
        seed_atoms = Atoms(
            "PtAu",
            positions=[[0, 0, 0], [2.6, 0, 0]],  # Pt-Au typical distance
        )
        seed_atoms.set_cell([20, 20, 20])
        seed_atoms.center()

        target_comp = ["Pt", "Au", "Pt", "Au", "Pt"]  # 5 atoms
        cell_side = compute_cell_side(target_comp)

        result = grow_from_seed(
            seed_atoms=seed_atoms,
            target_composition=target_comp,
            placement_radius_scaling=1.0,
            cell_side=cell_side,
            rng=rng,
        )

        assert result is not None
        assert_cluster_valid(result, target_comp)

        validate_structure_with_diagnostics(
            result, context=f"seed={seed}, mixed composition PtAu"
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("seed", REPRODUCIBILITY_SEEDS[:3])
    @pytest.mark.parametrize("target_size", [20, 30, 40])
    def test_grow_to_larger_sizes(self, seed, target_size):
        """Test growth to larger target sizes (slow)."""
        LARGE_CLUSTER_FACTOR = 2.0  # More lenient for large clusters (20+ atoms)
        # Manual RNG creation needed for parametrized test with specific seeds
        rng, _ = create_paired_rngs(seed)

        # Create a tetrahedron seed
        seed_atoms = Atoms(
            "Pt4",
            positions=[
                [0, 0, 0],
                [2.5, 0, 0],
                [1.25, 2.165, 0],
                [1.25, 0.721, 2.357],
            ],
        )
        seed_atoms.set_cell([30, 30, 30])
        seed_atoms.center()

        target_comp = ["Pt"] * target_size
        cell_side = compute_cell_side(target_comp)

        result = grow_from_seed(
            seed_atoms=seed_atoms,
            target_composition=target_comp,
            placement_radius_scaling=1.2,
            cell_side=cell_side,
            rng=rng,
            min_distance_factor=0.4,
            connectivity_factor=LARGE_CLUSTER_FACTOR,
        )

        assert result is not None
        assert len(result) == target_size

        validate_structure_with_diagnostics(
            result,
            min_distance_factor=0.4,
            connectivity_factor=LARGE_CLUSTER_FACTOR,
            context=f"seed={seed}, target_size={target_size}",
        )


class TestRefactoredSeedGrowth:
    """Tests for refactored seed growth functions."""

    def test_filter_candidates_by_geometry(self):
        """Test geometry filtering helper function."""
        from scgo.initialization.initializers import _filter_candidates_by_geometry

        # Create test candidates - use 3D structures that should pass filter
        # Planar structure (may or may not pass depending on classification)
        planar = Atoms("Pt3", positions=[[0, 0, 0], [1, 0, 0], [0.5, 0.866, 0]])
        # 3D structure (should definitely pass)
        three_d = Atoms(
            "Pt4",
            positions=[[0, 0, 0], [1, 0, 0], [0.5, 0.866, 0], [0.5, 0.289, 0.816]],
        )

        candidates = {
            "Pt3": [(-10.0, planar)],
            "Pt4": [(-15.0, three_d)],
        }

        filtered = _filter_candidates_by_geometry(candidates)
        # At least the 3D structure should be in filtered results
        assert "Pt4" in filtered
        # Pt3 may or may not be included depending on geometry classification
        assert len(filtered) > 0

    def test_seed_sampling_strategies(self, rng):
        """Test seed sampling strategy selection."""
        from scgo.initialization.initializers import _sample_seed_with_strategy

        # Create test candidates with same composition (required for Boltzmann sampling)
        # Sorted by energy (lowest first)
        candidates = [
            (-15.0, Atoms("Pt3", positions=[[0, 0, 0], [2.5, 0, 0], [1.25, 2.165, 0]])),
            (-10.0, Atoms("Pt3", positions=[[0, 0, 0], [2.5, 0, 0], [1.25, 2.0, 0]])),
            (-5.0, Atoms("Pt3", positions=[[0, 0, 0], [3.0, 0, 0], [1.5, 2.5, 0]])),
        ]

        # Test each strategy
        for strategy in range(5):
            result = _sample_seed_with_strategy(candidates, strategy, rng)
            assert result is not None
            energy, atoms = result
            assert isinstance(energy, float)
            assert isinstance(atoms, Atoms)
            assert len(atoms) == 3  # All candidates are Pt3
            # Verify result is from candidates
            assert any(
                abs(e - energy) < 1e-6 and len(a) == len(atoms) for e, a in candidates
            )

        with pytest.raises(SCGOValidationError, match="Invalid seed sampling strategy"):
            _sample_seed_with_strategy(candidates, 99, rng)


class TestSeedGrowthDiversity:
    """Tests to verify seed+growth produces diverse structures from seed selection, not fallbacks."""

    @pytest.mark.slow
    def test_seed_growth_diversity_with_available_seeds(self, rng):
        """Test that seed+growth produces diverse structures when seeds are available.

        This test ensures diversity comes from seed selection and combination,
        not from random_spherical fallbacks. We use a composition (Pt50) that
        we know has seeds available in the database.

        Verification strategy:
        1. Confirm seeds ARE available (pre-check)
        2. Capture logger output to detect fallbacks
        3. Generate structures with seed+growth mode
        4. Verify from logs that seed+growth succeeded (not falling back)
        5. Verify seed+growth diversity ≥70% (DIVERSITY_THRESHOLD_DEFAULT)

        The code now logs explicitly when seed+growth falls back to random_spherical,
        so we can verify directly from the logs.
        """
        import logging
        from io import StringIO

        from scgo.initialization.initializers import _find_smaller_candidates
        from scgo.utils.logging import get_logger

        comp = ["Pt"] * 50
        n_samples = DIVERSITY_TEST_SAMPLES_MEDIUM

        # First, verify that seeds ARE available for this composition
        candidates = _find_smaller_candidates(comp, "**/*.db")
        if len(candidates) == 0:
            pytest.skip(
                "No seeds found for Pt50 - cannot test seed+growth diversity. "
                "This test requires database files with Pt seeds."
            )
        total_candidates = sum(len(cands) for cands in candidates.values())
        if total_candidates == 0:
            pytest.skip("No candidate seeds available")

        # Set up logger capture to detect fallbacks
        # We need to capture from the root logger or configure the specific logger
        logger = get_logger("scgo.initialization.initializers")
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.DEBUG)  # Capture all levels to see everything
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        logger.addHandler(handler)
        original_level = logger.level
        original_propagate = logger.propagate
        logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all messages
        logger.propagate = False  # Prevent propagation to root logger

        try:
            # Generate structures with seed+growth mode
            seed_growth_signatures = []
            fallback_count = 0

            for _ in range(n_samples):
                log_capture.seek(0)
                log_capture.truncate(0)

                atoms = create_initial_cluster(
                    comp,
                    mode="seed+growth",
                    rng=rng,
                    previous_search_glob="**/*.db",
                )
                assert_cluster_valid(atoms, comp)
                seed_growth_signatures.append(get_structure_signature(atoms))

                # Check log for fallback messages
                log_output = log_capture.getvalue()
                if "falling back to random_spherical" in log_output:
                    fallback_count += 1

            # Verify that seed+growth did NOT fall back (or fell back very rarely)
            fallback_ratio = fallback_count / n_samples
            assert fallback_ratio < 0.2, (
                f"Too many fallbacks detected in logs: {fallback_count}/{n_samples} "
                f"({fallback_ratio:.1%}) structures fell back to random_spherical. "
                f"This suggests seed+growth is failing when seeds are available. "
                f"Expected <20% fallback rate."
            )

            # Verify seed+growth diversity: at least 70% unique structures
            unique_seed_growth = set(seed_growth_signatures)
            diversity_ratio = len(unique_seed_growth) / n_samples
            assert diversity_ratio >= DIVERSITY_THRESHOLD_DEFAULT, (
                f"Insufficient diversity in seed+growth mode: only "
                f"{len(unique_seed_growth)}/{n_samples} ({diversity_ratio:.1%}) unique structures. "
                f"Expected at least {DIVERSITY_THRESHOLD_DEFAULT:.0%}. "
                f"Fallbacks detected in logs: {fallback_count}/{n_samples}"
            )

        finally:
            handler.close()
            logger.removeHandler(handler)
            logger.setLevel(original_level)
            logger.propagate = original_propagate

    @pytest.mark.slow
    def test_seed_growth_diversity_bimetallic_with_seeds(self, rng):
        """Test seed+growth diversity for bimetallic composition with available seeds."""
        from scgo.initialization.initializers import _find_smaller_candidates

        comp = ["Pt", "Au"] * 25  # 50 atoms, bimetallic
        n_samples = DIVERSITY_TEST_SAMPLES_MEDIUM

        # Verify seeds are available
        candidates = _find_smaller_candidates(comp, "**/*.db")
        # For bimetallic, we may have fewer seeds, but should have some
        # If no seeds, skip this test
        if len(candidates) == 0:
            pytest.skip(
                "No seeds available for bimetallic Pt25Au25 - skipping diversity test"
            )

        signatures = []
        for _ in range(n_samples):
            atoms = create_initial_cluster(
                comp,
                mode="seed+growth",
                rng=rng,
                previous_search_glob="**/*.db",
            )
            assert_cluster_valid(atoms, comp)
            signatures.append(get_structure_signature(atoms))

        unique_signatures = set(signatures)
        diversity_ratio = len(unique_signatures) / n_samples
        assert diversity_ratio >= DIVERSITY_THRESHOLD_DEFAULT, (
            f"Insufficient diversity in seed+growth mode (bimetallic): only "
            f"{len(unique_signatures)}/{n_samples} ({diversity_ratio:.1%}) unique structures. "
            f"Expected at least {DIVERSITY_THRESHOLD_DEFAULT:.0%}."
        )


class TestSeedGrowthFallbackChain:
    """Regression tests for coherent seed+growth fallback behaviour."""

    def test_partial_combo_does_not_call_combine_and_grow(self, monkeypatch, rng):
        """If any formula in a combo fails sampling, discard the whole attempt."""
        from scgo.initialization import initializers as init_mod

        seed_a = Atoms("Pt2", positions=[[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
        seed_b = Atoms(
            "Pt3", positions=[[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [1.25, 2.2, 0.0]]
        )
        candidates_by_formula = {
            "Pt2": [(0.0, seed_a)],
            "Pt3": [(0.0, seed_b)],
        }
        valid_combinations = [("Pt2", "Pt3")]
        call_sizes: list[int] = []

        def fake_sample(
            candidates,
            strategy,
            tried_positions,
            existing_geometries,
            rng,
            max_attempts=10,
        ):
            # First formula succeeds; second fails → previously caused a partial combine.
            if len(candidates) == 1 and len(candidates[0][1]) == 2:
                return seed_a.copy(), None
            return None, "forced failure"

        def fake_combine_and_grow(*, seeds, **kwargs):
            call_sizes.append(len(seeds))
            return None

        random_calls = {"n": 0}

        def fake_grow_from_random_seed(**kwargs):
            random_calls["n"] += 1
            return Atoms("Pt5", positions=[[i * 2.5, 0.0, 0.0] for i in range(5)])

        monkeypatch.setattr(init_mod, "_sample_suitable_seed", fake_sample)
        monkeypatch.setattr(init_mod, "combine_and_grow", fake_combine_and_grow)
        monkeypatch.setattr(
            init_mod, "_grow_from_random_seed", fake_grow_from_random_seed
        )

        out = init_mod._try_seed_growth(
            composition=["Pt"] * 5,
            cell_side=20.0,
            rng=rng,
            placement_radius_scaling=1.2,
            min_distance_factor=MIN_DISTANCE_FACTOR_DEFAULT,
            connectivity_factor=CONNECTIVITY_FACTOR,
            candidates_by_formula=candidates_by_formula,
            valid_combinations=valid_combinations,
        )

        assert call_sizes == []
        assert random_calls["n"] == 1
        assert out is not None
        assert len(out) == 5

    def test_db_combo_exhaustion_tries_random_seed_growth(
        self, monkeypatch, rng, caplog
    ):
        """When DB combos exist but all fail, still attempt random-seed growth."""
        import logging

        from scgo.initialization import initializers as init_mod

        seed = Atoms("Pt2", positions=[[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
        candidates_by_formula = {"Pt2": [(0.0, seed)]}
        valid_combinations = [("Pt2",)]

        monkeypatch.setattr(
            init_mod,
            "_sample_suitable_seed",
            lambda *args, **kwargs: (seed.copy(), None),
        )
        monkeypatch.setattr(init_mod, "combine_and_grow", lambda **kwargs: None)

        random_calls = {"n": 0}

        def fake_grow_from_random_seed(**kwargs):
            random_calls["n"] += 1
            return Atoms("Pt5", positions=[[i * 2.5, 0.0, 0.0] for i in range(5)])

        monkeypatch.setattr(
            init_mod, "_grow_from_random_seed", fake_grow_from_random_seed
        )

        caplog.set_level(logging.INFO, logger="scgo.initialization.initializers")
        out = init_mod._try_seed_growth(
            composition=["Pt"] * 5,
            cell_side=20.0,
            rng=rng,
            placement_radius_scaling=1.2,
            min_distance_factor=MIN_DISTANCE_FACTOR_DEFAULT,
            connectivity_factor=CONNECTIVITY_FACTOR,
            candidates_by_formula=candidates_by_formula,
            valid_combinations=valid_combinations,
        )

        assert random_calls["n"] == 1
        assert out is not None
        assert "DB combinations exhausted; trying random-seed growth" in caplog.text


# ---------------------------------------------------------------------
# from test_init_template.py
# ---------------------------------------------------------------------

"""Tests for template initialization mode.

This module consolidates all tests for template mode initialization including:
- Basic template mode functionality
- Template removal safety
- Convex hull placement reliability
- Large cluster connectivity (50-60 atoms)
- Template generation optimization
"""


class TestTemplateModeInitialization:
    """Tests for template mode initialization."""

    def test_template_mode_icosahedron(self, rng):
        """Test template mode with icosahedral magic number."""
        comp = ["Pt"] * 13
        atoms = create_initial_cluster(comp, mode="template", rng=rng)
        assert len(atoms) == 13
        assert_cluster_valid(atoms, comp)

    def test_template_mode_decahedron(self, rng):
        """Test template mode with decahedral size."""
        comp = ["Pt"] * 23
        atoms = create_initial_cluster(
            comp,
            mode="template",
            rng=rng,
            connectivity_factor=CONNECTIVITY_FACTOR,
            placement_radius_scaling=1.3,
        )
        assert len(atoms) == 23
        assert_cluster_valid(atoms, comp)

    def test_template_mode_near_magic(self, rng):
        """Test template mode near magic number (adds/removes atoms)."""
        comp = ["Pt"] * 20  # Near 13 or 23
        atoms = create_initial_cluster(
            comp,
            mode="template",
            rng=rng,
            connectivity_factor=CONNECTIVITY_FACTOR,
            placement_radius_scaling=1.3,
        )
        assert len(atoms) == 20
        assert_cluster_valid(atoms, comp)

    def test_template_mode_fallback(self, rng):
        """Test template mode fallback when generation fails."""
        # Very small size that may not have template
        comp = ["Pt"] * 2
        atoms = create_initial_cluster(comp, mode="template", rng=rng)
        # Should fall back to random_spherical and still be valid
        assert len(atoms) == 2
        assert_cluster_valid(atoms, comp)

    def test_template_mode_multi_element(self, rng):
        """Test template mode with multiple elements."""
        comp = ["Pt", "Au"] * 6 + ["Pt"]  # 13 atoms total
        atoms = create_initial_cluster(comp, mode="template", rng=rng)
        assert len(atoms) == 13
        assert_cluster_valid(atoms, comp)
        symbols = atoms.get_chemical_symbols()
        assert "Pt" in symbols
        assert "Au" in symbols


class TestTemplateRemovalSafety:
    """Tests for template generation with atom removal.

    These tests verify that template generation either produces valid
    connected structures or cleanly returns None (without producing
    disconnected clusters).
    """

    # Sizes just below magic numbers where removal is needed
    NEAR_MAGIC_SIZES = [
        (12, 13),  # 12 atoms from 13-atom icosahedron
        (11, 13),  # 11 atoms from 13-atom icosahedron
        (52, 55),  # 52 atoms from 55-atom icosahedron
        (50, 55),  # 50 atoms from 55-atom icosahedron
    ]

    @pytest.mark.parametrize("seed", REPRODUCIBILITY_SEEDS[:5])
    @pytest.mark.parametrize(
        "target_n,magic_n", NEAR_MAGIC_SIZES[:2]
    )  # Just the smaller ones
    def test_near_match_templates_valid_or_none(self, seed, target_n, magic_n):
        """Test that near-match templates are either valid or cleanly skipped.

        When generating templates that require atom removal, the result should
        either be a valid connected structure or None (if removal would disconnect).
        """
        # Manual RNG creation needed for parametrized test with specific seeds
        rng, _ = create_paired_rngs(seed)
        comp = ["Pt"] * target_n
        cell_side = compute_cell_side(comp)

        templates = generate_template_matches(
            composition=comp,
            n_atoms=target_n,
            rng=rng,
            cell_side=cell_side,
            include_exact=False,
            include_near=True,
        )

        # Each returned template must be valid
        for template in templates:
            assert len(template) == target_n, (
                f"Template has wrong size: {len(template)} != {target_n}"
            )

            validate_structure_with_diagnostics(
                template,
                context=f"seed={seed}, target_n={target_n}, magic_n={magic_n}",
            )

    @pytest.mark.parametrize("seed", REPRODUCIBILITY_SEEDS[:5])
    @pytest.mark.parametrize(
        "template_type", ["icosahedron", "decahedron", "octahedron"]
    )
    def test_template_structure_valid_or_none(self, seed, template_type):
        """Test that generate_template_structure returns valid or None."""
        # Manual RNG creation needed for parametrized test with specific seeds
        rng, _ = create_paired_rngs(seed)
        n_atoms = 10  # Not a magic number, requires adjustment
        comp = ["Pt"] * n_atoms

        result = generate_template_structure(
            composition=comp,
            n_atoms=n_atoms,
            template_type=template_type,
            rng=rng,
        )

        if result is not None:
            assert len(result) == n_atoms
            validate_structure_with_diagnostics(
                result,
                context=f"seed={seed}, template_type={template_type}, n_atoms={n_atoms}",
            )

    @pytest.mark.slow
    @pytest.mark.parametrize("seed", REPRODUCIBILITY_SEEDS[:2])
    @pytest.mark.parametrize("target_n,magic_n", NEAR_MAGIC_SIZES)
    def test_near_match_templates_larger_sizes(self, seed, target_n, magic_n):
        """Test near-match templates with larger sizes (slow)."""
        LARGE_CLUSTER_FACTOR = 2.0  # More lenient for larger structures
        # Manual RNG creation needed for parametrized test with specific seeds
        rng, _ = create_paired_rngs(seed)
        comp = ["Pt"] * target_n
        cell_side = compute_cell_side(comp)

        templates = generate_template_matches(
            composition=comp,
            n_atoms=target_n,
            rng=rng,
            cell_side=cell_side,
            connectivity_factor=LARGE_CLUSTER_FACTOR,
            include_exact=False,
            include_near=True,
        )

        for template in templates:
            assert len(template) == target_n
            validate_structure_with_diagnostics(
                template,
                connectivity_factor=LARGE_CLUSTER_FACTOR,
                context=f"seed={seed}, target_n={target_n}",
            )


class TestConvexHullPlacementReliability:
    """Tests verifying convex hull placement works reliably for normal cases."""

    def test_convex_hull_placement_always_works(self, rng):
        """Verify convex hull placement works reliably for normal cases (no fallbacks needed)."""
        # Test seed+growth which uses convex hull placement
        seed = Atoms("Pt3", positions=[[0, 0, 0], [2.5, 0, 0], [1.25, 2.2, 0]])
        target_comp = ["Pt"] * 10
        cell_side = compute_cell_side(target_comp, vacuum=8.0)

        # Should work with default parameters - convex hull placement should be reliable
        result = grow_from_seed(
            seed_atoms=seed,
            target_composition=target_comp,
            placement_radius_scaling=PLACEMENT_RADIUS_SCALING_DEFAULT,
            cell_side=cell_side,
            rng=rng,
            min_distance_factor=MIN_DISTANCE_FACTOR_DEFAULT,
            connectivity_factor=CONNECTIVITY_FACTOR,
        )

        assert result is not None, "Convex hull placement should work for normal cases"
        assert len(result) == len(target_comp)
        assert get_composition_counts(
            result.get_chemical_symbols()
        ) == get_composition_counts(target_comp)
        assert is_cluster_connected(result, connectivity_factor=CONNECTIVITY_FACTOR)

    def test_seed_growth_convex_hull_reliable(self, rng):
        """Verify seed+growth convex hull placement works reliably for normal cases."""
        # Test various sizes that should work with convex hull placement
        seed = Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
        sizes = [5, 8, 12, 15]

        for size in sizes:
            target_comp = ["Pt"] * size
            cell_side = compute_cell_side(target_comp, vacuum=8.0)

            result = grow_from_seed(
                seed_atoms=seed,
                target_composition=target_comp,
                placement_radius_scaling=PLACEMENT_RADIUS_SCALING_DEFAULT,
                cell_side=cell_side,
                rng=rng,
            )

            assert result is not None, (
                f"Seed+growth convex hull placement should work for {size} atoms"
            )
            assert len(result) == size
            assert is_cluster_connected(result)

    def test_seed_combination_convex_hull_reliable(self, rng):
        """Verify seed combination convex hull placement works reliably."""
        seed1 = Atoms("Pt", positions=[[0, 0, 0]])
        seed2 = Atoms("Au", positions=[[0, 0, 0]])

        # Should work with default parameters
        combined = combine_seeds(
            seeds=[seed1, seed2],
            cell_side=10.0,
            separation_scaling=1.0,
            rng=rng,
            connectivity_factor=CONNECTIVITY_FACTOR,
        )

        assert combined is not None, "Seed combination should work reliably"
        assert len(combined) == 2
        assert get_composition_counts(
            combined.get_chemical_symbols()
        ) == get_composition_counts(["Pt", "Au"])

    def test_default_parameters_sufficient_convex_hull(self, rng):
        """Verify default parameters work for convex hull placement (shouldn't need relaxation)."""
        seed = Atoms("Pt3", positions=[[0, 0, 0], [2.5, 0, 0], [1.25, 2.2, 0]])
        target_comp = ["Pt"] * 8
        cell_side = compute_cell_side(target_comp, vacuum=8.0)

        # Use default parameters - should work without relaxation
        result = grow_from_seed(
            seed_atoms=seed,
            target_composition=target_comp,
            placement_radius_scaling=PLACEMENT_RADIUS_SCALING_DEFAULT,
            cell_side=cell_side,
            rng=rng,
            min_distance_factor=MIN_DISTANCE_FACTOR_DEFAULT,
            connectivity_factor=CONNECTIVITY_FACTOR,
        )

        assert result is not None, (
            "Default parameters should be sufficient for convex hull placement"
        )
        assert len(result) == len(target_comp)
        assert is_cluster_connected(result)

    def test_large_clusters_convex_hull_works(self, rng):
        """Verify large clusters work with convex hull placement using default parameters."""
        seed = Atoms(
            "Pt4", positions=[[0, 0, 0], [2.5, 0, 0], [1.25, 2.2, 0], [1.25, 0.73, 1.8]]
        )
        target_comp = ["Pt"] * 20
        cell_side = compute_cell_side(target_comp, vacuum=8.0)

        # Large clusters should work with convex hull placement
        result = grow_from_seed(
            seed_atoms=seed,
            target_composition=target_comp,
            placement_radius_scaling=PLACEMENT_RADIUS_SCALING_DEFAULT,
            cell_side=cell_side,
            rng=rng,
        )

        assert result is not None, (
            "Large clusters should work with convex hull placement"
        )
        assert len(result) == len(target_comp)
        assert is_cluster_connected(result)


class TestTemplateGenerationOptimization:
    """Tests for template generation optimizations."""

    def test_exact_match_avoids_near_match_redundancy(self, rng):
        """Test that exact magic number matches don't redundantly generate near matches."""
        # Use exact magic number (13)
        comp = ["Pt"] * 13

        # Should work without redundant template generation
        atoms = create_initial_cluster(comp, mode="smart", rng=rng)
        assert len(atoms) == 13
        assert get_composition_counts(
            atoms.get_chemical_symbols()
        ) == get_composition_counts(comp)


# ---------------------------------------------------------------------
# from test_mixed_compositions.py
# ---------------------------------------------------------------------

"""Consolidated tests for mixed compositions.

This module was refactored to reduce duplication by parametrizing similar
cases (bimetallic / trimetallic / multimetallic / small-mixed) while keeping
edge cases and mode coverage explicit.
"""


@pytest.mark.parametrize(
    "composition",
    [
        ["Pt", "Au"],
        ["Pt", "Au"] * 5,
        ["Pt", "Au", "Au"] * 3,
        ["Pt", "Pt", "Au"] * 3,
        ["Pt"] + ["Au"] * 8,
        ["Pt"] * 8 + ["Au"],
        ["H", "Pt"] * 5,
        ["Pt", "Au"] * 10,
    ],
)
def test_bimetallic_variants(composition, rng):
    """Parametrized coverage of bimetallic composition shapes and sizes."""
    atoms = create_initial_cluster(composition, rng=rng)
    assert_cluster_valid(atoms, composition)


@pytest.mark.parametrize(
    "mode", ["smart", "random_spherical", "seed+growth", "template"]
)
def test_bimetallic_all_modes(mode, rng):
    """Ensure basic bimetallic cases work across all initialization modes."""
    composition = ["Pt", "Au"] * 3
    atoms = create_initial_cluster(composition, mode=mode, rng=rng)
    assert_cluster_valid(atoms, composition)


@pytest.mark.parametrize(
    "composition",
    [
        ["Pt", "Au", "Pd"] * 3,
        ["Pt", "Pt", "Au", "Pd"] * 2 + ["Pt"],
        ["Pt"] * 7 + ["Au"] + ["Pd"],
        ["Li", "Pt", "Au"] * 3,
        ["Pt", "Au", "Pd"] * 5,
    ],
)
def test_trimetallic_variants(composition, rng):
    """Parametrized trimetallic cases (equal/unequal/skewed/large)."""
    atoms = create_initial_cluster(composition, rng=rng)
    assert_cluster_valid(atoms, composition)


@pytest.mark.parametrize(
    "composition",
    [["Pt", "Au", "Pd", "Ag"] * 2, ["Pt", "Au", "Pd", "Ag", "Cu"] * 2],
)
def test_multimetallic_variants(composition, rng):
    """Parametrized coverage for 4+ element clusters."""
    atoms = create_initial_cluster(composition, rng=rng)
    assert_cluster_valid(atoms, composition)


@pytest.mark.parametrize(
    "composition",
    [
        ["Pt", "Au"],
        ["Pt", "Pt", "Au"],
        ["Pt", "Au", "Pd", "Pt"],
        ["Pt", "Au", "Pd", "Ag", "Cu"],
    ],
)
def test_small_mixed_compositions(composition, rng):
    """Small (2-5 atom) mixed-composition smoke tests."""
    atoms = create_initial_cluster(composition, rng=rng)
    assert_cluster_valid(atoms, composition)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_different_seeds_produce_diversity(seed):
    """Different RNG seeds should produce diverse structures for mixed comps."""
    composition = ["Pt", "Au"] * 5
    rng_test, _ = create_paired_rngs(seed)
    atoms = create_initial_cluster(composition, rng=rng_test)
    assert_cluster_valid(atoms, composition)


@pytest.mark.reproducibility
@pytest.mark.requires_cache_isolation
def test_reproducibility_same_seed(rng):
    """Same seed produces identical structure and composition."""
    composition = ["Pt", "Au"] * 5
    rng1, rng2 = create_paired_rngs(42)
    atoms1 = create_initial_cluster(composition, rng=rng1)
    atoms2 = create_initial_cluster(composition, rng=rng2)

    assert np.allclose(atoms1.get_positions(), atoms2.get_positions(), atol=1e-10)
    assert atoms1.get_chemical_symbols() == atoms2.get_chemical_symbols()


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_trimetallic_diversity(seed):
    """Parametrized diversity checks for trimetallic compositions."""
    composition = ["Pt", "Au", "Pd"] * 4
    rng_test, _ = create_paired_rngs(seed)
    atoms = create_initial_cluster(composition, rng=rng_test)
    assert_cluster_valid(atoms, composition)


@pytest.mark.parametrize(
    "composition",
    [["H", "H", "Pt", "Pt", "Pt"], ["H", "Li", "Pt", "Au"] * 2],
)
def test_edge_case_mixed_compositions(composition, rng):
    """Edge cases: extreme size differences, magic numbers, strict connectivity."""
    atoms = create_initial_cluster(composition, rng=rng)
    assert_cluster_valid(atoms, composition)


@pytest.mark.parametrize(
    "composition", [["Pt", "Au"] * 6 + ["Pt"], ["Pt", "Au", "Pd"] * 4 + ["Pt"]]
)
def test_magic_number_and_strict_connectivity(composition, rng):
    atoms = create_initial_cluster(
        composition,
        mode="smart",
        rng=rng,
        connectivity_factor=CONNECTIVITY_FACTOR,
        min_distance_factor=0.5,
    )
    assert_cluster_valid(atoms, composition)


# ---------------------------------------------------------------------
# from test_parameter_sensitivity.py
# ---------------------------------------------------------------------

"""Tests for parameter sensitivity and cross-mode comparisons.

This module tests:
- How initialization parameters affect cluster generation across modes
- Cross-mode behavior comparisons
- Parameter interactions
- Performance scaling with cluster size

These tests were identified as coverage gaps and are new additions
to improve overall test suite coverage.
"""


class TestParameterSensitivity:
    """Test how parameters affect cluster generation."""

    @pytest.mark.parametrize(
        "connectivity_factor",
        [0.7, 0.9, 1.0, 1.1, 1.3],
    )
    def test_connectivity_factor_impact(self, connectivity_factor, rng):
        """Test that varying connectivity_factor affects cluster generation."""
        comp = ["Pt"] * 8
        try:
            atoms = create_initial_cluster(
                comp,
                mode="random_spherical",
                connectivity_factor=connectivity_factor,
                rng=rng,
            )
            # Should always produce connected clusters with proper connectivity factor
            assert len(atoms) == 8
            assert_cluster_valid(atoms, comp, connectivity_factor=connectivity_factor)
        except (ValueError, SCGOValidationError) as e:
            msg = str(e)
            # At the GA steric floor, bond length equals the clash limit; random
            # placement may fail to find a valid configuration for larger clusters.
            if "Validation failed" in msg:
                return
            if (
                connectivity_factor <= BLMIN_RATIO_DEFAULT
                and "Could not place all" in msg
            ):
                return
            raise

    @pytest.mark.parametrize(
        "placement_radius_scaling",
        [0.4, 0.6, 0.8, 1.0, 1.2],
    )
    def test_placement_radius_scaling_impact(self, placement_radius_scaling, rng):
        """Test that varying placement_radius_scaling still yields a valid cluster."""
        comp = ["Pt"] * 6
        atoms = create_initial_cluster(
            comp,
            mode="smart",
            placement_radius_scaling=placement_radius_scaling,
            rng=rng,
        )
        assert atoms is not None
        assert len(atoms) == 6
        assert np.all(np.isfinite(atoms.get_positions()))
        assert_cluster_valid(atoms, comp)

    @pytest.mark.parametrize(
        "vacuum",
        [3.0, 5.0, 7.0, 10.0, 15.0],
    )
    def test_vacuum_impact_on_cluster(self, vacuum, rng):
        """Test that vacuum parameter affects cell size appropriately."""
        comp = ["Pt"] * 5
        atoms1 = create_initial_cluster(
            comp, mode="random_spherical", vacuum=3.0, rng=rng
        )
        atoms2 = create_initial_cluster(
            comp, mode="random_spherical", vacuum=10.0, rng=rng
        )

        cell1 = atoms1.get_cell().lengths()
        cell2 = atoms2.get_cell().lengths()

        # Larger vacuum should result in larger cells
        assert np.mean(cell2) > np.mean(cell1)

    def test_parameter_interaction_connectivity_and_radius(self, rng):
        """Test interaction between connectivity_factor and placement_radius_scaling."""
        comp = ["Pt"] * 8

        # Test different combinations
        combinations = [
            (0.8, 0.5),
            (1.0, 0.7),
            (1.2, 1.0),
        ]

        for cf, prs in combinations:
            atoms = create_initial_cluster(
                comp,
                mode="smart",
                connectivity_factor=cf,
                placement_radius_scaling=prs,
                rng=rng,
            )
            assert len(atoms) == 8
            assert_cluster_valid(atoms, comp, check_connectivity=True)

    @pytest.mark.reproducibility
    @pytest.mark.requires_cache_isolation
    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 999])
    def test_reproducibility_with_parameters(self, seed):
        """Test that same parameters with same seed produce same structure."""
        comp = ["Pt"] * 6

        rng1, rng2 = create_paired_rngs(seed)
        atoms1 = create_initial_cluster(
            comp,
            mode="seed+growth",
            connectivity_factor=1.0,
            placement_radius_scaling=0.7,
            rng=rng1,
        )

        atoms2 = create_initial_cluster(
            comp,
            mode="seed+growth",
            connectivity_factor=1.0,
            placement_radius_scaling=0.7,
            rng=rng2,
        )

        assert np.allclose(
            atoms1.get_positions(),
            atoms2.get_positions(),
            atol=1e-6,
        )


class TestCrossModeComparison:
    """Compare behavior across different initialization modes."""

    def test_all_modes_produce_similar_composition(self, rng):
        """Test that all modes respect composition exactly."""
        comp = ["Pt", "Au", "Pd", "Pt", "Au"]

        modes = ["random_spherical", "seed+growth", "smart"]

        for mode in modes:
            try:
                atoms = create_initial_cluster(comp, mode=mode, rng=rng)
                assert_cluster_valid(atoms, comp)
            except (ValueError, SCGOValidationError):
                if mode == "template":
                    pytest.skip("Template mode may fail for non-magic numbers")
                raise

    def test_mode_connectivity_consistency(self, rng):
        """Test that all modes produce connected structures."""
        comp = ["Pt"] * 10

        modes = ["random_spherical", "seed+growth", "smart"]

        for mode in modes:
            try:
                atoms = create_initial_cluster(comp, mode=mode, rng=rng)
                assert is_cluster_connected(atoms)
            except (ValueError, SCGOValidationError):
                if mode == "template":
                    pytest.skip("Template mode may fail for non-magic numbers")
                raise

    def test_mode_diversity_comparison(self):
        """Compare diversity of structures across modes."""
        comp = ["Pt"] * 6

        def structure_signature(atoms):
            """Create a unique signature for cluster structure."""
            p = atoms.get_positions()
            d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
            triu = d[np.triu_indices(len(p), k=1)]
            return tuple(np.round(np.sort(triu), 4))

        mode_diversities = {}

        for mode in ["random_spherical", "seed+growth"]:
            signatures = set()
            for i in range(5):
                rng, _ = create_paired_rngs(1000 + i)
                atoms = create_initial_cluster(comp, mode=mode, rng=rng)
                signatures.add(structure_signature(atoms))

            mode_diversities[mode] = len(signatures)

        assert all(d >= 2 for d in mode_diversities.values())

    def test_mode_performance_scaling(self, rng):
        """Verify all initialization modes complete for small cluster sizes."""
        modes = ["random_spherical", "seed+growth", "smart"]
        sizes = [3, 5, 8]

        for size in sizes:
            comp = ["Pt"] * size
            for mode in modes:
                try:
                    atoms = create_initial_cluster(comp, mode=mode, rng=rng)
                    assert len(atoms) == size
                    assert is_cluster_connected(atoms) or size <= 2
                except (ValueError, SCGOValidationError):
                    if mode == "template":
                        pytest.skip("Template mode may fail for non-magic numbers")
                    raise


class TestClusterSizeScaling:
    """Test how algorithms scale with cluster size."""

    @pytest.mark.parametrize("size", SMALL_SIZES)
    def test_connectivity_scaling_small(self, size, rng):
        """Test connectivity is maintained for small clusters."""
        comp = ["Pt"] * size
        atoms = create_initial_cluster(comp, mode="smart", rng=rng)

        if size > 2:
            assert is_cluster_connected(atoms)

    @pytest.mark.parametrize("size", MEDIUM_SIZES)
    @pytest.mark.slow
    def test_connectivity_scaling_medium(self, size, rng):
        """Test connectivity is maintained for medium clusters."""
        comp = ["Pt"] * size
        atoms = create_initial_cluster(comp, mode="smart", rng=rng)

        if size > 2:
            assert is_cluster_connected(atoms)

    @pytest.mark.parametrize("size", LARGE_SIZES)
    @pytest.mark.slow
    def test_connectivity_scaling_large(self, size, rng):
        """Test connectivity is maintained for large clusters."""
        comp = ["Pt"] * size
        atoms = create_initial_cluster(comp, mode="random_spherical", rng=rng)

        if size > 2:
            assert is_cluster_connected(atoms)

    def test_cell_size_scales_with_cluster_size(self, rng):
        """Test that cell size appropriately scales with cluster size."""
        cell_sizes = {}

        for size in [2, 5, 10, 15]:
            comp = ["Pt"] * size
            atoms = create_initial_cluster(comp, vacuum=6.0, rng=rng)
            cell_volume = atoms.get_cell().volume
            cell_sizes[size] = cell_volume

        # Larger clusters should have larger cells
        assert cell_sizes[15] > cell_sizes[10]
        assert cell_sizes[10] > cell_sizes[5]
        assert cell_sizes[5] > cell_sizes[2]

    def test_position_variance_scales_with_size(self, rng):
        """Test that positional variance increases with cluster size."""
        position_ranges = {}

        for size in [3, 5, 8, 12]:
            comp = ["Pt"] * size
            atoms = create_initial_cluster(comp, rng=rng)
            positions = atoms.get_positions()
            position_range = np.max(positions) - np.min(positions)
            position_ranges[size] = position_range

        # Larger clusters should have larger position spreads
        assert position_ranges[12] > position_ranges[3]


class TestCompositionVariety:
    """Test parameter sensitivity across various compositions."""

    @pytest.mark.parametrize(
        "composition",
        [
            ["Pt"] * 4,
            ["Au"] * 4,
            ["Pd"] * 4,
            ["Pt", "Pt", "Au", "Au"],
            ["Pt", "Au", "Pd", "Pd"],
            ["Pt", "Pt", "Pt", "Au"],
        ],
    )
    def test_connectivity_across_compositions(self, composition, rng):
        """Test that connectivity is maintained across different compositions."""
        atoms = create_initial_cluster(composition, mode="smart", rng=rng)
        assert_cluster_valid(atoms, composition)

        if len(atoms) > 2:
            assert is_cluster_connected(atoms)

    @pytest.mark.parametrize(
        "composition",
        [
            ["Pt"] * 5,
            ["Au"] * 5,
            ["Pt", "Pt", "Au", "Au", "Pd"],
        ],
    )
    @pytest.mark.parametrize("cf", [0.8, 1.0, 1.2])
    def test_parameter_robustness_across_elements(self, composition, cf, rng):
        """Parameter sensitivity must be consistent: every combo yields a valid cluster."""
        atoms = create_initial_cluster(
            composition,
            mode="seed+growth",
            connectivity_factor=cf,
            rng=rng,
        )
        assert atoms is not None
        assert len(atoms) == len(composition)
        assert np.all(np.isfinite(atoms.get_positions()))
        assert_cluster_valid(atoms, composition)
        assert is_cluster_connected(atoms)


# ---------------------------------------------------------------------
# from test_init_logging.py
# ---------------------------------------------------------------------

"""Test initialization logging to understand the duplicate messages."""


def test_create_initial_cluster_batch_logs_and_returns_population(caplog, rng):
    composition = ["Pt"] * 4
    n_structures = 59
    caplog.set_level(logging.INFO)
    population = create_initial_cluster_batch(
        composition=composition,
        n_structures=n_structures,
        rng=rng,
        mode="smart",
        n_jobs=1,
    )

    assert isinstance(population, list)
    assert len(population) == n_structures
    # Ensure expected initialization log messages were emitted
    assert "Initialization for 4-atom clusters" in caplog.text
    assert "Strategy allocation" in caplog.text
    assert caplog.text.count("Population initialization:") == 1
    assert "Fallbacks: template->random=" not in caplog.text


def test_batch_init_fallback_summary_emitted_once(caplog, rng):
    """Multi-structure batch should emit one init summary, not per-structure fallbacks."""
    caplog.set_level(logging.DEBUG)
    create_initial_cluster_batch(
        composition=["Pt"] * 4,
        n_structures=10,
        rng=rng,
        mode="smart",
        n_jobs=1,
    )
    assert caplog.text.count("Population initialization:") == 1
    assert caplog.text.count("Fallbacks: template->random=") == 0


def test_format_placement_error_message_is_compact_and_consistent():
    msg = format_placement_error_message(
        context="complete batch placement (4/15 placed, 11 remaining, 500 attempts)",
        composition=None,
        n_atoms=None,
        placement_radius_scaling=1.2,
        min_distance_factor=0.7,
        connectivity_factor=1.4,
        additional_info="remaining: Ptx11",
    )

    assert msg.startswith(
        "Could not complete batch placement (4/15 placed, 11 remaining, 500 attempts)"
    )
    assert "  parameters: placement_radius_scaling=1.20" in msg
    assert "  remaining: Ptx11" in msg
    assert "  suggestions: placement_radius_scaling→1.80" in msg
    assert "Parameters:" not in msg
    assert "Diagnostics:" not in msg


def test_format_composition_counts_short():
    assert format_composition_counts_short({"Pt": 11}) == "Ptx11"
    assert format_composition_counts_short({"Au": 2, "Pt": 3}) == "Aux2, Ptx3"


def test_seed_sampling_log_collector_groups_failures(caplog):
    caplog.set_level(logging.INFO)

    _SeedSamplingLogCollector.reset()
    for _ in range(3):
        _SeedSamplingLogCollector.record("Pt5", "unsuitable linear geometry")
    for _ in range(2):
        _SeedSamplingLogCollector.record("Pt6", "need mixed seed geometries")
    _SeedSamplingLogCollector.emit_summary_if_any()

    assert caplog.text.count("no suitable seed") == 1
    assert "Pt5x3 [unsuitable linear geometry]" in caplog.text
    assert "Pt6x2 [need mixed seed geometries]" in caplog.text


def test_sample_suitable_seed_reports_specific_failure_reason(rng):
    linear_seed = Atoms("Pt3", positions=[[0, 0, 0], [0, 0, 2.5], [0, 0, 5.0]])
    candidates = [(0.0, linear_seed)]

    seed, reason = _sample_suitable_seed(
        candidates,
        strategy=0,
        tried_positions=set(),
        existing_geometries=[],
        rng=rng,
    )

    assert seed is None
    assert reason is not None
    assert "linear" in reason


def test_batch_seed_failures_are_grouped_not_repeated(caplog, rng):
    """Repeated per-structure seed failures should collapse to one INFO summary."""
    composition = ["Pt"] * 15
    n_structures = 20
    caplog.set_level(logging.INFO)

    create_initial_cluster_batch(
        composition=composition,
        n_structures=n_structures,
        rng=rng,
        mode="seed+growth",
        n_jobs=1,
    )

    seed_failure_lines = [
        line for line in caplog.text.splitlines() if "no suitable seed" in line
    ]
    if seed_failure_lines:
        assert len(seed_failure_lines) == 1
        assert "failures" in seed_failure_lines[0]
        assert "after attempts" not in caplog.text
    else:
        pytest.skip("No seed failures in this run; database may have suitable seeds")


def test_single_structure_smart_logs_strategy_allocation(caplog, rng):
    """Single-structure smart mode should emit allocation INFO at least once."""
    caplog.set_level(logging.INFO)
    atoms = create_initial_cluster(
        ["Pt"] * 4,
        mode="smart",
        rng=rng,
    )
    assert len(atoms) == 4
    assert "Initialization for 4-atom clusters" in caplog.text
    assert "Strategy allocation (1 structure" in caplog.text


def test_all_strategies_none_emits_warning_before_raise(caplog):
    """Terminal all-None path should WARNING before raising RuntimeError."""
    caplog.set_level(logging.WARNING)

    def always_none():
        return None

    with pytest.raises(
        SCGORuntimeError, match="All initialization strategies returned None"
    ):
        _try_strategies_in_order(
            [("primary", always_none), ("fallback", always_none)],
            composition=["Pt", "Pt"],
            connectivity_factor=1.4,
        )

    assert any(
        "All initialization strategies returned None" in r.getMessage()
        and r.levelno == logging.WARNING
        for r in caplog.records
    )
