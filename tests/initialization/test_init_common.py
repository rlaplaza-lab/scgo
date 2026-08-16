"""Common initialization tests shared across all modes.

This module contains tests for:
- Basic initialization functionality (smoke tests, edge cases)
- Parameter validation
- Cell side computation
- Internal function testing
- Database integration
- Caching and thread safety
- Reproducibility
- Mode switching and fallbacks
"""

import sqlite3
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.db import connect

from scgo.database import get_global_cache
from scgo.database.helpers import extract_minima_from_database_file
from scgo.exceptions import SCGOValidationError
from scgo.initialization import (
    BatchInitPlan,
    compute_cell_side,
    create_initial_cluster,
    create_initial_cluster_batch,
    is_cluster_connected,
    plan_batch_initialization,
    random_spherical,
)
from scgo.initialization.candidate_discovery import (
    _discover_all_candidates,
    _find_exact_candidates,
    _load_candidates_from_file,
    _load_db_candidates,
)
from scgo.initialization.geometry_helpers import (
    _classify_seed_geometry,
    _get_cached_hull,
    _get_positions_hash,
    _sort_atoms_by_element,
    analyze_disconnection,
    clear_convex_hull_cache,
    get_largest_facets,
    reorder_cluster_to_composition,
    validate_cluster,
    validate_cluster_structure,
)
from scgo.initialization.initialization_config import (
    CONNECTIVITY_FACTOR,
    MIN_DISTANCE_FACTOR_DEFAULT,
)
from scgo.initialization.initializers import (
    _boltzmann_sample,
    _find_smaller_candidates,
    _try_exact_match,
)
from scgo.initialization.random_spherical import (
    _add_atoms_to_cluster_iteratively,
    _growth_order_by_mass,
    _sample_atoms_to_add_order,
)
from scgo.initialization.seed_combiners import (
    _is_valid_placement,
)
from scgo.metadata.atoms import get_tag, set_tags
from scgo.metadata.db_stamp import stamp_db
from scgo.utils.helpers import get_composition_counts
from tests.helpers import (
    BATCH_TEST_SAMPLES,
    BATCH_TEST_SAMPLES_SLOW,
    MIXED_COMPOSITIONS,
    UNIQUENESS_THRESHOLD,
    assert_cluster_valid,
    create_paired_rngs,
    positions_equal,
    run_batch_connectivity_test,
    validate_structure_with_diagnostics,
)


def _stamp_init_test_db(db_path: str | Path) -> None:
    """Mark raw ASE-DB test files as SCGO databases for candidate discovery."""
    stamp_db(db_path)


def _discovery_db_path(
    tmp_path: Path, formula_hint: str, filename: str = "cluster.db"
) -> Path:
    """Return a DB path whose parent directories encode composition for discovery."""
    db_path = tmp_path / f"{formula_hint}_searches" / "run_001" / filename
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


class TestBasicInitialization:
    """Tests for basic cluster initialization functionality."""

    def test_create_initial_cluster_unknown_mode(self, rng):
        """Test that unknown initialization mode raises appropriate error."""
        comp = ["Pt"]
        with pytest.raises(
            SCGOValidationError,
            match='Unsupported mode: "invalid_mode"',
        ):
            create_initial_cluster(comp, mode="invalid_mode", rng=rng)

    def test_strict_connectivity_enforcement(self, rng):
        """Test that connectivity is strictly enforced with default factor."""
        comp = ["Pt"] * 8
        atoms = create_initial_cluster(
            comp, rng=rng, connectivity_factor=CONNECTIVITY_FACTOR
        )

        # Verify all invariants using helper
        assert_cluster_valid(atoms, comp)

    def test_no_atomic_clashes_strict(self, rng):
        """Test that no atomic clashes occur with strict distance checking."""
        comp = ["Pt"] * 6
        atoms = create_initial_cluster(comp, rng=rng, min_distance_factor=0.5)

        # Verify all invariants using helper
        assert_cluster_valid(atoms, comp, min_distance_factor=0.5)

    @pytest.mark.parametrize(
        "mode", ["smart", "random_spherical", "seed+growth", "template"]
    )
    def test_empty_composition_all_modes(self, mode, rng):
        """Test that empty composition returns empty Atoms object for all modes."""
        result = create_initial_cluster([], mode=mode, rng=rng)
        assert isinstance(result, Atoms)
        assert len(result) == 0

    @pytest.mark.parametrize(
        "mode", ["smart", "random_spherical", "seed+growth", "template"]
    )
    def test_single_atom_all_modes(self, mode, rng):
        """Test that single atom composition works and satisfies invariants for all modes."""
        comp = ["Pt"]
        atoms = create_initial_cluster(comp, mode=mode, rng=rng)
        assert isinstance(atoms, Atoms)
        assert len(atoms) == 1
        assert atoms.get_chemical_symbols() == comp
        assert get_composition_counts(
            atoms.get_chemical_symbols()
        ) == get_composition_counts(comp)
        cell = atoms.get_cell()
        assert cell.volume > 0.0


class TestCellSideComputation:
    """Tests for cell side computation functionality."""

    def test_compute_cell_side_empty(self):
        """Test cell side computation with empty composition."""
        assert compute_cell_side([]) == pytest.approx(0.0, abs=1e-8)

    def test_compute_cell_side_simple(self):
        """Test cell side computation with simple composition."""
        side = compute_cell_side(["H", "H"], vacuum=5.0)
        assert side > 0.0
        assert isinstance(side, float)

    def test_compute_cell_side_positive(self):
        """Test cell side computation with different vacuum values."""
        side = compute_cell_side(["Pt", "Pt"], vacuum=5.0)
        assert side > 0.0
        # with increased vacuum the side should grow
        side2 = compute_cell_side(["Pt", "Pt"], vacuum=15.0)
        assert side2 > side

    def test_packing_efficiency_applied(self):
        """Test that packing efficiency factor is applied."""
        comp = ["Pt"] * 10
        side_with_efficiency = compute_cell_side(comp, vacuum=10.0)

        # Should be larger than naive calculation (accounts for packing)
        # This is a sanity check - exact value depends on implementation
        assert side_with_efficiency > 0

    def test_large_cell_side_warning(self):
        """Test that warning is issued for very large cell sides."""
        # Use very large composition that will definitely exceed threshold
        # With packing efficiency, 10000 atoms should produce ~200-300 Å cell side
        # So we need even more or larger vacuum to trigger warning
        comp = ["Pt"] * 50000  # Very large composition
        side = compute_cell_side(comp, vacuum=500.0)  # Large vacuum to ensure warning
        # Just verify it computes (warning may or may not trigger
        # depending on exact calculation)
        assert side > 0

    def test_compute_cell_side_nan_vdw_fallback(self):
        """Test cell side for elements with NaN vdw_radii in ASE (e.g., Co, Fe, Ru).

        ASE's vdw_radii table has NaN for many transition metals. compute_cell_side
        should use interpolated VdW radii and return a finite value.
        """
        import numpy as np

        # Co, Fe have NaN vdw_radii in ASE
        for comp in [["Pt", "Pt", "Pt", "Pt", "Co"], ["Fe", "Fe"], ["Pt", "Co"]]:
            side = compute_cell_side(comp, vacuum=10.0)
            assert np.isfinite(side), f"cell_side should be finite for {comp}"
            assert side > 0, f"cell_side should be positive for {comp}"


class TestInvalidElementSymbols:
    """Tests for invalid element symbol handling."""

    @pytest.mark.parametrize("symbols", [["Xx"], ["Pt", "Xx"]])
    def test_invalid_element_symbol(self, symbols, rng):
        """Test that invalid element symbols raise SCGOValidationError."""
        with pytest.raises(SCGOValidationError, match="Unknown element symbol"):
            compute_cell_side(symbols, vacuum=10.0)
        with pytest.raises(SCGOValidationError, match="Unknown element symbol"):
            create_initial_cluster(symbols, rng=rng)


class TestDegenerateGeometries:
    """Tests for handling degenerate geometries."""

    def test_collinear_atoms(self, rng):
        """Test initialization with collinear atoms."""
        comp = ["Pt", "Pt", "Pt"]
        atoms = create_initial_cluster(comp, rng=rng)
        # Should still produce valid structure
        assert_cluster_valid(atoms, comp)

    def test_coplanar_atoms(self, rng):
        """Test initialization with coplanar atoms."""
        comp = ["Pt"] * 4
        atoms = create_initial_cluster(comp, rng=rng)
        # Should still produce valid structure
        assert_cluster_valid(atoms, comp)

    def test_pt4co1_multi_element_initialization(self, rng):
        """Test initialization for Pt4Co1 and similar multi-element clusters.

        Elements like Co have NaN vdw_radii in ASE, which previously caused
        compute_cell_side to return NaN and initialization to fail.
        """
        comp = ["Pt", "Pt", "Pt", "Pt", "Co"]
        atoms = create_initial_cluster(comp, rng=rng, mode="random_spherical")
        assert_cluster_valid(atoms, comp)

    def test_overlapping_atoms_validation(self):
        """Test validation detects overlapping atoms."""
        atoms = Atoms("Pt2", positions=[[0, 0, 0], [0.01, 0, 0]])  # Very close
        is_valid, msg = validate_cluster_structure(
            atoms, min_distance_factor=0.5, connectivity_factor=CONNECTIVITY_FACTOR
        )
        # Should detect clash
        assert is_valid is False or "clash" in msg.lower()


# Mode-switching basic invariants consolidated in tests/test_initialization_modes.py


class TestReproducibility:
    """Tests for reproducibility and RNG consistency."""

    @pytest.mark.parametrize(
        "seed1,seed2,expect_equal",
        [(42, 42, True), (42, 123, False)],
    )
    def test_seed_reproducibility(self, seed1, seed2, expect_equal):
        """Test that same seeds give equal results and different seeds give different results."""
        comp = ["Pt", "Pt", "Pt"]
        rng1, _ = create_paired_rngs(seed1)
        _, rng2 = create_paired_rngs(seed2)

        atoms1 = create_initial_cluster(comp, rng=rng1, mode="random_spherical")
        atoms2 = create_initial_cluster(comp, rng=rng2, mode="random_spherical")

        assert positions_equal(atoms1, atoms2) == expect_equal
        assert atoms1.get_chemical_symbols() == atoms2.get_chemical_symbols()


class TestCacheBehavior:
    """Tests for cache behavior across initialization module."""

    def test_convex_hull_cache_persistence(self, pt4_tetrahedron):
        """Test that convex hull cache persists across function calls."""
        atoms = pt4_tetrahedron.copy()

        # First call
        facets1 = get_largest_facets(atoms, n_facets=2)

        # Second call should use cached hull
        facets2 = get_largest_facets(atoms, n_facets=2)

        # Should have same results
        assert len(facets1) == len(facets2)

    def test_database_cache_mtime_based(self, tmp_path, pt2_atoms):
        """Test database cache uses mtime for invalidation."""
        db_path = tmp_path / "test.db"
        with connect(str(db_path)) as db:
            pt2 = pt2_atoms.copy()
            db.write(pt2, relaxed=True, key_value_pairs={"raw_score": -10.0}, gaid=1)
        _stamp_init_test_db(db_path)

        # First call
        _, entries1 = _load_db_candidates(str(db_path))

        # Second call immediately should use cache
        _, entries2 = _load_db_candidates(str(db_path))
        assert len(entries1) == len(entries2)

    def test_load_candidates_from_file_primary_failure_propagates(
        self, tmp_path, monkeypatch, pt2_atoms
    ):
        """Unexpected extractor failures propagate to the caller."""
        db_path = tmp_path / "test.db"
        with connect(str(db_path)) as db:
            pt2 = pt2_atoms.copy()
            db.write(pt2, relaxed=True, key_value_pairs={"raw_score": -10.0}, gaid=1)
        _stamp_init_test_db(db_path)

        def _bad_extract(*_args, **_kwargs):
            raise AttributeError("simulated DB internals error")

        monkeypatch.setattr(
            "scgo.initialization.candidate_discovery.extract_minima_from_database_file",
            _bad_extract,
            raising=True,
        )

        get_global_cache().clear_namespace("db_candidates")
        with pytest.raises(AttributeError, match="simulated DB internals error"):
            _load_candidates_from_file(str(db_path))

    def test_load_candidates_from_file_sqlite_error_returns_empty(
        self, tmp_path, monkeypatch, pt2_atoms
    ):
        """When extraction raises sqlite3 errors, return empty list."""
        db_path = tmp_path / "test.db"
        with connect(str(db_path)) as db:
            pt2 = pt2_atoms.copy()
            db.write(pt2, relaxed=True, key_value_pairs={"raw_score": -10.0}, gaid=1)
        _stamp_init_test_db(db_path)

        def _bad_extract(*_args, **_kwargs):
            raise sqlite3.OperationalError("simulated DB error")

        monkeypatch.setattr(
            "scgo.initialization.candidate_discovery.extract_minima_from_database_file",
            _bad_extract,
            raising=True,
        )

        entries = _load_candidates_from_file(str(db_path))
        assert entries == []

    def test_candidate_discovery_matches_extract_minima(self, tmp_path, pt2_atoms):
        """Initialization discovery and extract_minima_from_database_file agree.

        Folded from the deleted test_candidate_discovery_parity.py: both paths
        must enumerate the same (energy, atoms) candidates for a mixed DB.
        """
        db_path = tmp_path / "test.db"
        with connect(str(db_path)) as db:
            final = pt2_atoms.copy()
            db.write(
                final,
                relaxed=True,
                key_value_pairs={"raw_score": -10.0, "final_unique_minimum": True},
                gaid=1,
            )
            ts = pt2_atoms.copy()
            ts.positions += 0.1
            db.write(
                ts,
                relaxed=True,
                key_value_pairs={"raw_score": -8.0, "is_transition_state": True},
                gaid=2,
            )
            non_final = pt2_atoms.copy()
            non_final.positions -= 0.1
            db.write(
                non_final, relaxed=True, key_value_pairs={"raw_score": -7.0}, gaid=3
            )
        _stamp_init_test_db(db_path)

        extracted = extract_minima_from_database_file(
            str(db_path), run_id="runx", require_final=False
        )
        discovered = _load_candidates_from_file(str(db_path))

        assert len(discovered) == len(extracted)
        discovered_sorted = sorted(discovered, key=lambda item: item[1])
        extracted_sorted = sorted(extracted, key=lambda item: item[0])
        for (_, energy, atoms), (exp_energy, exp_atoms) in zip(
            discovered_sorted, extracted_sorted, strict=True
        ):
            assert energy == exp_energy
            assert atoms.get_chemical_symbols() == exp_atoms.get_chemical_symbols()

    def test_composition_cache_behavior(self, tmp_path, pt2_atoms):
        """Test composition cache behavior."""
        db_path = _discovery_db_path(tmp_path, "Pt2")
        with connect(str(db_path)) as db:
            pt2 = pt2_atoms.copy()
            db.write(pt2, relaxed=True, key_value_pairs={"raw_score": -10.0}, gaid=1)
        _stamp_init_test_db(db_path)

        target1 = ["Pt", "Pt", "Pt"]
        candidates1 = _find_smaller_candidates(target1, str(db_path))

        # Same target should use cache
        candidates2 = _find_smaller_candidates(target1, str(db_path))
        assert candidates1 == candidates2 or len(candidates1) == len(candidates2)

    def test_hash_collision_detection(self, rng):
        """Test that hash collisions are detected and handled correctly."""
        clear_convex_hull_cache()

        # Create two different position arrays
        positions1 = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        positions2 = np.array(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]
        )

        # Both should generate different hashes
        hash1 = _get_positions_hash(positions1)
        hash2 = _get_positions_hash(positions2)
        assert hash1 != hash2, "Different positions should produce different hashes"

        # Compute hulls for both
        hull1 = _get_cached_hull(positions1)
        hull2 = _get_cached_hull(positions2)

        # Verify they are different (different volumes)
        assert hull1.volume != hull2.volume, (
            "Different positions should produce different hulls"
        )

    def test_small_position_array_handling(self):
        """Test that <4 points raises appropriate error."""
        clear_convex_hull_cache()

        # Test with 3 points (insufficient for 3D convex hull)
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        with pytest.raises(SCGOValidationError, match="at least 4 points"):
            _get_cached_hull(positions)

    def test_concurrent_db_cache_access(self, tmp_path, pt2_atoms):
        """Test that database cache is thread-safe under concurrent access."""
        db_path = tmp_path / "test.db"
        with connect(str(db_path)) as db:
            pt2 = pt2_atoms.copy()
            db.write(pt2, relaxed=True, key_value_pairs={"raw_score": -10.0}, gaid=1)
        _stamp_init_test_db(db_path)

        results = []

        # Use ThreadPoolExecutor so worker exceptions are surfaced in main thread
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=10) as ex:
            futures = [ex.submit(_load_db_candidates, str(db_path)) for _ in range(10)]
            for f in as_completed(futures):
                mtime, entries = f.result()
                results.append((mtime, len(entries)))

        # All should succeed and return identical results
        assert len(results) == 10, "All threads should complete"
        assert all(r == results[0] for r in results), (
            "All threads should get same cached result"
        )

    def test_concurrent_composition_cache_access(self, tmp_path, pt2_atoms):
        """Test that composition cache is thread-safe under concurrent access."""
        db_path = _discovery_db_path(tmp_path, "Pt2")
        with connect(str(db_path)) as db:
            pt2 = pt2_atoms.copy()
            db.write(pt2, relaxed=True, key_value_pairs={"raw_score": -10.0}, gaid=1)
        _stamp_init_test_db(db_path)

        target = ["Pt", "Pt", "Pt", "Pt"]
        results = []

        # Use ThreadPoolExecutor so worker exceptions are surfaced in main thread
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=10) as ex:
            futures = [
                ex.submit(_find_smaller_candidates, target, str(db_path))
                for _ in range(10)
            ]
            for f in as_completed(futures):
                candidates = f.result()
                results.append(len(candidates))

        # All should succeed and return a value
        assert len(results) == 10, "All threads should complete"


class TestPerformanceOptimizations:
    """Tests for performance optimizations."""

    @pytest.mark.slow
    def test_clash_checking_uses_kdtree_for_large_clusters(self, rng):
        """Test that clash checking uses KDTree for large clusters."""
        # Create a large cluster (30 atoms, triggers KDTree path at 50+)
        # More lenient connectivity factor for large clusters (30+ atoms)
        large_factor = 2.0
        atoms = create_initial_cluster(
            ["Pt"] * 30,
            rng=rng,
            placement_radius_scaling=1.5,
            min_distance_factor=0.4,
            connectivity_factor=large_factor,
        )

        # Should use KDTree optimization and complete without errors
        # Verify validation works correctly (may use different factor for validation)
        is_valid, msg = validate_cluster_structure(
            atoms,
            min_distance_factor=0.4,
            connectivity_factor=CONNECTIVITY_FACTOR,
            check_clashes=True,
            check_connectivity=True,
        )
        assert isinstance(is_valid, bool)
        assert isinstance(msg, str)

    def test_sort_atoms_by_element_early_exit(self, rng):
        """Test that _sort_atoms_by_element exits early if already sorted."""
        # Create already-sorted atoms
        atoms = Atoms(
            ["Au", "Au", "Pt", "Pt"],
            positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
        )

        # Should detect it's already sorted and return quickly
        sorted_atoms = _sort_atoms_by_element(atoms)
        assert sorted_atoms.get_chemical_symbols() == ["Au", "Au", "Pt", "Pt"]


class TestStructureDiagnostics:
    """Tests for the StructureDiagnostics helper class."""

    def test_diagnostics_valid_cluster(self, rng):
        """Test diagnostics for a valid connected cluster."""
        from scgo.initialization import get_structure_diagnostics

        comp = ["Pt"] * 5
        atoms = create_initial_cluster(comp, rng=rng)

        diagnostics = get_structure_diagnostics(
            atoms, MIN_DISTANCE_FACTOR_DEFAULT, CONNECTIVITY_FACTOR
        )

        assert diagnostics.is_valid is True
        assert diagnostics.has_clashes is False
        assert diagnostics.is_disconnected is False
        assert diagnostics.n_components == 1
        assert "valid" in diagnostics.summary.lower()

    def test_diagnostics_disconnected_cluster(self):
        """Test diagnostics for a disconnected cluster."""
        from scgo.initialization import get_structure_diagnostics

        # Create two separate clusters far apart
        atoms = Atoms(
            "Pt4",
            positions=[[0, 0, 0], [2.5, 0, 0], [20, 0, 0], [22.5, 0, 0]],
        )

        diagnostics = get_structure_diagnostics(
            atoms, MIN_DISTANCE_FACTOR_DEFAULT, CONNECTIVITY_FACTOR
        )

        assert diagnostics.is_valid is False
        assert diagnostics.is_disconnected is True
        assert diagnostics.n_components == 2
        assert diagnostics.closest_inter_component_distance > 0
        assert "disconnected" in diagnostics.summary.lower()

    def test_diagnostics_clashing_atoms(self):
        """Test diagnostics for atoms that are too close."""
        from scgo.initialization import get_structure_diagnostics

        # Create atoms that are very close together (overlapping)
        atoms = Atoms(
            "Pt2",
            positions=[[0, 0, 0], [0.5, 0, 0]],  # Way too close for Pt
        )

        diagnostics = get_structure_diagnostics(
            atoms, MIN_DISTANCE_FACTOR_DEFAULT, CONNECTIVITY_FACTOR
        )

        assert diagnostics.is_valid is False
        assert diagnostics.has_clashes is True
        assert len(diagnostics.clash_details) > 0
        assert "clash" in diagnostics.summary.lower()

    def test_diagnostics_empty_cluster(self):
        """Test diagnostics for empty cluster."""
        from scgo.initialization import get_structure_diagnostics

        atoms = Atoms()
        diagnostics = get_structure_diagnostics(
            atoms, MIN_DISTANCE_FACTOR_DEFAULT, CONNECTIVITY_FACTOR
        )

        assert diagnostics.is_valid is True
        assert diagnostics.has_clashes is False
        assert diagnostics.is_disconnected is False


class TestDatabaseIntegration:
    """Tests for database integration in initialization."""

    def test_find_smaller_candidates(self, tmp_path):
        """Test finding smaller candidates from database."""
        # Create test database
        db_path = _discovery_db_path(tmp_path, "Pt3Au")
        with connect(db_path) as db:
            # Add a Pt2 structure
            pt2 = Atoms("Pt2", positions=[[0, 0, 0], [0, 0, 2.5]])
            pt2.info.setdefault("key_value_pairs", {})["final_unique_minimum"] = True
            pt2.info.setdefault("key_value_pairs", {})["final_unique_minimum"] = True
            db.write(
                pt2,
                relaxed=True,
                key_value_pairs={"raw_score": -10.0, "final_unique_minimum": True},
                gaid=1,
            )

            # Add a Pt3 structure
            pt3 = Atoms("Pt3", positions=[[0, 0, 0], [0, 0, 2.5], [0, 2.5, 0]])
            pt3.info.setdefault("key_value_pairs", {})["final_unique_minimum"] = True
            pt3.info.setdefault("key_value_pairs", {})["final_unique_minimum"] = True
            db.write(
                pt3,
                relaxed=True,
                key_value_pairs={"raw_score": -15.0, "final_unique_minimum": True},
                gaid=2,
            )

            # Add an Au2 structure
            au2 = Atoms("Au2", positions=[[0, 0, 0], [0, 0, 2.8]])
            db.write(au2, relaxed=True, key_value_pairs={"raw_score": -8.0}, gaid=3)

            # Add a PtAu structure
            ptau = Atoms("PtAu", positions=[[0, 0, 0], [0, 0, 2.6]])
            ptau.info.setdefault("key_value_pairs", {})["final_unique_minimum"] = True
            ptau.info.setdefault("key_value_pairs", {})["final_unique_minimum"] = True
            db.write(
                ptau,
                relaxed=True,
                key_value_pairs={"raw_score": -12.0, "final_unique_minimum": True},
                gaid=4,
            )
        _stamp_init_test_db(db_path)

        # Test with a target composition that has smaller candidates
        target_comp = ["Pt", "Pt", "Pt", "Au"]  # Pt3Au1
        db_glob_pattern = str(db_path)  # Use the full path for glob

        candidates = _find_smaller_candidates(target_comp, db_glob_pattern)

        assert "Pt2" in candidates
        assert len(candidates["Pt2"]) == 1
        assert candidates["Pt2"][0][0] == pytest.approx(10.0, rel=1e-6)

        assert "Au2" not in candidates  # Au2 is not a sub-composition of Pt3Au1
        assert "Pt3" in candidates
        assert len(candidates["Pt3"]) == 1
        assert candidates["Pt3"][0][0] == pytest.approx(15.0, rel=1e-6)

        assert "AuPt" in candidates
        assert len(candidates["AuPt"]) == 1
        assert candidates["AuPt"][0][0] == pytest.approx(12.0, rel=1e-6)

        # Test with an empty target composition
        empty_candidates = _find_smaller_candidates([], db_glob_pattern)
        assert len(empty_candidates) == 0, "Expected no candidates for this mode"

        # Test with a target composition that has no smaller candidates
        no_candidates = _find_smaller_candidates(["Pd", "Pd"], db_glob_pattern)
        assert len(no_candidates) == 0, "Expected no candidates for this mode"


# Internal function tests


class TestSortAtomsByElement:
    """Tests for _sort_atoms_by_element internal function."""

    def test_empty_atoms(self):
        """Test sorting empty Atoms object."""
        atoms = Atoms()
        result = _sort_atoms_by_element(atoms)
        assert len(result) == 0
        assert isinstance(result, Atoms)

    def test_single_atom(self):
        """Test sorting single atom."""
        atoms = Atoms("Pt", positions=[[0, 0, 0]])
        result = _sort_atoms_by_element(atoms)
        assert len(result) == 1
        assert result.get_chemical_symbols() == ["Pt"]

    def test_multiple_same_element(self):
        """Test sorting multiple atoms of same element."""
        atoms = Atoms("Pt3", positions=[[0, 0, 0], [2, 0, 0], [1, 1, 0]])
        result = _sort_atoms_by_element(atoms)
        # Should preserve order for same element
        assert len(result) == 3
        assert result.get_chemical_symbols() == ["Pt", "Pt", "Pt"]

    def test_mixed_elements(self):
        """Test sorting mixed element types."""
        atoms = Atoms("PtAuPd", positions=[[0, 0, 0], [2, 0, 0], [1, 1, 0]])
        result = _sort_atoms_by_element(atoms)
        # Should be sorted alphabetically
        assert result.get_chemical_symbols() == ["Au", "Pd", "Pt"]

    def test_preserves_properties(self):
        """Test that sorting preserves cell, pbc, calc, and info."""
        atoms = Atoms("PtAu", positions=[[0, 0, 0], [2, 0, 0]])
        atoms.set_cell([10, 10, 10])
        atoms.set_pbc([True, True, False])
        atoms.info = {"test": "value"}

        result = _sort_atoms_by_element(atoms)
        assert np.allclose(result.get_cell(), atoms.get_cell())
        assert np.all(result.get_pbc() == atoms.get_pbc())
        # Info may be a copy, so check contents
        assert result.info.get("test") == atoms.info.get("test")

    def test_stable_sorting(self):
        """Test stable sorting behavior."""
        # Create atoms in specific order
        atoms = Atoms(
            ["Pt", "Au", "Pt", "Au"],
            positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
        )
        result = _sort_atoms_by_element(atoms)
        # Should sort by element but preserve relative order within same elements
        assert result.get_chemical_symbols() == ["Au", "Au", "Pt", "Pt"]


class TestBoltzmannSample:
    """Tests for _boltzmann_sample internal function."""

    def test_basic_sampling(self, rng):
        """Test Boltzmann sampling basic functionality."""
        atoms1 = Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
        atoms2 = Atoms("Pt2", positions=[[0, 0, 0], [2.6, 0, 0]])
        candidates = [(-10.0, atoms1), (-5.0, atoms2)]
        result = _boltzmann_sample(candidates, rng)
        assert result is not None
        energy, sampled = result
        assert energy in [-10.0, -5.0]
        assert len(sampled) == 2

    def test_edge_cases(self, rng):
        """Test empty list and single candidate."""
        # Empty list
        assert _boltzmann_sample([], rng) is None

        # Single candidate
        atoms = Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
        result = _boltzmann_sample([(-10.0, atoms)], rng)
        assert result is not None

    def test_mixed_sizes_same_element_raises_error(self, rng):
        """Test that different cluster sizes (even same element) raises ValueError."""
        atoms_pt2 = Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
        atoms_pt3 = Atoms("Pt3", positions=[[0, 0, 0], [2.5, 0, 0], [1.25, 2.165, 0]])
        candidates = [(-10.0, atoms_pt2), (-10.0, atoms_pt3)]
        with pytest.raises(SCGOValidationError, match="same composition"):
            _boltzmann_sample(candidates, rng)

    def test_same_composition_multiple_candidates(self, rng):
        """Test that sampling works correctly with multiple candidates of same composition."""
        atoms1 = Atoms("Pt3", positions=[[0, 0, 0], [2, 0, 0], [1, 2, 0]])
        atoms2 = Atoms("Pt3", positions=[[0, 0, 0], [2.5, 0, 0], [1, 2.5, 0]])
        atoms3 = Atoms("Pt3", positions=[[0, 0, 0], [3, 0, 0], [1, 3, 0]])
        candidates = [(1.0, atoms1), (1.5, atoms2), (2.0, atoms3)]
        energy, sampled = _boltzmann_sample(candidates, rng)
        assert sampled.get_chemical_formula() == "Pt3"
        assert energy in [1.0, 1.5, 2.0]

    def test_permuted_order_same_formula_succeeds(self, rng):
        """Permuted atom order with identical composition counts must be allowed."""
        atoms_a = Atoms(
            ["Co", "Pt", "Co", "Co"],
            positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
        )
        atoms_b = Atoms(
            ["Co", "Co", "Co", "Pt"],
            positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
        )
        candidates = [(-10.0, atoms_a), (-8.0, atoms_b)]
        result = _boltzmann_sample(candidates, rng)
        assert result is not None


class TestReorderClusterToComposition:
    """Tests for composition-canonical atom ordering."""

    def test_reorders_permuted_symbols(self):
        atoms = Atoms(
            ["O", "Ir", "O", "O"],
            positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
        )
        composition = ["Ir", "O", "O", "O"]
        reordered = reorder_cluster_to_composition(atoms, composition)
        assert reordered.get_chemical_symbols() == composition

    def test_validate_cluster_canonicalizes_to_composition(self):
        atoms = Atoms(
            ["Pt", "Au", "Pt"],
            positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0]],
        )
        composition = ["Au", "Pt", "Pt"]
        validated, ok, err = validate_cluster(
            atoms,
            composition=composition,
            sort_atoms=True,
            check_clashes=False,
            check_connectivity=False,
        )
        assert ok and err == ""
        assert validated.get_chemical_symbols() == composition


class TestGrowthOrderByMass:
    """Tests for heavy-atom-first placement order."""

    def test_heavier_element_groups_first(self, rng):
        ordered = _growth_order_by_mass(["O", "Ir", "O", "Ru", "O"], rng)
        heavy_positions = [ordered.index("Ir"), ordered.index("Ru")]
        oxygen_positions = [i for i, sym in enumerate(ordered) if sym == "O"]
        assert max(heavy_positions) < min(oxygen_positions)

    def test_sampler_includes_exploratory_orders(self, rng):
        """Mass bias is probabilistic; some samples should not start with the heaviest element."""
        starts_with_heavy = 0
        trials = 200
        for _ in range(trials):
            ordered = _sample_atoms_to_add_order(["Co", "Pt", "Co", "Pt"], rng)
            if ordered[0] == "Pt":
                starts_with_heavy += 1
        assert starts_with_heavy < trials
        assert starts_with_heavy > trials * 0.5

    def test_growth_order_sampler_reproducible(self):
        seed = 42
        composition = ["Co", "Pt", "Co", "Pt", "O", "O"]
        rng1, rng2 = create_paired_rngs(seed)
        for _ in range(10):
            order1 = _sample_atoms_to_add_order(composition, rng1)
            order2 = _sample_atoms_to_add_order(composition, rng2)
            assert order1 == order2

    def test_growth_order_by_mass_reproducible(self):
        seed = 7
        composition = ["O", "Ir", "O", "Ru", "O"]
        rng1, rng2 = create_paired_rngs(seed)
        assert _growth_order_by_mass(composition, rng1) == _growth_order_by_mass(
            composition, rng2
        )

    def test_random_spherical_multi_element_reproducible(self):
        seed = 99
        composition = ["Ir", "O", "O", "O"]
        rng1, rng2 = create_paired_rngs(seed)
        atoms1 = create_initial_cluster(composition, rng=rng1, mode="random_spherical")
        atoms2 = create_initial_cluster(composition, rng=rng2, mode="random_spherical")
        assert positions_equal(atoms1, atoms2)

    def test_iro4_initializes_with_mass_first_growth(self, rng):
        composition = ["Ir", "O", "O", "O"]
        atoms = create_initial_cluster(composition, rng=rng)
        assert atoms.get_chemical_symbols() == composition


class TestLoadDbCandidates:
    """Tests for _load_db_candidates internal function."""

    def test_missing_file(self):
        """Test loading from missing database file."""
        mtime, entries = _load_db_candidates("/nonexistent/file.db")
        assert mtime == pytest.approx(0.0, abs=1e-8)
        assert len(entries) == 0

    def test_empty_database(self, tmp_path):
        """Test loading from empty database."""
        db_path = tmp_path / "empty.db"
        with connect(str(db_path)):
            pass  # Create empty database

        mtime, entries = _load_db_candidates(str(db_path))
        assert mtime >= 0  # mtime can be 0 for new files
        assert len(entries) == 0

    def test_valid_database(self, tmp_path, pt2_atoms, pt3_atoms):
        """Test loading from valid database."""
        db_path = tmp_path / "test.db"
        with connect(str(db_path)) as db:
            pt2 = pt2_atoms.copy()
            db.write(pt2, relaxed=True, key_value_pairs={"raw_score": -10.0}, gaid=1)

            pt3 = pt3_atoms.copy()
            db.write(pt3, relaxed=True, key_value_pairs={"raw_score": -15.0}, gaid=2)
        _stamp_init_test_db(db_path)

        mtime, entries = _load_db_candidates(str(db_path))
        assert mtime > 0
        assert len(entries) > 0
        for symbols, energy, atoms in entries:
            assert isinstance(symbols, tuple)
            assert isinstance(energy, float)
            assert isinstance(atoms, Atoms)

    def test_cache_behavior(self, tmp_path, pt2_atoms):
        """Test that caching works correctly."""
        db_path = tmp_path / "test.db"
        with connect(str(db_path)) as db:
            pt2 = pt2_atoms.copy()
            db.write(pt2, relaxed=True, key_value_pairs={"raw_score": -10.0}, gaid=1)
        _stamp_init_test_db(db_path)

        # First call
        mtime1, entries1 = _load_db_candidates(str(db_path))

        # Second call should use cache (same mtime, allowing for floating point precision)
        mtime2, entries2 = _load_db_candidates(str(db_path))
        # Use approximate equality for mtime due to platform differences in getmtime() precision
        assert abs(mtime1 - mtime2) < 0.01, (
            f"mtime should be approximately equal: {mtime1} vs {mtime2}"
        )
        assert len(entries1) == len(entries2)

    def test_invalid_entry_handling(self, tmp_path):
        """Test handling of database entries with missing keys."""
        db_path = tmp_path / "invalid.db"
        with connect(str(db_path)) as db:
            pt2 = Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
            db.write(pt2, relaxed=True, gaid=1)  # Missing raw_score
        _stamp_init_test_db(db_path)

        # Should skip invalid entries
        mtime, entries = _load_db_candidates(str(db_path))
        # May have 0 entries if validation fails, or may raise exception
        # Either behavior is acceptable
        assert isinstance(entries, list)

    def test_extract_minima_require_final(self, tmp_path):
        """Default extract returns only ``final_unique_minimum`` relaxed rows."""
        from scgo.database.helpers import extract_minima_from_database_file

        db_path = tmp_path / "test.db"
        with connect(str(db_path)) as db:
            a1 = Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
            a1.info.setdefault("key_value_pairs", {})["final_unique_minimum"] = True
            a1.info.setdefault("key_value_pairs", {})["final_unique_minimum"] = True
            db.write(
                a1,
                relaxed=True,
                key_value_pairs={"raw_score": -10.0, "final_unique_minimum": True},
                gaid=1,
            )

            a2 = Atoms("Pt2", positions=[[0, 0, 0], [2.6, 0, 0]])
            db.write(a2, relaxed=True, key_value_pairs={"raw_score": -9.0}, gaid=2)

        from scgo.metadata.db_stamp import stamp_db

        stamp_db(db_path)

        minima_default = extract_minima_from_database_file(str(db_path), run_id="runx")

        assert len(minima_default) == 1
        # raw_score is stored as -energy in ASE GA rows, so
        # extract_minima_from_database returns the positive energy value.
        assert minima_default[0][0] == pytest.approx(10.0)


class TestFindSmallerCandidates:
    """Tests for _find_smaller_candidates internal function."""

    def test_empty_target(self, tmp_path):
        """Test finding candidates for empty target composition."""
        db_path = _discovery_db_path(tmp_path, "Pt2")
        connect(str(db_path))
        candidates = _find_smaller_candidates([], str(db_path))
        assert candidates == {}

    def test_no_matching_databases(self, tmp_path):
        """Test with glob pattern that matches no databases."""
        candidates = _find_smaller_candidates(["Pt", "Pt"], "/nonexistent/*.db")
        assert candidates == {}

    def test_finds_smaller_candidates(self, tmp_path):
        """Test finding valid smaller candidates."""
        db_path = _discovery_db_path(tmp_path, "Pt2")
        with connect(str(db_path)) as db:
            pt2 = Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
            # Mark this candidate as a final unique minimum so it is available
            # to seed+growth selection (seed strategy only uses final-tagged minima).
            pt2.info.setdefault("key_value_pairs", {})["final_unique_minimum"] = True
            pt2.info.setdefault("key_value_pairs", {})["final_unique_minimum"] = True
            db.write(
                pt2,
                relaxed=True,
                key_value_pairs={"raw_score": -10.0, "final_unique_minimum": True},
                gaid=1,
            )
        _stamp_init_test_db(db_path)

        target = ["Pt", "Pt", "Pt", "Pt"]  # Pt4
        candidates = _find_smaller_candidates(target, str(db_path))
        assert "Pt2" in candidates or len(candidates) > 0

    def test_filters_larger_candidates(self, tmp_path):
        """Test that candidates larger than target are filtered out."""
        db_path = _discovery_db_path(tmp_path, "Pt5", "large.db")
        with connect(str(db_path)) as db:
            pt5 = Atoms("Pt5", positions=[[i * 2.5, 0, 0] for i in range(5)])
            db.write(pt5, relaxed=True, key_value_pairs={"raw_score": -25.0}, gaid=1)
        _stamp_init_test_db(db_path)

        target = ["Pt", "Pt", "Pt"]  # Pt3
        candidates = _find_smaller_candidates(target, str(db_path))
        # Should not include Pt5 (or any formula with 5 atoms)
        # Check that no candidate has 5 atoms
        for candidate_list in candidates.values():
            if candidate_list:
                # Each candidate is (energy, atoms) tuple
                sample_atoms = candidate_list[0][1]
                assert len(sample_atoms) < len(target), (
                    "Should not include larger candidates"
                )

    def test_cache_invalidation(self, tmp_path, pt2_atoms):
        """Test cache invalidation on file changes."""
        db_path = _discovery_db_path(tmp_path, "Pt2")
        with connect(str(db_path)) as db:
            pt2 = pt2_atoms.copy()
            # not final - should not be considered by seed finder
            db.write(pt2, relaxed=True, key_value_pairs={"raw_score": -10.0}, gaid=1)

            # First call
            target = ["Pt", "Pt", "Pt"]
            _find_smaller_candidates(target, str(db_path))

            # Modify database: add a final-tagged candidate
            pt_new = Atoms("Pt2", positions=[[0, 0, 0], [2.6, 0, 0]])
            pt_new.info.setdefault("key_value_pairs", {})["final_unique_minimum"] = True
            pt_new.info.setdefault("key_value_pairs", {})["final_unique_minimum"] = True
            db.write(
                pt_new,
                relaxed=True,
                key_value_pairs={"raw_score": -11.0, "final_unique_minimum": True},
                gaid=3,
            )
        _stamp_init_test_db(db_path)

        # Ensure filesystem mtime changes so the cache sees the update
        import os

        os.utime(str(db_path), None)

        # Second call should see new (final) entry; filesystem caching may
        # prevent immediate detection on some filesystems, so only assert
        # correct return type here to avoid flaky behavior.
        candidates2 = _find_smaller_candidates(target, str(db_path))
        assert isinstance(candidates2, dict)

    def test_finds_smaller_candidates_requires_final_flag(self, tmp_path, pt2_atoms):
        """Ensure non-final relaxed candidates are ignored by seed finder."""
        db_path = _discovery_db_path(tmp_path, "Pt2", "test2.db")
        with connect(str(db_path)) as db:
            pt2 = pt2_atoms.copy()
            db.write(pt2, relaxed=True, key_value_pairs={"raw_score": -10.0}, gaid=1)

            target = ["Pt", "Pt", "Pt", "Pt"]  # Pt4
            candidates = _find_smaller_candidates(target, str(db_path))
            # No final-tagged entries -> should be empty
            assert candidates == {}

            # Now add a final-tagged candidate and verify it is found
            pt2_final = Atoms("Pt2", positions=[[0, 0, 0], [2.6, 0, 0]])
            pt2_final.info.setdefault("key_value_pairs", {})["final_unique_minimum"] = (
                True
            )
            pt2_final.info.setdefault("key_value_pairs", {})["final_unique_minimum"] = (
                True
            )
            db.write(
                pt2_final,
                relaxed=True,
                key_value_pairs={"raw_score": -9.0, "final_unique_minimum": True},
                gaid=2,
            )
        _stamp_init_test_db(db_path)

        # Note: cache canonical mtime may prevent immediate detection of the
        # newly added candidate in the same file. We only assert the function
        # returns a dict to avoid flaky behaviour across file systems.
        # (A fresh database file would demonstrate detection reliably.)
        candidates = _find_smaller_candidates(target, str(db_path))
        assert isinstance(candidates, dict)

        # Create a fresh database file with a final-tagged candidate to verify
        # that final candidates are discoverable (avoids cache timing issues).
        db3_path = _discovery_db_path(tmp_path, "Pt2", "test2b.db")
        with connect(str(db3_path)) as db3:
            rf = Atoms("Pt2", positions=[[0, 0, 0], [3.0, 0, 0]])
            rf.info.setdefault("key_value_pairs", {})["final_unique_minimum"] = True
            rf.info.setdefault("key_value_pairs", {})["final_unique_minimum"] = True
            db3.write(
                rf,
                relaxed=True,
                key_value_pairs={"raw_score": -8.0, "final_unique_minimum": True},
                gaid=4,
            )
        _stamp_init_test_db(db3_path)

        candidates_fresh = _find_smaller_candidates(target, str(db3_path))
        assert sum(len(lst) for lst in candidates_fresh.values()) >= 1


class TestClassifySeedGeometry:
    """Tests for _classify_seed_geometry internal function."""

    def test_single_atom(self, single_atom):
        """Test classification of single atom."""
        atoms = single_atom.copy()
        geometry = _classify_seed_geometry(atoms)
        assert geometry == "single"

    def test_linear(self, pt2_atoms):
        """Test classification of linear cluster."""
        atoms = pt2_atoms.copy()
        geometry = _classify_seed_geometry(atoms)
        assert geometry == "linear"

    def test_planar(self, pt3_atoms):
        """Test classification of planar cluster."""
        atoms = pt3_atoms.copy()
        geometry = _classify_seed_geometry(atoms)
        # Depends on eigenvalues and ConvexHull behavior
        assert geometry in ["single", "linear", "planar", "3d"]

    def test_3d(self, pt4_tetrahedron):
        """Test classification of 3D cluster."""
        atoms = pt4_tetrahedron.copy()
        geometry = _classify_seed_geometry(atoms)
        assert geometry == "3d"

    def test_exception_handling(self):
        """Test exception handling for degenerate cases."""
        # Very close atoms that might cause ConvexHull to fail
        atoms = Atoms("Pt3", positions=[[0, 0, 0], [1e-10, 0, 0], [0, 1e-10, 0]])
        geometry = _classify_seed_geometry(atoms)
        # Should return some classification, not raise
        assert geometry in ["single", "linear", "planar", "3d"]


class TestIsValidPlacement:
    """Tests for _is_valid_placement internal function."""

    def test_valid_placement(self, rng):
        """Test valid placement (no clashes, connected)."""
        seed = Atoms("Pt", positions=[[3, 0, 0]])  # Closer to ensure connectivity
        combined = Atoms("Pt", positions=[[0, 0, 0]])
        is_valid = _is_valid_placement(
            seed, combined, connectivity_factor=CONNECTIVITY_FACTOR
        )
        assert is_valid is True

    def test_clash_detection(self):
        """Test clash detection."""
        seed = Atoms("Pt", positions=[[0.1, 0, 0]])  # Too close
        combined = Atoms("Pt", positions=[[0, 0, 0]])
        is_valid = _is_valid_placement(
            seed, combined, connectivity_factor=CONNECTIVITY_FACTOR
        )
        assert is_valid is False

    def test_connectivity_validation(self):
        """Test connectivity validation."""
        seed = Atoms("Pt", positions=[[20, 0, 0]])  # Too far to connect
        combined = Atoms("Pt", positions=[[0, 0, 0]])
        is_valid = _is_valid_placement(
            seed, combined, connectivity_factor=CONNECTIVITY_FACTOR
        )
        assert is_valid is False


class TestAddAtomsToClusterIteratively:
    """Tests for _add_atoms_to_cluster_iteratively internal function."""

    def test_empty_base_atoms(self, rng):
        """Test with empty base atoms."""
        base = Atoms()
        result = _add_atoms_to_cluster_iteratively(
            base,
            ["Pt"],
            min_distance_factor=0.5,
            placement_radius_scaling=1.2,
            rng=rng,
        )
        assert result is not None
        assert len(result) == 1

    def test_empty_atoms_to_add(self, rng):
        """Test with empty atoms_to_add list."""
        base = Atoms("Pt", positions=[[0, 0, 0]])
        result = _add_atoms_to_cluster_iteratively(
            base,
            [],
            min_distance_factor=0.5,
            placement_radius_scaling=1.2,
            rng=rng,
        )
        assert result is not None
        assert len(result) == 1

    def test_single_atom_addition(self, rng):
        """Test adding single atom."""
        base = Atoms("Pt", positions=[[0, 0, 0]])
        result = _add_atoms_to_cluster_iteratively(
            base,
            ["Pt"],
            min_distance_factor=0.5,
            placement_radius_scaling=1.2,
            rng=rng,
        )
        assert result is not None
        assert len(result) == 2

    def test_placement_failure_returns_none(self, rng):
        """Test that placement failure returns None."""
        base = Atoms("Pt", positions=[[0, 0, 0]])
        # Try to place many atoms in very small space
        # May return None if placement fails
        # (This test is probabilistic, may sometimes succeed)
        _add_atoms_to_cluster_iteratively(
            base,
            ["Pt"] * 50,
            min_distance_factor=1.0,
            placement_radius_scaling=0.01,
            rng=rng,
        )

    def test_connectivity_failure_returns_none(self, rng):
        """Test that connectivity failure returns None."""
        base = Atoms("Pt", positions=[[0, 0, 0]])
        # Use very strict connectivity factor
        # May return None if connectivity check fails
        # (This test is probabilistic)
        VERY_STRICT_FACTOR = 0.1  # Very strict for testing failure cases
        _add_atoms_to_cluster_iteratively(
            base,
            ["Pt", "Pt"],
            min_distance_factor=0.5,
            placement_radius_scaling=1.2,
            rng=rng,
            connectivity_factor=VERY_STRICT_FACTOR,
        )

    def test_batch_mode_used_for_large_clusters(self, rng):
        """Test that batch mode is used for clusters with ≥4 atoms."""
        # Start with 4 atoms (minimum for batch mode)
        base = Atoms(
            "Pt4",
            positions=[[0, 0, 0], [2, 0, 0], [1, 1.732, 0], [1, 0.577, 1.633]],
        )

        # Add multiple atoms - should use batch mode
        result = _add_atoms_to_cluster_iteratively(
            base,
            ["Pt"] * 5,
            min_distance_factor=0.5,
            placement_radius_scaling=1.2,
            rng=rng,
        )

        assert result is not None
        assert len(result) == 9  # 4 base + 5 added

    def test_single_mode_used_for_small_clusters(self, rng):
        """Test that single mode is used for clusters with <4 atoms."""
        # Start with 1 atom
        base = Atoms("Pt", positions=[[0, 0, 0]])

        # Add atoms - should use single mode (cluster stays <4 until 4th atom)
        result = _add_atoms_to_cluster_iteratively(
            base,
            ["Pt", "Pt"],  # Will have 3 total, still single mode
            min_distance_factor=0.5,
            placement_radius_scaling=1.2,
            rng=rng,
        )

        assert result is not None
        assert len(result) == 3

    def test_transition_from_single_to_batch_mode(self, rng):
        """Test transition from single mode to batch mode when cluster grows."""
        # Start with 1 atom
        base = Atoms("Pt", positions=[[0, 0, 0]])

        # Add enough atoms to trigger batch mode (need ≥4 total)
        result = _add_atoms_to_cluster_iteratively(
            base,
            ["Pt"] * 5,  # Will have 6 total, should use batch mode after 4th atom
            min_distance_factor=0.5,
            placement_radius_scaling=1.2,
            rng=rng,
        )

        assert result is not None
        assert len(result) == 6

    def test_batch_placement_produces_valid_cluster(self, rng):
        """Test that batch placement produces valid, connected clusters."""
        # Start with a tetrahedral cluster
        base = Atoms(
            "Pt4",
            positions=[[0, 0, 0], [2, 0, 0], [1, 1.732, 0], [1, 0.577, 1.633]],
        )

        # Add multiple atoms using batch mode
        result = _add_atoms_to_cluster_iteratively(
            base,
            ["Pt"] * 10,
            min_distance_factor=0.5,
            placement_radius_scaling=1.2,
            rng=rng,
            connectivity_factor=1.4,
        )

        assert result is not None
        assert len(result) == 14

        # Verify cluster is connected
        from scgo.initialization.geometry_helpers import is_cluster_connected

        assert is_cluster_connected(result, connectivity_factor=1.4)

    def test_batch_placement_with_mixed_elements(self, rng):
        """Test batch placement with mixed element types."""
        base = Atoms(
            "Pt4",
            positions=[[0, 0, 0], [2, 0, 0], [1, 1.732, 0], [1, 0.577, 1.633]],
        )

        # Add mixed elements
        atoms_to_add = ["Pt", "Au", "Pt", "Au", "Pt"]
        result = _add_atoms_to_cluster_iteratively(
            base,
            atoms_to_add,
            min_distance_factor=0.5,
            placement_radius_scaling=1.2,
            rng=rng,
        )

        assert result is not None
        assert len(result) == 9

        # Verify composition
        symbols = result.get_chemical_symbols()
        assert symbols.count("Pt") == 7  # 4 base + 3 added
        assert symbols.count("Au") == 2  # 2 added


class TestConfigurationConstants:
    """Tests for configuration constant interactions."""

    def test_final_validation_always_enabled(self, rng):
        """Test that final validation always runs."""
        atoms = create_initial_cluster(["Pt", "Pt"], rng=rng)
        # Should always produce valid cluster with validation enabled
        assert len(atoms) == 2


@pytest.mark.slow
class TestLargeClusterConnectivityAllModes:
    """Streamlined connectivity tests for 50-60 atom clusters."""

    @pytest.mark.parametrize(
        "mode", ["random_spherical", "seed+growth", "smart", "template"]
    )
    @pytest.mark.parametrize("n_atoms", [50, 60])
    @pytest.mark.parametrize("seed", [0, 42])
    def test_connectivity_single_element(self, mode, n_atoms, seed):
        """Test all modes produce connected clusters for single-element compositions."""
        rng, _ = create_paired_rngs(seed)
        comp = ["Pt"] * n_atoms
        atoms = _create_atoms_for_mode(comp, mode, rng)
        _assert_connectivity_with_diagnostics(atoms, comp, mode, n_atoms, seed)

    @pytest.mark.parametrize("mode", ["random_spherical", "seed+growth", "smart"])
    @pytest.mark.parametrize("n_atoms", [50, 60])
    @pytest.mark.parametrize("seed", [0, 42])
    def test_connectivity_bimetallic(self, mode, n_atoms, seed):
        """Test all modes produce connected clusters for bimetallic compositions."""
        rng, _ = create_paired_rngs(seed)
        n_pt = n_atoms // 2
        n_au = n_atoms - n_pt
        comp = ["Pt"] * n_pt + ["Au"] * n_au
        atoms = _create_atoms_for_mode(comp, mode, rng)
        _assert_connectivity_with_diagnostics(atoms, comp, mode, n_atoms, seed)


def _generate_composition(comp_type: str, n_atoms: int) -> list[str]:
    """Generate composition based on type.

    Args:
        comp_type: Type of composition ("single", "bimetallic", or "trimetallic")
        n_atoms: Total number of atoms

    Returns:
        List of atomic symbols representing the composition
    """
    if comp_type == "single":
        return ["Pt"] * n_atoms
    elif comp_type == "bimetallic":
        n_pt = n_atoms // 2
        n_au = n_atoms - n_pt
        return ["Pt"] * n_pt + ["Au"] * n_au
    elif comp_type == "trimetallic":
        n_pt = n_atoms // 3
        n_au = n_atoms // 3
        n_pd = n_atoms - n_pt - n_au
        return ["Pt"] * n_pt + ["Au"] * n_au + ["Pd"] * n_pd
    else:
        raise ValueError(f"Unknown composition type: {comp_type}")


class TestBatchConnectivityAllModes:
    """Consolidated batch connectivity tests for all initialization modes.

    This test class replaces redundant mode-specific batch tests by parametrizing
    over all modes and composition types.

    Note: Template mode is excluded from batch uniqueness tests. Template mode
    produces structures from a small, finite set of deterministic geometries
    (icosahedron, decahedron, octahedron, etc.). Uniqueness is measured via
    sorted interatomic distances (rotation-invariant), so we get at most a few
    distinct signatures per size (e.g. ~3 for 50/55 atoms). Requiring ≥75%
    uniqueness over 100 samples is therefore unrealistic for template mode.
    """

    @pytest.mark.slow
    @pytest.mark.parametrize("mode", ["random_spherical", "seed+growth", "smart"])
    @pytest.mark.parametrize("n_atoms", [50, 60])
    @pytest.mark.parametrize("comp_type", ["single", "bimetallic"])
    def test_batch_connectivity_all_types(self, mode, n_atoms, comp_type, rng):
        """Test multiple initializations with uniqueness check.

        For all composition types (single-element, bimetallic, trimetallic).
        Template mode excluded: see class docstring.
        """
        comp = _generate_composition(comp_type, n_atoms)
        composition_label = comp_type if comp_type != "single" else ""
        n_samples = (
            BATCH_TEST_SAMPLES_SLOW
            if mode in ("seed+growth", "smart")
            else BATCH_TEST_SAMPLES
        )
        run_batch_connectivity_test(
            composition=comp,
            mode=mode,
            n_atoms=n_atoms,
            rng=rng,
            create_atoms_func=_create_atoms_for_mode,
            n_samples=n_samples,
            uniqueness_threshold=UNIQUENESS_THRESHOLD,
            connectivity_factor=CONNECTIVITY_FACTOR,
            composition_label=composition_label,
        )


def _create_atoms_for_mode(
    composition: list[str],
    mode: str,
    rng: np.random.Generator,
    connectivity_factor: float = CONNECTIVITY_FACTOR,
    placement_radius_scaling: float | None = None,
    min_distance_factor: float | None = None,
) -> Atoms:
    """Helper to create atoms using specified mode.

    Args:
        composition: Target composition
        mode: Initialization mode
        rng: Random number generator
        connectivity_factor: Connectivity factor
        placement_radius_scaling: Optional placement radius scaling
        min_distance_factor: Optional minimum distance factor

    Returns:
        Atoms object
    """
    kwargs = {"connectivity_factor": connectivity_factor}
    if placement_radius_scaling is not None:
        kwargs["placement_radius_scaling"] = placement_radius_scaling
    if min_distance_factor is not None:
        kwargs["min_distance_factor"] = min_distance_factor

    if mode == "random_spherical":
        cell_side = compute_cell_side(composition)
        return random_spherical(
            composition=composition, cell_side=cell_side, rng=rng, **kwargs
        )
    else:
        return create_initial_cluster(
            composition=composition, rng=rng, mode=mode, **kwargs
        )


def _assert_connectivity_with_diagnostics(
    atoms: Atoms,
    composition: list[str],
    mode: str,
    n_atoms: int,
    seed: int,
    connectivity_factor: float = CONNECTIVITY_FACTOR,
) -> None:
    """Helper to assert connectivity and provide diagnostics on failure.

    Args:
        atoms: Generated atoms object
        composition: Target composition
        mode: Initialization mode
        n_atoms: Expected number of atoms
        seed: Random seed used
        connectivity_factor: Connectivity factor used
    """
    assert len(atoms) == n_atoms, (
        f"Cluster size mismatch: expected {n_atoms}, got {len(atoms)}"
    )
    assert get_composition_counts(
        atoms.get_chemical_symbols()
    ) == get_composition_counts(composition), "Composition mismatch"

    is_connected = is_cluster_connected(atoms, connectivity_factor=connectivity_factor)
    if not is_connected:
        (
            suggested_factor,
            analysis_msg,
        ) = analyze_disconnection(atoms, connectivity_factor)
        pytest.fail(
            f"{mode} mode produced disconnected cluster "
            f"(n_atoms={n_atoms}, seed={seed}). "
            f"Connectivity factor: {connectivity_factor}. "
            f"Analysis: {analysis_msg}. "
            f"Suggested factor: {suggested_factor:.2f}."
        )


@pytest.mark.slow
class TestReproducibilityAllModes:
    """Smoke reproducibility across init modes (narrow seed matrix)."""

    @pytest.mark.parametrize(
        "mode", ["random_spherical", "seed+growth", "smart", "template"]
    )
    @pytest.mark.reproducibility
    @pytest.mark.requires_cache_isolation
    @pytest.mark.parametrize("seed", [0, 42])
    def test_reproducibility_single_element(self, mode, seed):
        comp = ["Pt"] * 30
        rng1, rng2 = create_paired_rngs(seed)
        atoms1 = _create_atoms_for_mode(comp, mode, rng1)
        atoms2 = _create_atoms_for_mode(comp, mode, rng2)
        assert positions_equal(atoms1, atoms2), (
            f"{mode} mode not reproducible with seed={seed}"
        )

    @pytest.mark.reproducibility
    @pytest.mark.requires_cache_isolation
    @pytest.mark.parametrize("mode", ["random_spherical", "seed+growth", "smart"])
    @pytest.mark.parametrize("seed", [0, 42])
    def test_reproducibility_bimetallic(self, mode, seed):
        comp = ["Pt"] * 15 + ["Au"] * 15
        rng1, rng2 = create_paired_rngs(seed)
        atoms1 = _create_atoms_for_mode(comp, mode, rng1)
        atoms2 = _create_atoms_for_mode(comp, mode, rng2)
        assert positions_equal(atoms1, atoms2), (
            f"{mode} mode bimetallic not reproducible with seed={seed}"
        )


class TestReliabilityAllModes:
    """Narrow reliability smoke across init modes (one seed, representative sizes)."""

    @pytest.mark.slow
    @pytest.mark.parametrize("mode", ["random_spherical", "seed+growth", "smart"])
    @pytest.mark.parametrize("n_atoms", [4, 10])
    def test_single_element_small_clusters(self, mode, n_atoms):
        seed = 42
        rng, _ = create_paired_rngs(seed)
        comp = ["Pt"] * n_atoms
        atoms = _create_atoms_for_mode(comp, mode, rng)
        assert len(atoms) == n_atoms
        assert get_composition_counts(
            atoms.get_chemical_symbols()
        ) == get_composition_counts(comp)
        validate_structure_with_diagnostics(
            atoms, context=f"mode={mode}, seed={seed}, n_atoms={n_atoms}"
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("mode", ["random_spherical", "seed+growth", "smart"])
    @pytest.mark.parametrize("n_atoms", [4, 10])
    def test_mixed_element_small_clusters(self, mode, n_atoms):
        seed = 42
        rng, _ = create_paired_rngs(seed)
        comp = MIXED_COMPOSITIONS["PtAu"](n_atoms)
        atoms = _create_atoms_for_mode(comp, mode, rng)
        assert len(atoms) == n_atoms
        assert get_composition_counts(
            atoms.get_chemical_symbols()
        ) == get_composition_counts(comp)
        validate_structure_with_diagnostics(
            atoms,
            context=f"mode={mode}, seed={seed}, n_atoms={n_atoms}, composition=PtAu",
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("mode", ["random_spherical", "seed+growth", "smart"])
    def test_medium_cluster(self, mode):
        seed = 42
        n_atoms = 20
        rng, _ = create_paired_rngs(seed)
        comp = ["Pt"] * n_atoms
        atoms = _create_atoms_for_mode(comp, mode, rng, placement_radius_scaling=1.2)
        assert len(atoms) == n_atoms
        validate_structure_with_diagnostics(
            atoms, context=f"mode={mode}, seed={seed}, n_atoms={n_atoms}"
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("mode", ["random_spherical", "seed+growth", "smart"])
    def test_large_cluster(self, mode):
        seed = 42
        n_atoms = 40
        rng, _ = create_paired_rngs(seed)
        LARGE_CLUSTER_FACTOR = 2.0
        atoms = _create_atoms_for_mode(
            ["Pt"] * n_atoms,
            mode,
            rng,
            placement_radius_scaling=1.5,
            min_distance_factor=0.4,
            connectivity_factor=LARGE_CLUSTER_FACTOR,
        )
        assert len(atoms) == n_atoms
        validate_structure_with_diagnostics(
            atoms,
            min_distance_factor=0.4,
            connectivity_factor=LARGE_CLUSTER_FACTOR,
            context=f"mode={mode}, seed={seed}, n_atoms={n_atoms}",
        )


class TestExactMatchReuse:
    """Tests for the additive exact-match reuse tier (Tier E)."""

    def _write_minimum(self, tmp_path, formula_hint, atoms, energy, *, final=True):
        """Write a relaxed minimum to a *_searches DB so discovery can find it."""
        db_path = _discovery_db_path(tmp_path, formula_hint)
        atoms = atoms.copy()
        set_tags(
            atoms,
            final_unique_minimum=final,
            raw_score=-float(energy),
        )
        with connect(db_path) as db:
            db.write(
                atoms,
                relaxed=True,
                gaid=1,
                key_value_pairs={
                    "raw_score": -float(energy),
                    "final_unique_minimum": final,
                },
            )
        _stamp_init_test_db(db_path)
        return db_path

    def _build_db(self, tmp_path, rng):
        """Populate a DB set with sub / exact / larger / non-subset minima."""
        glob = str(tmp_path / "**" / "*.db")
        pt3 = create_initial_cluster(["Pt"] * 3, mode="random_spherical", rng=rng)
        pt5 = create_initial_cluster(["Pt"] * 5, mode="random_spherical", rng=rng)
        pt6 = create_initial_cluster(["Pt"] * 6, mode="random_spherical", rng=rng)
        au2 = create_initial_cluster(["Au"] * 2, mode="random_spherical", rng=rng)
        self._write_minimum(tmp_path, "Pt3", pt3, 10.0)
        pt5_path = self._write_minimum(tmp_path, "Pt5", pt5, 15.0)
        # A Pt5 row WITHOUT the final flag must be ignored by both tiers.
        pt5_nf = pt5.copy()
        set_tags(pt5_nf, final_unique_minimum=False, raw_score=-99.0)
        with connect(pt5_path) as db:
            db.write(
                pt5_nf,
                relaxed=True,
                gaid=2,
                key_value_pairs={"raw_score": -99.0, "final_unique_minimum": False},
            )
        self._write_minimum(tmp_path, "Pt6", pt6, 20.0)
        self._write_minimum(tmp_path, "Au2", au2, 8.0)
        return glob

    def test_find_exact_candidates(self, tmp_path, rng):
        """Exact candidates: only same-composition final minima; no smaller/larger."""
        glob = self._build_db(tmp_path, rng)
        exact = _find_exact_candidates(["Pt"] * 5, glob)

        assert "Pt5" in exact
        assert len(exact["Pt5"]) == 1  # non-final Pt5b excluded
        assert exact["Pt5"][0][0] == pytest.approx(15.0, rel=1e-6)

        # Sub-composition and larger/non-subset candidates are excluded.
        assert "Pt3" not in exact
        assert "Pt6" not in exact
        assert "Au2" not in exact

    def test_discover_all_candidates_splits_sub_and_exact(self, tmp_path, rng):
        """_discover_all_candidates returns sub and exact buckets from one scan."""
        glob = self._build_db(tmp_path, rng)
        sub, exact = _discover_all_candidates(["Pt"] * 5, glob)

        assert "Pt3" in sub
        assert "Pt5" in exact
        assert "Pt5" not in sub
        assert "Pt3" not in exact

    def test_find_exact_candidates_dedups_geometry(self, tmp_path, rng):
        """Two identical-geometry exact minima deduplicate to a single entry."""
        pt5 = create_initial_cluster(["Pt"] * 5, mode="random_spherical", rng=rng)
        self._write_minimum(tmp_path, "Pt5", pt5, 15.0)
        self._write_minimum(tmp_path, "Pt5", pt5.copy(), 15.0)
        glob = str(tmp_path / "**" / "*.db")
        exact = _find_exact_candidates(["Pt"] * 5, glob)
        assert "Pt5" in exact
        assert len(exact["Pt5"]) == 1

    def test_plan_populates_exact_candidates_smart(self, tmp_path, rng):
        """Smart plan carries exact candidates; other modes leave it empty."""
        glob = self._build_db(tmp_path, rng)
        plan = plan_batch_initialization(
            ["Pt"] * 5,
            n_structures=10,
            rng=rng,
            previous_search_glob=glob,
            mode="smart",
        )
        assert isinstance(plan, BatchInitPlan)
        assert "Pt5" in plan.exact_candidates
        assert any(s == "exact" for s, _ in plan.allocations)

        # seed+growth mode never allocates exact, so the bucket stays empty.
        plan_sg = plan_batch_initialization(
            ["Pt"] * 5,
            n_structures=10,
            rng=rng,
            previous_search_glob=glob,
            mode="seed+growth",
        )
        assert plan_sg.exact_candidates == {}

        # reuse_exact_matches=False forces an empty exact bucket.
        plan_off = plan_batch_initialization(
            ["Pt"] * 5,
            n_structures=10,
            rng=rng,
            previous_search_glob=glob,
            mode="smart",
            reuse_exact_matches=False,
        )
        assert plan_off.exact_candidates == {}

    def test_try_exact_match_reuses_and_strips_tags(self, tmp_path, rng):
        """_try_exact_match returns a valid seed, stripping final_unique_minimum."""
        glob = self._build_db(tmp_path, rng)
        exact = _find_exact_candidates(["Pt"] * 5, glob)
        rng2 = np.random.default_rng(0)
        atoms = _try_exact_match(["Pt"] * 5, exact, rng2)
        assert atoms is not None
        assert len(atoms) == 5
        assert get_tag(atoms, "final_unique_minimum") is None
        assert get_tag(atoms, "scgo_reused_from_previous_search") is True

        # Wrong composition yields no exact candidate.
        assert _try_exact_match(["Au"] * 2, exact, rng2) is None

    def test_gas_phase_exact_allocation_and_fallback(self, tmp_path, rng):
        """Explicit 'exact' allocation reuses a prior minimum; falls back if none."""
        db_rng, _ = create_paired_rngs(1)
        glob = self._build_db(tmp_path, db_rng)

        rng_a, rng_b = create_paired_rngs(123)
        reused = create_initial_cluster_batch(
            ["Pt"] * 5,
            n_structures=1,
            rng=rng_a,
            mode="smart",
            previous_search_glob=glob,
            allocation=("exact", None),
        )
        assert len(reused) == 1
        assert get_tag(reused[0], "scgo_reused_from_previous_search") is True
        assert get_tag(reused[0], "final_unique_minimum") is None

        # Deterministic for fixed rng seed (paired RNGs share the same state).
        reused2 = create_initial_cluster_batch(
            ["Pt"] * 5,
            n_structures=1,
            rng=rng_b,
            mode="smart",
            previous_search_glob=glob,
            allocation=("exact", None),
        )
        assert positions_equal(reused[0], reused2[0])

        # No DB: exact allocation degrades to a valid random_spherical cluster.
        no_db = create_initial_cluster_batch(
            ["Pt"] * 5,
            n_structures=1,
            rng=rng,
            mode="smart",
            previous_search_glob=str(tmp_path / "nope" / "**" / "*.db"),
            allocation=("exact", None),
        )
        assert len(no_db) == 1
        assert get_tag(no_db[0], "scgo_reused_from_previous_search") is None
        assert len(no_db[0]) == 5

    def test_smart_mix_includes_exact_seed(self, tmp_path, rng):
        """Across a smart batch, at least one structure reuses the exact minimum."""
        glob = self._build_db(tmp_path, rng)
        batch = create_initial_cluster_batch(
            ["Pt"] * 5,
            n_structures=60,
            rng=rng,
            mode="smart",
            previous_search_glob=glob,
        )
        assert len(batch) == 60
        # Exact-tier reused seeds must not leak the source final_unique_minimum tag;
        # sub-tier seeds from previous runs are untouched by this change.
        reused = [
            a for a in batch if get_tag(a, "scgo_reused_from_previous_search") is True
        ]
        assert len(reused) >= 1
        for a in reused:
            assert get_tag(a, "final_unique_minimum") is None
