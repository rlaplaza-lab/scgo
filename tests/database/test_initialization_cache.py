"""Tests for the unified cache system.

These tests cover basic semantics, eviction, and concurrent get_or_compute
to ensure the double-check logic prevents redundant computations under contention.
"""

from __future__ import annotations

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest
from ase import Atoms

from scgo.database import (
    SCGODatabaseManager,
    close_data_connection,
    setup_database,
)
from scgo.database.cache import UnifiedCache, reset_global_cache
from scgo.exceptions import SCGOValidationError


def _final_kvp(raw_score: float) -> dict[str, float | bool]:
    """``key_value_pairs`` for relaxed rows that are canonical final minima."""
    return {"raw_score": raw_score, "final_unique_minimum": True}


def test_unified_cache_max_size_validation():
    with pytest.raises(
        SCGOValidationError, match="max_size must be a positive integer"
    ):
        UnifiedCache(max_size=0)


def test_unified_cache_eviction_and_len():
    reset_global_cache()  # Clean slate
    c = UnifiedCache(max_size=2)
    namespace = "test_ns"

    c.set(namespace, "a", 1)
    c.set(namespace, "b", 2)
    c.set(namespace, "c", 3)

    assert c.get(namespace, "a") is None  # Evicted
    assert c.get(namespace, "b") == 2
    assert c.get(namespace, "c") == 3


def test_unified_cache_get_or_compute_concurrent():
    reset_global_cache()  # Clean slate
    calls = 0
    lock = threading.Lock()
    start_event = threading.Event()

    def compute():
        nonlocal calls
        # Wait for explicit signal to simulate expensive computation (deterministic)
        start_event.wait(timeout=1)
        with lock:
            calls += 1
        return "computed_value"

    cache = UnifiedCache(max_size=10)
    namespace = "test_ns"

    def worker():
        return cache.get_or_compute(namespace, "key", compute)

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(worker) for _ in range(8)]
        # Allow compute to proceed now that workers are ready
        start_event.set()
        results = [f.result() for f in as_completed(futures)]

    assert all(r == "computed_value" for r in results)
    assert calls == 1, f"Compute function called {calls} times (expected 1)"


def test_unified_cache_namespace_isolation():
    reset_global_cache()  # Clean slate
    c = UnifiedCache(max_size=20)

    # Set values in different namespaces
    c.set("ns1", "key", "value1")
    c.set("ns2", "key", "value2")

    # Should be isolated
    assert c.get("ns1", "key") == "value1"
    assert c.get("ns2", "key") == "value2"


def test_unified_cache_concurrent_get_or_compute():
    reset_global_cache()  # Clean slate
    c = UnifiedCache(max_size=10)

    calls = 0
    lock = threading.Lock()
    start_event = threading.Event()

    def compute():
        nonlocal calls
        # Wait for an explicit signal to simulate expensive computation (deterministic)
        start_event.wait(timeout=1)
        with lock:
            calls += 1
        return "V"

    def worker():
        return c.get_or_compute("test_ns", "x", compute)

    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = [ex.submit(worker) for _ in range(6)]
        # Allow compute to proceed now that workers are ready
        start_event.set()
        results = [f.result() for f in as_completed(futures)]

    assert all(r == "V" for r in results)
    assert calls == 1


def _make_run_with_minima(base_dir: Path, n_rows: int) -> None:
    """Create ``base_dir/run_*/ga_go.db`` holding ``n_rows`` final minima."""
    run_dir = base_dir / "run_20250101_000000_000000"
    run_dir.mkdir(parents=True)
    template = Atoms("Pt2", positions=[[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
    da = setup_database(run_dir, "ga_go.db", template, initial_candidate=template)
    try:
        for i in range(n_rows):
            a = template.copy()
            a.positions[1][0] += 0.05 * i
            a.info["key_value_pairs"] = {
                "raw_score": -10.0 - i,
                "final_unique_minimum": True,
            }
            a.info["data"] = {"tag": f"row_{i}"}
            da.add_relaxed_step(a)
    finally:
        close_data_connection(da)


@pytest.mark.requires_cache_isolation
def test_database_manager_cache_is_isolated_per_base_dir(tmp_path):
    """Two managers over different base dirs must not poison each other's cache."""
    dir_a = tmp_path / "search_a"
    dir_b = tmp_path / "search_b"
    _make_run_with_minima(dir_a, 1)
    _make_run_with_minima(dir_b, 3)

    manager_a = SCGODatabaseManager(
        base_dir=dir_a, enable_caching=True, cache_ttl_seconds=600
    )
    manager_b = SCGODatabaseManager(
        base_dir=dir_b, enable_caching=True, cache_ttl_seconds=600
    )
    try:
        first_a = manager_a.load_previous_results(
            composition=["Pt", "Pt"], current_run_id="run_current"
        )
        assert len(first_a) == 1

        from_b = manager_b.load_previous_results(
            composition=["Pt", "Pt"], current_run_id="run_current"
        )
        assert len(from_b) == 3

        second_a = manager_a.load_previous_results(
            composition=["Pt", "Pt"], current_run_id="run_current"
        )
        assert len(second_a) == 1, "manager A returned manager B's cached results"
    finally:
        manager_a.close()
        manager_b.close()


@pytest.mark.requires_cache_isolation
class TestDatabaseManagerCaching:
    """Comprehensive tests for enhanced database manager features."""

    def test_caching_behavior(self, tmp_path, rng):
        """Test result caching and cache invalidation."""
        run_dir = tmp_path / "run_001"
        run_dir.mkdir(parents=True)
        atoms = Atoms("Pt3", positions=[[0, 0, 0], [2.5, 0, 0], [1.25, 2.0, 0]])
        da = setup_database(run_dir, "test.db", atoms, initial_candidate=atoms)

        for i in range(5):
            a = atoms.copy()
            a.positions += rng.random((3, 3)) * 0.1
            a.info["key_value_pairs"] = _final_kvp(-30.0 - i)
            a.info["data"] = {"tag": f"test_{i}"}
            da.add_relaxed_step(a)

        close_data_connection(da)
        del da

        manager = SCGODatabaseManager(
            base_dir=tmp_path, enable_caching=True, cache_ttl_seconds=10
        )

        result1 = manager.load_previous_results(
            composition=["Pt", "Pt", "Pt"],
            current_run_id="run_test",
        )

        result2 = manager.load_previous_results(
            composition=["Pt", "Pt", "Pt"],
            current_run_id="run_test",
        )

        assert len(result1) == len(result2)
        assert result1 is result2

        result3 = manager.load_previous_results(
            composition=["Pt", "Pt", "Pt"],
            current_run_id="run_test",
            force_reload=True,
        )

        assert len(result3) == len(result1)
        assert result3 is not result1

        manager.clear_cache()

        result4 = manager.load_previous_results(
            composition=["Pt", "Pt", "Pt"],
            current_run_id="run_test",
        )

        assert len(result4) == len(result1)
        manager.close()

    def test_cache_ttl_expiration(self, tmp_path, rng):
        """Test that cache expires after TTL."""
        run_dir = tmp_path / "run_001"
        run_dir.mkdir(parents=True)
        atoms = Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
        da = setup_database(run_dir, "test.db", atoms, initial_candidate=atoms)

        for i in range(3):
            a = atoms.copy()
            a.positions += rng.random((2, 3)) * 0.1
            a.info["key_value_pairs"] = _final_kvp(-10.0 - i)
            a.info["data"] = {"tag": f"test_{i}"}
            da.add_relaxed_step(a)

        close_data_connection(da)
        del da

        manager = SCGODatabaseManager(
            base_dir=tmp_path,
            enable_caching=True,
            cache_ttl_seconds=1,
        )

        result1 = manager.load_previous_results(
            composition=["Pt", "Pt"],
            current_run_id="run_test",
        )

        result2 = manager.load_previous_results(
            composition=["Pt", "Pt"],
            current_run_id="run_test",
        )
        assert result1 is result2

        # Simulate TTL expiry deterministically
        manager.clear_cache()

        result3 = manager.load_previous_results(
            composition=["Pt", "Pt"],
            current_run_id="run_test",
        )
        assert result1 is not result3

        manager.close()

    def test_concurrent_manager_access(self, tmp_path, rng):
        """Test thread-safe manager operations."""
        run_dir = tmp_path / "run_000"
        run_dir.mkdir(parents=True)
        atoms = Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
        da = setup_database(run_dir, "test.db", atoms, initial_candidate=atoms)

        for i in range(10):
            a = atoms.copy()
            a.positions += rng.random((2, 3)) * 0.1
            a.info["key_value_pairs"] = _final_kvp(-10.0 - i)
            a.info["data"] = {"tag": f"test_{i}"}
            da.add_relaxed_step(a)

        close_data_connection(da)
        del da

        manager = SCGODatabaseManager(base_dir=tmp_path, enable_caching=True)

        results = []
        errors = []

        def load_data(thread_id):
            try:
                data = manager.load_previous_results(
                    composition=["Pt", "Pt"],
                    current_run_id=f"run_{thread_id}",
                )
                results.append((thread_id, len(data)))
            except sqlite3.OperationalError as e:
                errors.append((thread_id, e))

        threads = []
        for i in range(5):
            t = threading.Thread(target=load_data, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 5

        lengths = [length for _, length in results]
        assert all(val == lengths[0] for val in lengths)

        manager.close()

    def test_load_reference_structures_uses_datetime_run_id(self, tmp_path, rng):
        """Reference structures inherit run_id from parent run_* directory."""
        from scgo.database.helpers import load_reference_structures
        from scgo.metadata.atoms import get_tag

        run_id = "run_20250124_143022_123456"
        run_dir = tmp_path / run_id
        run_dir.mkdir(parents=True)

        atoms = Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
        da = setup_database(run_dir, "ga_go.db", atoms, initial_candidate=atoms)
        a = atoms.copy()
        a.info["key_value_pairs"] = _final_kvp(-5.0)
        a.info["data"] = {"tag": "test_min"}
        da.add_relaxed_step(a)
        close_data_connection(da)
        del da

        refs = load_reference_structures(
            f"{run_id}/ga_go.db",
            composition=["Pt", "Pt"],
            max_structures=10,
            base_dir=tmp_path,
        )
        assert len(refs) == 1
        assert get_tag(refs[0], "run_id") == run_id

    def test_diversity_references_caching(self, tmp_path, rng):
        """Test diversity reference loading with caching."""
        for i in range(3):
            run_dir = tmp_path / f"run_{i:03d}"
            run_dir.mkdir(parents=True)

            _ = run_dir / f"ref_{i}.db"
            atoms = Atoms("Pt3", positions=[[0, 0, 0], [2.5, 0, 0], [1.25, 2.0, 0]])
            da = setup_database(run_dir, f"ref_{i}.db", atoms, initial_candidate=atoms)

            for j in range(5):
                a = atoms.copy()
                a.positions += rng.random((3, 3)) * 0.1
                a.info["key_value_pairs"] = _final_kvp(-30.0 - j)
                a.info["data"] = {"tag": f"test_{j}"}
                da.add_relaxed_step(a)

            close_data_connection(da)
        del da

        manager = SCGODatabaseManager(base_dir=tmp_path, enable_caching=True)

        refs1 = manager.load_reference_structures(
            "run_*/ref_*.db",
            composition=["Pt", "Pt", "Pt"],
            max_structures=20,
        )

        from scgo.metadata.atoms import get_tag

        assert len(refs1) > 0
        assert all(isinstance(a, Atoms) for a in refs1)
        # Ensure only final-unique-minimum tagged structures were loaded
        assert all(get_tag(a, "final_unique_minimum", False) for a in refs1)

        refs2 = manager.load_reference_structures(
            "run_*/ref_*.db",
            composition=["Pt", "Pt", "Pt"],
            max_structures=20,
        )

        assert refs1 is refs2

        refs3 = manager.load_reference_structures(
            "run_*/ref_*.db",
            composition=["Pt", "Pt", "Pt"],
            max_structures=20,
            force_reload=True,
        )

        assert refs1 is not refs3

        manager.close()

    @pytest.mark.requires_multicore
    @pytest.mark.xdist_group
    def test_load_previous_run_results_parallel_integration(self, tmp_path, rng):
        """Ensure the parallel-capable loader returns the same minima set (integration).

        Uses >=4 run directories so the parallel branch is exercised.
        """
        # Create 4 runs each with a trial containing 3 relaxed structures
        for i in range(4):
            run_dir = tmp_path / f"run_{i:03d}"
            run_dir.mkdir(parents=True)

            atoms = Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
            da = setup_database(run_dir, "test.db", atoms, initial_candidate=atoms)

            for j in range(3):
                a = atoms.copy()
                a.positions += rng.random((2, 3)) * 0.1
                a.info["key_value_pairs"] = _final_kvp(-10.0 - j)
                a.info["data"] = {"tag": f"test_{j}"}
                da.add_relaxed_step(a)

            close_data_connection(da)
        del da

        from scgo.database import helpers

        minima = helpers.load_previous_run_results(
            base_output_dir=tmp_path,
            composition=["Pt", "Pt"],
            current_run_id="run_999",
        )

        # 4 runs * 3 structures each = 12 minima expected
        assert len(minima) == 12
        assert all(
            isinstance(e, float) and hasattr(a, "get_chemical_symbols")
            for e, a in minima
        )


def test_manager_cache_metadata_folded_into_value(tmp_path):
    """4.2: cache metadata lives in the value tuple; no parallel dict."""
    from scgo.database.manager import SCGODatabaseManager

    manager = SCGODatabaseManager(
        base_dir=tmp_path, enable_caching=True, cache_ttl_seconds=10
    )
    try:
        # The redundant side-car metadata dict is gone.
        assert not hasattr(manager, "_cache_entries")

        key = (
            str(manager.base_dir.resolve()),
            "prev_results",
            ("Pt", "Pt"),
            None,
            None,
            True,
        )
        fp = manager._compute_files_fingerprint([])
        value = [("e", None)]

        # Store folds (value, fingerprint, timestamp) into the cache.
        manager._store_cached(key, value, fp)
        # Read returns just the value.
        assert manager._get_cached(key) == value
        # Stored object is returned by reference (no copy).
        assert manager._get_cached(key) is value

        # Valid when fingerprint matches and TTL is fresh.
        assert manager._is_cache_valid(key, fp) is True
        # Invalid on fingerprint mismatch.
        assert manager._is_cache_valid(key, ("other",)) is False
        # Invalid after TTL expiry.
        manager.cache_ttl_seconds = -1
        assert manager._is_cache_valid(key, fp) is False

        # clear_cache wipes the value.
        manager._store_cached(key, value, fp)
        manager.clear_cache()
        assert manager._get_cached(key) is None
    finally:
        manager.close()
