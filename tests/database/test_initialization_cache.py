"""Tests for the unified cache system.

These tests cover basic semantics, eviction, and concurrent get_or_compute
to ensure the double-check logic prevents redundant computations under contention.
"""

from __future__ import annotations

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


def test_unified_cache_max_size_validation():
    with pytest.raises(SCGOValidationError):
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
