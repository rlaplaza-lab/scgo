"""Test initialization logging to understand the duplicate messages."""

import logging

import numpy as np
import pytest
from ase import Atoms

from scgo.exceptions import SCGORuntimeError
from scgo.initialization import create_initial_cluster, create_initial_cluster_batch
from scgo.initialization.geometry_helpers import (
    format_composition_counts_short,
    format_placement_error_message,
)
from scgo.initialization.initializers import (
    _sample_suitable_seed,
    _SeedSamplingLogCollector,
    _try_strategies_in_order,
)


def test_create_initial_cluster_batch_logs_and_returns_population(caplog):
    composition = ["Pt"] * 4
    n_structures = 59
    rng = np.random.default_rng(42)
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


def test_batch_init_fallback_summary_emitted_once(caplog):
    """Multi-structure batch should emit one init summary, not per-structure fallbacks."""
    caplog.set_level(logging.DEBUG)
    create_initial_cluster_batch(
        composition=["Pt"] * 4,
        n_structures=10,
        rng=np.random.default_rng(99),
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
        additional_info="remaining: Pt×11",
    )

    assert msg.startswith(
        "Could not complete batch placement (4/15 placed, 11 remaining, 500 attempts)"
    )
    assert "  parameters: placement_radius_scaling=1.20" in msg
    assert "  remaining: Pt×11" in msg
    assert "  suggestions: placement_radius_scaling→1.80" in msg
    assert "Parameters:" not in msg
    assert "Diagnostics:" not in msg


def test_format_composition_counts_short():
    assert format_composition_counts_short({"Pt": 11}) == "Pt×11"
    assert format_composition_counts_short({"Au": 2, "Pt": 3}) == "Au×2, Pt×3"


def test_seed_sampling_log_collector_groups_failures(caplog):
    caplog.set_level(logging.INFO)

    _SeedSamplingLogCollector.reset()
    for _ in range(3):
        _SeedSamplingLogCollector.record("Pt5", "unsuitable linear geometry")
    for _ in range(2):
        _SeedSamplingLogCollector.record("Pt6", "need mixed seed geometries")
    _SeedSamplingLogCollector.emit_summary_if_any()

    assert caplog.text.count("no suitable seed") == 1
    assert "Pt5×3 [unsuitable linear geometry]" in caplog.text
    assert "Pt6×2 [need mixed seed geometries]" in caplog.text


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


def test_batch_seed_failures_are_grouped_not_repeated(caplog):
    """Repeated per-structure seed failures should collapse to one INFO summary."""
    composition = ["Pt"] * 15
    n_structures = 20
    rng = np.random.default_rng(7)
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


def test_single_structure_smart_logs_strategy_allocation(caplog):
    """Single-structure smart mode should emit allocation INFO at least once."""
    caplog.set_level(logging.INFO)
    atoms = create_initial_cluster(
        ["Pt"] * 4,
        mode="smart",
        rng=np.random.default_rng(0),
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
