"""Tests for transition state network metadata output."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from scgo.ts_search.ts_network import save_ts_network_metadata


@pytest.fixture
def sample_ts_results():
    """Create sample TS results for testing."""
    return [
        {
            "pair_id": "0_1",
            "status": "success",
            "reactant_energy": -5.0,
            "product_energy": -4.8,
            "ts_energy": -4.5,
            "barrier_height": 0.5,
            "barrier_forward": 0.5,
            "barrier_reverse": 0.3,
            "neb_converged": True,
            "n_images": 5,
        },
        {
            "pair_id": "1_2",
            "status": "success",
            "reactant_energy": -4.8,
            "product_energy": -4.6,
            "ts_energy": -4.3,
            "barrier_height": 0.5,
            "barrier_forward": 0.5,
            "barrier_reverse": 0.7,
            "neb_converged": True,
            "n_images": 5,
        },
        {
            "pair_id": "0_2",
            "status": "success",
            "reactant_energy": -5.0,
            "product_energy": -4.6,
            "ts_energy": -4.2,
            "barrier_height": 0.8,
            "barrier_forward": 0.8,
            "barrier_reverse": 1.4,
            "neb_converged": False,
            "n_images": 5,
        },
        {
            "pair_id": "2_3",
            "status": "failed",
            "error": "NEB did not converge",
        },
    ]


def test_save_ts_network_metadata(sample_ts_results):
    """Test saving TS network metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = save_ts_network_metadata(
            sample_ts_results,
            tmpdir,
            composition=["Cu", "Cu", "Cu"],
            minima_count=4,
        )

        assert os.path.exists(output_path)
        assert "ts_network_metadata.json" in output_path

        with open(output_path) as f:
            metadata = json.load(f)

        assert metadata["formula"] == "Cu3"
        assert metadata["num_minima"] == 4
        assert len(metadata["ts_connections"]) == 3  # Only successful ones
        assert metadata["statistics"]["successful_ts"] == 3
        assert metadata["statistics"]["converged_ts"] == 2


def test_network_with_no_connections():
    """Test network metadata with no successful TS found."""
    ts_results = [
        {
            "pair_id": "0_1",
            "status": "failed",
            "error": "NEB did not converge",
        },
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        network_file = save_ts_network_metadata(
            ts_results,
            tmpdir,
            composition=["Cu", "Cu"],
            minima_count=2,
        )

        with open(network_file) as f:
            metadata = json.load(f)

        assert len(metadata["ts_connections"]) == 0
        assert metadata["statistics"]["successful_ts"] == 0


def test_network_statistics(sample_ts_results):
    """Test network statistics calculation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        network_file = save_ts_network_metadata(
            sample_ts_results,
            tmpdir,
            composition=["Cu", "Cu", "Cu"],
            minima_count=4,
        )

        with open(network_file) as f:
            metadata = json.load(f)

        stats = metadata["statistics"]

        # Should have: 3 successful, 2 converged
        assert stats["total_ts_found"] == 3
        assert stats["successful_ts"] == 3
        assert stats["converged_ts"] == 2

        # Barrier stats
        assert stats["min_barrier"] == pytest.approx(0.5, rel=1e-6)
        assert stats["max_barrier"] == pytest.approx(0.8, rel=1e-6)
        assert stats["avg_barrier"] == pytest.approx((0.5 + 0.5 + 0.8) / 3, rel=1e-6)
