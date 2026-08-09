"""Tests for transition state network building and analysis."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from scgo.ts_search.ts_network import (
    _build_graph_from_connections,
    build_connectivity_graph,
    build_connectivity_graph_from_final_unique_ts,
    find_minimum_barrier_path,
    find_shortest_path,
    get_connected_components,
    save_ts_network_metadata,
)


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


def test_build_connectivity_graph(sample_ts_results):
    """Test building connectivity graph from TS results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        network_file = save_ts_network_metadata(
            sample_ts_results,
            tmpdir,
            composition=["Cu", "Cu", "Cu"],
            minima_count=4,
        )

        graph = build_connectivity_graph(network_file)

        assert graph["num_nodes"] == 4
        assert graph["num_edges"] == 3

        # Check adjacency
        assert 1 in graph["adjacency"][0]  # 0 connects to 1
        assert 0 in graph["adjacency"][1]  # 1 connects to 0 (bidirectional)
        assert 2 in graph["adjacency"][1]  # 1 connects to 2

        # Check edge metadata
        edge_key = (0, 1)
        assert edge_key in graph["edge_metadata"]
        assert graph["edge_metadata"][edge_key]["barrier"] == pytest.approx(
            0.5, rel=1e-6
        )


def test_build_connectivity_graph_ignores_duplicate_connections(sample_ts_results):
    """Duplicate `pair_id` entries in metadata should not create duplicate
    adjacency entries or inflate the reported unique-edge count.
    """
    # Add a duplicate of the first successful connection
    ts_results_dup = sample_ts_results + [sample_ts_results[0].copy()]

    with tempfile.TemporaryDirectory() as tmpdir:
        network_file = save_ts_network_metadata(
            ts_results_dup, tmpdir, composition=["Cu", "Cu", "Cu"], minima_count=4
        )

        graph = build_connectivity_graph(network_file)

        # num_edges reports unique edges (deduplicated)
        assert graph["num_edges"] == 3

        # Adjacency lists contain unique neighbors only
        assert graph["adjacency"][0].count(1) == 1
        assert 1 in graph["adjacency"][0]


def test_get_connected_components(sample_ts_results):
    """Test finding connected components."""
    with tempfile.TemporaryDirectory() as tmpdir:
        network_file = save_ts_network_metadata(
            sample_ts_results,
            tmpdir,
            composition=["Cu", "Cu", "Cu"],
            minima_count=4,
        )

        graph = build_connectivity_graph(network_file)
        components = get_connected_components(graph)

        # Should have 2 components: {0, 1, 2} and {3} (since pair_2_3 failed)
        assert len(components) == 2
        assert {0, 1, 2} in components
        assert {3} in components


def test_get_connected_components_multiple():
    """Test finding multiple connected components."""
    ts_results = [
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
        # No connection to 2, 3
        {
            "pair_id": "2_3",
            "status": "success",
            "reactant_energy": -6.0,
            "product_energy": -5.8,
            "ts_energy": -5.5,
            "barrier_height": 0.5,
            "barrier_forward": 0.5,
            "barrier_reverse": 0.3,
            "neb_converged": True,
            "n_images": 5,
        },
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        network_file = save_ts_network_metadata(
            ts_results,
            tmpdir,
            composition=["Cu", "Cu"],
            minima_count=4,
        )

        graph = build_connectivity_graph(network_file)
        components = get_connected_components(graph)

        # Should have 3 components: {0,1}, {2,3}, {4} (if 4 exists in graph)
        assert len(components) >= 2


def test_find_shortest_path(sample_ts_results):
    """Test finding shortest path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        network_file = save_ts_network_metadata(
            sample_ts_results,
            tmpdir,
            composition=["Cu", "Cu", "Cu"],
            minima_count=4,
        )

        graph = build_connectivity_graph(network_file)

        # Path from 0 to 2 should go through 1
        path = find_shortest_path(graph, 0, 2)
        assert path is not None
        assert path[0] == 0
        assert path[-1] == 2

        # Same node should return self
        path = find_shortest_path(graph, 0, 0)
        assert path == [0]

        # No path between disconnected nodes (0 to 3, since 3 is isolated)
        path = find_shortest_path(graph, 0, 3)
        assert path is None


def test_find_minimum_barrier_path(sample_ts_results):
    """Test finding path with minimum barrier."""
    with tempfile.TemporaryDirectory() as tmpdir:
        network_file = save_ts_network_metadata(
            sample_ts_results,
            tmpdir,
            composition=["Cu", "Cu", "Cu"],
            minima_count=4,
        )

        graph = build_connectivity_graph(network_file)

        # Path from 0 to 2 via lowest barriers
        result = find_minimum_barrier_path(graph, 0, 2)
        assert result is not None
        path, total_barrier = result

        assert path[0] == 0
        assert path[-1] == 2
        assert isinstance(total_barrier, float)
        assert total_barrier >= 0

        # Same node should return self with zero barrier
        result = find_minimum_barrier_path(graph, 0, 0)
        assert result == ([0], 0.0)

        # No path between disconnected nodes
        result = find_minimum_barrier_path(graph, 0, 3)
        assert result is None


def test_pathfinders_handle_invalid_nodes(sample_ts_results):
    """Invalid node IDs should return no path instead of raising."""
    with tempfile.TemporaryDirectory() as tmpdir:
        network_file = save_ts_network_metadata(
            sample_ts_results,
            tmpdir,
            composition=["Cu", "Cu", "Cu"],
            minima_count=4,
        )
        graph = build_connectivity_graph(network_file)

        assert find_shortest_path(graph, 99, 2) is None
        assert find_shortest_path(graph, 0, 99) is None
        assert find_minimum_barrier_path(graph, 99, 2) is None
        assert find_minimum_barrier_path(graph, 0, 99) is None


def test_find_minimum_barrier_path_sparse_node_ids():
    """Dijkstra should support sparse/non-contiguous node keys."""
    sparse_graph = {
        "adjacency": {2: [10], 10: [2]},
        "edge_metadata": {(2, 10): {"barrier": 0.75}},
        "num_nodes": 2,
    }
    result = find_minimum_barrier_path(sparse_graph, 2, 10)
    assert result == ([2, 10], pytest.approx(0.75))


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


def test_build_connectivity_graph_from_final_unique_ts():
    """Graph reconstruction should work from final_unique_ts summary."""
    summary = {
        "formula": "Cu3",
        "unique_ts": [
            {
                "ts_energy": -4.5,
                "connected_edges": [
                    {
                        "pair_id": "0_1",
                        "minima_indices": [0, 1],
                        "barrier_height": 0.5,
                        "barrier_forward": 0.5,
                        "barrier_reverse": 0.3,
                        "neb_converged": True,
                    },
                    {
                        "pair_id": "1_2",
                        "minima_indices": [1, 2],
                        "barrier_height": 0.5,
                        "barrier_forward": 0.5,
                        "barrier_reverse": 0.7,
                        "neb_converged": True,
                    },
                ],
            },
            {
                "ts_energy": -4.2,
                "connected_edges": [
                    {
                        "pair_id": "0_2",
                        "minima_indices": [0, 2],
                        "barrier_height": 0.8,
                        "neb_converged": False,
                    }
                ],
            },
        ],
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "final_unique_ts_summary.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        graph = build_connectivity_graph_from_final_unique_ts(path)

    assert graph["num_nodes"] == 3
    assert graph["num_edges"] == 3
    assert sorted(graph["adjacency"][0]) == [1, 2]
    assert sorted(graph["adjacency"][1]) == [0, 2]
    assert sorted(graph["adjacency"][2]) == [0, 1]
    assert graph["edge_metadata"][(0, 1)]["pair_id"] == "0_1"
    assert graph["edge_metadata"][(1, 2)]["barrier"] == pytest.approx(0.5, rel=1e-6)
    assert graph["edge_metadata"][(0, 2)]["converged"] is False


# ---------------------------------------------------------------------------
# T4: connected components must iterate adjacency keys, not range(num_nodes)
# ---------------------------------------------------------------------------


def _graph_with_sparse_high_index_edge():
    """adjacency keys {0,1,2,5,6}: 0-1-2 connected, plus a separate 5-6 edge."""
    connections = [
        {"pair_id": "0_1", "minima_indices": [0, 1], "barrier_height": 0.5},
        {"pair_id": "1_2", "minima_indices": [1, 2], "barrier_height": 0.5},
        {"pair_id": "5_6", "minima_indices": [5, 6], "barrier_height": 0.5},
    ]
    return _build_graph_from_connections(
        ts_connections=connections,
        num_minima=3,
        composition=["Cu", "Cu", "Cu"],
        formula="Cu3",
        statistics=None,
    )


def test_get_connected_components_uses_adjacency_keys_not_range():
    """High-index adjacency keys must not be dropped, no phantom singletons.

    ``num_nodes = len(adjacency) = 5`` here, but the real keys are {0,1,2,5,6}.
    Iterating ``range(num_nodes)`` invented {3}/{4} and never visited 5/6.
    """
    graph = _graph_with_sparse_high_index_edge()
    assert set(graph["adjacency"].keys()) == {0, 1, 2, 5, 6}
    assert graph["num_nodes"] == 5  # len(adjacency), keys go up to 6

    components = get_connected_components(graph)

    assert {0, 1, 2} in components
    assert {5, 6} in components
    assert len(components) == 2
    # No phantom singletons for the missing dense indices 3 and 4.
    assert {3} not in components
    assert {4} not in components
    # Every real node is covered exactly once (no dropped high-index node).
    covered = set().union(*components)
    assert covered == {0, 1, 2, 5, 6}


# ---------------------------------------------------------------------------
# T5: duplicate TS edges keep the minimal barrier
# ---------------------------------------------------------------------------


def _graph_from_barriers(barriers):
    """Build a graph with repeated (0, 1) connections carrying ``barriers``."""
    connections = [
        {"pair_id": "0_1", "minima_indices": [0, 1], "barrier_height": b}
        for b in barriers
    ]
    return _build_graph_from_connections(
        ts_connections=connections,
        num_minima=2,
        composition=["Cu", "Cu"],
        formula="Cu2",
        statistics=None,
    )


def test_duplicate_edges_keep_minimal_barrier_high_then_low():
    """Second (lower) barrier must replace the first (higher) one."""
    graph = _graph_from_barriers([0.9, 0.4])
    assert graph["num_edges"] == 1
    assert graph["edge_metadata"][(0, 1)]["barrier"] == pytest.approx(0.4)


def test_duplicate_edges_keep_minimal_barrier_low_then_high():
    """First (lower) barrier must survive a later (higher) duplicate."""
    graph = _graph_from_barriers([0.4, 0.9])
    assert graph["num_edges"] == 1
    assert graph["edge_metadata"][(0, 1)]["barrier"] == pytest.approx(0.4)


def test_duplicate_edges_none_barrier_loses_to_numeric():
    """A missing (None) barrier must lose to any numeric barrier, either order."""
    graph_none_first = _graph_from_barriers([None, 0.7])
    assert graph_none_first["num_edges"] == 1
    assert graph_none_first["edge_metadata"][(0, 1)]["barrier"] == pytest.approx(0.7)

    graph_none_last = _graph_from_barriers([0.7, None])
    assert graph_none_last["num_edges"] == 1
    assert graph_none_last["edge_metadata"][(0, 1)]["barrier"] == pytest.approx(0.7)


def test_duplicate_edges_two_none_barriers_keep_first():
    """Two missing barriers keep the first entry (by pair_id)."""
    connections = [
        {"pair_id": "first", "minima_indices": [0, 1], "barrier_height": None},
        {"pair_id": "second", "minima_indices": [0, 1], "barrier_height": None},
    ]
    graph = _build_graph_from_connections(
        ts_connections=connections,
        num_minima=2,
        composition=["Cu", "Cu"],
        formula="Cu2",
        statistics=None,
    )
    assert graph["num_edges"] == 1
    assert graph["edge_metadata"][(0, 1)]["barrier"] is None
    assert graph["edge_metadata"][(0, 1)]["pair_id"] == "first"
