"""Tests for label_propagation with iteration cap and oscillation detection."""

import logging
from unittest.mock import patch

from graphiti_core.utils.maintenance.community_operations import (
    MAX_ITERATIONS,
    OSCILLATION_WINDOW,
    Neighbor,
    label_propagation,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_constants_exist():
    """MAX_ITERATIONS and OSCILLATION_WINDOW must be importable module constants."""
    assert isinstance(MAX_ITERATIONS, int)
    assert MAX_ITERATIONS > 0
    assert isinstance(OSCILLATION_WINDOW, int)
    assert OSCILLATION_WINDOW > 0


def test_constants_values():
    """Verify expected default values."""
    assert MAX_ITERATIONS == 100
    assert OSCILLATION_WINDOW == 5


# ---------------------------------------------------------------------------
# Simple convergence
# ---------------------------------------------------------------------------


def test_triangle_converges():
    """Fully connected triangle should form a single cluster.

    With synchronous updates all three nodes see two neighbors that vote for
    the same (highest) label, so they all adopt it on the first iteration.
    """
    projection = {
        'a': [Neighbor(node_uuid='b', edge_count=2), Neighbor(node_uuid='c', edge_count=2)],
        'b': [Neighbor(node_uuid='a', edge_count=2), Neighbor(node_uuid='c', edge_count=2)],
        'c': [Neighbor(node_uuid='a', edge_count=2), Neighbor(node_uuid='b', edge_count=2)],
    }
    clusters = label_propagation(projection)
    assert len(clusters) == 1
    assert set(clusters[0]) == {'a', 'b', 'c'}


def test_complete_four_converges():
    """Complete graph on four nodes (K4) converges to a single cluster.

    Every node sees three neighbors. On the first iteration all nodes adopt the
    highest-numbered initial label (the node that was enumerated last gets the
    highest community id; all others see a majority vote for it).
    """
    nodes = ['a', 'b', 'c', 'd']
    projection: dict[str, list[Neighbor]] = {}
    for n in nodes:
        projection[n] = [
            Neighbor(node_uuid=m, edge_count=2) for m in nodes if m != n
        ]

    clusters = label_propagation(projection)
    assert len(clusters) == 1
    assert set(clusters[0]) == set(nodes)


# ---------------------------------------------------------------------------
# Pair oscillation (pre-existing algorithm behavior)
# ---------------------------------------------------------------------------


def test_two_connected_nodes_terminates():
    """Two mutually-connected nodes oscillate under synchronous update.

    The algorithm should still terminate (via oscillation detection) and
    return all nodes. Due to the synchronous label swap, the pair may end
    up in separate clusters.
    """
    projection = {
        'a': [Neighbor(node_uuid='b', edge_count=2)],
        'b': [Neighbor(node_uuid='a', edge_count=2)],
    }
    clusters = label_propagation(projection)
    all_nodes = {n for c in clusters for n in c}
    assert all_nodes == {'a', 'b'}


def test_three_node_chain_terminates():
    """A-B-C chain terminates.

    End nodes each see one neighbor and swap labels; the algorithm terminates
    via oscillation detection. We verify all nodes are accounted for.
    """
    projection = {
        'a': [Neighbor(node_uuid='b', edge_count=2)],
        'b': [Neighbor(node_uuid='a', edge_count=2), Neighbor(node_uuid='c', edge_count=2)],
        'c': [Neighbor(node_uuid='b', edge_count=2)],
    }
    clusters = label_propagation(projection)
    all_nodes = {n for c in clusters for n in c}
    assert all_nodes == {'a', 'b', 'c'}


# ---------------------------------------------------------------------------
# Disconnected nodes
# ---------------------------------------------------------------------------


def test_disconnected_nodes_separate_clusters():
    """Nodes with no neighbors should each form their own cluster."""
    projection: dict[str, list[Neighbor]] = {
        'a': [],
        'b': [],
        'c': [],
    }
    clusters = label_propagation(projection)
    assert len(clusters) == 3
    cluster_sets = [set(c) for c in clusters]
    assert {'a'} in cluster_sets
    assert {'b'} in cluster_sets
    assert {'c'} in cluster_sets


def test_two_disconnected_components_terminates():
    """Two separate connected components. Each pair may oscillate under
    synchronous update. Verify all nodes are present and no node is lost."""
    projection = {
        'a': [Neighbor(node_uuid='b', edge_count=2)],
        'b': [Neighbor(node_uuid='a', edge_count=2)],
        'c': [Neighbor(node_uuid='d', edge_count=2)],
        'd': [Neighbor(node_uuid='c', edge_count=2)],
    }
    clusters = label_propagation(projection)
    all_nodes = {n for c in clusters for n in c}
    assert all_nodes == {'a', 'b', 'c', 'd'}


def test_two_disconnected_triangles():
    """Two separate triangles should produce exactly two clusters.

    Triangles converge under synchronous update (each node sees two
    neighbors voting for the same label).
    """
    projection = {
        'a': [Neighbor(node_uuid='b', edge_count=2), Neighbor(node_uuid='c', edge_count=2)],
        'b': [Neighbor(node_uuid='a', edge_count=2), Neighbor(node_uuid='c', edge_count=2)],
        'c': [Neighbor(node_uuid='a', edge_count=2), Neighbor(node_uuid='b', edge_count=2)],
        'd': [Neighbor(node_uuid='e', edge_count=2), Neighbor(node_uuid='f', edge_count=2)],
        'e': [Neighbor(node_uuid='d', edge_count=2), Neighbor(node_uuid='f', edge_count=2)],
        'f': [Neighbor(node_uuid='d', edge_count=2), Neighbor(node_uuid='e', edge_count=2)],
    }
    clusters = label_propagation(projection)
    assert len(clusters) == 2
    cluster_sets = [set(c) for c in clusters]
    assert {'a', 'b', 'c'} in cluster_sets
    assert {'d', 'e', 'f'} in cluster_sets


# ---------------------------------------------------------------------------
# Single node
# ---------------------------------------------------------------------------


def test_single_node():
    """A single node with no neighbors should produce one cluster."""
    projection: dict[str, list[Neighbor]] = {'x': []}
    clusters = label_propagation(projection)
    assert len(clusters) == 1
    assert clusters[0] == ['x']


# ---------------------------------------------------------------------------
# Empty projection
# ---------------------------------------------------------------------------


def test_empty_projection():
    """Empty input should return no clusters."""
    clusters = label_propagation({})
    assert clusters == []


# ---------------------------------------------------------------------------
# Iteration cap prevents infinite loop
# ---------------------------------------------------------------------------


def test_iteration_cap_prevents_infinite_loop(caplog):
    """When MAX_ITERATIONS is artificially lowered and OSCILLATION_WINDOW set
    high enough to not trigger, the loop must terminate via the iteration cap
    and log a warning."""
    # A long chain that oscillates — with a high oscillation window it won't
    # be detected via hashing, forcing the iteration cap to kick in.
    projection = {
        'a': [Neighbor(node_uuid='b', edge_count=2)],
        'b': [Neighbor(node_uuid='a', edge_count=2), Neighbor(node_uuid='c', edge_count=2)],
        'c': [Neighbor(node_uuid='b', edge_count=2), Neighbor(node_uuid='d', edge_count=2)],
        'd': [Neighbor(node_uuid='c', edge_count=2), Neighbor(node_uuid='e', edge_count=2)],
        'e': [Neighbor(node_uuid='d', edge_count=2)],
    }

    with (
        patch('graphiti_core.utils.maintenance.community_operations.MAX_ITERATIONS', 2),
        patch('graphiti_core.utils.maintenance.community_operations.OSCILLATION_WINDOW', 1000),
    ):
        with caplog.at_level(logging.WARNING):
            clusters = label_propagation(projection)

    # Must still return some result (not hang)
    assert isinstance(clusters, list)
    assert len(clusters) >= 1
    # All nodes accounted for
    all_nodes = {n for c in clusters for n in c}
    assert all_nodes == {'a', 'b', 'c', 'd', 'e'}
    # Warning about max iterations should have been logged
    assert any('maximum iterations' in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# Oscillation detection
# ---------------------------------------------------------------------------


def test_oscillation_detection_logs_warning(caplog):
    """When the state oscillates, the algorithm should detect it, break,
    and log a warning."""
    # A bipartite graph where synchronous updates cause oscillation:
    # Two groups {a,b} and {c,d} fully cross-connected but not intra-connected.
    # Under synchronous update a,b adopt cd's label and cd adopt ab's label
    # (flip-flop). With edge_count=2 this triggers the candidate_rank > 1 path.
    projection = {
        'a': [Neighbor(node_uuid='c', edge_count=2), Neighbor(node_uuid='d', edge_count=2)],
        'b': [Neighbor(node_uuid='c', edge_count=2), Neighbor(node_uuid='d', edge_count=2)],
        'c': [Neighbor(node_uuid='a', edge_count=2), Neighbor(node_uuid='b', edge_count=2)],
        'd': [Neighbor(node_uuid='a', edge_count=2), Neighbor(node_uuid='b', edge_count=2)],
    }

    with caplog.at_level(logging.WARNING):
        clusters = label_propagation(projection)

    # Must terminate (not hang)
    assert isinstance(clusters, list)
    all_nodes = {n for c in clusters for n in c}
    assert all_nodes == {'a', 'b', 'c', 'd'}

    # Oscillation should have been detected and logged
    assert any('oscillation' in r.message.lower() for r in caplog.records)


def test_pair_oscillation_detected(caplog):
    """A simple pair of nodes oscillates under synchronous update.

    Verify oscillation detection triggers and a warning is logged.
    """
    projection = {
        'a': [Neighbor(node_uuid='b', edge_count=2)],
        'b': [Neighbor(node_uuid='a', edge_count=2)],
    }

    with caplog.at_level(logging.WARNING):
        clusters = label_propagation(projection)

    all_nodes = {n for c in clusters for n in c}
    assert all_nodes == {'a', 'b'}
    assert any('oscillation' in r.message.lower() for r in caplog.records)
