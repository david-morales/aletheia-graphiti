"""Tests for community update result flattening in Graphiti.add_episode.

The semaphore_gather call in add_episode returns a list of tuples, one per
node.  Each tuple is (list[CommunityNode], list[CommunityEdge]).  The code
must flatten these into two flat lists.  This file validates the flattening
logic for 0, 1, and multiple nodes.

Regression test for #1085.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Helpers that reproduce the flattening logic under test
# ---------------------------------------------------------------------------

def flatten_community_results(
    community_results: list[tuple[list[Any], list[Any]]],
) -> tuple[list[Any], list[Any]]:
    """Reproduce the flattening logic from graphiti.py add_episode."""
    communities: list[Any] = []
    community_edges: list[Any] = []
    for result_communities, result_edges in community_results:
        communities.extend(result_communities)
        community_edges.extend(result_edges)
    return communities, community_edges


def _make_community(name: str) -> MagicMock:
    node = MagicMock(name=f"CommunityNode-{name}")
    node.mock_name = name
    return node


def _make_edge(label: str) -> MagicMock:
    edge = MagicMock(name=f"CommunityEdge-{label}")
    edge.mock_label = label
    return edge


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCommunityResultFlattening:
    """Verify that semaphore_gather results are correctly flattened."""

    def test_empty_results(self) -> None:
        """No nodes processed -> empty lists."""
        results: list[tuple[list[Any], list[Any]]] = []
        communities, edges = flatten_community_results(results)
        assert communities == []
        assert edges == []

    def test_single_node_returns_single_community(self) -> None:
        """One node yields one (community, edge) tuple."""
        c1 = _make_community("c1")
        e1 = _make_edge("e1")
        results = [([c1], [e1])]
        communities, edges = flatten_community_results(results)
        assert communities == [c1]
        assert edges == [e1]

    def test_single_node_no_community(self) -> None:
        """One node where update_community returned ([], [])."""
        results = [([], [])]
        communities, edges = flatten_community_results(results)
        assert communities == []
        assert edges == []

    def test_multiple_nodes_flattened(self) -> None:
        """Multiple nodes -> flat concatenation of all communities and edges."""
        c1, c2, c3 = _make_community("c1"), _make_community("c2"), _make_community("c3")
        e1, e2 = _make_edge("e1"), _make_edge("e2")
        # Node 1 -> community c1, edge e1
        # Node 2 -> community c2, no new edge (not new to community)
        # Node 3 -> community c3, edge e2
        results = [
            ([c1], [e1]),
            ([c2], []),
            ([c3], [e2]),
        ]
        communities, edges = flatten_community_results(results)
        assert communities == [c1, c2, c3]
        assert edges == [e1, e2]

    def test_multiple_nodes_some_empty(self) -> None:
        """Some nodes return no community (entity has no community)."""
        c1 = _make_community("c1")
        e1 = _make_edge("e1")
        results = [
            ([], []),       # node with no community
            ([c1], [e1]),   # node with community
            ([], []),       # another with no community
        ]
        communities, edges = flatten_community_results(results)
        assert communities == [c1]
        assert edges == [e1]

    def test_old_destructuring_fails_with_multiple_nodes(self) -> None:
        """Demonstrate the original bug: direct destructuring crashes with >1 result."""
        c1, c2 = _make_community("c1"), _make_community("c2")
        e1, e2 = _make_edge("e1"), _make_edge("e2")
        results = [([c1], [e1]), ([c2], [e2])]

        # The old code did:  communities, community_edges = await semaphore_gather(...)
        # which is equivalent to:  communities, community_edges = results
        # This should raise ValueError for len(results) != 2 items to unpack
        # Actually with 2 results it would "work" but assign wrong types:
        # communities = ([c1], [e1])  and community_edges = ([c2], [e2])
        # which is a tuple, not a list of CommunityNode.
        communities_bad, _edges_bad = results  # "works" but assigns tuples
        assert isinstance(communities_bad, tuple)  # BUG: should be list of nodes
        assert communities_bad == ([c1], [e1])      # wrong: got a full result tuple

        # With 3 results it crashes:
        results_3 = [([c1], [e1]), ([c2], [e2]), ([], [])]
        try:
            _a, _b = results_3
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected: too many values to unpack

        # The fix flattens correctly:
        communities, edges = flatten_community_results(results)
        assert communities == [c1, c2]
        assert edges == [e1, e2]
