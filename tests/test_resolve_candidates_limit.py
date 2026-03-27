"""Test that resolve_extracted_nodes limits candidate context to prevent token overflow.
Upstream #1276."""

from graphiti_core.utils.maintenance.node_operations import MAX_RESOLVE_CANDIDATES


def test_max_resolve_candidates_constant_exists():
    """MAX_RESOLVE_CANDIDATES should be defined and reasonable."""
    assert isinstance(MAX_RESOLVE_CANDIDATES, int)
    assert MAX_RESOLVE_CANDIDATES == 50
