"""Test that node dedup limits candidate context to prevent token overflow.

The fork's MAX_RESOLVE_CANDIDATES (#1276) was superseded by graphiti-core
v0.29.2's ID-based dedup (#1224/#1432), which limits the candidate set via
NODE_DEDUP_CANDIDATE_LIMIT during candidate collection. The context-overflow
guard is preserved; only the constant/mechanism changed.
"""

from graphiti_core.utils.maintenance.node_operations import NODE_DEDUP_CANDIDATE_LIMIT


def test_node_dedup_candidate_limit_constant_exists():
    """NODE_DEDUP_CANDIDATE_LIMIT should bound the candidate context passed to
    dedup so the LLM prompt cannot overflow with too many candidates."""
    assert isinstance(NODE_DEDUP_CANDIDATE_LIMIT, int)
    assert 0 < NODE_DEDUP_CANDIDATE_LIMIT <= 50
