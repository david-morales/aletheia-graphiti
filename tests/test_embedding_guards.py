"""Test that similarity search queries include null/invalid embedding guards.
Upstream #1332."""

import inspect

from graphiti_core.search import search_utils


def test_node_similarity_search_has_null_guard():
    source = inspect.getsource(search_utils.node_similarity_search)
    assert 'n.name_embedding IS NOT NULL' in source
    assert 'size(n.name_embedding)' in source


def test_edge_similarity_search_has_null_guard():
    source = inspect.getsource(search_utils.edge_similarity_search)
    assert 'e.fact_embedding IS NOT NULL' in source
    assert 'size(e.fact_embedding)' in source


def test_community_similarity_search_has_null_guard():
    source = inspect.getsource(search_utils.community_similarity_search)
    assert 'c.name_embedding IS NOT NULL' in source
    assert 'size(c.name_embedding)' in source
