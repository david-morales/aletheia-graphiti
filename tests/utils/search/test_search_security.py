import pytest
from pydantic import ValidationError

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.errors import GroupIdValidationError, NodeLabelValidationError
from graphiti_core.search.search_filters import (
    SearchFilters,
    edge_search_filter_query_constructor,
    node_search_filter_query_constructor,
)
from graphiti_core.search.search_utils import fulltext_query


def test_search_filters_reject_unsafe_node_labels():
    with pytest.raises(ValidationError, match='node_labels must start with a letter or underscore'):
        SearchFilters(node_labels=['Entity`) WITH n MATCH (x) DETACH DELETE x //'])


def test_node_search_filter_constructor_keeps_valid_label_expression():
    filters = SearchFilters(node_labels=['Person', 'Organization'])
    filter_queries, filter_params = node_search_filter_query_constructor(
        filters, GraphProvider.NEO4J
    )
    assert filter_queries == ['n:Person|Organization']
    assert filter_params == {}


def test_node_search_filter_constructor_rejects_unsafe_labels_bypassing_pydantic():
    filters = SearchFilters.model_construct(node_labels=['Entity`) DETACH DELETE x //'])
    with pytest.raises(NodeLabelValidationError):
        node_search_filter_query_constructor(filters, GraphProvider.NEO4J)


def test_edge_search_filter_constructor_rejects_unsafe_labels_bypassing_pydantic():
    filters = SearchFilters.model_construct(node_labels=['Entity`) DETACH DELETE x //'])
    with pytest.raises(NodeLabelValidationError):
        edge_search_filter_query_constructor(filters, GraphProvider.NEO4J)


def test_fulltext_query_rejects_unsafe_group_ids():
    mock_driver = type('MockDriver', (), {'provider': GraphProvider.NEO4J})()
    with pytest.raises(GroupIdValidationError):
        fulltext_query('test query', ['valid', 'inv@lid!'], mock_driver)
