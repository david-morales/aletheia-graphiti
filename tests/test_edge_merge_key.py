"""Verify that edge save queries use (source, target, name) as MERGE key, not uuid."""

from graphiti_core.models.edges.edge_db_queries import (
    get_entity_edge_save_query,
    get_entity_edge_save_bulk_query,
    EPISODIC_EDGE_SAVE,
    get_episodic_edge_save_bulk_query,
)
from graphiti_core.driver.driver import GraphProvider


class TestEntityEdgeMergeKey:
    def test_falkordb_single_merges_on_name(self):
        query = get_entity_edge_save_query(GraphProvider.FALKORDB)
        assert '{uuid: $edge_data.uuid}' not in query
        assert 'name: $edge_data.name' in query

    def test_falkordb_bulk_merges_on_name(self):
        query = get_entity_edge_save_bulk_query(GraphProvider.FALKORDB)
        assert '{uuid: edge.uuid}' not in query
        assert 'name: edge.name' in query

    def test_neo4j_single_merges_on_name(self):
        query = get_entity_edge_save_query(GraphProvider.NEO4J)
        assert '{uuid: $edge_data.uuid}' not in query
        assert 'name: $edge_data.name' in query

    def test_neo4j_bulk_merges_on_name(self):
        query = get_entity_edge_save_bulk_query(GraphProvider.NEO4J)
        assert '{uuid: edge.uuid}' not in query
        assert 'name: edge.name' in query


class TestEpisodicEdgeMergeKey:
    def test_single_merges_without_uuid(self):
        assert '{uuid: $uuid}' not in EPISODIC_EDGE_SAVE

    def test_bulk_merges_without_uuid(self):
        for provider in [GraphProvider.FALKORDB, GraphProvider.NEO4J]:
            query = get_episodic_edge_save_bulk_query(provider)
            assert '{uuid: edge.uuid}' not in query, f"Failed for {provider}"
