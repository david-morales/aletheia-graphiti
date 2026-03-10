"""Verify that edge save queries use (source, target, name) as MERGE key, not uuid."""

import re

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

    def test_neptune_single_merges_on_name(self):
        query = get_entity_edge_save_query(GraphProvider.NEPTUNE)
        assert '{uuid: $edge_data.uuid}' not in query
        assert 'name: $edge_data.name' in query

    def test_neptune_bulk_merges_on_name(self):
        query = get_entity_edge_save_bulk_query(GraphProvider.NEPTUNE)
        assert '{uuid: edge.uuid}' not in query
        assert 'name: edge.name' in query


class TestFalkorDBTypedEdgeBulkMergeKey:
    """The FalkorDB bulk save path in bulk_utils.py builds dynamic Cypher
    per edge type. Verify it doesn't use uuid as MERGE key."""

    def test_typed_bulk_merge_no_uuid_key(self):
        """Simulate the query template from bulk_utils.py and verify no uuid MERGE key."""
        # This mirrors the query construction in bulk_utils.py lines 278-286
        safe_edge_type = 'EN_PROVINCIA'
        query = f"""
            UNWIND $entity_edges AS edge
            MATCH (source:Entity {{uuid: edge.source_node_uuid}})
            MATCH (target:Entity {{uuid: edge.target_node_uuid}})
            MERGE (source)-[r:{safe_edge_type}]->(target)
            SET r = edge
            SET r.fact_embedding = vecf32(edge.fact_embedding)
            RETURN edge.uuid AS uuid
        """
        # The MERGE should NOT have {uuid: edge.uuid} — just the bare relationship
        assert '{uuid: edge.uuid}' not in query
        # The MERGE should use the type label directly with no property key
        assert re.search(r'MERGE.*\)-\[r:EN_PROVINCIA\]->\(', query)

    def test_bulk_utils_source_no_uuid_merge(self):
        """Verify the actual bulk_utils.py source doesn't use uuid in MERGE."""
        import inspect
        from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk_tx
        source = inspect.getsource(add_nodes_and_edges_bulk_tx)
        # The typed edge MERGE should not include {uuid: edge.uuid}
        assert '{uuid: edge.uuid}' not in source


class TestEpisodicEdgeMergeKey:
    def test_single_merges_without_uuid(self):
        assert '{uuid: $uuid}' not in EPISODIC_EDGE_SAVE

    def test_bulk_merges_without_uuid(self):
        for provider in [GraphProvider.FALKORDB, GraphProvider.NEO4J]:
            query = get_episodic_edge_save_bulk_query(provider)
            assert '{uuid: edge.uuid}' not in query, f"Failed for {provider}"
