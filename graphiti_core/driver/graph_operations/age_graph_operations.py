"""GraphOperationsInterface implementation for the PostgreSQL + Apache AGE driver.

Phase 0 spike: methods are implemented incrementally, driven by the tests that
exercise `add_episode` + hybrid `search`. Anything not yet implemented inherits
the base `raise NotImplementedError` and is out of scope for this phase
(communities, saga nodes, BFS, next/has-episode edges beyond MENTIONS).
"""

from typing import Any

from graphiti_core.driver.graph_operations.graph_operations import GraphOperationsInterface


class AGEGraphOperations(GraphOperationsInterface):
    """Native-SQL graph mutation/read operations against AGE + pgvector shadow tables."""

    async def clear_data(self, driver: Any, group_ids: list[str] | None = None) -> None:
        """Delete all graph data (and shadow rows), or only the given groups."""
        if group_ids:
            # group_ids are internal identifiers; quote-escape for the Cypher IN-list.
            quoted = ', '.join("'" + str(g).replace("'", "''") + "'" for g in group_ids)
            await driver.execute_query(f'MATCH (n) WHERE n.group_id IN [{quoted}] DETACH DELETE n')
            await driver.execute_sql(
                f'DELETE FROM {driver._node_tbl} WHERE group_id = ANY($1::text[])', group_ids
            )
            await driver.execute_sql(
                f'DELETE FROM {driver._edge_tbl} WHERE group_id = ANY($1::text[])', group_ids
            )
        else:
            await driver.execute_query('MATCH (n) DETACH DELETE n')
            await driver.execute_sql(f'TRUNCATE {driver._node_tbl}, {driver._edge_tbl}')
