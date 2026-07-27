"""SearchInterface implementation for the PostgreSQL + Apache AGE driver.

Phase 0 spike: implements the fulltext/similarity methods that hybrid `search`
exercises, in native SQL over pgvector (similarity) and tsvector (fulltext)
shadow tables. Rerankers and community/BFS search inherit the base
`raise NotImplementedError` and are out of scope for this phase.

Search runs against the uuid-keyed shadow tables, then hydrates full
EntityNode/EntityEdge objects from the AGE graph (via the graph_operations
interface), preserving the shadow-table ranking order. SearchFilters are
ignored in this spike (broader results are acceptable for the gate).
"""

from typing import Any

from graphiti_core.driver.graph_operations.age_graph_operations import _vec
from graphiti_core.driver.search_interface.search_interface import SearchInterface


class AGESearch(SearchInterface):
    """Native-SQL search over pgvector + tsvector shadow tables."""

    # ---------------------------------------------------------- hydration helpers
    async def _hydrate_nodes_in_order(self, driver: Any, uuids: list[str]) -> list[Any]:
        from graphiti_core.nodes import EntityNode

        if not uuids:
            return []
        nodes = await driver.graph_operations_interface.node_get_by_uuids(
            EntityNode, driver, uuids
        )
        order = {u: i for i, u in enumerate(uuids)}
        nodes.sort(key=lambda n: order.get(n.uuid, len(uuids)))
        return nodes

    async def _hydrate_edges_in_order(self, driver: Any, uuids: list[str]) -> list[Any]:
        from graphiti_core.edges import EntityEdge

        if not uuids:
            return []
        edges = await driver.graph_operations_interface.edge_get_by_uuids(
            EntityEdge, driver, uuids
        )
        order = {u: i for i, u in enumerate(uuids)}
        edges.sort(key=lambda e: order.get(e.uuid, len(uuids)))
        return edges

    # ------------------------------------------------------------- vector search
    async def node_similarity_search(
        self,
        driver: Any,
        search_vector: list[float],
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
        min_score: float = 0.7,
    ) -> list[Any]:
        args: list[Any] = [_vec(search_vector)]
        group_clause = ''
        if group_ids:
            group_clause = 'AND group_id = ANY($2::text[])'
            args.append(group_ids)
        rows = await driver.execute_sql(
            f"""SELECT uuid, 1 - (name_embedding <=> $1::vector) AS score
                FROM {driver._node_tbl}
                WHERE name_embedding IS NOT NULL {group_clause}
                ORDER BY name_embedding <=> $1::vector
                LIMIT {int(limit)}""",
            *args,
        )
        ranked = [
            r['uuid'] for r in rows if r['score'] is not None and float(r['score']) >= min_score
        ]
        return await self._hydrate_nodes_in_order(driver, ranked)

    async def node_summary_similarity_search(
        self,
        driver: Any,
        search_vector: list[float],
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
        min_score: float = 0.7,
    ) -> list[Any]:
        # Spike simplification: summaries are not embedded in the shadow table
        # (only name embeddings are indexed), so summary-similarity contributes
        # no candidates. Node dedup/resolution still works via name fulltext +
        # name similarity. Indexing summary embeddings is a later refinement.
        return []

    async def edge_similarity_search(
        self,
        driver: Any,
        search_vector: list[float],
        source_node_uuid: str | None,
        target_node_uuid: str | None,
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
        min_score: float = 0.7,
    ) -> list[Any]:
        args: list[Any] = [_vec(search_vector)]
        conds = ['fact_embedding IS NOT NULL']
        idx = 2
        if group_ids:
            conds.append(f'group_id = ANY(${idx}::text[])')
            args.append(group_ids)
            idx += 1
        if source_node_uuid:
            conds.append(f'source_node_uuid = ${idx}')
            args.append(source_node_uuid)
            idx += 1
        if target_node_uuid:
            conds.append(f'target_node_uuid = ${idx}')
            args.append(target_node_uuid)
            idx += 1
        where = ' AND '.join(conds)
        rows = await driver.execute_sql(
            f"""SELECT uuid, 1 - (fact_embedding <=> $1::vector) AS score
                FROM {driver._edge_tbl}
                WHERE {where}
                ORDER BY fact_embedding <=> $1::vector
                LIMIT {int(limit)}""",
            *args,
        )
        ranked = [
            r['uuid'] for r in rows if r['score'] is not None and float(r['score']) >= min_score
        ]
        return await self._hydrate_edges_in_order(driver, ranked)

    # ----------------------------------------------------------- keyword search
    async def node_fulltext_search(
        self,
        driver: Any,
        query: str,
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        if not query or not query.strip():
            return []
        args: list[Any] = [query]
        group_clause = ''
        if group_ids:
            group_clause = 'AND group_id = ANY($2::text[])'
            args.append(group_ids)
        rows = await driver.execute_sql(
            f"""SELECT uuid, ts_rank_cd(tsv, plainto_tsquery('simple', $1)) AS rank
                FROM {driver._node_tbl}
                WHERE tsv @@ plainto_tsquery('simple', $1) {group_clause}
                ORDER BY rank DESC
                LIMIT {int(limit)}""",
            *args,
        )
        return await self._hydrate_nodes_in_order(driver, [r['uuid'] for r in rows])

    async def edge_fulltext_search(
        self,
        driver: Any,
        query: str,
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
        edge_types: list[str] | None = None,
    ) -> list[Any]:
        if not query or not query.strip():
            return []
        args: list[Any] = [query]
        group_clause = ''
        if group_ids:
            group_clause = 'AND group_id = ANY($2::text[])'
            args.append(group_ids)
        rows = await driver.execute_sql(
            f"""SELECT uuid, ts_rank_cd(tsv, plainto_tsquery('simple', $1)) AS rank
                FROM {driver._edge_tbl}
                WHERE tsv @@ plainto_tsquery('simple', $1) {group_clause}
                ORDER BY rank DESC
                LIMIT {int(limit)}""",
            *args,
        )
        return await self._hydrate_edges_in_order(driver, [r['uuid'] for r in rows])

    async def episode_fulltext_search(
        self,
        driver: Any,
        query: str,
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        """Fulltext search over episode content.

        Episodic (raw-document) nodes are not mirrored into a tsvector shadow
        table on the AGE backend — only Entity nodes and edges are (see
        node_fulltext_search / edge_fulltext_search) — so there is nothing to
        keyword-rank here. Return an empty list rather than raising, so the
        combined/hybrid search recipes — which fan a node + edge + episode
        sub-search out concurrently via asyncio.gather — still return their node
        and edge results on AGE instead of the whole call failing. (Mirrors
        node_summary_similarity_search, likewise a documented shadow-table gap.)
        A future task can add an episode shadow table populated on
        episodic_node_save to make this live.
        """
        return []

    async def community_fulltext_search(
        self,
        driver: Any,
        query: str,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        """Fulltext search over community names.

        Communities (built by build_communities) are not projected into AGE nor
        mirrored into a shadow table, so there is nothing to rank. Return [] —
        NOT-implemented would be caught by the caller and fall through to a
        FalkorDB/Neo4j-specific fulltext Cypher (`YIELD node …`) that AGE cannot
        parse ("syntax error at or near '.'"), breaking every combined/hybrid
        search. Returning [] keeps the node + edge results flowing on AGE.
        """
        return []

    async def community_similarity_search(
        self,
        driver: Any,
        search_vector: list[float],
        group_ids: list[str] | None = None,
        limit: int = 100,
        min_score: float = 0.6,
    ) -> list[Any]:
        """Vector similarity search over community name embeddings.

        Same rationale as community_fulltext_search: no communities are stored on
        AGE, and the caller falls through to backend-specific Cypher on
        NotImplementedError, so return [] to keep combined/hybrid search working.
        """
        return []
