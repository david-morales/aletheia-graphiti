"""SearchInterface implementation for the PostgreSQL + Apache AGE driver.

Implements the fulltext/similarity methods that hybrid `search` exercises, in
native SQL over pgvector (similarity) and tsvector (fulltext) shadow tables,
plus the graph-traversal methods (BFS + node-distance reranking) in AGE Cypher.
Community search inherits documented no-op overrides.

Search runs against the uuid-keyed shadow tables, then hydrates full
EntityNode/EntityEdge objects from the AGE graph (via the graph_operations
interface), preserving the shadow-table ranking order.

SearchFilters support, in full:
  * `edge_types`   — HONOURED by `edge_bfs_search` (`explore_node(edge_types=…)`
                     feeds it straight in).
  * `node_labels`  — DROPPED. The generic constructor emits `n:A|B`, Neo4j
                     syntax AGE cannot parse, and AGE vertices carry only their
                     leaf label anyway; a correct version has to read the
                     `labels` property. Not implemented.
  * `edge_uuids`   — DROPPED.
  * temporal filters (`valid_at` / `invalid_at` / `created_at` / `expired_at`)
                   — DROPPED.
  * The shadow-table legs (fulltext / similarity) ignore SearchFilters entirely.
Dropped filters BROADEN results; they never narrow them wrongly. Callers that
need real filtering must post-filter.

The traversal methods MUST live here rather than fall through to the
provider-generic Cypher in `search_utils`: that Cypher filters on the `:Entity`
node label and the `:RELATES_TO` edge type, and AGE — a single-label store —
carries NEITHER. `age_graph_operations._node_label()` writes each entity vertex
under its LEAF ontology class (`Persona`, `Identificacion`, …) and
`_edge_label()` writes each entity edge under its typed relationship name
(`ES_IDENTIFICADO`, `EN_PARTE`, …). The generic queries parse fine on AGE and
match zero rows, so the failure was silent (2026-08-03, q5 residual F3).
"""

from typing import Any

from graphiti_core.driver.graph_operations.age_graph_operations import _cy, _vec
from graphiti_core.driver.search_interface.search_interface import SearchInterface

# "Is an entity vertex" is asked POSITIVELY, of the `labels` property that
# `node_save` always writes, not by excluding the labels we happen to know about
# today. Measured equivalent on both live AGE graphs (identical result sets:
# 1358/1358 on policia_partes_bench, 564/564 on policia_partes_real), and it
# stays correct if `build_communities` ever writes `:Community` vertices through
# the generic fallback — those have no `labels` property, so they cannot leak
# into BFS and produce `HAS_MEMBER` edges that `EntityEdge` cannot even validate.
_ENTITY_BASE_LABEL = 'Entity'

# Edges get the mirror-image treatment for the opposite reason: AGE edge labels
# are the ONTOLOGY relationship names, an open set with no shared marker, so
# there is no positive predicate to write. What IS closed is Graphiti's own
# structural edge types — the two that are not entity-to-entity facts.
_NON_ENTITY_EDGE_LABELS = ('MENTIONS', 'HAS_MEMBER')

# The similarity score, on the scale `min_score` is calibrated for (BUG-98).
#
# `DEFAULT_MIN_SCORE = 0.6` (search_utils) is a NORMALIZED [0, 1] cosine, because
# that is what every other provider produces: Neo4j's `vector.similarity.cosine`
# is normalized by definition, and FalkorDB rescales explicitly —
# `graph_queries.get_vector_cosine_func_query` emits
# `(2 - vec.cosineDistance(a, b)) / 2`, i.e. `(1 + cos) / 2`.
#
# pgvector's `<=>` is cosine DISTANCE (`1 - cos`), so the obvious `1 - (a <=> b)`
# is the RAW cosine in [-1, 1]. Gating that on 0.6 ran an effective ~0.8 floor on
# AGE while every other arm ran 0.6, and the AGE `search` tool answered only
# near-literal corpus strings: nine phrasings of the same question returned 5
# nodes each on falkor and 0 on AGE for every abstract paraphrase. The legs had
# agreed on the ranking all along — 'tipo de hecho' put
# OTROS HECHOS DE INTERES POLICIAL first on both arms — and disagreed only on the
# scale the shared floor was applied to (642/1336 nodes cleared it on falkor,
# 0/1361 on AGE).
#
# `(2 - dist) / 2` is the same algebra as the FalkorDB branch, so the two arms are
# now comparable by construction rather than by coincidence. Note this changes the
# SCORE only, never the ORDER: `ORDER BY <=>` is a monotone transform of it, and
# is kept as the bare distance so the hnsw `vector_cosine_ops` index still serves
# the sort.
#
# The gate against `min_score` is STRICT (`>`), matching every other provider —
# each of them writes `WHERE score > $min_score` in-query (see
# `falkordb/operations/search_ops.py` and the generic Cypher in `search_utils`).
# On the normalized scale the boundary is reachable in practice rather than
# theoretical: an ORTHOGONAL vector scores exactly `(1 + 0) / 2 = 0.5`, so a `>=`
# gate at `min_score=0.5` would admit a node with nothing in common with the
# query. Under the old raw scale that same node scored 0.0 and was excluded by
# arithmetic, which is why the looser comparison never showed.
_NORMALIZED_COSINE = '(2 - ({col} <=> $1::vector)) / 2'


def _cy_list(values: list[str]) -> str:
    """A Cypher list literal, for inlining into AGE Cypher.

    Delegates to the write path's `_cy` serializer so escaping has exactly one
    implementation on this driver.
    """
    return _cy(list(values))


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
            f"""SELECT uuid, {_NORMALIZED_COSINE.format(col='name_embedding')} AS score
                FROM {driver._node_tbl}
                WHERE name_embedding IS NOT NULL {group_clause}
                ORDER BY name_embedding <=> $1::vector
                LIMIT {int(limit)}""",
            *args,
        )
        ranked = [
            r['uuid'] for r in rows if r['score'] is not None and float(r['score']) > min_score
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
            f"""SELECT uuid, {_NORMALIZED_COSINE.format(col='fact_embedding')} AS score
                FROM {driver._edge_tbl}
                WHERE {where}
                ORDER BY fact_embedding <=> $1::vector
                LIMIT {int(limit)}""",
            *args,
        )
        ranked = [
            r['uuid'] for r in rows if r['score'] is not None and float(r['score']) > min_score
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

    # ---------------------------------------------------------- graph traversal
    async def node_bfs_search(
        self,
        driver: Any,
        bfs_origin_node_uuids: list[str] | None,
        search_filter: Any,
        bfs_max_depth: int,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        """Entity nodes within `bfs_max_depth` OUTGOING hops of the origins.

        Mirrors the generic leg exactly — directed traversal, same-partition
        targets, entity-only results — with AGE's label model: the generic
        `(n:Entity)` becomes `'Entity' IN n.labels` (see `_ENTITY_BASE_LABEL`).
        The direction is not incidental: the generic node leg is directed while
        the edge leg is not, and the two flavours must agree.

        Returns uuids only, then hydrates through the same path as the other legs.
        """
        if not bfs_origin_node_uuids or bfs_max_depth < 1:
            return []

        where = [
            f'origin.uuid IN {_cy_list(bfs_origin_node_uuids)}',
            'n.group_id = origin.group_id',
            f'{_cy(_ENTITY_BASE_LABEL)} IN n.labels',
        ]
        if group_ids:
            # Both conjuncts mirror the generic leg. The first is logically
            # redundant here (`n.group_id = origin.group_id` above, plus the
            # second, already implies it) — a mutation that drops it survives by
            # equivalence, not for want of a test. Kept so the two flavours read
            # the same and so a future edit to the same-partition clause cannot
            # silently widen the partition.
            where.append(f'n.group_id IN {_cy_list(group_ids)}')
            where.append(f'origin.group_id IN {_cy_list(group_ids)}')

        records, _, _ = await driver.execute_query(
            f'MATCH (origin)-[*1..{int(bfs_max_depth)}]->(n) '
            f'WHERE {" AND ".join(where)} '
            f'RETURN DISTINCT n.uuid AS uuid LIMIT {int(limit)}',
            columns=['uuid'],
        )
        return await self._hydrate_nodes_in_order(driver, [r['uuid'] for r in records])

    async def edge_bfs_search(
        self,
        driver: Any,
        bfs_origin_node_uuids: list[str] | None,
        bfs_max_depth: int,
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        """Entity edges lying on any UNDIRECTED path of up to `bfs_max_depth`
        hops from the origins.

        Direction matches the generic leg: node BFS is directed, edge BFS is not
        (an edge is "near" the origin whichever way it points). Graphiti's own
        structural edges are excluded (`_NON_ENTITY_EDGE_LABELS`) because they
        are not entity-to-entity facts — the generic leg says the same thing as
        `(n:Entity)-[e]-(m:Entity)`.
        """
        if not bfs_origin_node_uuids or bfs_max_depth < 1:
            return []

        rel_where = [f'NOT type(rel) IN {_cy_list(_NON_ENTITY_EDGE_LABELS)}']
        if group_ids:
            rel_where.append(f'rel.group_id IN {_cy_list(group_ids)}')
        edge_types = getattr(search_filter, 'edge_types', None)
        if edge_types:
            # `explore_node(edge_types=[...])` reaches us here; the generic leg
            # filters the same property (`e.name in $edge_types`).
            rel_where.append(f'rel.name IN {_cy_list(edge_types)}')

        records, _, _ = await driver.execute_query(
            f'MATCH p = (origin)-[*1..{int(bfs_max_depth)}]-(m) '
            f'WHERE origin.uuid IN {_cy_list(bfs_origin_node_uuids)} '
            f'UNWIND relationships(p) AS rel '
            f'WITH rel WHERE {" AND ".join(rel_where)} '
            f'RETURN DISTINCT rel.uuid AS uuid LIMIT {int(limit)}',
            columns=['uuid'],
        )
        return await self._hydrate_edges_in_order(driver, [r['uuid'] for r in records])

    async def node_distance_reranker(
        self,
        driver: Any,
        node_uuids: list[str],
        center_node_uuid: str,
        min_score: float = 0,
    ) -> tuple[list[str], list[float]]:
        """Rank candidates by adjacency to the center node.

        Same scoring contract as the generic reranker (adjacent -> 1.0, center
        -> 0.1, unconnected -> 1/inf), which on AGE scored EVERY candidate 0.0
        because it matched `(:Entity)-[:RELATES_TO]-(:Entity)`. That made
        `explore_node`'s documented "ranked by proximity to the center node" a
        no-op, and truncation to `limit` then dropped arbitrary results.
        """
        filtered_uuids = [u for u in node_uuids if u != center_node_uuid]
        scores: dict[str, float] = {center_node_uuid: 0.0}

        if filtered_uuids:
            records, _, _ = await driver.execute_query(
                f'MATCH (center)-[rel]-(n) '
                f'WHERE center.uuid = {_cy(center_node_uuid)} '
                f'AND n.uuid IN {_cy_list(filtered_uuids)} '
                f'AND NOT type(rel) IN {_cy_list(_NON_ENTITY_EDGE_LABELS)} '
                f'RETURN DISTINCT n.uuid AS uuid',
                columns=['uuid'],
            )
            for record in records:
                scores[record['uuid']] = 1.0

        for uuid in filtered_uuids:
            scores.setdefault(uuid, float('inf'))

        filtered_uuids.sort(key=lambda cur_uuid: scores[cur_uuid])

        if center_node_uuid in node_uuids:
            scores[center_node_uuid] = 0.1
            filtered_uuids = [center_node_uuid] + filtered_uuids

        return (
            [uuid for uuid in filtered_uuids if (1 / scores[uuid]) >= min_score],
            [1 / scores[uuid] for uuid in filtered_uuids if (1 / scores[uuid]) >= min_score],
        )
