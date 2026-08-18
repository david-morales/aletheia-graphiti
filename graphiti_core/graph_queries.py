"""
Database query utilities for different graph database backends.

This module provides database-agnostic query generation for Neo4j and FalkorDB,
supporting index creation, fulltext search, and bulk operations.
"""

from typing_extensions import LiteralString

from graphiti_core.driver.driver import GraphProvider

# Mapping from Neo4j fulltext index names to FalkorDB node labels
NEO4J_TO_FALKORDB_MAPPING = {
    'node_name_and_summary': 'Entity',
    'community_name': 'Community',
    'episode_content': 'Episodic',
    'edge_name_and_fact': 'RELATES_TO',
}
# Mapping from fulltext index names to Kuzu node labels
INDEX_TO_LABEL_KUZU_MAPPING = {
    'node_name_and_summary': 'Entity',
    'community_name': 'Community',
    'episode_content': 'Episodic',
    'edge_name_and_fact': 'RelatesToNode_',
}

# The relationship type entity edges are written under everywhere except the
# FalkorDB bulk path, which MERGEs each edge under its own `name` instead.
DEFAULT_ENTITY_EDGE_TYPE = 'RELATES_TO'

# The providers on which an entity edge's relationship type is OPEN — an
# ontology name, not `RELATES_TO` (BUG-62, BUG-87).
#
#   * FALKORDB — `bulk_utils`' edge writer MERGEs each edge under the extracted
#     edge's own `name` (`DETIENE`, `INTERVIENE_AGENTE`, …). The single-edge
#     writer still uses `RELATES_TO`, so one graph holds an open-ended MIX.
#   * AGE — `age_graph_operations._edge_label()` stores every entity edge under
#     its typed relationship name; `RELATES_TO` never appears at all.
#
# On Neo4j, Kuzu and Neptune the type genuinely is the constant, so a pattern
# that names it stays both correct and index-served there. Any query that pins
# `:RELATES_TO` matches ZERO rows on the providers listed here, and matching
# nothing is not an error — which is why every instance of this bug has been
# silent.
TYPED_EDGE_LABEL_PROVIDERS = frozenset({GraphProvider.FALKORDB, GraphProvider.AGE})


def entity_edge_pattern_type(provider: GraphProvider) -> str:
    """The relationship-type part of an `(:Entity)-[e…]-(:Entity)` pattern.

    `':RELATES_TO'` where that constant is the truth, and the EMPTY string —
    an untyped relationship — where it is not (`TYPED_EDGE_LABEL_PROVIDERS`).

    Dropping the type does not widen the match set in any way that matters,
    because the ENDPOINTS are what separate entity edges from Graphiti's
    structural ones: MENTIONS runs Episodic->Entity, HAS_MEMBER
    Community->Entity, HAS_EPISODE Saga->Episodic and NEXT_EPISODE
    Episodic->Episodic, so none of them can match Entity->Entity. That is an
    invariant of the structural WRITERS rather than of the names — the bulk
    writer takes an edge's type from extraction, so a genuine fact edge may well
    be called `HAS_MEMBER` and belongs in the results. This is the same argument
    `get_entity_edge_types_query` and the BUG-62 similarity-leg cure rest on, and
    using one helper is what keeps the legs from drifting apart again.

    Callers must place this immediately after the `e` variable:
    ``f'MATCH (n:Entity)-[e{entity_edge_pattern_type(p)} {{…}}]->(m:Entity)'``.
    """
    return '' if provider in TYPED_EDGE_LABEL_PROVIDERS else f':{DEFAULT_ENTITY_EDGE_TYPE}'


def sanitize_edge_type(edge_type: str) -> str:
    """Reduce a relationship type to what can be interpolated into Cypher.

    Relationship types cannot be parameterised, so every path that builds a typed
    pattern has to inline the name. Mirrors the sanitisation the FalkorDB bulk
    writer applies when it MERGEs the edge, so a type survives the round trip
    from write to search unchanged.
    """
    safe = ''.join(c for c in edge_type if c.isalnum() or c == '_')
    return safe or DEFAULT_ENTITY_EDGE_TYPE


def get_entity_edge_types_query() -> LiteralString:
    """The relationship types that run between two Entity nodes.

    Entity edges are identified by their ENDPOINTS, not by their names. The
    FalkorDB bulk writer takes the relationship type straight from the extracted
    edge's ``name``, so a perfectly ordinary fact edge can arrive called
    ``HAS_MEMBER`` or ``MENTIONS`` — a blocklist of structural names would discard
    it. Graphiti's own structural edges are excluded by their endpoints instead:
    MENTIONS runs Episodic->Entity, HAS_MEMBER Community->Entity, HAS_EPISODE
    Saga->Episodic and NEXT_EPISODE Episodic->Episodic, so none of them can match
    an Entity->Entity pattern.

    This is deliberately the same pattern ``edge_similarity_search`` matches on,
    so the bm25 and cosine legs cannot disagree about what an entity edge is.
    """
    return 'MATCH (n:Entity)-[e]->(m:Entity) RETURN DISTINCT type(e) AS edge_type'


def get_range_indices(provider: GraphProvider) -> list[LiteralString]:
    if provider == GraphProvider.FALKORDB:
        return [
            # Entity node
            'CREATE INDEX FOR (n:Entity) ON (n.uuid, n.group_id, n.name, n.created_at)',
            # Episodic node
            'CREATE INDEX FOR (n:Episodic) ON (n.uuid, n.group_id, n.created_at, n.valid_at)',
            # Community node
            'CREATE INDEX FOR (n:Community) ON (n.uuid)',
            # Saga node
            'CREATE INDEX FOR (n:Saga) ON (n.uuid, n.group_id, n.name)',
            # RELATES_TO edge
            'CREATE INDEX FOR ()-[e:RELATES_TO]-() ON (e.uuid, e.group_id, e.name, e.created_at, e.expired_at, e.valid_at, e.invalid_at)',
            # MENTIONS edge
            'CREATE INDEX FOR ()-[e:MENTIONS]-() ON (e.uuid, e.group_id)',
            # HAS_MEMBER edge
            'CREATE INDEX FOR ()-[e:HAS_MEMBER]-() ON (e.uuid)',
            # HAS_EPISODE edge
            'CREATE INDEX FOR ()-[e:HAS_EPISODE]-() ON (e.uuid, e.group_id)',
            # NEXT_EPISODE edge
            'CREATE INDEX FOR ()-[e:NEXT_EPISODE]-() ON (e.uuid, e.group_id)',
        ]

    if provider == GraphProvider.KUZU:
        return []

    return [
        'CREATE INDEX entity_uuid IF NOT EXISTS FOR (n:Entity) ON (n.uuid)',
        'CREATE INDEX episode_uuid IF NOT EXISTS FOR (n:Episodic) ON (n.uuid)',
        'CREATE INDEX community_uuid IF NOT EXISTS FOR (n:Community) ON (n.uuid)',
        'CREATE INDEX saga_uuid IF NOT EXISTS FOR (n:Saga) ON (n.uuid)',
        'CREATE INDEX relation_uuid IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.uuid)',
        'CREATE INDEX mention_uuid IF NOT EXISTS FOR ()-[e:MENTIONS]-() ON (e.uuid)',
        'CREATE INDEX has_member_uuid IF NOT EXISTS FOR ()-[e:HAS_MEMBER]-() ON (e.uuid)',
        'CREATE INDEX has_episode_uuid IF NOT EXISTS FOR ()-[e:HAS_EPISODE]-() ON (e.uuid)',
        'CREATE INDEX next_episode_uuid IF NOT EXISTS FOR ()-[e:NEXT_EPISODE]-() ON (e.uuid)',
        'CREATE INDEX entity_group_id IF NOT EXISTS FOR (n:Entity) ON (n.group_id)',
        'CREATE INDEX episode_group_id IF NOT EXISTS FOR (n:Episodic) ON (n.group_id)',
        'CREATE INDEX community_group_id IF NOT EXISTS FOR (n:Community) ON (n.group_id)',
        'CREATE INDEX saga_group_id IF NOT EXISTS FOR (n:Saga) ON (n.group_id)',
        'CREATE INDEX relation_group_id IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.group_id)',
        'CREATE INDEX mention_group_id IF NOT EXISTS FOR ()-[e:MENTIONS]-() ON (e.group_id)',
        'CREATE INDEX has_episode_group_id IF NOT EXISTS FOR ()-[e:HAS_EPISODE]-() ON (e.group_id)',
        'CREATE INDEX next_episode_group_id IF NOT EXISTS FOR ()-[e:NEXT_EPISODE]-() ON (e.group_id)',
        'CREATE INDEX name_entity_index IF NOT EXISTS FOR (n:Entity) ON (n.name)',
        'CREATE INDEX saga_name IF NOT EXISTS FOR (n:Saga) ON (n.name)',
        'CREATE INDEX created_at_entity_index IF NOT EXISTS FOR (n:Entity) ON (n.created_at)',
        'CREATE INDEX created_at_episodic_index IF NOT EXISTS FOR (n:Episodic) ON (n.created_at)',
        'CREATE INDEX valid_at_episodic_index IF NOT EXISTS FOR (n:Episodic) ON (n.valid_at)',
        'CREATE INDEX name_edge_index IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.name)',
        'CREATE INDEX created_at_edge_index IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.created_at)',
        'CREATE INDEX expired_at_edge_index IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.expired_at)',
        'CREATE INDEX valid_at_edge_index IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.valid_at)',
        'CREATE INDEX invalid_at_edge_index IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.invalid_at)',
    ]


def get_fulltext_indices(provider: GraphProvider) -> list[LiteralString]:
    if provider == GraphProvider.FALKORDB:
        from typing import cast

        from graphiti_core.driver.falkordb import STOPWORDS

        # Convert to string representation for embedding in queries
        stopwords_str = str(STOPWORDS)

        # Use type: ignore to satisfy LiteralString requirement while maintaining single source of truth
        return cast(
            list[LiteralString],
            [
                f"""CALL db.idx.fulltext.createNodeIndex(
                                                {{
                                                    label: 'Episodic',
                                                    stopwords: {stopwords_str}
                                                }},
                                                'content', 'source', 'source_description', 'group_id'
                                                )""",
                f"""CALL db.idx.fulltext.createNodeIndex(
                                                {{
                                                    label: 'Entity',
                                                    stopwords: {stopwords_str}
                                                }},
                                                'name', 'summary', 'group_id'
                                                )""",
                f"""CALL db.idx.fulltext.createNodeIndex(
                                                {{
                                                    label: 'Community',
                                                    stopwords: {stopwords_str}
                                                }},
                                                'name', 'group_id'
                                                )""",
                """CREATE FULLTEXT INDEX FOR ()-[e:RELATES_TO]-() ON (e.name, e.fact, e.group_id)""",
            ],
        )

    if provider == GraphProvider.KUZU:
        return [
            "CALL CREATE_FTS_INDEX('Episodic', 'episode_content', ['content', 'source', 'source_description']);",
            "CALL CREATE_FTS_INDEX('Entity', 'node_name_and_summary', ['name', 'summary']);",
            "CALL CREATE_FTS_INDEX('Community', 'community_name', ['name']);",
            "CALL CREATE_FTS_INDEX('RelatesToNode_', 'edge_name_and_fact', ['name', 'fact']);",
        ]

    return [
        """CREATE FULLTEXT INDEX episode_content IF NOT EXISTS
        FOR (e:Episodic) ON EACH [e.content, e.source, e.source_description, e.group_id]""",
        """CREATE FULLTEXT INDEX node_name_and_summary IF NOT EXISTS
        FOR (n:Entity) ON EACH [n.name, n.summary, n.group_id]""",
        """CREATE FULLTEXT INDEX community_name IF NOT EXISTS
        FOR (n:Community) ON EACH [n.name, n.group_id]""",
        """CREATE FULLTEXT INDEX edge_name_and_fact IF NOT EXISTS
        FOR ()-[e:RELATES_TO]-() ON EACH [e.name, e.fact, e.group_id]""",
    ]


def get_nodes_query(name: str, query: str, limit: int, provider: GraphProvider) -> str:
    if provider == GraphProvider.FALKORDB:
        label = NEO4J_TO_FALKORDB_MAPPING[name]
        return f"CALL db.idx.fulltext.queryNodes('{label}', {query})"

    if provider == GraphProvider.KUZU:
        label = INDEX_TO_LABEL_KUZU_MAPPING[name]
        return f"CALL QUERY_FTS_INDEX('{label}', '{name}', {query}, TOP := $limit)"

    return f'CALL db.index.fulltext.queryNodes("{name}", {query}, {{limit: $limit}})'


def get_vector_cosine_func_query(vec1, vec2, provider: GraphProvider) -> str:
    if provider == GraphProvider.FALKORDB:
        # FalkorDB uses a different syntax for regular cosine similarity and Neo4j uses normalized cosine similarity
        return f'(2 - vec.cosineDistance({vec1}, vecf32({vec2})))/2'

    if provider == GraphProvider.KUZU:
        return f'array_cosine_similarity({vec1}, {vec2})'

    return f'vector.similarity.cosine({vec1}, {vec2})'


def get_relationships_query(
    name: str, limit: int, provider: GraphProvider, edge_type: str | None = None
) -> str:
    if provider == GraphProvider.FALKORDB:
        # Use provided edge_type or fall back to mapping
        label = edge_type or NEO4J_TO_FALKORDB_MAPPING[name]
        return f"CALL db.idx.fulltext.queryRelationships('{label}', $query)"

    if provider == GraphProvider.KUZU:
        label = INDEX_TO_LABEL_KUZU_MAPPING[name]
        return f"CALL QUERY_FTS_INDEX('{label}', '{name}', cast($query AS STRING), TOP := $limit)"

    return f'CALL db.index.fulltext.queryRelationships("{name}", $query, {{limit: $limit}})'
