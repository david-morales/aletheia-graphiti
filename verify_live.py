#!/usr/bin/env python3
"""BUG-62 live probe — does FalkorDB edge search return facts again?

Reproduces the 2026-08-15 measurement that opened this lane: on the FalkorDB
flavour, `edges` mode returned 0 edges for every probe query and the default
`combined` mode returned nodes but zero facts, while the AGE arm returned 4-10
edges for the same queries against the same data.

READ-ONLY. Every statement it issues is a CALL/MATCH read; it never writes,
never creates an index, never drops anything.

NO API KEYS NEEDED. The cosine leg needs a query vector, so instead of embedding
the query text the script borrows a `fact_embedding` already stored on an edge in
the graph. That makes the cosine and combined numbers a proof that the leg
*reaches typed edges at all* rather than a semantic ranking — which is exactly
what BUG-62 was about, since the leg returned nothing whatsoever. Pass --embed
(with OPENAI_API_KEY set) to use a real embedder and get semantically meaningful
combined-mode numbers.

Usage:
    python verify_live.py --graph <graph_name> [--host H] [--port P] [--embed]

Environment fallbacks:
    FALKORDB_HOST (default 127.0.0.1), FALKORDB_PORT (default 6379),
    FALKORDB_GRAPH / FALKORDB_DATABASE (the graph to probe)

Point it at a THROWAWAY stack. The default port is the operator's reserved
container; the script refuses port 6379 unless --allow-reserved-port is passed.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

# The queries measured on 2026-08-15 over the MCP wire.
PROBE_QUERIES = [
    'robo',
    'robo con violencia',
    'detenido',
    'agente interviene',
    'pertenece a',
    'víctima del hecho',
]

RESERVED_PORT = 6379


class _Clients:
    """The three attributes graphiti_core.search.search() reads off clients."""

    def __init__(self, driver, embedder=None, cross_encoder=None):
        self.driver = driver
        self.embedder = embedder
        self.cross_encoder = cross_encoder


def _bm25_only(config):
    """Strip every method that needs a query vector, so no embedder is called.

    search() only reaches for the embedder when a cosine/summary method or an mmr
    reranker is configured; with those gone it uses a zero vector and never calls
    it. Used when the graph has no stored embedding to borrow.
    """
    from graphiti_core.search.search_config import (
        CommunitySearchMethod,
        EdgeSearchMethod,
        NodeSearchMethod,
    )

    stripped = config.model_copy(deep=True)
    if stripped.edge_config:
        stripped.edge_config.search_methods = [
            m for m in stripped.edge_config.search_methods if m != EdgeSearchMethod.cosine_similarity
        ]
    if stripped.node_config:
        stripped.node_config.search_methods = [
            m
            for m in stripped.node_config.search_methods
            if m not in (NodeSearchMethod.cosine_similarity, NodeSearchMethod.summary_similarity)
        ]
    if stripped.community_config:
        stripped.community_config.search_methods = [
            m
            for m in stripped.community_config.search_methods
            if m != CommunitySearchMethod.cosine_similarity
        ]
    return stripped


async def _borrow_fact_embedding(driver) -> list[float] | None:
    """Take a fact_embedding off any entity edge already in the graph.

    Untyped on purpose: the whole point of BUG-62 is that entity edges are not
    stored under RELATES_TO on this flavour.
    """
    query = """
        MATCH (n:Entity)-[e]->(m:Entity)
        WHERE e.fact_embedding IS NOT NULL
        RETURN e.fact_embedding AS v
        LIMIT 1
    """
    try:
        records, _, _ = await driver.execute_query(query, routing_='r')
    except Exception as exc:  # pragma: no cover - live-only path
        print(f'  ! could not borrow a fact_embedding: {exc}')
        return None
    if not records:
        return None
    value = records[0]['v'] if isinstance(records[0], dict) else records[0][0]
    return [float(x) for x in value] if value else None


async def main() -> int:
    parser = argparse.ArgumentParser(description='BUG-62 live probe (read-only)')
    parser.add_argument('--graph', default=os.environ.get('FALKORDB_GRAPH') or os.environ.get('FALKORDB_DATABASE'))
    parser.add_argument('--host', default=os.environ.get('FALKORDB_HOST', '127.0.0.1'))
    parser.add_argument('--port', type=int, default=int(os.environ.get('FALKORDB_PORT', '6379')))
    parser.add_argument('--group-id', action='append', dest='group_ids', default=None)
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--embed', action='store_true', help='use a real embedder (needs OPENAI_API_KEY)')
    parser.add_argument('--allow-reserved-port', action='store_true')
    args = parser.parse_args()

    if not args.graph:
        parser.error('--graph (or FALKORDB_GRAPH) is required')

    if args.port == RESERVED_PORT and not args.allow_reserved_port:
        print(
            f'REFUSING port {RESERVED_PORT}: that is the reserved FalkorDB container.\n'
            'Point --port at a throwaway stack, or pass --allow-reserved-port deliberately.'
        )
        return 2

    from graphiti_core.driver.falkordb_driver import FalkorDriver
    from graphiti_core.search.search import search
    from graphiti_core.search.search_config_recipes import (
        COMBINED_HYBRID_SEARCH_RRF,
        EDGE_HYBRID_SEARCH_RRF,
    )
    from graphiti_core.search.search_filters import SearchFilters
    from graphiti_core.search.search_utils import resolve_entity_edge_types

    driver = FalkorDriver(host=args.host, port=args.port, database=args.graph)

    print(f'target   : falkordb://{args.host}:{args.port} graph={args.graph}')
    print(f'group_ids: {args.group_ids or "(none — all groups)"}')

    edge_types = await resolve_entity_edge_types(driver)
    print(f'entity edge types resolved from the graph ({len(edge_types)}): {sorted(edge_types)}')
    if edge_types == ['RELATES_TO']:
        print('  NOTE: only RELATES_TO — either a non-bulk-ingested graph, or enumeration failed.')

    query_vector = None
    if args.embed:
        from graphiti_core.embedder.openai import OpenAIEmbedder

        embedder = OpenAIEmbedder()
        print('cosine leg: real embedder (--embed)')
    else:
        embedder = None
        query_vector = await _borrow_fact_embedding(driver)
        if query_vector is None:
            print('cosine leg: SKIPPED — no edge in this graph carries a fact_embedding')
        else:
            print(f'cosine leg: borrowed a stored fact_embedding (dim={len(query_vector)})')
            print('            counts prove the leg reaches typed edges, not semantic rank')

    clients = _Clients(driver, embedder=embedder)
    filters = SearchFilters()

    edges_config, combined_config = EDGE_HYBRID_SEARCH_RRF, COMBINED_HYBRID_SEARCH_RRF
    if not args.embed and query_vector is None:
        # Nothing to embed with and nothing to borrow: run the bm25 legs only,
        # which is the leg the 2026-08-15 measurement found returning zero.
        edges_config, combined_config = _bm25_only(edges_config), _bm25_only(combined_config)
        print('            falling back to bm25-only recipes (no vector available)')

    print()
    print(f'{"query":<24} {"edges-mode":>10} {"combined":>10} {"nodes":>7}')
    print('-' * 55)

    failures = 0
    for query in PROBE_QUERIES:
        vector = query_vector
        if args.embed:
            vector = await embedder.create(input_data=[query])

        edges_mode = await search(
            clients, query, args.group_ids, edges_config, filters, query_vector=vector
        )
        combined = await search(
            clients, query, args.group_ids, combined_config, filters, query_vector=vector
        )

        n_edges = len(edges_mode.edges)
        n_combined = len(combined.edges)
        print(f'{query:<24} {n_edges:>10} {n_combined:>10} {len(combined.nodes):>7}')
        if n_edges == 0 and n_combined == 0:
            failures += 1

    print('-' * 55)
    if failures == len(PROBE_QUERIES):
        print('FAIL: every query still returns zero edges — BUG-62 is NOT fixed here.')
        return 1
    if failures:
        print(f'PARTIAL: {failures}/{len(PROBE_QUERIES)} queries returned zero edges.')
        print('         Expected for a query whose terms are absent from this graph.')
        return 0
    print('PASS: every probe query returns edges on both edges-mode and combined.')
    return 0


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
