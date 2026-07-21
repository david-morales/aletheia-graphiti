"""Tests for the PostgreSQL + Apache AGE + pgvector Graphiti driver (Phase 0 spike).

Requires the docker-compose.age.yml service running and reachable at AGE_TEST_DSN
(default postgresql://age:age@localhost:5433/age_test).
"""

import os

import asyncpg
import pytest

AGE_DSN = os.environ.get('AGE_TEST_DSN', 'postgresql://age:age@localhost:5433/age_test')


@pytest.mark.asyncio
async def test_extensions_available():
    """Both extensions load and AGE's agtype is usable."""
    conn = await asyncpg.connect(AGE_DSN)
    try:
        exts = {
            r['extname']
            for r in await conn.fetch(
                "SELECT extname FROM pg_extension WHERE extname IN ('age','vector')"
            )
        }
        assert exts == {'age', 'vector'}, f'missing extensions, found: {exts}'

        # AGE requires ag_catalog on search_path and its library loaded per session.
        await conn.execute("LOAD 'age'; SET search_path = ag_catalog, '$user', public;")
        val = await conn.fetchval("SELECT '1'::agtype")
        assert str(val) == '1'
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_driver_connects_and_runs_cypher(age_driver):
    """AGEDriver connects, declares provider AGE, and runs a Cypher aggregation."""
    from graphiti_core.driver.driver import GraphProvider

    assert age_driver.provider == GraphProvider.AGE
    records, header, _ = await age_driver.execute_query('MATCH (n) RETURN count(n) AS n')
    assert records == [{'n': 0}]
    assert header == ['n']


@pytest.mark.asyncio
async def test_build_indices_idempotent_and_clear_data_empties(age_driver):
    """build_indices is idempotent (fixture already ran it once); clear_data empties the graph."""
    # second call must not error (shadow tables + indexes use IF NOT EXISTS)
    await age_driver.build_indices_and_constraints()

    await age_driver.execute_query("CREATE (n:Entity {uuid: 'x', group_id: 'g'})")
    records, _, _ = await age_driver.execute_query('MATCH (n) RETURN count(n) AS n')
    assert records == [{'n': 1}]

    await age_driver.graph_operations_interface.clear_data(age_driver, group_ids=None)
    records, _, _ = await age_driver.execute_query('MATCH (n) RETURN count(n) AS n')
    assert records == [{'n': 0}]


@pytest.mark.asyncio
async def test_node_save_and_get_roundtrip(age_driver):
    from datetime import datetime, timezone

    from graphiti_core.nodes import EntityNode

    ops = age_driver.graph_operations_interface
    node = EntityNode(
        name='KHADIJA DAOUD',
        group_id='g',
        labels=['Entity', 'Persona'],
        created_at=datetime.now(timezone.utc),
        summary='persona detenida',
        attributes={'nacionalidad': 'MA'},
    )
    node.name_embedding = [0.1] * age_driver.embedding_dim

    await ops.node_save(node, age_driver)

    got = await ops.node_get_by_uuid(EntityNode, age_driver, node.uuid)
    assert got.uuid == node.uuid
    assert got.name == 'KHADIJA DAOUD'
    assert 'Persona' in got.labels
    assert got.attributes.get('nacionalidad') == 'MA'

    # embeddings load lazily from the pgvector shadow table
    got.name_embedding = None
    await ops.node_load_embeddings(got, age_driver)
    assert got.name_embedding is not None
    assert len(got.name_embedding) == age_driver.embedding_dim


@pytest.mark.asyncio
async def test_edge_save_and_get_between_nodes(age_driver):
    from datetime import datetime, timezone

    from graphiti_core.edges import EntityEdge
    from graphiti_core.nodes import EntityNode

    ops = age_driver.graph_operations_interface
    now = datetime.now(timezone.utc)
    dim = age_driver.embedding_dim
    a = EntityNode(name='A', group_id='g', labels=['Entity'], created_at=now)
    b = EntityNode(name='B', group_id='g', labels=['Entity'], created_at=now)
    a.name_embedding = [0.0] * dim
    b.name_embedding = [0.0] * dim
    await ops.node_save(a, age_driver)
    await ops.node_save(b, age_driver)

    edge = EntityEdge(
        source_node_uuid=a.uuid,
        target_node_uuid=b.uuid,
        name='RELATES_TO',
        fact='A relates to B',
        group_id='g',
        created_at=now,
    )
    edge.fact_embedding = [0.2] * dim
    await ops.edge_save(edge, age_driver)

    got = await ops.edge_get_between_nodes(EntityEdge, age_driver, a.uuid, b.uuid)
    assert len(got) == 1
    assert got[0].uuid == edge.uuid
    assert got[0].fact == 'A relates to B'

    got[0].fact_embedding = None
    await ops.edge_load_embeddings(got[0], age_driver)
    assert got[0].fact_embedding is not None and len(got[0].fact_embedding) == dim


@pytest.mark.asyncio
async def test_node_similarity_search_orders_by_cosine(age_driver):
    from datetime import datetime, timezone

    from graphiti_core.nodes import EntityNode
    from graphiti_core.search.search_filters import SearchFilters

    ops = age_driver.graph_operations_interface
    search = age_driver.search_interface
    now = datetime.now(timezone.utc)
    d = age_driver.embedding_dim

    near = EntityNode(name='near', group_id='g', labels=['Entity'], created_at=now)
    far = EntityNode(name='far', group_id='g', labels=['Entity'], created_at=now)
    near.name_embedding = [1.0] + [0.0] * (d - 1)
    far.name_embedding = [0.0] * (d - 1) + [1.0]
    await ops.node_save(near, age_driver)
    await ops.node_save(far, age_driver)

    results = await search.node_similarity_search(
        age_driver,
        search_vector=[1.0] + [0.0] * (d - 1),
        search_filter=SearchFilters(),
        group_ids=['g'],
        limit=10,
        min_score=0.5,
    )
    assert results, 'expected at least the near node'
    assert results[0].uuid == near.uuid
    assert far.uuid not in [r.uuid for r in results]  # cosine 0 < min_score 0.5


@pytest.mark.asyncio
async def test_node_fulltext_search_matches_terms(age_driver):
    from datetime import datetime, timezone

    from graphiti_core.nodes import EntityNode
    from graphiti_core.search.search_filters import SearchFilters

    ops = age_driver.graph_operations_interface
    search = age_driver.search_interface
    now = datetime.now(timezone.utc)
    d = age_driver.embedding_dim
    for nm, summ in [('KHADIJA DAOUD', 'detenida por hurto'), ('JUAN PEREZ', 'testigo de accidente')]:
        n = EntityNode(name=nm, group_id='g', labels=['Entity'], created_at=now, summary=summ)
        n.name_embedding = [0.0] * d
        await ops.node_save(n, age_driver)

    hits = await search.node_fulltext_search(
        age_driver, 'KHADIJA', SearchFilters(), group_ids=['g'], limit=10
    )
    assert [h.name for h in hits][:1] == ['KHADIJA DAOUD']


@pytest.mark.skipif(
    os.environ.get('AGE_RUN_LIVE_LLM') != '1',
    reason='Live gate: set AGE_RUN_LIVE_LLM=1 with ANTHROPIC_API_KEY + OPENAI_API_KEY to run.',
)
@pytest.mark.asyncio
async def test_add_episode_then_hybrid_search_live(age_driver):
    """Phase 0 GATE (live): real Graphiti add_episode + hybrid search on AGE+pgvector.

    Proves the full pipeline round-trips: structured + narrative extraction into the
    AGE graph, then hybrid retrieval. Uses Anthropic for the LLM + OpenAI for
    embeddings (route matches the aletheia deploy config). Skipped by default.
    """
    from datetime import datetime, timezone

    from graphiti_core import Graphiti
    from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
    from graphiti_core.llm_client.anthropic_client import AnthropicClient
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.nodes import EpisodeType

    llm = AnthropicClient(
        config=LLMConfig(
            api_key=os.environ['ANTHROPIC_API_KEY'],
            model='claude-haiku-4-5-20251001',
            small_model='claude-haiku-4-5-20251001',
        )
    )
    emb = OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key=os.environ['OPENAI_API_KEY'],
            embedding_model='text-embedding-3-small',
            embedding_dim=age_driver.embedding_dim,
        )
    )
    g = Graphiti(graph_driver=age_driver, llm_client=llm, embedder=emb)

    await g.add_episode(
        name='parte-1',
        episode_body=(
            'KHADIJA DAOUD fue detenida por hurto en Madrid. '
            'Menciono a un tal "El Rubio", con quien contacto por el telefono 612345678.'
        ),
        source_description='parte de intervencion',
        reference_time=datetime.now(timezone.utc),
        source=EpisodeType.text,
        group_id='live',
    )

    recs, _, _ = await age_driver.execute_query('MATCH (n:Entity) RETURN count(n) AS n')
    assert recs[0]['n'] >= 2  # structured entity + at least one narrative-derived entity

    results = await g.search('KHADIJA DAOUD', group_ids=['live'])
    assert results, 'hybrid search returned nothing after ingest'
    joined = ' '.join((getattr(r, 'fact', '') or '') for r in results).upper()
    assert 'KHADIJA' in joined
