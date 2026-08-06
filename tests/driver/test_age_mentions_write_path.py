"""Regression: the AGE MENTIONS writers must not constrain the target to `:Entity`.

Bug: both episodic-edge writers matched their target as `(n:Entity {uuid: …})` —

  * `episodic_edge_save` (age_graph_operations.py:526-538), the projection path;
  * `_write_episodic_edge_from_fields` (:683-695), which `episodic_edge_save_bulk`
    loops over and which is therefore the path `add_episode` uses.

AGE gives a vertex exactly ONE label, and `_node_label` (:94-105) picks the LEAF
ontology class, so a typed entity is labelled `Persona` / `Detencion` / … and never
`Entity`. The MATCH found nothing, the MERGE no-opped, and no error surfaced: on the
live bench graph 101 episodes over 1358 entities produced 24 MENTIONS edges, and a
from-scratch ingest produced 0.

Fix: match the target label-free by uuid in a WHERE, mirroring the entity-edge writer
at :662-667. uuids are globally unique in the graph, so dropping the label constraint
changes nothing but which vertices the pattern can reach. A zero-row MERGE now also
logs a warning, so the next member of this bug class is loud instead of silent.

The offline tests gate every run; the live tests reproduce the exact trigger (a
leaf-labelled entity vertex) on the AGE bed through the shared `age_driver` fixture.
"""

import logging
from datetime import datetime, timezone
from typing import Any

import pytest

from graphiti_core.driver.graph_operations.age_graph_operations import (
    AGEGraphOperations,
    _node_label,
)
from graphiti_core.edges import EpisodicEdge
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode

GROUP = 'mentionswrite'

# The trigger: a label list whose leaf is NOT 'Entity', so the AGE vertex carries
# 'Detencion' and a `(n:Entity {uuid: …})` pattern cannot reach it.
TYPED_LABELS = ['Entity', 'RolInvolucramiento', 'Detencion']


def _episodic_edge(uuid: str, source_uuid: str, target_uuid: str) -> EpisodicEdge:
    return EpisodicEdge(
        uuid=uuid,
        group_id=GROUP,
        source_node_uuid=source_uuid,
        target_node_uuid=target_uuid,
        created_at=datetime.now(timezone.utc),
    )


# ------------------------------------------------------------------------ offline
# Query-shape guards driven through the REAL writers against a capturing double,
# in the style of tests/driver/test_age_columns_from_return_offline.py: no backend,
# so they gate every CI run and not only live ones.


class _CapturingDriver:
    """Records every Cypher body the writers emit; returns one row by default."""

    def __init__(self, records: list[dict[str, Any]] | None = None) -> None:
        self.queries: list[str] = []
        self.ensured: list[tuple[str, str]] = []
        self._records = [{'uuid': 'merged'}] if records is None else records
        self._node_tbl = 'age_nodes'
        self._edge_tbl = 'age_edges'

    async def execute_query(self, cypher: str, **_kwargs: Any):
        self.queries.append(cypher)
        return list(self._records), ['uuid'], None

    async def execute_sql(self, _sql: str, *_args: Any):
        return []

    # Label materialisation is part of the AGE driver contract the writers call
    # into (BUG-38); a double that omitted it would just fail on attribute access.
    async def ensure_vertex_label(self, label: str) -> None:
        self.ensured.append(('v', label))

    async def ensure_edge_label(self, label: str) -> None:
        self.ensured.append(('e', label))


def _merge_query(driver: _CapturingDriver) -> str:
    merges = [q for q in driver.queries if 'MERGE' in q and 'MENTIONS' in q]
    assert len(merges) == 1, driver.queries
    return merges[0]


def _assert_target_is_label_free(query: str) -> None:
    """The target must be reachable whatever its leaf label is."""
    assert ':Entity' not in query, query
    # The source stays label-scoped — `Episodic` IS the stored label for episodes.
    assert ':Episodic' in query, query
    # …and both endpoints are still pinned by uuid, so the label-free pattern
    # cannot widen into a cartesian write.
    assert 'MERGE (e)-[r:MENTIONS' in query, query
    assert query.count('.uuid = ') == 2, query


@pytest.mark.asyncio
async def test_episodic_edge_save_does_not_constrain_the_target_by_label():
    driver = _CapturingDriver()
    edge = _episodic_edge('mw-off-1', 'mw-ep-1', 'mw-ent-1')
    await AGEGraphOperations().episodic_edge_save(edge, driver)
    _assert_target_is_label_free(_merge_query(driver))


@pytest.mark.asyncio
async def test_bulk_episodic_edge_write_does_not_constrain_the_target_by_label():
    driver = _CapturingDriver()
    await AGEGraphOperations()._write_episodic_edge_from_fields(
        driver, _episodic_edge('mw-off-2', 'mw-ep-2', 'mw-ent-2').model_dump()
    )
    _assert_target_is_label_free(_merge_query(driver))


@pytest.mark.asyncio
async def test_both_writers_emit_the_same_merge_shape():
    """The projection and bulk writers are two copies of one query; a fix applied
    to only one of them is the bug half-shipped."""
    single, bulk = _CapturingDriver(), _CapturingDriver()
    edge = _episodic_edge('mw-same', 'mw-ep-s', 'mw-ent-s')  # ONE edge: same created_at
    await AGEGraphOperations().episodic_edge_save(edge, single)
    await AGEGraphOperations()._write_episodic_edge_from_fields(bulk, edge.model_dump())
    assert _merge_query(single) == _merge_query(bulk)


@pytest.mark.parametrize(
    'writer', ['episodic_edge_save', '_write_episodic_edge_from_fields']
)
@pytest.mark.asyncio
async def test_the_target_excludes_bookkeeping_vertices(writer):
    """Dropping the label constraint widens the target to EVERY vertex, including
    Graphiti's own `:Episodic` bookkeeping ones. `n.labels IS NOT NULL` narrows it
    back to entities: every entity write persists a `labels` list (verified live for
    typed, untyped, empty and no-Entity-base shapes, through both writer paths),
    while `episodic_node_save` never writes one. Same discriminator the MCP AGE
    flavour uses for its profile probes (mcp_server/src/flavours/age.py:660-673)."""
    driver = _CapturingDriver()
    ops = AGEGraphOperations()
    edge = _episodic_edge('mw-bk', 'mw-ep-bk', 'mw-ent-bk')
    if writer == 'episodic_edge_save':
        await ops.episodic_edge_save(edge, driver)
    else:
        await ops._write_episodic_edge_from_fields(driver, edge.model_dump())
    assert 'n.labels IS NOT NULL' in _merge_query(driver), _merge_query(driver)


@pytest.mark.parametrize(
    'writer', ['episodic_edge_save', '_write_episodic_edge_from_fields']
)
@pytest.mark.asyncio
async def test_a_zero_row_merge_is_logged_not_swallowed(writer, caplog):
    """Hardening: this bug class was silent. A MERGE that matched nothing must say so."""
    driver = _CapturingDriver(records=[])
    ops = AGEGraphOperations()
    edge = _episodic_edge('mw-miss', 'mw-ep-miss', 'mw-ent-miss')
    with caplog.at_level(logging.WARNING):
        if writer == 'episodic_edge_save':
            await ops.episodic_edge_save(edge, driver)
        else:
            await ops._write_episodic_edge_from_fields(driver, edge.model_dump())
    assert any(r.levelno >= logging.WARNING for r in caplog.records), caplog.records
    assert 'mw-ep-miss' in caplog.text and 'mw-ent-miss' in caplog.text, caplog.text


@pytest.mark.parametrize(
    'writer', ['episodic_edge_save', '_write_episodic_edge_from_fields']
)
@pytest.mark.asyncio
async def test_the_happy_path_logs_no_warning(writer, caplog):
    driver = _CapturingDriver()
    ops = AGEGraphOperations()
    edge = _episodic_edge('mw-hit', 'mw-ep-hit', 'mw-ent-hit')
    with caplog.at_level(logging.WARNING):
        if writer == 'episodic_edge_save':
            await ops.episodic_edge_save(edge, driver)
        else:
            await ops._write_episodic_edge_from_fields(driver, edge.model_dump())
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_the_trigger_label_list_is_actually_typed():
    """Guard for the guards: if `_node_label` ever returned 'Entity' for this list,
    every live test below would pass against the unfixed writer."""
    assert _node_label(TYPED_LABELS) == 'Detencion'


# --------------------------------------------------------------------------- live
# Through the shared `age_driver` fixture (tests/driver/conftest.py): a per-test
# `g_<hex>` graph on AGE_TEST_DSN, dropped in teardown, skipped only when the store
# is unreachable.


async def _typed_entity(driver, uuid: str, labels: list[str] | None = None) -> None:
    """A leaf-labelled entity vertex, written through the production writer."""
    node = EntityNode(
        uuid=uuid,
        name=f'ENTIDAD {uuid}',
        group_id=GROUP,
        labels=list(TYPED_LABELS if labels is None else labels),
        created_at=datetime.now(timezone.utc),
    )
    node.name_embedding = [0.1] * driver.embedding_dim
    await driver.graph_operations_interface.node_save(node, driver)


async def _episode(driver, uuid: str) -> None:
    await driver.graph_operations_interface.episodic_node_save(
        EpisodicNode(
            uuid=uuid,
            name=f'PARTE {uuid}',
            group_id=GROUP,
            source=EpisodeType.json,
            source_description='regression',
            content='{}',
            created_at=datetime.now(timezone.utc),
            valid_at=datetime.now(timezone.utc),
        ),
        driver,
    )


async def _count_mentions(driver) -> int:
    records, _, _ = await driver.execute_query(
        'MATCH (:Episodic)-[r:MENTIONS]->() RETURN count(r) AS n'
    )
    return int(records[0]['n'])


async def _mentions_targets(driver, episode_uuid: str) -> list[str]:
    records, _, _ = await driver.execute_query(
        f"MATCH (e:Episodic)-[r:MENTIONS]->(n) WHERE e.uuid = '{episode_uuid}' "
        f'RETURN n.uuid AS uuid'
    )
    return sorted(str(r['uuid']) for r in records)


@pytest.mark.asyncio
async def test_projection_path_writes_mentions_to_a_typed_entity(age_driver):
    """The exact trigger: the target vertex is labelled `Detencion`, not `Entity`."""
    ops = age_driver.graph_operations_interface
    await _episode(age_driver, 'mw-ep-a')
    await _typed_entity(age_driver, 'mw-ent-a')

    await ops.episodic_edge_save(_episodic_edge('mw-me-a', 'mw-ep-a', 'mw-ent-a'), age_driver)

    assert await _mentions_targets(age_driver, 'mw-ep-a') == ['mw-ent-a']
    assert await _count_mentions(age_driver) == 1


@pytest.mark.asyncio
async def test_bulk_path_writes_mentions_to_a_typed_entity(age_driver):
    """`add_episode` reaches AGE through `episodic_edge_save_bulk`, so the bulk
    writer is the one that produced 24 MENTIONS for 101 episodes in the bench graph."""
    ops = age_driver.graph_operations_interface
    await _episode(age_driver, 'mw-ep-b')
    await _typed_entity(age_driver, 'mw-ent-b')

    await ops.episodic_edge_save_bulk(
        None, age_driver, None, [_episodic_edge('mw-me-b', 'mw-ep-b', 'mw-ent-b').model_dump()]
    )

    assert await _mentions_targets(age_driver, 'mw-ep-b') == ['mw-ent-b']
    assert await _count_mentions(age_driver) == 1


@pytest.mark.asyncio
async def test_an_untyped_entity_target_still_works(age_driver):
    """The only class the old pattern COULD reach (leaf falls back to 'Entity').
    It must keep working — the fix widens the pattern, it must not narrow it."""
    ops = age_driver.graph_operations_interface
    await _episode(age_driver, 'mw-ep-c')
    await _typed_entity(age_driver, 'mw-ent-c', labels=['Entity'])

    await ops.episodic_edge_save(_episodic_edge('mw-me-c', 'mw-ep-c', 'mw-ent-c'), age_driver)

    assert await _mentions_targets(age_driver, 'mw-ep-c') == ['mw-ent-c']


@pytest.mark.asyncio
async def test_every_bulk_episodic_edge_lands(age_driver):
    """Acceptance: N episodic edges saved through the bulk path => exactly N MENTIONS
    edges in the graph, each on its own (episode, entity) pair. Baseline was 0."""
    ops = age_driver.graph_operations_interface
    n = 5
    for i in range(n):
        await _episode(age_driver, f'mw-ep-{i}')
        await _typed_entity(age_driver, f'mw-ent-{i}')
    rows = [
        _episodic_edge(f'mw-me-{i}', f'mw-ep-{i}', f'mw-ent-{i}').model_dump() for i in range(n)
    ]

    await ops.episodic_edge_save_bulk(None, age_driver, None, rows)

    assert await _count_mentions(age_driver) == n
    for i in range(n):
        assert await _mentions_targets(age_driver, f'mw-ep-{i}') == [f'mw-ent-{i}']


@pytest.mark.asyncio
async def test_resaving_the_same_episodic_edge_does_not_duplicate_it(age_driver):
    """MERGE keys on uuid; re-ingesting an episode must not fan out MENTIONS edges."""
    ops = age_driver.graph_operations_interface
    await _episode(age_driver, 'mw-ep-d')
    await _typed_entity(age_driver, 'mw-ent-d')
    edge = _episodic_edge('mw-me-d', 'mw-ep-d', 'mw-ent-d')

    await ops.episodic_edge_save(edge, age_driver)
    await ops.episodic_edge_save(edge, age_driver)
    await ops.episodic_edge_save_bulk(None, age_driver, None, [edge.model_dump()])

    assert await _count_mentions(age_driver) == 1


@pytest.mark.asyncio
async def test_a_missing_target_writes_nothing_and_warns(age_driver, caplog):
    """Live half of the loud-miss guard: a genuinely absent target must not create
    a vertex, must not create an edge, and must be visible in the logs."""
    ops = age_driver.graph_operations_interface
    await _episode(age_driver, 'mw-ep-e')

    with caplog.at_level(logging.WARNING):
        await ops.episodic_edge_save(
            _episodic_edge('mw-me-e', 'mw-ep-e', 'mw-ent-absent'), age_driver
        )

    assert await _count_mentions(age_driver) == 0
    assert 'mw-ent-absent' in caplog.text, caplog.text


@pytest.mark.asyncio
async def test_an_episodic_vertex_is_never_a_mentions_target(age_driver, caplog):
    """The cost of a label-free target: a corrupt `target_node_uuid` pointing at an
    EPISODE used to be unreachable (an episode is not `:Entity`), and would now
    resolve unless the writer excludes bookkeeping vertices. Episode->episode is not
    a MENTIONS relationship in any reading, so it must be refused and logged."""
    ops = age_driver.graph_operations_interface
    await _episode(age_driver, 'mw-ep-g')
    await _episode(age_driver, 'mw-ep-g2')

    with caplog.at_level(logging.WARNING):
        await ops.episodic_edge_save(
            _episodic_edge('mw-me-g', 'mw-ep-g', 'mw-ep-g2'), age_driver
        )

    assert await _count_mentions(age_driver) == 0
    assert 'mw-ep-g2' in caplog.text, caplog.text


@pytest.mark.asyncio
async def test_an_episode_cannot_mention_itself(age_driver, caplog):
    """Degenerate case of the same hole: source uuid == target uuid. The label-free
    pattern would happily MERGE a self-loop on the episode."""
    ops = age_driver.graph_operations_interface
    await _episode(age_driver, 'mw-ep-h')

    with caplog.at_level(logging.WARNING):
        await ops.episodic_edge_save(
            _episodic_edge('mw-me-h', 'mw-ep-h', 'mw-ep-h'), age_driver
        )

    assert await _count_mentions(age_driver) == 0
    loops, _, _ = await age_driver.execute_query(
        "MATCH (n)-[r]->(n) WHERE n.uuid = 'mw-ep-h' RETURN r.uuid AS uuid"
    )
    assert loops == [], loops
    assert 'mw-ep-h' in caplog.text, caplog.text


@pytest.mark.asyncio
async def test_the_bulk_path_also_refuses_an_episodic_target(age_driver):
    """`add_episode` writes through the bulk path, so the guard has to hold there too."""
    ops = age_driver.graph_operations_interface
    await _episode(age_driver, 'mw-ep-i')
    await _episode(age_driver, 'mw-ep-i2')
    await _typed_entity(age_driver, 'mw-ent-i')

    await ops.episodic_edge_save_bulk(
        None,
        age_driver,
        None,
        [
            _episodic_edge('mw-me-i-bad', 'mw-ep-i', 'mw-ep-i2').model_dump(),
            _episodic_edge('mw-me-i-ok', 'mw-ep-i', 'mw-ent-i').model_dump(),
        ],
    )

    # The good row still lands: the guard rejects, it does not abort the batch.
    assert await _mentions_targets(age_driver, 'mw-ep-i') == ['mw-ent-i']
    assert await _count_mentions(age_driver) == 1


@pytest.mark.asyncio
async def test_mentions_are_readable_through_get_mentioned_nodes(age_driver):
    """End of the pipe: what the fix writes is what the reader returns. Without it
    `get_mentioned_nodes` had nothing to traverse."""
    ops = age_driver.graph_operations_interface
    await _episode(age_driver, 'mw-ep-f')
    await _typed_entity(age_driver, 'mw-ent-f')
    await ops.episodic_edge_save(_episodic_edge('mw-me-f', 'mw-ep-f', 'mw-ent-f'), age_driver)

    episode = EpisodicNode(
        uuid='mw-ep-f',
        name='PARTE mw-ep-f',
        group_id=GROUP,
        source=EpisodeType.json,
        source_description='regression',
        content='{}',
        created_at=datetime.now(timezone.utc),
        valid_at=datetime.now(timezone.utc),
    )
    mentioned = await ops.get_mentioned_nodes(age_driver, [episode])
    assert [n.uuid for n in mentioned] == ['mw-ent-f']
