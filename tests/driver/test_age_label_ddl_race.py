"""BUG-38: concurrent first use of a new label raced on AGE's implicit DDL.

AGE materialises a label's backing table the first time a MERGE/CREATE names it,
and it does so implicitly — from inside whatever statement happens to touch the
label first. Nothing serialises that. With N concurrent saves sharing a label
that does not exist yet, several sessions run the same CREATE TABLE at once and
the losers abort the whole save:

    DuplicateTableError: relation "BROADER" already exists          (42P07)
    UniqueViolationError: duplicate key value violates unique
        constraint "pg_class_relname_nsp_index"
        Key (relname, relnamespace)=(BROADER_id_seq, …) already exists   (23505)

The second SQLSTATE is the reason a "catch 42P07 and retry" fix is not enough:
the collision can land on the label's table OR on its sequence, and there is no
bounded list of the relations AGE creates per label. Live trigger (2026-08-05):
5 of 428 skos:broader edges were lost on the ontology load, and the recorded
workaround was to drop the loader to ALETHEIA_ONTOLOGY_LOADER_MAX_CONCURRENT=1.

FalkorDB has no label DDL step at all, which is why only the AGE flavour ever
showed this.

Fix under test: the driver materialises a label explicitly, once, behind a
transaction-scoped advisory lock keyed on (graph, label), and every label-bearing
write declares its labels first — so the implicit path is never reached
concurrently. Reproducing the race needs the live bed, so these run through the
shared `age_driver` fixture (tests/driver/conftest.py).
"""

import asyncio
from datetime import datetime, timezone
from typing import Any

import pytest

from graphiti_core.driver.graph_operations.age_graph_operations import AGEGraphOperations

CONCURRENCY = 24


# ---------------------------------------------------------------------- offline
# Every writer that NAMES a label must declare it, or that label's first use is
# back on AGE's unsynchronised implicit path. Checked through the real writers
# against a capturing double so it gates every CI run, not only live ones.


class _CapturingDriver:
    def __init__(self) -> None:
        self.queries: list[str] = []
        self.ensured: list[tuple[str, str]] = []
        self._node_tbl = 'age_nodes'
        self._edge_tbl = 'age_edges'

    async def execute_query(self, cypher: str, **_kwargs: Any):
        self.queries.append(cypher)
        return [{'uuid': 'merged'}], ['uuid'], None

    async def execute_sql(self, _sql: str, *_args: Any):
        return []

    async def ensure_vertex_label(self, label: str) -> None:
        self.ensured.append(('v', label))

    async def ensure_edge_label(self, label: str) -> None:
        self.ensured.append(('e', label))


def _labels_named_in_merges(driver: _CapturingDriver) -> set[tuple[str, str]]:
    """The labels the emitted MERGE patterns actually name: `(n:Foo` / `[r:BAR`."""
    import re

    named: set[tuple[str, str]] = set()
    for query in driver.queries:
        for merge in re.findall(r'MERGE\s+(.+?)(?:\s+SET\b|$)', query):
            named |= {('v', m) for m in re.findall(r'\(\s*\w*\s*:(\w+)', merge)}
            named |= {('e', m) for m in re.findall(r'\[\s*\w*\s*:(\w+)', merge)}
    return named


@pytest.mark.asyncio
@pytest.mark.parametrize('writer', ['node_save', 'edge_save', 'episodic_node_save',
                                    'episodic_edge_save', 'bulk_node', 'bulk_edge',
                                    'bulk_episode'])
async def test_every_writer_declares_the_labels_its_merge_names(writer):
    from graphiti_core.edges import EntityEdge, EpisodicEdge
    from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode

    driver = _CapturingDriver()
    ops = AGEGraphOperations()
    now = datetime.now(timezone.utc)
    node = _node('N', 'Persona')
    edge = _edge(node, node, 'ES_DETENIDO', 'fact')
    episode = EpisodicNode(
        name='ep', group_id='bug38', source=EpisodeType.text, source_description='',
        content='c', valid_at=now, created_at=now,
    )

    if writer == 'node_save':
        await ops.node_save(node, driver)
    elif writer == 'edge_save':
        await ops.edge_save(edge, driver)
    elif writer == 'episodic_node_save':
        await ops.episodic_node_save(episode, driver)
    elif writer == 'episodic_edge_save':
        await ops.episodic_edge_save(
            EpisodicEdge(
                source_node_uuid=episode.uuid, target_node_uuid=node.uuid,
                group_id='bug38', created_at=now,
            ),
            driver,
        )
    elif writer == 'bulk_node':
        await ops.node_save_bulk(EntityNode, driver, None, [node.model_dump()])
    elif writer == 'bulk_edge':
        await ops.edge_save_bulk(EntityEdge, driver, None, [edge.model_dump()])
    elif writer == 'bulk_episode':
        await ops.episodic_node_save_bulk(EpisodicNode, driver, None, [episode.model_dump()])

    named = _labels_named_in_merges(driver)
    assert named, f'{writer} emitted no MERGE to check'
    undeclared = named - set(driver.ensured)
    assert not undeclared, f'{writer} MERGEs on undeclared labels: {sorted(undeclared)}'


# ------------------------------------------------------------------------- live


def _node(name: str, label: str):
    from graphiti_core.nodes import EntityNode

    node = EntityNode(
        name=name,
        group_id='bug38',
        labels=['Entity', label],
        created_at=datetime.now(timezone.utc),
        summary='',
        attributes={},
    )
    node.name_embedding = None
    return node


def _edge(src, tgt, name: str, fact: str):
    from graphiti_core.edges import EntityEdge

    edge = EntityEdge(
        source_node_uuid=src.uuid,
        target_node_uuid=tgt.uuid,
        name=name,
        group_id='bug38',
        fact=fact,
        created_at=datetime.now(timezone.utc),
    )
    edge.fact_embedding = None
    return edge


async def _label_kinds(driver, label: str) -> list[str]:
    rows = await driver.execute_sql(
        'SELECT l.kind::text AS kind FROM ag_catalog.ag_label l '
        'JOIN ag_catalog.ag_graph g ON g.graphid = l.graph '
        'WHERE g.name = $1 AND l.name = $2',
        driver._database,
        label,
    )
    return [r['kind'] for r in rows]


@pytest.mark.asyncio
async def test_concurrent_first_use_of_a_new_edge_label_persists_every_edge(age_driver):
    """The recorded shape: >=20 concurrent saves sharing a brand-new edge label."""
    ops = age_driver.graph_operations_interface
    nodes = [_node(f'C{i}', 'OntologyClass') for i in range(CONCURRENCY + 1)]
    for node in nodes:
        await ops.node_save(node, age_driver)

    edges = [
        _edge(nodes[i], nodes[i + 1], 'BROADER', f'C{i} broader C{i + 1}')
        for i in range(CONCURRENCY)
    ]
    results = await asyncio.gather(
        *(ops.edge_save(edge, age_driver) for edge in edges), return_exceptions=True
    )

    failures = [r for r in results if isinstance(r, BaseException)]
    assert not failures, f'{len(failures)}/{CONCURRENCY} saves raced on label DDL: {failures[:3]}'
    records, _, _ = await age_driver.execute_query(
        'MATCH ()-[r:BROADER]->() RETURN count(r) AS n'
    )
    assert records[0]['n'] == CONCURRENCY


@pytest.mark.asyncio
async def test_concurrent_first_use_of_a_new_vertex_label_persists_every_node(age_driver):
    """Vertex labels take the identical implicit-DDL path, so they race identically."""
    ops = age_driver.graph_operations_interface
    nodes = [_node(f'P{i}', 'Persona') for i in range(CONCURRENCY)]
    results = await asyncio.gather(
        *(ops.node_save(node, age_driver) for node in nodes), return_exceptions=True
    )

    failures = [r for r in results if isinstance(r, BaseException)]
    assert not failures, f'{len(failures)}/{CONCURRENCY} saves raced on label DDL: {failures[:3]}'
    records, _, _ = await age_driver.execute_query('MATCH (n:Persona) RETURN count(n) AS n')
    assert records[0]['n'] == CONCURRENCY


@pytest.mark.asyncio
async def test_concurrent_saves_of_several_new_labels_at_once(age_driver):
    """Distinct labels created concurrently must not deadlock or lose writes —
    the lock is keyed per label, so these proceed in parallel."""
    ops = age_driver.graph_operations_interface
    labels = [f'Tipo{i}' for i in range(8)]
    nodes = [_node(f'{label}-{j}', label) for label in labels for j in range(3)]
    results = await asyncio.gather(
        *(ops.node_save(node, age_driver) for node in nodes), return_exceptions=True
    )

    failures = [r for r in results if isinstance(r, BaseException)]
    assert not failures, f'{len(failures)} saves failed: {failures[:3]}'
    for label in labels:
        assert await _label_kinds(age_driver, label) == ['v']


@pytest.mark.asyncio
async def test_labels_are_materialised_with_the_right_kind(age_driver):
    """A vertex label and an edge label of the same name are different objects;
    the guard must create each as the kind its writer needs."""
    ops = age_driver.graph_operations_interface
    src, tgt = _node('A', 'Cosa'), _node('B', 'Cosa')
    await ops.node_save(src, age_driver)
    await ops.node_save(tgt, age_driver)
    await ops.edge_save(_edge(src, tgt, 'RELACION', 'a relates to b'), age_driver)

    assert await _label_kinds(age_driver, 'Cosa') == ['v']
    assert await _label_kinds(age_driver, 'RELACION') == ['e']


@pytest.mark.asyncio
async def test_ensure_label_is_idempotent_and_cached(age_driver):
    """Repeat calls must neither error nor re-issue DDL; the second is a cache hit."""
    await age_driver.ensure_vertex_label('Repetida')
    await age_driver.ensure_vertex_label('Repetida')
    await age_driver.ensure_edge_label('REPETIDA')
    await age_driver.ensure_edge_label('REPETIDA')

    assert await _label_kinds(age_driver, 'Repetida') == ['v']
    assert await _label_kinds(age_driver, 'REPETIDA') == ['e']
    assert ('v', 'Repetida') in age_driver._known_labels
    assert ('e', 'REPETIDA') in age_driver._known_labels


@pytest.mark.asyncio
async def test_rebuilding_the_graph_forgets_the_label_cache(age_driver):
    """`build_indices_and_constraints(delete_existing=True)` drops the graph, and
    with it every label. A cache that survived that would let the first write
    after a rebuild fall back into the implicit-DDL path it is there to avoid."""
    await age_driver.ensure_vertex_label('Efimera')
    assert ('v', 'Efimera') in age_driver._known_labels

    await age_driver.build_indices_and_constraints(delete_existing=True)
    assert ('v', 'Efimera') not in age_driver._known_labels
    assert await _label_kinds(age_driver, 'Efimera') == []

    await age_driver.ensure_vertex_label('Efimera')
    assert await _label_kinds(age_driver, 'Efimera') == ['v']


@pytest.mark.asyncio
async def test_bulk_writers_declare_their_labels_too(age_driver):
    """The bulk save paths issue the same MERGEs and need the same guard.

    They take FLAT DICTS, not node objects (add_episode persists through them)."""
    from graphiti_core.nodes import EntityNode

    ops = age_driver.graph_operations_interface
    rows = [
        {
            'uuid': f'bulk-masiva-{i}',
            'name': f'B{i}',
            'group_id': 'bug38',
            'summary': '',
            'labels': ['Entity', 'Masiva'],
            'created_at': datetime.now(timezone.utc),
            'name_embedding': None,
        }
        for i in range(6)
    ]
    await ops.node_save_bulk(EntityNode, age_driver, None, rows)

    assert await _label_kinds(age_driver, 'Masiva') == ['v']
    records, _, _ = await age_driver.execute_query('MATCH (n:Masiva) RETURN count(n) AS n')
    assert records[0]['n'] == 6
