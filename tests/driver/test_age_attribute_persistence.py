"""Regression: AGE re-saves must not erase stored attributes.

2026-08-01 attribute-loss bug: a node saved with attributes, then re-saved
(as entity resolution does) with an empty in-memory attributes map, ended
with attributes = {} in the graph. AGE stores attributes as ONE nested map
and `SET n += {...}` replaces a nested map wholesale.

Edges have the identical shape (`SET r += {...}` over an `attributes` nested
map), so both writer families — node and edge, single and bulk — are covered
here.

Live-gated through the shared `age_driver` fixture (tests/driver/conftest.py):
a per-test `g_<hex>` graph on AGE_TEST_DSN, dropped in teardown, skipped when
the store is unreachable.
"""

from datetime import datetime, timezone

import pytest


def _entity(uuid, attributes, summary='', labels=('Entity', 'Persona')):
    from graphiti_core.nodes import EntityNode

    n = EntityNode(
        uuid=uuid, name='PERSONA UNO', group_id='attrloss',
        labels=list(labels), created_at=datetime.now(timezone.utc),
        summary=summary, attributes=attributes,
    )
    n.name_embedding = None
    return n


@pytest.mark.asyncio
async def test_resave_with_empty_attributes_preserves_stored(age_driver):
    from graphiti_core.nodes import EntityNode

    ops = age_driver.graph_operations_interface
    await ops.node_save(
        _entity('attrloss-1', {'nacimiento_fecha': '1980-01-01', 'telefono': '600000000'}),
        age_driver,
    )
    await ops.node_save(_entity('attrloss-1', {}, summary='seen again'), age_driver)
    got = await ops.node_get_by_uuid(EntityNode, age_driver, 'attrloss-1')
    assert got.attributes.get('nacimiento_fecha') == '1980-01-01'
    assert got.attributes.get('telefono') == '600000000'
    assert got.summary == 'seen again'  # non-attribute props must still update


@pytest.mark.asyncio
async def test_resave_merges_per_key_incoming_wins(age_driver):
    from graphiti_core.nodes import EntityNode

    ops = age_driver.graph_operations_interface
    await ops.node_save(_entity('attrloss-2', {'telefono': '600000000', 'sexo': 'F'}), age_driver)
    await ops.node_save(_entity('attrloss-2', {'telefono': '699999999'}), age_driver)
    got = await ops.node_get_by_uuid(EntityNode, age_driver, 'attrloss-2')
    assert got.attributes.get('telefono') == '699999999'  # incoming non-empty wins
    assert got.attributes.get('sexo') == 'F'              # stored key survives


def _bulk_fields(uuid, attributes, summary='', labels=('Entity', 'Persona')):
    """The flat dict shape the bulk path passes to `_write_entity_from_fields`:
    known node keys plus the node's attributes flattened at the top level
    (see `neo4j/operations/entity_node_ops.py::save_bulk`)."""
    d = {
        'uuid': uuid,
        'name': 'PERSONA UNO',
        'group_id': 'attrloss',
        'summary': summary,
        'created_at': datetime.now(timezone.utc),
        'name_embedding': None,
        'labels': list(labels),
    }
    d.update(attributes)
    return d


@pytest.mark.asyncio
async def test_bulk_resave_with_empty_attributes_preserves_stored(age_driver):
    """The narrative path writes through `node_save_bulk` -> `_write_entity_from_fields`,
    which has the same `SET n +=` shape and so must carry the same guard."""
    from graphiti_core.nodes import EntityNode

    ops = age_driver.graph_operations_interface
    await ops.node_save(
        _entity('attrloss-3', {'nacimiento_fecha': '1980-01-01', 'sexo': 'F'}), age_driver
    )
    await ops.node_save_bulk(
        EntityNode, age_driver, None, [_bulk_fields('attrloss-3', {}, summary='seen again')]
    )
    got = await ops.node_get_by_uuid(EntityNode, age_driver, 'attrloss-3')
    assert got.attributes.get('nacimiento_fecha') == '1980-01-01'
    assert got.attributes.get('sexo') == 'F'
    assert got.summary == 'seen again'


@pytest.mark.asyncio
async def test_bulk_resave_merges_per_key_incoming_wins(age_driver):
    from graphiti_core.nodes import EntityNode

    ops = age_driver.graph_operations_interface
    await ops.node_save(_entity('attrloss-4', {'telefono': '600000000', 'sexo': 'F'}), age_driver)
    await ops.node_save_bulk(
        EntityNode, age_driver, None, [_bulk_fields('attrloss-4', {'telefono': '699999999'})]
    )
    got = await ops.node_get_by_uuid(EntityNode, age_driver, 'attrloss-4')
    assert got.attributes.get('telefono') == '699999999'
    assert got.attributes.get('sexo') == 'F'


# ------------------------------------------------- duplicate uuids across labels
# AGE's MERGE is label-scoped, so `MERGE (n:Persona {uuid: X})` and
# `MERGE (n:Detencion {uuid: X})` create TWO vertices. Graphiti reaches this
# routinely: bulk_utils builds `labels` via `list(set(...))`, whose order is not
# stable across processes, so `_node_label`'s leaf pick can differ run to run for
# the same logical node. A stored-attribute read that is not scoped to the same
# label the write MERGEs on would read one vertex and write its attributes onto
# the other.
@pytest.mark.asyncio
async def test_resave_does_not_bleed_attributes_across_same_uuid_labels(age_driver):
    ops = age_driver.graph_operations_interface

    await ops.node_save(_entity('attrloss-dup', {'origen': 'persona'}), age_driver)
    await ops.node_save(
        _entity('attrloss-dup', {'origen': 'detencion'}, labels=('Entity', 'Detencion')),
        age_driver,
    )
    # Re-save ONLY the Persona vertex, with empty attributes.
    await ops.node_save(_entity('attrloss-dup', {}, summary='seen again'), age_driver)

    records, _, _ = await age_driver.execute_query(
        "MATCH (n) WHERE n.uuid = 'attrloss-dup' "
        'RETURN n.attributes AS attributes, n.summary AS summary, n.labels AS labels'
    )
    by_leaf = {tuple(r['labels'])[-1]: r for r in records}
    assert set(by_leaf) == {'Persona', 'Detencion'}, by_leaf
    assert by_leaf['Persona']['attributes'].get('origen') == 'persona'
    assert by_leaf['Persona']['summary'] == 'seen again'
    # The untouched vertex keeps its own attributes AND its own summary.
    assert by_leaf['Detencion']['attributes'].get('origen') == 'detencion'
    assert by_leaf['Detencion']['summary'] == ''


@pytest.mark.asyncio
async def test_bulk_resave_does_not_bleed_attributes_across_same_uuid_labels(age_driver):
    from graphiti_core.nodes import EntityNode

    ops = age_driver.graph_operations_interface

    await ops.node_save(_entity('attrloss-dup2', {'origen': 'persona'}), age_driver)
    await ops.node_save(
        _entity('attrloss-dup2', {'origen': 'detencion'}, labels=('Entity', 'Detencion')),
        age_driver,
    )
    await ops.node_save_bulk(
        EntityNode, age_driver, None, [_bulk_fields('attrloss-dup2', {}, summary='seen again')]
    )

    records, _, _ = await age_driver.execute_query(
        "MATCH (n) WHERE n.uuid = 'attrloss-dup2' "
        'RETURN n.attributes AS attributes, n.labels AS labels'
    )
    by_leaf = {tuple(r['labels'])[-1]: r for r in records}
    assert by_leaf['Persona']['attributes'].get('origen') == 'persona'
    assert by_leaf['Detencion']['attributes'].get('origen') == 'detencion'


# ------------------------------------------- repeated uuids inside a single batch
@pytest.mark.asyncio
async def test_bulk_accumulates_repeated_uuid_within_one_batch(age_driver):
    """Every row in a batch reads from the SAME pre-batch snapshot, so without
    folding each merge back in as the loop advances, the middle write's keys are
    lost."""
    from graphiti_core.nodes import EntityNode

    ops = age_driver.graph_operations_interface
    await ops.node_save_bulk(
        EntityNode, age_driver, None,
        [
            _bulk_fields('attrloss-rep', {'base': 'B'}),
            _bulk_fields('attrloss-rep', {'first': 'F'}),
            _bulk_fields('attrloss-rep', {'second': 'S'}),
        ],
    )
    got = await ops.node_get_by_uuid(EntityNode, age_driver, 'attrloss-rep')
    assert got.attributes.get('base') == 'B'
    assert got.attributes.get('first') == 'F'
    assert got.attributes.get('second') == 'S'


# ================================================================= edge writers
# Edges carry an `attributes` nested map exactly as nodes do, written through the
# same `SET r += {...}` shape — so the same wipe applies and the same guard must.
def _edge(uuid, attributes, fact='', name='ES_DETENIDO'):
    from graphiti_core.edges import EntityEdge

    e = EntityEdge(
        uuid=uuid, source_node_uuid='attrloss-src', target_node_uuid='attrloss-tgt',
        name=name, fact=fact, group_id='attrloss',
        created_at=datetime.now(timezone.utc), attributes=attributes,
    )
    e.fact_embedding = None
    return e


def _edge_bulk_fields(uuid, attributes, fact='', name='ES_DETENIDO'):
    """The flat dict shape the bulk path passes to `_write_entity_edge_from_fields`:
    known edge keys plus the edge's attributes flattened at the top level
    (see `neo4j/operations/entity_edge_ops.py::save_bulk`)."""
    d = {
        'uuid': uuid,
        'source_node_uuid': 'attrloss-src',
        'target_node_uuid': 'attrloss-tgt',
        'name': name,
        'fact': fact,
        'fact_embedding': None,
        'group_id': 'attrloss',
        'episodes': [],
        'created_at': datetime.now(timezone.utc),
        'expired_at': None,
        'valid_at': None,
        'invalid_at': None,
    }
    d.update(attributes)
    return d


async def _endpoints(driver):
    """Both edge writers do `MATCH (a), (b)`, so the endpoints must exist first."""
    ops = driver.graph_operations_interface
    await ops.node_save(_entity('attrloss-src', {}), driver)
    await ops.node_save(_entity('attrloss-tgt', {}), driver)
    return ops


@pytest.mark.asyncio
async def test_edge_resave_with_empty_attributes_preserves_stored(age_driver):
    from graphiti_core.edges import EntityEdge

    ops = await _endpoints(age_driver)
    await ops.edge_save(
        _edge('attrloss-e1', {'motivo': 'flagrante', 'lugar': 'via publica'}), age_driver
    )
    await ops.edge_save(_edge('attrloss-e1', {}, fact='visto de nuevo'), age_driver)
    got = await ops.edge_get_by_uuid(EntityEdge, age_driver, 'attrloss-e1')
    assert got.attributes.get('motivo') == 'flagrante'
    assert got.attributes.get('lugar') == 'via publica'
    assert got.fact == 'visto de nuevo'  # non-attribute props must still update


@pytest.mark.asyncio
async def test_edge_resave_merges_per_key_incoming_wins(age_driver):
    from graphiti_core.edges import EntityEdge

    ops = await _endpoints(age_driver)
    await ops.edge_save(_edge('attrloss-e2', {'motivo': 'flagrante', 'turno': 'noche'}), age_driver)
    await ops.edge_save(_edge('attrloss-e2', {'motivo': 'orden judicial'}), age_driver)
    got = await ops.edge_get_by_uuid(EntityEdge, age_driver, 'attrloss-e2')
    assert got.attributes.get('motivo') == 'orden judicial'  # incoming non-empty wins
    assert got.attributes.get('turno') == 'noche'            # stored key survives


@pytest.mark.asyncio
async def test_bulk_edge_resave_with_empty_attributes_preserves_stored(age_driver):
    """The narrative path writes edges through `edge_save_bulk` ->
    `_write_entity_edge_from_fields`, which has the same `SET r +=` shape."""
    from graphiti_core.edges import EntityEdge

    ops = await _endpoints(age_driver)
    await ops.edge_save(_edge('attrloss-e3', {'motivo': 'flagrante', 'turno': 'noche'}), age_driver)
    await ops.edge_save_bulk(
        EntityEdge, age_driver, None,
        [_edge_bulk_fields('attrloss-e3', {}, fact='visto de nuevo')],
    )
    got = await ops.edge_get_by_uuid(EntityEdge, age_driver, 'attrloss-e3')
    assert got.attributes.get('motivo') == 'flagrante'
    assert got.attributes.get('turno') == 'noche'
    assert got.fact == 'visto de nuevo'


@pytest.mark.asyncio
async def test_bulk_edge_resave_merges_per_key_incoming_wins(age_driver):
    from graphiti_core.edges import EntityEdge

    ops = await _endpoints(age_driver)
    await ops.edge_save(_edge('attrloss-e4', {'motivo': 'flagrante', 'turno': 'noche'}), age_driver)
    await ops.edge_save_bulk(
        EntityEdge, age_driver, None,
        [_edge_bulk_fields('attrloss-e4', {'motivo': 'orden judicial'})],
    )
    got = await ops.edge_get_by_uuid(EntityEdge, age_driver, 'attrloss-e4')
    assert got.attributes.get('motivo') == 'orden judicial'
    assert got.attributes.get('turno') == 'noche'


@pytest.mark.asyncio
async def test_edge_resave_does_not_bleed_attributes_across_same_uuid_labels(age_driver):
    """`MERGE (a)-[r:TYPE {uuid: X}]->(b)` is label-scoped just like the node MERGE:
    the same edge uuid under two relationship types is two edges. Narrative edges
    fall back to RELATES_TO while projected edges keep their ontology type, so one
    uuid under two labels is reachable in practice."""
    ops = await _endpoints(age_driver)

    await ops.edge_save(_edge('attrloss-edup', {'origen': 'proyeccion'}), age_driver)
    await ops.edge_save(
        _edge('attrloss-edup', {'origen': 'narrativa'}, name='relacionado con'), age_driver
    )
    # Re-save ONLY the typed edge, with empty attributes.
    await ops.edge_save(_edge('attrloss-edup', {}, fact='visto de nuevo'), age_driver)

    records, _, _ = await age_driver.execute_query(
        "MATCH ()-[r]->() WHERE r.uuid = 'attrloss-edup' "
        'RETURN r.attributes AS attributes, r.name AS name'
    )
    by_name = {r['name']: r for r in records}
    assert set(by_name) == {'ES_DETENIDO', 'relacionado con'}, by_name
    assert by_name['ES_DETENIDO']['attributes'].get('origen') == 'proyeccion'
    assert by_name['relacionado con']['attributes'].get('origen') == 'narrativa'


@pytest.mark.asyncio
async def test_bulk_edge_accumulates_repeated_uuid_within_one_batch(age_driver):
    from graphiti_core.edges import EntityEdge

    ops = await _endpoints(age_driver)
    await ops.edge_save_bulk(
        EntityEdge, age_driver, None,
        [
            _edge_bulk_fields('attrloss-erep', {'base': 'B'}),
            _edge_bulk_fields('attrloss-erep', {'first': 'F'}),
            _edge_bulk_fields('attrloss-erep', {'second': 'S'}),
        ],
    )
    got = await ops.edge_get_by_uuid(EntityEdge, age_driver, 'attrloss-erep')
    assert got.attributes.get('base') == 'B'
    assert got.attributes.get('first') == 'F'
    assert got.attributes.get('second') == 'S'
