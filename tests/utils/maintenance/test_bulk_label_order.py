"""Regression: the bulk writer must preserve the ORDER of a node's label list.

`add_nodes_and_edges_bulk_tx` built each row's `labels` with
`list(set(node.labels + ['Entity']))`. A set has no order, so the row carried an
arbitrary permutation of a list whose LAST entry is the ontology leaf.

On backends that apply every label to the vertex (Neo4j, FalkorDB, Neptune,
Kuzu — see models/nodes/node_db_queries.py:194-267) that is harmless. On AGE,
which stores ONE label per vertex and picks it with `_node_label` ("the last
identifier-safe non-'Entity' entry"), the bulk path and the projection path
(`EntityNode.save` -> `node_save`, which passes the RAW ordered list) could pick
DIFFERENT labels for the same node. AGE's `MERGE (n:<label> {uuid: ...})` is
label-scoped, so two picks mean two vertices sharing one uuid: 62 duplicated
node uuids in production, 194 in the bench graph.

The fix is an order-preserving dedupe: `list(dict.fromkeys(...))`.

Runs without any backend, so it gates every run and not only live ones.
"""

from typing import Any

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.nodes import EntityNode
from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk_tx

# Two permutations of the SAME six identifier-safe labels. Six, and two orders,
# is what makes this test's failure deterministic rather than hash luck: set
# iteration order follows the hash-table layout, not insertion order, so the
# buggy code hands BOTH rows the same permutation and can satisfy at most one of
# the two assertions below. (Insertion order could only survive a set if all six
# strings collided into one slot, ~32**-5.)
LABELS_A = ['Entity', 'PhysicalObject', 'Vehiculo', 'Turismo', 'Matriculado', 'Incautado']
LABELS_B = ['Entity', 'Incautado', 'Matriculado', 'Turismo', 'Vehiculo', 'PhysicalObject']


class _CapturingOps:
    """Stands in for `driver.graph_operations_interface`, recording the node rows."""

    def __init__(self) -> None:
        self.node_rows: list[dict[str, Any]] = []

    async def episodic_node_save_bulk(self, _cls, driver, tx, episodes) -> None:
        return None

    async def node_save_bulk(self, _cls, driver, tx, nodes) -> None:
        self.node_rows = nodes

    async def episodic_edge_save_bulk(self, _cls, driver, tx, edges) -> None:
        return None

    async def edge_save_bulk(self, _cls, driver, tx, edges) -> None:
        return None


class _FakeDriver:
    """Minimal driver double: the bulk tx reads only `provider` and
    `graph_operations_interface`."""

    def __init__(self, ops: _CapturingOps) -> None:
        self.provider = GraphProvider.AGE
        self.graph_operations_interface = ops


def _node(uuid: str, labels: list[str]) -> EntityNode:
    node = EntityNode(uuid=uuid, name='VEHICULO UNO', group_id='labelorder', labels=list(labels))
    # Pre-set so the bulk path never calls an embedder (it only embeds when None).
    node.name_embedding = [0.0, 0.0, 0.0, 0.0]
    return node


async def _bulk_rows(nodes: list[EntityNode]) -> list[dict[str, Any]]:
    """The entity rows `add_nodes_and_edges_bulk_tx` hands to `node_save_bulk`."""
    ops = _CapturingOps()
    driver = _FakeDriver(ops)
    await add_nodes_and_edges_bulk_tx(None, [], [], nodes, [], None, driver)
    return ops.node_rows


@pytest.mark.asyncio
async def test_bulk_row_preserves_label_order_for_both_permutations():
    rows = await _bulk_rows([_node('labelorder-a', LABELS_A), _node('labelorder-b', LABELS_B)])
    by_uuid = {r['uuid']: r for r in rows}
    assert by_uuid['labelorder-a']['labels'] == LABELS_A
    assert by_uuid['labelorder-b']['labels'] == LABELS_B


@pytest.mark.asyncio
async def test_bulk_row_appends_entity_when_absent_and_keeps_the_leaf_last():
    rows = await _bulk_rows([_node('labelorder-c', ['PhysicalObject', 'Vehiculo'])])
    assert rows[0]['labels'] == ['PhysicalObject', 'Vehiculo', 'Entity']


@pytest.mark.asyncio
async def test_bulk_row_does_not_duplicate_entity_when_it_is_already_first():
    rows = await _bulk_rows([_node('labelorder-d', ['Entity', 'Vehiculo'])])
    assert rows[0]['labels'] == ['Entity', 'Vehiculo']


@pytest.mark.asyncio
async def test_bulk_row_does_not_duplicate_entity_when_it_is_already_last():
    rows = await _bulk_rows([_node('labelorder-e', ['Vehiculo', 'Entity'])])
    assert rows[0]['labels'] == ['Vehiculo', 'Entity']


@pytest.mark.asyncio
async def test_bulk_row_for_empty_labels_is_entity_only():
    rows = await _bulk_rows([_node('labelorder-f', [])])
    assert rows[0]['labels'] == ['Entity']


@pytest.mark.asyncio
async def test_bulk_row_repeated_label_keeps_the_first_occurrence():
    """Pinned semantics: `dict.fromkeys` keeps the FIRST occurrence, so a label
    list that repeats an entry collapses onto that first position. See
    `test_repeated_leaf_label_is_the_one_known_disagreement` in
    tests/driver/test_age_writer_label_determinism.py for the consequence."""
    rows = await _bulk_rows(
        [_node('labelorder-g', ['Entity', 'Vehiculo', 'PhysicalObject', 'Vehiculo'])]
    )
    assert rows[0]['labels'] == ['Entity', 'Vehiculo', 'PhysicalObject']


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'labels',
    [
        LABELS_A,                                            # Entity first, no repeats
        ['Entity', 'PhysicalObject', 'Vehiculo'],            # ordered projection shape
        ['Vehiculo', 'PhysicalObject', 'Entity'],            # Entity last
        ['Vehiculo', 'PhysicalObject'],                      # Entity absent -> appended
        ['Entity'],                                          # Entity only
        [],                                                  # empty -> Entity only
        ['Entity', 'Vehiculo', 'PhysicalObject', 'Vehiculo'],  # repeated leaf -> collapsed
    ],
)
async def test_bulk_row_label_membership_is_unchanged_for_the_other_backends(labels):
    """No-regression bar for Neo4j / FalkorDB / Neptune / Kuzu: their bulk queries
    apply EVERY entry of this list to the vertex (node_db_queries.py:204, :224,
    :260) or store it as a property, so only membership matters — and membership
    is exactly what an order-preserving dedupe leaves alone. Parametrized over the
    classes where the dedupe semantics actually changed (Entity absent -> appended;
    repeated entry -> collapsed), not just the identity class (ship review F2)."""
    rows = await _bulk_rows([_node('labelorder-h', labels)])
    assert set(rows[0]['labels']) == set(labels) | {'Entity'}
    assert len(rows[0]['labels']) == len(set(rows[0]['labels']))
