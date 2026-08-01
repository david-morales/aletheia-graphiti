"""Regression: both AGE writer paths must choose the SAME vertex label.

AGE stores ONE label per vertex, chosen by `_node_label` = the last
identifier-safe non-'Entity' entry of the label list. Two writers reach that
chooser with two different lists:

  * projection path — `EntityNode.save` -> `AGEGraphOperations.node_save`
    (age_graph_operations.py:234-235), which passes the node's RAW ordered
    `labels`;
  * bulk / narrative path — `add_nodes_and_edges_bulk_tx` (bulk_utils.py:153)
    -> `node_save_bulk` (age_graph_operations.py:695-710) ->
    `_write_entity_from_fields` (:571-583, MERGE at :600), which passes the
    row's `labels`.

While the row was built with `list(set(...))` the two lists differed in order and
the picks could differ. `MERGE (n:<label> {uuid: ...})` is label-scoped on AGE, so
a differing pick created a SECOND vertex for the same uuid.

The offline tests below drive the real bulk path against a capturing double and
compare its label pick with the projection path's. The live test appended in
Task 3 proves the whole thing end to end on the AGE bed.
"""

import pytest

from graphiti_core.driver.graph_operations.age_graph_operations import _node_label

# Reuse the bulk-path harness from the bulk-row order regression instead of
# duplicating the driver double: same fixture, different question.
from tests.utils.maintenance.test_bulk_label_order import LABELS_A, LABELS_B, _bulk_rows, _node


@pytest.mark.asyncio
async def test_both_paths_agree_for_two_permutations_of_the_same_labels():
    """The deterministic form of the bug. Two nodes carry the same six labels in
    different orders, so their projection picks differ ('Incautado' vs
    'PhysicalObject'). A set hands both bulk rows the SAME permutation, hence the
    same bulk pick — which can match at most one of the two projection picks."""
    assert _node_label(LABELS_A) == 'Incautado'
    assert _node_label(LABELS_B) == 'PhysicalObject'

    rows = await _bulk_rows([_node('crosspath-a', LABELS_A), _node('crosspath-b', LABELS_B)])
    by_uuid = {r['uuid']: r for r in rows}
    assert _node_label(by_uuid['crosspath-a']['labels']) == 'Incautado'
    assert _node_label(by_uuid['crosspath-b']['labels']) == 'PhysicalObject'


@pytest.mark.parametrize(
    'labels',
    [
        ['Entity', 'PhysicalObject', 'Vehiculo'],
        ['PhysicalObject', 'Vehiculo', 'Entity'],
        ['PhysicalObject', 'Vehiculo'],
        ['Entity', 'Incautado', 'Matriculado', 'Turismo', 'Vehiculo', 'PhysicalObject'],
        ['Entity'],
        [],
    ],
)
@pytest.mark.asyncio
async def test_writer_paths_agree_for_every_distinct_label_list(labels):
    node = _node('crosspath-' + ('-'.join(labels) or 'empty'), labels)
    rows = await _bulk_rows([node])
    assert _node_label(rows[0]['labels']) == _node_label(node.labels)


@pytest.mark.asyncio
async def test_repeated_leaf_label_is_the_one_known_disagreement():
    """A label list that REPEATS its leaf earlier is the single input class where
    the two paths still differ: keep-first dedupe collapses the leaf onto its
    first position, so the bulk pick becomes the preceding label. Nothing in
    graphiti_core emits repeated labels (dedup_helpers.py:210-215 already dedupes
    keep-first, node_operations.py:364 works on sets, and the projection pipeline
    renders an ontology hierarchy leaf-last), so this is pinned as known behaviour
    rather than fixed here — the sibling-merge repair CLI remains the backstop. If
    a future change makes the dedupe keep-last, this test is the one to revisit."""
    node = _node('crosspath-dup', ['Entity', 'Vehiculo', 'PhysicalObject', 'Vehiculo'])
    rows = await _bulk_rows([node])
    assert _node_label(node.labels) == 'Vehiculo'
    assert _node_label(rows[0]['labels']) == 'PhysicalObject'
