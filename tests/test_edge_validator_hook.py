"""Tests for the validator hook inside resolve_extracted_edges."""

from unittest.mock import MagicMock

import pytest

from graphiti_core.validation import EdgeDecision, ValidationContext


class _DropAllValidator:
    name = "drop-all"

    def validate_edge(self, edge, source_node, target_node, context):
        return EdgeDecision(action="drop", reason="dropped by test validator")


class _KeepAllValidator:
    name = "keep-all"

    def validate_edge(self, edge, source_node, target_node, context):
        return EdgeDecision(action="keep")


class _WarnFirstValidator:
    name = "warn-first"
    called = 0

    def validate_edge(self, edge, source_node, target_node, context):
        _WarnFirstValidator.called += 1
        return EdgeDecision(action="warn", reason="test warn")


class _BrokenValidator:
    name = "broken"

    def validate_edge(self, edge, source_node, target_node, context):
        raise RuntimeError("oops")


def _make_edge(name: str, source_uuid: str, target_uuid: str):
    """Build a mock edge with the fields the hook reads."""
    edge = MagicMock()
    edge.uuid = f"edge-{name}-{source_uuid}-{target_uuid}"
    edge.source_node_uuid = source_uuid
    edge.target_node_uuid = target_uuid
    edge.name = name
    return edge


def _make_node(uuid: str, label: str):
    """Build a mock node with the fields the hook reads."""
    node = MagicMock()
    node.uuid = uuid
    node.labels = [label, "Entity"]
    node.name = f"{label}-{uuid}"
    return node


@pytest.fixture
def context():
    return ValidationContext(
        episode_uuid="ep1",
        knowledge_graph="test",
        edge_type_map={},
        entity_types={},
    )


@pytest.fixture
def basic_uuid_map():
    return {
        "s1": _make_node("s1", "Source"),
        "t1": _make_node("t1", "Target"),
    }


@pytest.mark.asyncio
async def test_apply_validators_empty_chain_passes_all_through(context, basic_uuid_map):
    """When validators list is empty, the helper must return the input unchanged."""
    from graphiti_core.utils.maintenance.edge_operations import _apply_edge_validators

    edges = [_make_edge("REL_A", "s1", "t1"), _make_edge("REL_B", "s1", "t1")]
    kept = await _apply_edge_validators(edges, basic_uuid_map, [], context)
    assert len(kept) == 2


@pytest.mark.asyncio
async def test_apply_validators_keeps_edge_when_chain_returns_keep(context, basic_uuid_map):
    from graphiti_core.utils.maintenance.edge_operations import _apply_edge_validators

    edges = [_make_edge("REL_A", "s1", "t1")]
    kept = await _apply_edge_validators(edges, basic_uuid_map, [_KeepAllValidator()], context)
    assert len(kept) == 1
    assert kept[0].name == "REL_A"


@pytest.mark.asyncio
async def test_apply_validators_drops_edge_when_chain_returns_drop(context, basic_uuid_map):
    from graphiti_core.utils.maintenance.edge_operations import _apply_edge_validators

    edges = [_make_edge("REL_A", "s1", "t1")]
    kept = await _apply_edge_validators(edges, basic_uuid_map, [_DropAllValidator()], context)
    assert kept == []


@pytest.mark.asyncio
async def test_apply_validators_warn_does_not_drop(context, basic_uuid_map):
    from graphiti_core.utils.maintenance.edge_operations import _apply_edge_validators

    edges = [_make_edge("REL_A", "s1", "t1")]
    _WarnFirstValidator.called = 0
    kept = await _apply_edge_validators(edges, basic_uuid_map, [_WarnFirstValidator()], context)
    assert len(kept) == 1
    assert _WarnFirstValidator.called == 1


@pytest.mark.asyncio
async def test_apply_validators_exception_fails_open(context, basic_uuid_map):
    """A validator that raises is caught; the edge is kept (fail-open)."""
    from graphiti_core.utils.maintenance.edge_operations import _apply_edge_validators

    edges = [_make_edge("REL_A", "s1", "t1")]
    kept = await _apply_edge_validators(edges, basic_uuid_map, [_BrokenValidator()], context)
    assert len(kept) == 1  # fail-open: edge kept despite exception


@pytest.mark.asyncio
async def test_apply_validators_missing_nodes_keeps_edge(context):
    """If source or target node is missing from uuid_map, the edge is kept (can't validate)."""
    from graphiti_core.utils.maintenance.edge_operations import _apply_edge_validators

    edges = [_make_edge("REL_A", "missing_src", "missing_tgt")]
    # uuid_map is empty → source/target lookup returns None
    kept = await _apply_edge_validators(edges, {}, [_DropAllValidator()], context)
    # With no nodes to pass to the validator, the hook must keep the edge
    # (let downstream handle it). Otherwise we'd drop legitimate edges that
    # the uuid_map hasn't fully populated yet.
    assert len(kept) == 1


@pytest.mark.asyncio
async def test_apply_validators_drop_shortcircuits_remaining_validators(context, basic_uuid_map):
    """Once a drop decision is returned, remaining validators in the chain don't run."""
    from graphiti_core.utils.maintenance.edge_operations import _apply_edge_validators

    called = []

    class Drop:
        name = "drop"
        def validate_edge(self, edge, source_node, target_node, context):
            called.append("drop")
            return EdgeDecision(action="drop", reason="first")

    class ShouldNotRun:
        name = "should-not-run"
        def validate_edge(self, edge, source_node, target_node, context):
            called.append("should-not-run")
            return EdgeDecision(action="keep")

    edges = [_make_edge("REL_A", "s1", "t1")]
    kept = await _apply_edge_validators(edges, basic_uuid_map, [Drop(), ShouldNotRun()], context)
    assert kept == []
    assert called == ["drop"]  # ShouldNotRun never executed
