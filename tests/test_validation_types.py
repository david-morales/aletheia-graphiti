"""Tests for the validator contract types exposed by graphiti_core.validation."""

import pytest


def test_edge_decision_keep():
    from graphiti_core.validation import EdgeDecision
    d = EdgeDecision(action="keep")
    assert d.action == "keep"
    assert d.reason == ""


def test_edge_decision_drop_with_reason():
    from graphiti_core.validation import EdgeDecision
    d = EdgeDecision(action="drop", reason="illegal type pair")
    assert d.action == "drop"
    assert d.reason == "illegal type pair"


def test_edge_decision_warn_with_reason():
    from graphiti_core.validation import EdgeDecision
    d = EdgeDecision(action="warn", reason="unknown edge type")
    assert d.action == "warn"
    assert d.reason == "unknown edge type"


def test_edge_decision_action_values():
    """Document the contract: action is one of keep, drop, warn."""
    from graphiti_core.validation import EdgeDecision
    for action in ("keep", "drop", "warn"):
        d = EdgeDecision(action=action)
        assert d.action == action


def test_edge_validator_protocol_is_importable():
    from graphiti_core.validation import EdgeValidator
    assert EdgeValidator is not None


def test_validation_context_fields():
    from graphiti_core.validation import ValidationContext
    ctx = ValidationContext(
        episode_uuid="abc-123",
        knowledge_graph="test_graph",
        edge_type_map={("A", "B"): ["X"]},
        entity_types={"A": dict, "B": dict},
    )
    assert ctx.episode_uuid == "abc-123"
    assert ctx.knowledge_graph == "test_graph"
    assert ctx.edge_type_map == {("A", "B"): ["X"]}
    assert ctx.entity_types == {"A": dict, "B": dict}


def test_validation_context_default_fields():
    from graphiti_core.validation import ValidationContext
    ctx = ValidationContext(episode_uuid="x", knowledge_graph="y")
    assert ctx.edge_type_map == {}
    assert ctx.entity_types == {}


def test_validator_protocol_can_be_implemented():
    """A concrete class that conforms to the protocol should be acceptable."""
    from graphiti_core.validation import EdgeValidator, EdgeDecision

    class AlwaysKeep:
        name = "always-keep"

        def validate_edge(self, edge, source_node, target_node, context):
            return EdgeDecision(action="keep")

    v: EdgeValidator = AlwaysKeep()
    assert v.name == "always-keep"


def test_protocol_runtime_checkable():
    """The protocol should be runtime-checkable so we can validate at construction time."""
    from graphiti_core.validation import EdgeValidator

    class GoodValidator:
        name = "good"
        def validate_edge(self, edge, source_node, target_node, context):
            pass

    class BadValidator:
        # missing name and validate_edge
        pass

    assert isinstance(GoodValidator(), EdgeValidator)
    assert not isinstance(BadValidator(), EdgeValidator)
