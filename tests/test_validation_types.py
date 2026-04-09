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


def _mock_clients():
    """Build MagicMocks that pass Pydantic's isinstance validation for GraphitiClients."""
    from unittest.mock import MagicMock

    from graphiti_core.cross_encoder.client import CrossEncoderClient
    from graphiti_core.driver.driver import GraphDriver
    from graphiti_core.embedder import EmbedderClient
    from graphiti_core.llm_client import LLMClient
    from graphiti_core.tracer import Tracer

    mock_driver = MagicMock(spec=GraphDriver)
    mock_driver.provider = "falkordb"
    return {
        "graph_driver": mock_driver,
        "llm_client": MagicMock(spec=LLMClient),
        "embedder": MagicMock(spec=EmbedderClient),
        "cross_encoder": MagicMock(spec=CrossEncoderClient),
        "tracer": MagicMock(spec=Tracer),
    }


def test_graphiti_accepts_edge_validators_parameter():
    """Graphiti.__init__ accepts edge_validators as a keyword argument.

    Default (None) must keep _edge_validators as an empty list,
    preserving backward compatibility.
    """
    from graphiti_core.graphiti import Graphiti

    mocks = _mock_clients()

    # Default: edge_validators is None → stored as empty list
    g = Graphiti(
        graph_driver=mocks["graph_driver"],
        llm_client=mocks["llm_client"],
        embedder=mocks["embedder"],
        cross_encoder=mocks["cross_encoder"],
    )
    assert g._edge_validators == []
    assert g.clients.edge_validators == []


def test_graphiti_stores_explicit_edge_validators():
    """When edge_validators is passed, Graphiti stores them on _edge_validators and clients."""
    from graphiti_core.graphiti import Graphiti
    from graphiti_core.validation import EdgeDecision

    class DummyValidator:
        name = "dummy"

        def validate_edge(self, edge, source_node, target_node, context):
            return EdgeDecision(action="keep")

    mocks = _mock_clients()

    g = Graphiti(
        graph_driver=mocks["graph_driver"],
        llm_client=mocks["llm_client"],
        embedder=mocks["embedder"],
        cross_encoder=mocks["cross_encoder"],
        edge_validators=[DummyValidator()],
    )
    assert len(g._edge_validators) == 1
    assert g._edge_validators[0].name == "dummy"
    assert g.clients.edge_validators == g._edge_validators


def test_graphiti_clients_has_edge_validators_field():
    """GraphitiClients Pydantic model exposes edge_validators as a field."""
    from graphiti_core.graphiti_types import GraphitiClients

    field_names = set(GraphitiClients.model_fields.keys())
    assert "edge_validators" in field_names


def test_graphiti_clients_default_edge_validators_is_empty_list():
    """GraphitiClients constructor defaults edge_validators to []."""
    from graphiti_core.graphiti_types import GraphitiClients

    mocks = _mock_clients()
    clients = GraphitiClients(
        driver=mocks["graph_driver"],
        llm_client=mocks["llm_client"],
        embedder=mocks["embedder"],
        cross_encoder=mocks["cross_encoder"],
        tracer=mocks["tracer"],
    )
    assert clients.edge_validators == []


def test_observer_protocol_is_runtime_checkable():
    from graphiti_core.validation import EdgeValidationObserver

    class _StubObserver:
        def record(self, edge, source_node, target_node, context, validator_name, decision, dry_run=False):
            pass

        def record_error(self, edge, source_node, target_node, context, validator_name, error_message, dry_run=False):
            pass

    assert isinstance(_StubObserver(), EdgeValidationObserver)


def test_observer_protocol_rejects_non_conforming():
    from graphiti_core.validation import EdgeValidationObserver

    class _Bad:
        pass

    assert not isinstance(_Bad(), EdgeValidationObserver)
