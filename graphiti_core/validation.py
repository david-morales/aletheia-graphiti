"""Contract types for the edge validator hook.

Aletheia (or any other graphiti consumer) implements validators conforming
to the EdgeValidator Protocol and passes them to Graphiti via the
edge_validators constructor parameter. Graphiti calls validators inside
resolve_extracted_edges before its own rename-to-RELATES_TO fallback.

The validator contract is intentionally minimal: KEEP, DROP, or WARN.
Rename is handled by graphiti's own fallback; validators do not influence
it directly.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from graphiti_core.edges import EntityEdge
    from graphiti_core.nodes import EntityNode


EdgeAction = Literal["keep", "drop", "warn"]


@dataclass
class EdgeDecision:
    """The outcome of validating one extracted edge.

    Attributes
    ----------
    action : Literal["keep", "drop", "warn"]
        - "keep": edge proceeds unchanged.
        - "drop": edge is removed from the extracted set; no further validators see it.
        - "warn": edge is kept but a metric event is logged with the given reason.
    reason : str
        Human-readable explanation used by metrics and logs. Required for "drop"
        and "warn"; optional for "keep".
    """

    action: EdgeAction
    reason: str = ""


@dataclass
class ValidationContext:
    """Context passed to each validator call.

    Carries enough information for validators to make schema-based decisions
    without needing access to the Graphiti instance itself.
    """

    episode_uuid: str
    knowledge_graph: str
    edge_type_map: dict[tuple[str, str], list[str]] = field(default_factory=dict)
    entity_types: dict[str, type] = field(default_factory=dict)


@runtime_checkable
class EdgeValidator(Protocol):
    """Protocol that validator implementations must satisfy.

    Attributes
    ----------
    name : str
        A stable identifier for metric events (e.g., "schema-type", "self-loop").

    Methods
    -------
    validate_edge(edge, source_node, target_node, context) -> EdgeDecision
        Called once per extracted edge. Must return an EdgeDecision.
        Must not raise — if it does, graphiti fails open (logs the error,
        treats the edge as KEEP, and records decision='error' in metrics).
    """

    name: str

    def validate_edge(
        self,
        edge: "EntityEdge",
        source_node: "EntityNode",
        target_node: "EntityNode",
        context: ValidationContext,
    ) -> EdgeDecision: ...


@runtime_checkable
class EdgeValidationObserver(Protocol):
    """Protocol for observing validator decisions.

    Consumers (e.g., aletheia's ValidationMetricsWriter) implement this
    to record every keep/drop/warn/error decision for observability.
    """

    def record(
        self,
        edge: "EntityEdge",
        source_node: "EntityNode",
        target_node: "EntityNode",
        context: ValidationContext,
        validator_name: str,
        decision: EdgeDecision,
        dry_run: bool = False,
    ) -> None: ...

    def record_error(
        self,
        edge: "EntityEdge",
        source_node: "EntityNode",
        target_node: "EntityNode",
        context: ValidationContext,
        validator_name: str,
        error_message: str,
        dry_run: bool = False,
    ) -> None: ...
