"""Response type definitions for Graphiti MCP Server."""

from typing import Any

from typing_extensions import TypedDict


class ErrorResponse(TypedDict):
    error: str


class SuccessResponse(TypedDict):
    message: str


class EpisodeAddedResponse(TypedDict):
    message: str
    node_uuids: list[str]
    edge_uuids: list[str]


class NodeResult(TypedDict):
    uuid: str
    name: str
    labels: list[str]
    created_at: str | None
    summary: str | None
    group_id: str
    attributes: dict[str, Any]


class NodeSearchResponse(TypedDict):
    message: str
    nodes: list[NodeResult]


class FactSearchResponse(TypedDict):
    message: str
    facts: list[dict[str, Any]]


class EpisodeSearchResponse(TypedDict):
    message: str
    episodes: list[dict[str, Any]]


class StatusResponse(TypedDict):
    status: str
    message: str


class EdgeResult(TypedDict):
    uuid: str
    name: str
    fact: str
    source_node_uuid: str
    target_node_uuid: str
    created_at: str | None
    valid_at: str | None
    invalid_at: str | None
    group_id: str


class CommunityResult(TypedDict):
    uuid: str
    name: str
    summary: str
    member_count: int
    group_id: str


class SearchResponse(TypedDict):
    message: str
    nodes: list[NodeResult]
    edges: list[EdgeResult]
    communities: list[CommunityResult]


class ExploreResponse(TypedDict):
    message: str
    center_node: NodeResult | None
    nodes: list[NodeResult]
    edges: list[EdgeResult]
    communities: list[CommunityResult]


class EpisodeContextResponse(TypedDict):
    message: str
    nodes: list[NodeResult]
    edges: list[EdgeResult]


class CommunityBuildResponse(TypedDict):
    message: str
    community_count: int
    communities: list[CommunityResult]


class SchemaNodeInfo(TypedDict, total=False):
    count: int
    attribute_keys: list[str]  # canonical ADR-019 R5: domain-queryable keys
    properties: list[str]      # full top-level keys (feeds cypher_quality schema_match)
    sampled: bool
    description: str
    sample_names: list[str]


class SchemaRelationshipInfo(TypedDict, total=False):
    count: int
    patterns: list[list[str]]
    description: str


class SchemaResponse(TypedDict, total=False):
    """Canonical get_schema payload (ADR-019 R5) + retained fork extras + error path.

    total=False so FastMCP's structuredContent preserves every returned key (undeclared keys are
    silently dropped from structuredContent) without requiring any of them.
    """
    type: str  # Always "schema"
    graph_name: str
    domain: str
    dialect: str                 # short id, e.g. "falkordb-cypher" | "age-opencypher"
    dialect_reference: str       # full dialect teaching text
    node_labels: dict[str, SchemaNodeInfo]
    relationship_types: dict[str, SchemaRelationshipInfo]
    cypher_reference: str        # back-compat alias of dialect_reference (one release)
    tool_capabilities: dict[str, Any]
    analysis_notes: list[str]
    error: str                   # ADR-015 R4 error path


class CypherResultResponse(TypedDict, total=False):
    """run_cypher envelope (ADR-019 R2) — success + error keys in one type.

    A single flat TypedDict (not a Success|Error union) so FastMCP's structuredContent stays flat
    (a union return would nest it under `result`). total=False so nothing is required and no
    returned key is dropped.
    """
    query: str
    auto_fixes: list[str]
    type: str  # "scalar", "tabular", "graph", "path", "error"
    result: Any
    columns: list[str]
    rows: list[list[Any]]
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    steps: list[dict[str, Any]]
    row_count: int
    truncated: bool
    limit_applied: int
    execution_ms: float
    cypher_quality: dict[str, Any]
    # error path (ADR-015 R4): top-level `error` is a STRING; hint + error_detail are additive.
    error: str
    hint: str
    error_detail: dict[str, str]
