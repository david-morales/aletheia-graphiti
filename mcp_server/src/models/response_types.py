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


class SchemaNodeInfo(TypedDict):
    count: int
    properties: list[str]
    sampled: bool


class SchemaRelationshipInfo(TypedDict):
    count: int
    patterns: list[list[str]]


class SchemaResponse(TypedDict):
    type: str  # Always "schema"
    graph_name: str
    domain: str
    node_labels: dict[str, SchemaNodeInfo]
    relationship_types: dict[str, SchemaRelationshipInfo]


class CypherResultResponse(TypedDict, total=False):
    query: str
    auto_fixes: list[str]
    type: str  # "scalar", "tabular", "graph", "path", "error"
    result: Any
    columns: list[str]
    rows: list[list[Any]]
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    steps: list[dict[str, Any]]
    error: dict[str, str]
    row_count: int
    truncated: bool
    limit_applied: int
    execution_ms: float
