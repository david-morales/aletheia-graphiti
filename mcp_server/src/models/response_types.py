"""Response type definitions for Graphiti MCP Server."""

from typing import Any

from typing_extensions import TypedDict


class ErrorResponse(TypedDict):
    error: str


class SuccessResponse(TypedDict):
    message: str


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
