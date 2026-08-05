"""Response type definitions for Graphiti MCP Server."""

from typing import Any

from typing_extensions import NotRequired, TypedDict


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
    execution_ms: NotRequired[float]


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


# ---------------------------------------------------------------------------
# Typed tool-output TypedDicts (ADR-019 R2). EVERY field is `X | None`.
#
# WHY nullable: these are declared as tool return annotations (get_schema ->
# SchemaResponse, run_cypher -> CypherResultResponse), so FastMCP validates the
# structuredContent against their outputSchema over the wire. FastMCP (mcp
# <=1.28.1) builds a pydantic model from a total=False TypedDict with EVERY
# optional field defaulted to None, then convert_result dumps it WITHOUT
# exclude_unset — so any field ABSENT from a returned payload is emitted as None
# in structuredContent. A non-nullable field type then makes the lowlevel server
# reject that None ("None is not of type 'string'"), which broke EVERY
# over-the-wire caller of get_schema (`error`) and run_cypher (`truncated`, ...).
# Declaring the fields nullable lets the injected None validate. The fork's
# capability-level tests call the functions directly and never hit this path;
# test_typed_tool_outputs_survive_fastmcp_output_validation guards it.
#
# DO NOT drop the `| None` — it is load-bearing. Consumers read the UNSTRUCTURED
# content (the original payload, no injected None), so nullability here is
# invisible to them (see aletheia per_call_client._parse_mcp_response).
# ---------------------------------------------------------------------------


class SchemaNodeInfo(TypedDict, total=False):
    count: int | None
    attribute_keys: list[str] | None  # canonical ADR-019 R5: domain-queryable keys
    properties: list[str] | None      # full top-level keys (feeds cypher_quality schema_match)
    sampled: bool | None
    description: str | None
    sample_names: list[str] | None


class SchemaRelationshipInfo(TypedDict, total=False):
    count: int | None
    patterns: list[list[str]] | None
    description: str | None


class SchemaResponse(TypedDict, total=False):
    """Canonical get_schema payload (ADR-019 R5) + retained fork extras + error path.

    total=False so FastMCP's structuredContent preserves every returned key (undeclared keys are
    silently dropped from structuredContent) without requiring any of them. Every field is
    nullable — see the module note above (FastMCP injects None for absent optional fields).
    """
    type: str | None  # Always "schema"
    graph_name: str | None
    domain: str | None
    dialect: str | None                 # short id, e.g. "falkordb-cypher" | "age-opencypher"
    dialect_reference: str | None       # full dialect teaching text
    node_labels: dict[str, SchemaNodeInfo] | None
    relationship_types: dict[str, SchemaRelationshipInfo] | None
    cypher_reference: str | None        # back-compat alias of dialect_reference (one release)
    tool_capabilities: dict[str, Any] | None
    analysis_notes: list[str] | None
    error: str | None                   # ADR-015 R4 error path


class SubgraphNode(TypedDict, total=False):
    uuid: str | None
    name: str | None
    # Full logical hierarchy on BOTH flavours (internal labels included). UNORDERED —
    # see the SubgraphResponse docstring: set-filter, never index positionally.
    labels: list[str] | None
    created_at: str | None
    summary: str | None
    group_id: str | None


class SubgraphEdge(TypedDict, total=False):
    uuid: str | None
    name: str | None           # type(r)
    fact: str | None
    source_node_uuid: str | None
    target_node_uuid: str | None
    created_at: str | None


class SubgraphResponse(TypedDict, total=False):
    """sample_subgraph payload (Step 2 UI alignment): a flavour-normalized node/edge sample.

    `labels` carries the full logical hierarchy on both flavours (FalkorDB:
    labels(n); AGE: the stored n.labels list).

    The list is UNORDERED — its order is NOT a contract on either flavour, and
    position carries no meaning. Live bench evidence: most rows come back
    ['Entity', 'Actor', 'Persona'] (internal label first), but ['Droga', 'Entity']
    puts the domain label first. Consumers MUST set-filter the internal labels
    (Entity, Episodic, ...) to find the domain type — never index positionally
    (`labels[0]`, `labels[-1]`) and never assume general-to-specific ordering.

    Same nullability rule as the module note above — every field `X | None`.
    """
    type: str | None           # always "subgraph"
    graph_name: str | None
    nodes: list[SubgraphNode] | None
    edges: list[SubgraphEdge] | None
    error: str | None          # ADR-015 R4 error path


class CypherResultResponse(TypedDict, total=False):
    """run_cypher envelope (ADR-019 R2) — success + error keys in one type.

    A single flat TypedDict (not a Success|Error union) so FastMCP's structuredContent stays flat
    (a union return would nest it under `result`). total=False so nothing is required and no
    returned key is dropped. Every field is nullable — see the module note above.
    """
    query: str | None
    auto_fixes: list[str] | None
    type: str | None  # "scalar", "tabular", "graph", "path", "error"
    result: Any
    columns: list[str] | None
    rows: list[list[Any]] | None
    nodes: list[dict[str, Any]] | None
    edges: list[dict[str, Any]] | None
    steps: list[dict[str, Any]] | None
    row_count: int | None
    truncated: bool | None
    limit_applied: int | None
    execution_ms: float | None
    cypher_quality: dict[str, Any] | None
    # error path (ADR-015 R4): top-level `error` is a STRING; hint + error_detail are additive.
    error: str | None
    hint: str | None
    error_detail: dict[str, str] | None
