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
    """get_status. `version` is the CONNECTOR's build (A-D11), which since the SDK
    2.x migration is also what `serverInfo.version` carries — the two agree rather
    than the handshake reporting the SDK. Kept because it answers a different
    question than the handshake does, on a probe operators already make.

    Deliberately `total=True`: all three fields are always returned, so the SDK
    never injects None into them (see test_output_field_nullability)."""
    status: str
    message: str
    version: str


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
# SchemaResponse, graph_query -> CypherResultResponse), so MCPServer validates the
# structuredContent against their outputSchema over the wire. MCPServer (mcp
# <=1.28.1) builds a pydantic model from a total=False TypedDict with EVERY
# optional field defaulted to None, then convert_result dumps it WITHOUT
# exclude_unset — so any field ABSENT from a returned payload is emitted as None
# in structuredContent. A non-nullable field type then makes the lowlevel server
# reject that None ("None is not of type 'string'"), which broke EVERY
# over-the-wire caller of get_schema (`error`) and graph_query (`truncated`, ...).
# Declaring the fields nullable lets the injected None validate. The fork's
# capability-level tests call the functions directly and never hit this path;
# test_typed_tool_outputs_survive_mcpserver_output_validation guards it.
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
    # True when the label is HIERARCHY-ONLY: censusable and searchable, but NOT a
    # storage type — no vertex is stored under it, `MATCH (n:Label)` reaches
    # nothing, and it can appear in no relationship pattern. Schema and graph
    # views should render STORAGE TYPES ONLY; an unfiltered view draws these as
    # disconnected nodes. Absent/None means "storage type" (always so on
    # FalkorDB, where every censused label is stored).
    hierarchy: bool | None


class SchemaRelationshipInfo(TypedDict, total=False):
    count: int | None
    patterns: list[list[str]] | None
    description: str | None


class SchemaResponse(TypedDict, total=False):
    """Canonical get_schema payload (ADR-019 R5) + retained fork extras + error path.

    total=False so MCPServer's structuredContent preserves every returned key (undeclared keys are
    silently dropped from structuredContent) without requiring any of them. Every field is
    nullable — see the module note above (MCPServer injects None for absent optional fields).
    """
    type: str | None  # Always "schema"
    graph_name: str | None
    domain: str | None
    dialect: str | None                 # short id, e.g. "falkordb-cypher" | "age-opencypher"
    dialect_reference: str | None       # full dialect teaching text
    # The map this backend nests its DOMAIN fields in (AGE: "attributes"), or
    # null when they are top-level (FalkorDB / openCypher). Read it with
    # `node_labels[*].attribute_keys`: those keys are addressed as
    # `n.<attribute_container>.<key>` when it is set and as `n.<key>` when it is
    # not. Announced so no consumer has to map a dialect id to a storage shape.
    attribute_container: str | None
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
    # The node's single MOST SPECIFIC label — the one to type/colour by. Producer-
    # announced precisely because `labels` is unordered: deriving a type from its
    # first element collapsed 16 leaf types into 3 supertypes on the AGE arm.
    # Nullable per the module rule: null when no non-internal label exists.
    leaf: str | None
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

    `leaf` exists so consumers do not have to: it is the node's single most
    specific label, announced by the producer. TYPE AND COLOUR BY `leaf`, and
    fall back to set-filtering `labels` only when it is null. Deriving a type from
    the hierarchy's first element is the concrete bug this closes — on the AGE arm
    it collapsed 16 leaf types into 3 supertypes (Actor 94 / Event 67 /
    Ubicacion 39) and coloured the two backends differently for the same graph.

    Same nullability rule as the module note above — every field `X | None`.
    """
    type: str | None           # always "subgraph"
    graph_name: str | None
    nodes: list[SubgraphNode] | None
    edges: list[SubgraphEdge] | None
    error: str | None          # ADR-015 R4 error path


# ---------------------------------------------------------------------------
# ONE envelope family (A-D7). Each tool returns a single flat TypedDict that
# folds its success keys and the ADR-015 R4 error path into one type.
#
# WHY NOT `X | ErrorResponse`: MCPServer cannot publish a union as a structured
# output, so it wraps it — `structuredContent` arrives as `{"result": {...}}`.
# That split the surface into two envelope families (four flat tools, eleven
# wrapped), and made the ADR-015 R4 client rule `if "error" in result` true for
# only the flat four. The fork had already reasoned this out for `graph_query`
# (see CypherResultResponse below) and never applied it to the rest.
#
# The RETURNED payloads are unchanged — success returns carry only success keys,
# error returns carry only `error` (+ the permitted `hint`). Consumers reading
# the unstructured content see byte-identical dicts; `structuredContent` simply
# stops nesting. Same nullability rule as the module note above: total=False and
# every field `X | None`, because MCPServer injects None for absent fields.
# ---------------------------------------------------------------------------


class MutationResult(TypedDict, total=False):
    """delete_entity_edge / delete_episode / clear_graph — a message or an error."""
    message: str | None
    error: str | None


class AddMemoryResult(TypedDict, total=False):
    """add_memory. `node_uuids`/`edge_uuids` are present only in sync mode; the
    queued single-episode path returns the message alone."""
    message: str | None
    node_uuids: list[str] | None
    edge_uuids: list[str] | None
    error: str | None


class EpisodeContextResult(TypedDict, total=False):
    """get_episode_context — what the given episodes extracted."""
    message: str | None
    nodes: list[NodeResult] | None
    edges: list[EdgeResult] | None
    error: str | None


class CommunityBuildResult(TypedDict, total=False):
    """build_communities."""
    message: str | None
    community_count: int | None
    communities: list[CommunityResult] | None
    error: str | None


class EpisodeListResult(TypedDict, total=False):
    """get_episodes."""
    message: str | None
    episodes: list[dict[str, Any]] | None
    error: str | None


class SearchResult(TypedDict, total=False):
    """search / search_ontology."""
    message: str | None
    nodes: list[NodeResult] | None
    edges: list[EdgeResult] | None
    communities: list[CommunityResult] | None
    execution_ms: float | None
    error: str | None


class ExploreResult(TypedDict, total=False):
    """explore_entity. A miss is an answer, not an error: `center_node` is null and
    `message` says so, with `error` absent."""
    message: str | None
    center_node: NodeResult | None
    nodes: list[NodeResult] | None
    edges: list[EdgeResult] | None
    communities: list[CommunityResult] | None
    error: str | None


class OntologyClassEntry(TypedDict, total=False):
    """One ontology class, as served by the structure and documentation tiers.

    The two tiers differ by DEPTH, not by shape: the structure tier omits
    `identity`/`properties` (it is the cheap surface map) and the documentation
    tier fills them in. `source_entity`/`target_entity` are present only when
    `ontology_type == 'relationship_class'`.

    THE FOUR `Any` FIELDS ARE AUTHOR-OWNED, AND DELIBERATELY UNTYPED. They are
    whatever JSON the ontology author stored, handed back verbatim by
    `_parse_properties`. Typing them was a real regression: pydantic validates
    this payload inside `FuncMetadata.convert_result`, which the lowlevel server
    calls AFTER the tool returned — outside its `try/except` — so a class whose
    `properties` is a list of strings, or whose `required` is "sometimes", turned
    a working connector into one serving PROTOCOL errors, invisibly to the tool's
    own logging. Constraining them also silently dropped every key the model did
    not name, including `inherited_from`, which aletheia's own ontology generator
    emits on every property.

    A-D6 was about the TOP LEVEL publishing `{"type":"object"}` and hiding the
    error path. That goal is met by the typed keys here; it never required
    policing content the connector does not own.

    Conventional `properties` entry (documentation tier), by convention only:
    `{name, label, range, comment, required, inherited_from}`.
    """
    name: str | None
    ontology_type: str | None
    summary: str | None
    source_entity: str | None
    target_entity: str | None
    identity: bool | None
    # author-owned — see the note above
    alt_labels: Any
    inherits_from: Any
    examples: Any
    properties: Any


class OntologyStructureResponse(TypedDict, total=False):
    """get_ontology_structure payload — the whole-ontology SURFACE map."""
    ontology_graph: str | None
    entity_classes: list[OntologyClassEntry] | None
    relationship_classes: list[OntologyClassEntry] | None
    error: str | None  # ADR-015 R4 error path


class OntologyDocumentationResponse(TypedDict, total=False):
    """get_ontology_documentation payload — the FULL reference (large).

    Same three keys as the structure tier; the entries carry full prose,
    per-class property definitions and the identity flag.
    """
    ontology_graph: str | None
    entity_classes: list[OntologyClassEntry] | None
    relationship_classes: list[OntologyClassEntry] | None
    error: str | None  # ADR-015 R4 error path


class OntologyRelationshipRef(TypedDict, total=False):
    """A relationship touching the explored class. `source` on incoming,
    `target` on outgoing — the other end is the class itself."""
    name: str | None
    source: str | None
    target: str | None
    summary: str | None


class OntologyClassRef(TypedDict, total=False):
    """A neighbouring or related class, with a one-line preview.
    `via` names the relationship that reached it (neighbors only)."""
    name: str | None
    summary_line: str | None
    via: str | None


class OntologyRelationshipSets(TypedDict, total=False):
    outgoing: list[OntologyRelationshipRef] | None
    incoming: list[OntologyRelationshipRef] | None


class OntologyHierarchy(TypedDict, total=False):
    parents: list[OntologyClassRef] | None
    children: list[OntologyClassRef] | None
    siblings: list[OntologyClassRef] | None


class OntologyClassContextResponse(TypedDict, total=False):
    """explore_ontology payload — ONE class in full context."""
    center: OntologyClassEntry | None
    relationships: OntologyRelationshipSets | None
    hierarchy: OntologyHierarchy | None
    neighbors: list[OntologyClassRef] | None
    error: str | None  # ADR-015 R4 error path


class ProfileGraphResponse(TypedDict, total=False):
    """profile_data payload — property coverage, samples, languages, cardinality.

    The three sections stay `dict[str, Any]`-shaped inside: they are keyed by the
    graph's own labels and relationship types, which no static type can enumerate.
    Typing the TOP level is what consumers needed — the empty
    `{"type":"object"}` schema this tool used to publish constrained nothing at all
    and hid the ADR-015 R4 error path.
    """
    entity_profiles: dict[str, Any] | None
    relationship_profiles: dict[str, Any] | None
    language_summary: dict[str, Any] | None
    error: str | None  # ADR-015 R4 error path


class CypherResultResponse(TypedDict, total=False):
    """graph_query envelope (ADR-019 R2) — success + error keys in one type.

    A single flat TypedDict (not a Success|Error union) so MCPServer's structuredContent stays flat
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
