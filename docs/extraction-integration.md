# Extraction Integration Contract

This document is the integration contract for downstream services (initially `aletheia-extraction`) that consume the graphiti MCP server. It covers connection setup, the 3 tools downstream services consume, response shapes, error modes, and a placeholder shim recipe for development without a live MCP server.

**Status:** stable contract for `aletheia-extraction`'s use as of 2026-04-28. New tool additions follow a similar pattern; existing tool signatures and response shapes are stable.

**Audience:** developers building or maintaining downstream services that consume graphiti MCP.

---

## 1. Connection setup

The graphiti MCP server runs as a standalone service. Downstream services connect via **HTTP-SSE** (recommended for service-to-service) or stdio (supported but not recommended for service mesh patterns).

### 1.1 HTTP-SSE setup

```python
import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client

async def main() -> None:
    mcp_url = "http://graphiti:8000/sse"  # or wherever the MCP server listens
    async with sse_client(mcp_url) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools = await session.list_tools()
            # ... call tools via session.call_tool(name, arguments)

asyncio.run(main())
```

**MCP SDK version:** the contract is tested against `mcp>=1.9.4` (the version pinned in `mcp_server/pyproject.toml`). Downstream services should pin a compatible version.

### 1.2 stdio (alternative, not recommended for service mesh)

stdio is supported by the same FastMCP-backed server but requires the downstream service to spawn the graphiti MCP as a subprocess. This pattern fits Claude Desktop integrations but not service-to-service. See the upstream MCP SDK docs for the stdio client pattern.

---

## 2. `get_ontology_structure()`

### Signature

```python
async def get_ontology_structure() -> dict[str, Any]:
    """Return the full ontology class hierarchy in one call.

    Returns entity classes (with inheritance, alt_labels, descriptions)
    and relationship classes (with source/target constraints).
    """
```

The tool takes **no arguments**. The destination ontology graph is determined server-side from the connector's configured `ontology_graph` (set at server startup). Note: this tool is registered dynamically via `register_dynamic_tools()` (which calls `mcp.add_tool(get_ontology_structure)` at server startup), not via the `@mcp.tool()` decorator — but the wire-level contract is identical.

### Response shape

On success, the response is a `dict[str, Any]` with these keys:

| Key | Type | Description |
|---|---|---|
| `ontology_graph` | `str` | Name of the ontology graph (the FalkorDB graph name configured server-side). May be empty string if unset. |
| `entity_classes` | `list[dict]` | All ontology entity classes. Each item shape below. |
| `relationship_classes` | `list[dict]` | All ontology relationship classes. Each item has the entity-class fields plus `source_entity` and `target_entity`. |

Each `entity_classes[]` item:

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Class name (the ontology label). |
| `ontology_type` | `str` | Discriminator — typically `"class"` or `"abstract_class"` for entity classes; `"relationship_class"` for relationships. |
| `summary` | `str` | Free-form description from the ontology. |
| `alt_labels` | `list[str]` \| `str` | Alternative labels. May be empty. |
| `inherits_from` | `str` | Parent class name; empty if root. |
| `examples` | `list[str]` \| `str` | Example instances. May be empty. |

Each `relationship_classes[]` item has all the entity-class fields plus:

| Field | Type | Description |
|---|---|---|
| `source_entity` | `str` | Required source entity class. |
| `target_entity` | `str` | Required target entity class. |

On error, the response is `{"error": "<message>"}`. Possible errors include:
- `"Service not initialized. Please wait for startup to complete."` — server not yet ready.
- `"No ontology graph configured for this connector."` — connector lacks an `ontology_graph`.
- `"Failed to retrieve ontology structure: <exception>"` — driver-level failure.

### Worked example

```json
{
  "ontology_graph": "aviation_ontology",
  "entity_classes": [
    {
      "name": "Aircraft",
      "ontology_type": "class",
      "summary": "An aircraft involved in a safety occurrence.",
      "alt_labels": ["airplane", "plane"],
      "inherits_from": "Vehicle",
      "examples": ["PH-KZB", "G-EUUK"]
    },
    {
      "name": "Operator",
      "ontology_type": "class",
      "summary": "The operating organization of an aircraft.",
      "alt_labels": [],
      "inherits_from": "Organization",
      "examples": []
    }
  ],
  "relationship_classes": [
    {
      "name": "OPERATED_BY",
      "ontology_type": "relationship_class",
      "summary": "An aircraft is operated by an operator.",
      "alt_labels": [],
      "inherits_from": "",
      "examples": [],
      "source_entity": "Aircraft",
      "target_entity": "Operator"
    }
  ]
}
```

### OntologySpec adapter recipe

Downstream extraction code typically wraps the raw response in a typed `OntologySpec` for use as a Pydantic model in prompts and validators:

```python
from typing import Any
from pydantic import BaseModel

class OntologyClass(BaseModel):
    name: str
    ontology_type: str
    summary: str
    alt_labels: list[str] = []
    inherits_from: str = ""
    examples: list[str] = []

class RelationshipClass(OntologyClass):
    source_entity: str
    target_entity: str

class OntologySpec(BaseModel):
    target_graph: str          # supplied by caller, not from MCP
    ontology_graph: str        # from MCP response
    entity_classes: list[OntologyClass]
    relationship_classes: list[RelationshipClass]

    @property
    def entity_types(self) -> list[str]:
        return [c.name for c in self.entity_classes]

    @property
    def edge_types(self) -> list[str]:
        return [r.name for r in self.relationship_classes]

    @property
    def edge_type_map(self) -> dict[tuple[str, str], list[str]]:
        result: dict[tuple[str, str], list[str]] = {}
        for r in self.relationship_classes:
            key = (r.source_entity, r.target_entity)
            result.setdefault(key, []).append(r.name)
        return result

    @classmethod
    async def from_mcp(cls, session: "ClientSession", target_graph: str) -> "OntologySpec":
        # `_unwrap_tool_result` should prefer `result.structuredContent` over
        # `result.content` — that's the SDK's typical pattern for structured
        # tool returns, and it's the field the placeholder shim in Section 5
        # populates. Fall back to `content` only if `structuredContent` is
        # absent (older SDKs / non-structured responses).
        result = await session.call_tool("get_ontology_structure", {})
        payload: dict[str, Any] = _unwrap_tool_result(result)
        if "error" in payload:
            raise RuntimeError(f"get_ontology_structure failed: {payload['error']}")
        return cls(target_graph=target_graph, **payload)
```

**Cold-start handling:** when the ontology graph is empty (no classes loaded yet), the response has empty `entity_classes` and `relationship_classes` lists. Caller should fail-fast at run-start if the ontology is empty — extraction has no useful work to do without an ontology.

**Missing-field handling:** when a tool's response lacks an optional field, default to empty list / empty string. Note the response field types for `alt_labels` and `examples` are not strictly normalized — server may return either `list[str]` or `str` depending on how the ontology was loaded. The adapter should coerce defensively.

---

## 3. `get_schema()`

### Signature

```python
async def get_schema() -> dict[str, Any]:
    """Retrieve the structural schema of the knowledge graph.

    Returns node labels with property keys, relationship types with
    source->target patterns, and counts. Results are cached until
    new data is ingested via add_memory.
    """
```

The tool takes **no arguments**. The destination graph is the connector's configured `group_id` (set at server startup). Like `get_ontology_structure`, this is registered dynamically via `register_dynamic_tools()` (which calls `mcp.add_tool(get_schema)` at server startup).

### Response shape

On success, the response is a `dict[str, Any]` with these keys:

| Key | Type | Description |
|---|---|---|
| `type` | `str` | Always `"schema"`. Discriminator for clients that route on response type. |
| `graph_name` | `str` | Destination graph name (the connector's `group_id`). |
| `domain` | `str` | Display name of the domain — derived from `graph_name` (underscores → spaces, title-cased). |
| `node_labels` | `dict[str, dict]` | Per-label: `{count: int, properties: list[str], sampled: bool, description?: str, sample_names?: list[str]}`. The `name_embedding` property is excluded from `properties`. |
| `relationship_types` | `dict[str, dict]` | Per-relationship-type: `{count: int, patterns: list[list[str]], description?: str}`. Each pattern is a 2-element `[source_label, target_label]` list. |
| `analysis_notes` | `list[str]` (optional) | `IMPORTANT:` lines extracted from entity/relationship descriptions in the domain profile. Present only if any `IMPORTANT:` notes exist. |
| `tool_capabilities` | `dict` | Discovery metadata for the reasoning engine — describes which tools cover which fields, search strategies, rerankers, etc. Stable but not consumed by extraction today. |
| `cypher_reference` | `str` | A multi-line FalkorDB Cypher quick reference. Stable; safe to embed in prompts. |

On error: `{"error": "<message>"}`.

The response is cached server-side and invalidated by `add_memory()` calls. A single pipeline run that does an ingest pass can expect the cached shape to remain stable until it ingests.

### Worked example

```json
{
  "type": "schema",
  "graph_name": "aviation_safety",
  "domain": "Aviation Safety",
  "node_labels": {
    "Aircraft": {
      "count": 1247,
      "properties": ["model", "name", "operator", "registration", "uuid"],
      "sampled": true,
      "description": "Aircraft involved in a safety occurrence."
    },
    "Operator": {
      "count": 312,
      "properties": ["country", "iata_code", "name", "uuid"],
      "sampled": true
    }
  },
  "relationship_types": {
    "OPERATED_BY": {
      "count": 1180,
      "patterns": [["Aircraft", "Operator"]]
    }
  },
  "tool_capabilities": { "...": "..." },
  "cypher_reference": "..."
}
```

`tool_capabilities` and `cypher_reference` are elided here for brevity — see the schema table above for their types. `tool_capabilities` is a discovery-metadata dict consumed by the reasoning engine; `cypher_reference` is a multi-line FalkorDB Cypher quick-reference string safe to embed in prompts.

### Prompt-content recipe

```python
def render_live_data_shape(schema: dict, top_k: int = 10) -> str:
    """Render a get_schema() response as a markdown block for prompt priors."""
    node_labels = schema.get("node_labels", {})
    rel_types = schema.get("relationship_types", {})

    if not any(v.get("count", 0) for v in node_labels.values()) and not any(
        v.get("count", 0) for v in rel_types.values()
    ):
        return "**Live data shape:** the destination graph is empty; no live priors yet."

    sorted_nodes = sorted(
        node_labels.items(), key=lambda kv: kv[1].get("count", 0), reverse=True
    )[:top_k]
    sorted_rels = sorted(
        rel_types.items(), key=lambda kv: kv[1].get("count", 0), reverse=True
    )[:top_k]

    lines = ["**Live data shape (current destination graph):**", ""]
    lines.append("Top node labels by count:")
    for label, info in sorted_nodes:
        props = ", ".join(info.get("properties", []))
        lines.append(f"- `{label}` ({info.get('count', 0)} nodes; properties: {props})")

    lines.append("")
    lines.append("Top relationship types by count:")
    for rtype, info in sorted_rels:
        patterns = "; ".join(f"({s}, {t})" for s, t in info.get("patterns", []))
        lines.append(f"- `{rtype}` ({info.get('count', 0)} edges; patterns: {patterns})")

    return "\n".join(lines)
```

**Stable vs derived fields:** `count` is stable across calls during a single pipeline run (cache the response at run-start). `properties` and `patterns` are observational summaries computed at query time; they may shift between runs as the graph grows. Don't treat them as schema; treat them as priors.

**Cold-start handling:** see the helper's first branch — when `node_labels` and `relationship_types` are empty (or all counts are zero), the destination graph is fresh and the prompt should reflect that.

---

## 4. `add_memory()`

### Signature

```python
@mcp.tool()
async def add_memory(
    name: str | None = None,
    episode_body: str | None = None,
    group_id: str | None = None,
    source: Literal['text', 'json', 'message'] = 'text',
    source_description: str = '',
    uuid: str | None = None,
    sync: bool = False,
    episodes: list[dict] | None = None,
) -> SuccessResponse | EpisodeAddedResponse | ErrorResponse:
    ...
```

### Episode shape

The tool supports two modes:

**Single mode** (queued, async by default):
- `name` (str, required) — episode name.
- `episode_body` (str, required) — content to persist. When `source='json'`, must be a JSON string.
- `source` (literal `'text' | 'json' | 'message'`, default `'text'`).
- `source_description` (str, default `''`).
- `group_id` (str, optional) — graph partition. Falls back to the connector's default `group_id` when omitted.
- `uuid` (str, optional) — caller-supplied episode UUID.
- `sync` (bool, default `False`) — if `True`, bypasses the async queue and calls Graphiti directly. Returns `EpisodeAddedResponse` with extracted node and edge UUIDs. Otherwise returns `SuccessResponse` (queued).

**Bulk mode**:
- `episodes` (list of dicts, required for bulk mode).
- Each dict: `{"name": str, "content": str, "source": str, "source_description": str}`.
- Bulk mode is processed synchronously; the call returns when ingestion completes.
- The single-mode keys (`name`, `episode_body`, `sync`, `uuid`) are ignored when `episodes` is provided.

The response is one of (TypedDicts defined in `mcp_server/src/models/response_types.py`):

| Type | Shape | When |
|---|---|---|
| `SuccessResponse` | `{"message": str}` | Queued single episode, or completed bulk ingest. |
| `EpisodeAddedResponse` | `{"message": str, "node_uuids": list[str], "edge_uuids": list[str]}` | Synchronous single mode (`sync=True`). |
| `ErrorResponse` | `{"error": str}` | Validation or runtime failure. |

### Ingest semantics

`add_memory()` replaces the direct `graphiti_core.Graphiti.add_episode()` call that downstream services (e.g., today's `aletheia/extraction/nodes/ingest.py:157`) currently make. Same downstream effect: LLM-driven entity + relationship extraction into FalkorDB. Same idempotency semantics as Graphiti's `add_episode` (caller-supplied UUID is honored if provided).

After every successful ingest the server sets `_schema_dirty = True`, which invalidates the `get_schema()` cache so the next `get_schema()` call recomputes counts and patterns.

### Error modes

- **LLM refusal:** the underlying Graphiti `add_episode` raises a non-retryable refusal error. Caller should surface the error and skip the episode (do not retry).
- **Rate limit:** retryable. Caller's retry policy should use exponential backoff (e.g., the existing `with_retry` pattern from `aletheia/llm.py`).
- **Partial failure (bulk mode):** the bulk path returns a `SuccessResponse` summarizing total nodes/edges but does not detail per-episode failures inside Graphiti's bulk API. Caller should treat the bulk call as best-effort and rely on subsequent queries to verify state.
- **Validation errors:** the server returns `ErrorResponse` for missing required fields (e.g., bulk-mode dict missing `name` or `content`, or single-mode call missing `name`/`episode_body`). These are caller bugs, not retryable.
- **MCP transport error:** HTTP-SSE timeout or connection closed. Caller retries with reconnection.

---

## 5. Placeholder shim recipe

For development without a live graphiti MCP, downstream services can drop in this minimal shim. The shim mirrors the **return shapes** of the real `mcp` SDK `ClientSession` (`list_tools()` returns a `ListToolsResult`-shaped object with a `.tools` attribute; `call_tool()` returns a `CallToolResult`-shaped object whose `structuredContent` holds the response payload), so swapping shim ↔ real is a one-line edit at the construction site as long as the unwrap adapter prefers `structuredContent` (see Section 2's OntologySpec recipe).

```python
# placeholder_mcp_client.py — ~50 LOC
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _ShimCallToolResult:
    """Mimics mcp.types.CallToolResult for the shim's call_tool return.

    Real CallToolResult has structuredContent / content / isError fields;
    downstream code typically unwraps via a helper that prefers
    structuredContent. The shim populates structuredContent directly so an
    unwrap helper that prefers structuredContent gets the right answer.
    """

    structuredContent: dict[str, Any]
    content: list[Any] = field(default_factory=list)
    isError: bool = False


@dataclass
class _ShimTool:
    name: str


@dataclass
class _ShimListToolsResult:
    tools: list[_ShimTool]


class PlaceholderMCPClient:
    """Drop-in shim for the mcp SDK ClientSession during W3 development.

    Replace with a real SseClient at W3 step 5.
    """

    async def initialize(self) -> None:
        return None

    async def list_tools(self) -> _ShimListToolsResult:
        return _ShimListToolsResult(
            tools=[
                _ShimTool("get_ontology_structure"),
                _ShimTool("get_schema"),
                _ShimTool("add_memory"),
            ]
        )

    async def call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> _ShimCallToolResult:
        if name == "get_ontology_structure":
            return _ShimCallToolResult(structuredContent=_FAKE_ONTOLOGY_RESPONSE)
        if name == "get_schema":
            return _ShimCallToolResult(structuredContent=_FAKE_SCHEMA_RESPONSE)
        if name == "add_memory":
            return _ShimCallToolResult(
                structuredContent={
                    "message": "Episode 'fake' queued for processing in group 'fake_destination'"
                }
            )
        raise ValueError(f"unknown tool: {name}")


_FAKE_ONTOLOGY_RESPONSE = {
    "ontology_graph": "fake_ontology",
    "entity_classes": [
        {"name": "Person", "ontology_type": "class", "summary": "A person.", "alt_labels": [], "inherits_from": "", "examples": []},
        {"name": "Organization", "ontology_type": "class", "summary": "An organization.", "alt_labels": [], "inherits_from": "", "examples": []},
        {"name": "Location", "ontology_type": "class", "summary": "A location.", "alt_labels": [], "inherits_from": "", "examples": []},
    ],
    "relationship_classes": [
        {"name": "WORKS_AT", "ontology_type": "relationship_class", "summary": "Person works at Organization.", "alt_labels": [], "inherits_from": "", "examples": [], "source_entity": "Person", "target_entity": "Organization"},
        {"name": "LOCATED_IN", "ontology_type": "relationship_class", "summary": "Organization located in Location.", "alt_labels": [], "inherits_from": "", "examples": [], "source_entity": "Organization", "target_entity": "Location"},
    ],
}

_FAKE_SCHEMA_RESPONSE = {
    "type": "schema",
    "graph_name": "fake_destination",
    "domain": "Fake Destination",
    "node_labels": {
        "Person": {"count": 5, "properties": ["age", "name"], "sampled": True},
        "Organization": {"count": 2, "properties": ["industry", "name"], "sampled": True},
    },
    "relationship_types": {
        "WORKS_AT": {"count": 3, "patterns": [["Person", "Organization"]]},
    },
    "tool_capabilities": {"...": "..."},
    "cypher_reference": "...",
}
```

**Construction-site swap:**

```python
# in W3's MCP client construction site
import os
from typing import Any

client: Any
if MCP_URL := os.environ.get("EXTRACTION_MCP_URL"):
    # Real SSE client — see Section 1
    ...
else:
    client = PlaceholderMCPClient()
```

---

## 6. Cold-start + error handling

### 6.1 Cold-start (destination graph empty)

`get_schema()` returns zero-count `node_labels` and `relationship_types` (or empty dicts entirely if the graph contains no nodes). The prompt-content recipe (Section 3) emits a one-line "graph is empty" message instead of empty lists. `get_ontology_structure()` is unaffected by destination-graph state — returns the ontology regardless of whether the destination graph has data.

### 6.2 MCP unreachable at run-start

Master design §3.5 commits to **fail-fast**. Concrete behavior:

- The `mcp.client.sse.sse_client(...)` async context manager raises a connection error (typically `ConnectionRefusedError` wrapped in an `httpx`-class exception, depending on SDK version).
- Caller should not catch and degrade silently. Surface a clear "graphiti MCP unavailable at `{url}`" error and exit the run.
- This applies at run-start *only*. Mid-run transport errors (after the connection has been established) follow the per-call retry policy below.

### 6.3 Per-call retry / timeout posture

| Tool | Retry posture | Rationale |
|---|---|---|
| `get_ontology_structure` | None — read-once at run-start; failure is fail-fast | Run cannot proceed without ontology |
| `get_schema` | None — read-once at run-start; failure is fail-fast | Run cannot proceed without schema priors (Q1=B) |
| `add_memory` | Exponential backoff on rate-limit + transport errors; no retry on LLM refusal | Stage 12 ingest is many calls; transient failures shouldn't kill the run |

Caller owns the retry policy. The graphiti MCP server itself does not retry across the boundary.

---

## 7. Forward-looking

The graphiti MCP server exposes additional tools beyond the 3 documented above: `search_ontology`, `explore_ontology`, `run_cypher`, `profile_graph`. These are **available for future extraction work** (e.g., richer enrichment passes, ad-hoc Cypher escape hatches) but are **not part of the current extraction integration contract**. The server also registers `search` and `explore_node` via `register_dynamic_tools()` for free-text search and node exploration; those are reserved for reasoning-engine use cases not yet anticipated by extraction, so they're not listed alongside the four enrichment-relevant tools above. When a downstream service needs one of these, it should be added to this document with the same shape (signature, response, error modes) as Sections 2-4.
