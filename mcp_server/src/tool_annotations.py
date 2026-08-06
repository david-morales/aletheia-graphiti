"""Tool annotations for the served MCP surface (ADR-019 R3).

ONE table for the whole surface. The server registers tools through two paths —
static `@mcp.tool()` decorators and `register_dynamic_tools()`'s `add_tool()`
calls — and a per-call-site literal would drift between them the first time a
tool moved from one path to the other. Every registration site reads
`annotations_for(<name>)`; `annotations_for` raises on an unknown name so a typo
fails at import instead of silently serving a tool with no hints.

Truthfulness is the point. `readOnlyHint` is what a client gates writes on and
what a human-in-the-loop UI auto-approves against; `destructiveHint` separates
"adds data" from "removes data you cannot get back". The audit found all 18
tools serving `annotations: None`, which made `clear_graph` machine-
indistinguishable from `get_status`.

Per the protocol, `destructiveHint` and `idempotentHint` are only meaningful when
`readOnlyHint` is false, so the read-only tools leave them unset.
`openWorldHint` is False throughout: every tool's domain of interaction is this
connector's own graph, never an open set of external entities.
"""

from __future__ import annotations

from mcp.types import ToolAnnotations


def _read_only(title: str) -> ToolAnnotations:
    """Reads only. `run_cypher` qualifies by construction: writes are rejected by
    the Cypher validator and, on FalkorDB, again by DB-side `ro_query`."""
    return ToolAnnotations(title=title, readOnlyHint=True, openWorldHint=False)


def _additive_write(title: str, *, idempotent: bool = False) -> ToolAnnotations:
    """Mutates the graph by ADDING; never removes."""
    return ToolAnnotations(
        title=title,
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=idempotent,
        openWorldHint=False,
    )


def _destructive(title: str) -> ToolAnnotations:
    """Removes data that cannot be recovered from this connector.

    All three deletions are idempotent: deleting what is already gone changes
    nothing further.
    """
    return ToolAnnotations(
        title=title,
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=True,
        openWorldHint=False,
    )


TOOL_ANNOTATIONS: dict[str, ToolAnnotations] = {
    # --- read-only (13) ---------------------------------------------------
    'search': _read_only('Search the knowledge graph'),
    'explore_node': _read_only('Explore an entity neighborhood'),
    'search_ontology': _read_only('Search the ontology'),
    'explore_ontology': _read_only('Explore an ontology class'),
    'get_schema': _read_only('Get the graph schema'),
    'get_ontology_structure': _read_only('Get the ontology structure'),
    'get_ontology_documentation': _read_only('Get the full ontology reference'),
    'run_cypher': _read_only('Run a read-only Cypher query'),
    'profile_graph': _read_only('Profile graph properties'),
    'sample_subgraph': _read_only('Sample a renderable subgraph'),
    'get_episodes': _read_only('List recent episodes'),
    'get_episode_context': _read_only('Get what an episode extracted'),
    'get_status': _read_only('Check server and database health'),
    # --- additive writes (2) ----------------------------------------------
    'add_memory': _additive_write('Ingest an episode'),
    'build_communities': _additive_write('Rebuild entity communities'),
    # --- destructive (3) --------------------------------------------------
    'delete_entity_edge': _destructive('Delete a relationship'),
    'delete_episode': _destructive('Delete an episode and its extracted data'),
    'clear_graph': _destructive('Delete ALL data in a graph partition'),
}


def annotations_for(name: str) -> ToolAnnotations:
    """Return the annotations for a served tool.

    Raises:
        KeyError: the name is not part of the served surface — a registration-site
            typo, or a new tool that has not been classified yet. Both must fail
            loudly rather than serve a tool with no machine-readable hints.
    """
    try:
        return TOOL_ANNOTATIONS[name]
    except KeyError:
        raise KeyError(
            f'No ADR-019 R3 annotations declared for tool {name!r}. Add it to '
            f'TOOL_ANNOTATIONS in tool_annotations.py (read-only? additive write? '
            f'destructive?) before registering it.'
        ) from None
