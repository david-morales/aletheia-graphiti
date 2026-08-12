"""Tests for new response types."""
import sys
from pathlib import Path

# Add src to path so we can import models
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.response_types import (
    CommunityResult,
    CommunityBuildResponse,
    EdgeResult,
    EpisodeAddedResponse,
    EpisodeContextResponse,
    ExploreResponse,
    SearchResponse,
)


def test_search_response_structure():
    response = SearchResponse(
        message='Results found',
        nodes=[],
        edges=[],
        communities=[],
    )
    assert response['message'] == 'Results found'
    assert response['nodes'] == []
    assert response['edges'] == []
    assert response['communities'] == []


def test_explore_response_structure():
    response = ExploreResponse(
        message='Node explored',
        center_node=None,
        nodes=[],
        edges=[],
        communities=[],
    )
    assert response['center_node'] is None


def test_episode_context_response_structure():
    response = EpisodeContextResponse(
        message='Context retrieved',
        nodes=[],
        edges=[],
    )
    assert response['nodes'] == []


def test_community_build_response_structure():
    response = CommunityBuildResponse(
        message='Communities built',
        community_count=3,
        communities=[],
    )
    assert response['community_count'] == 3


def test_edge_result_structure():
    edge = EdgeResult(
        uuid='edge-1',
        name='SANCTION',
        fact='Entity A is related to Entity B',
        source_node_uuid='uuid-a',
        target_node_uuid='uuid-b',
        created_at='2026-01-01T00:00:00',
        valid_at=None,
        invalid_at=None,
        group_id='test_group',
    )
    assert edge['name'] == 'SANCTION'
    assert edge['fact'] == 'Entity A is related to Entity B'


def test_community_result_structure():
    community = CommunityResult(
        uuid='comm-1',
        name='Terror Network Cluster',
        summary='A cluster of related organizations',
        member_count=5,
        group_id='test_group',
    )
    assert community['member_count'] == 5


def test_episode_added_response_structure():
    response = EpisodeAddedResponse(
        message='Episode processed',
        node_uuids=['n-1', 'n-2'],
        edge_uuids=['e-1'],
    )
    assert response['message'] == 'Episode processed'
    assert response['node_uuids'] == ['n-1', 'n-2']
    assert response['edge_uuids'] == ['e-1']


def test_episode_added_response_empty_lists():
    response = EpisodeAddedResponse(
        message='No entities extracted',
        node_uuids=[],
        edge_uuids=[],
    )
    assert response['node_uuids'] == []
    assert response['edge_uuids'] == []


# --- ADR-019 R2: graph_query + get_schema publish a typed outputSchema ---

def test_query_and_schema_tools_publish_output_schema():
    import asyncio
    from mcp.server.mcpserver import MCPServer
    import graphiti_mcp_server as srv

    m = MCPServer("t")
    m.add_tool(srv.graph_query)
    m.add_tool(srv.get_schema)
    tools = {t.name: t for t in asyncio.run(m.list_tools())}

    rc = tools["graph_query"].output_schema
    assert rc is not None
    rc_props = rc["properties"]
    # rich envelope + ADR-015 R4 error path both surfaced in the schema
    for key in ("query", "auto_fixes", "type", "row_count", "truncated",
                "limit_applied", "execution_ms", "cypher_quality", "error", "hint", "error_detail"):
        assert key in rc_props, f"graph_query outputSchema missing {key}"

    gs = tools["get_schema"].output_schema
    assert gs is not None
    gs_props = gs["properties"]
    for key in ("dialect", "dialect_reference", "node_labels", "relationship_types"):
        assert key in gs_props, f"get_schema outputSchema missing {key}"


def test_typed_tool_outputs_survive_mcpserver_output_validation():
    """Regression: graph_query / get_schema / sample_subgraph success AND error payloads
    must pass MCPServer's structured-output path over the wire.

    MCPServer builds a pydantic model from each total=False TypedDict with every
    optional field defaulted to None, then convert_result dumps it WITHOUT
    exclude_unset (mcp <=1.28.1) — so any field absent from a real payload is
    emitted as None in structuredContent. If the outputSchema types that field
    non-nullable, the lowlevel server rejects it ("None is not of type 'string'"),
    which broke get_schema (`error`) and graph_query (`truncated`, ...) for every
    over-the-wire caller. The fork's capability-level tests never exercised this
    round-trip. This test reproduces it; the fix makes the TypedDict fields
    nullable so the injected None validates.
    """
    import jsonschema
    from mcp.server.mcpserver import MCPServer
    import graphiti_mcp_server as srv

    m = MCPServer("t")
    m.add_tool(srv.graph_query)
    m.add_tool(srv.get_schema)
    m.add_tool(srv.sample_subgraph)
    tools = m._tool_manager._tools

    get_schema_success = {
        "type": "schema", "graph_name": "g", "domain": "G",
        "dialect": "falkordb-cypher", "dialect_reference": "ref",
        "node_labels": {"Persona": {"count": 3, "attribute_keys": ["dni"],
                                    "properties": ["dni", "name"], "sampled": True},
                        # A hierarchy-only label: censusable, but no vertex is
                        # stored under it, so it belongs in no pattern.
                        "Actor": {"count": 9, "attribute_keys": [], "properties": [],
                                  "sampled": False, "hierarchy": True}},
        "relationship_types": {"ES_DETENIDO": {"count": 2,
                                               "patterns": [["Persona", "Detencion"]]}},
        "cypher_reference": "ref", "tool_capabilities": {},
    }
    get_schema_error = {"error": "Failed to retrieve schema: boom"}
    # Every SchemaNodeInfo field None — same rule as the sample_subgraph case
    # below: making `hierarchy` non-nullable would leave this green until a real
    # None reached convert_result, which fails OUTSIDE the tool's try/except and
    # so escapes the ADR-015 error envelope as a protocol error.
    get_schema_all_null = {
        "type": None, "graph_name": None, "domain": None,
        "dialect": None, "dialect_reference": None,
        "node_labels": {"Persona": {"count": None, "attribute_keys": None,
                                    "properties": None, "sampled": None,
                                    "description": None, "sample_names": None,
                                    "hierarchy": None}},
        "relationship_types": {"ES_DETENIDO": {"count": None, "patterns": None,
                                               "description": None}},
        "error": None,
    }
    run_cypher_success = {"query": "MATCH (n) RETURN n", "type": "tabular",
                          "columns": ["n"], "rows": [[1]], "row_count": 1}
    run_cypher_error = {"error": "syntax error", "hint": "use <>",
                        "error_detail": {"code": "SYNTAX"}}
    sample_subgraph_success = {
        "type": "subgraph", "graph_name": "g",
        "nodes": [{"uuid": "n1", "name": "Ada", "labels": ["Entity", "Persona"],
                   "leaf": "Persona",
                   "created_at": "2026-01-01T00:00:00Z", "summary": None,
                   "group_id": "g"}],
        "edges": [{"uuid": "e1", "name": "ES_DETENIDO", "fact": "f",
                   "source_node_uuid": "n1", "target_node_uuid": "n2",
                   "created_at": None}],
    }
    sample_subgraph_error = {"error": "boom"}
    # Every field None. The success fixture above only nulls `summary` and the edge's
    # `created_at`, so it pins nullability for those two alone — making SubgraphNode.name
    # non-nullable would leave it green while a real `name=None` row (Graphiti permits it)
    # dies in convert_result. That failure happens OUTSIDE the tool's try/except, so it
    # escapes the ADR-015 error envelope as a protocol error. total=False: nothing required.
    sample_subgraph_all_null = {
        "type": None, "graph_name": None,
        "nodes": [{"uuid": None, "name": None, "labels": None, "leaf": None,
                   "created_at": None, "summary": None, "group_id": None}],
        "edges": [{"uuid": None, "name": None, "fact": None,
                   "source_node_uuid": None, "target_node_uuid": None,
                   "created_at": None}],
        "error": None,
    }

    cases = [
        ("get_schema", get_schema_success),
        ("get_schema", get_schema_error),
        ("get_schema", get_schema_all_null),
        ("graph_query", run_cypher_success),
        ("graph_query", run_cypher_error),
        ("sample_subgraph", sample_subgraph_success),
        ("sample_subgraph", sample_subgraph_error),
        ("sample_subgraph", sample_subgraph_all_null),
    ]
    for name, payload in cases:
        meta = tools[name].fn_metadata
        # SDK 2.x returns a `CallToolResult`; 1.x returned `(content, structured)`.
        structured = meta.convert_result(payload).structured_content
        # Exactly what the lowlevel server validates before sending — must not raise.
        jsonschema.validate(structured, meta.output_schema)
