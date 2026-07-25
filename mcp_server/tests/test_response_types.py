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


# --- ADR-019 R2: run_cypher + get_schema publish a typed outputSchema ---

def test_query_and_schema_tools_publish_output_schema():
    import asyncio
    from mcp.server.fastmcp import FastMCP
    import graphiti_mcp_server as srv

    m = FastMCP("t")
    m.add_tool(srv.run_cypher)
    m.add_tool(srv.get_schema)
    tools = {t.name: t for t in asyncio.run(m.list_tools())}

    rc = tools["run_cypher"].outputSchema
    assert rc is not None
    rc_props = rc["properties"]
    # rich envelope + ADR-015 R4 error path both surfaced in the schema
    for key in ("query", "auto_fixes", "type", "row_count", "truncated",
                "limit_applied", "execution_ms", "cypher_quality", "error", "hint", "error_detail"):
        assert key in rc_props, f"run_cypher outputSchema missing {key}"

    gs = tools["get_schema"].outputSchema
    assert gs is not None
    gs_props = gs["properties"]
    for key in ("dialect", "dialect_reference", "node_labels", "relationship_types"):
        assert key in gs_props, f"get_schema outputSchema missing {key}"
