"""Tests for new response types."""
import sys
from pathlib import Path

# Add src to path so we can import models
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.response_types import (
    CommunityResult,
    CommunityBuildResponse,
    EdgeResult,
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
