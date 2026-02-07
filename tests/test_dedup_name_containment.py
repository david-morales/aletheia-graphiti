"""Tests for containment matching in LLM dedup duplicate_name resolution.

When the LLM dedup returns a duplicate_name like "RUSAL" but the existing
node is named "United Company RUSAL", the exact lookup fails. The containment
fallback matches abbreviated or extended LLM names to existing nodes using
case-insensitive substring matching.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.nodes import EntityNode
from graphiti_core.utils.maintenance.node_operations import _resolve_with_llm


def _make_node(name: str, uuid: str = '', labels: list[str] | None = None) -> EntityNode:
    return EntityNode(
        name=name,
        uuid=uuid or f'uuid-{name.lower().replace(" ", "-")}',
        labels=labels or ['Entity'],
        group_id='test',
    )


def _make_state(n: int):
    """Build a DedupResolutionState for n unresolved extracted nodes."""
    return SimpleNamespace(
        resolved_nodes=[None] * n,
        uuid_map={},
        unresolved_indices=list(range(n)),
        duplicate_pairs=[],
    )


def _make_indexes(existing_nodes: list[EntityNode]):
    return SimpleNamespace(existing_nodes=existing_nodes)


@pytest.mark.asyncio
async def test_abbreviated_duplicate_name_resolves():
    """LLM returns 'RUSAL' but existing node is 'United Company RUSAL'."""
    existing = _make_node('United Company RUSAL', uuid='existing-rusal')
    extracted = _make_node('UC RUSAL', uuid='new-rusal')

    state = _make_state(1)
    indexes = _make_indexes([existing])

    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(return_value={
        'entity_resolutions': [
            {'id': 0, 'name': 'UC RUSAL', 'duplicate_name': 'RUSAL'},
        ]
    })

    await _resolve_with_llm(
        llm_client, [extracted], indexes, state,
        episode=None, previous_episodes=None, entity_types=None,
    )

    assert state.resolved_nodes[0] == existing
    assert state.uuid_map[extracted.uuid] == existing.uuid


@pytest.mark.asyncio
async def test_extended_duplicate_name_resolves():
    """LLM returns 'Renova Group LLC' but existing node is 'Renova Group'."""
    existing = _make_node('Renova Group', uuid='existing-renova')
    extracted = _make_node('Renova', uuid='new-renova')

    state = _make_state(1)
    indexes = _make_indexes([existing])

    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(return_value={
        'entity_resolutions': [
            {'id': 0, 'name': 'Renova', 'duplicate_name': 'Renova Group LLC'},
        ]
    })

    await _resolve_with_llm(
        llm_client, [extracted], indexes, state,
        episode=None, previous_episodes=None, entity_types=None,
    )

    assert state.resolved_nodes[0] == existing
    assert state.uuid_map[extracted.uuid] == existing.uuid


@pytest.mark.asyncio
async def test_exact_match_still_preferred():
    """Exact match should be used when available, not containment."""
    existing_exact = _make_node('RUSAL', uuid='existing-rusal-exact')
    existing_long = _make_node('United Company RUSAL', uuid='existing-rusal-long')
    extracted = _make_node('Some RUSAL entity', uuid='new-rusal')

    state = _make_state(1)
    indexes = _make_indexes([existing_exact, existing_long])

    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(return_value={
        'entity_resolutions': [
            {'id': 0, 'name': 'UC RUSAL', 'duplicate_name': 'RUSAL'},
        ]
    })

    await _resolve_with_llm(
        llm_client, [extracted], indexes, state,
        episode=None, previous_episodes=None, entity_types=None,
    )

    # Exact match on "RUSAL" wins
    assert state.resolved_nodes[0] == existing_exact


@pytest.mark.asyncio
async def test_containment_picks_longest_match():
    """When multiple nodes match by containment, pick the longest."""
    existing_short = _make_node('RUSAL', uuid='existing-short')
    existing_long = _make_node('Company RUSAL International', uuid='existing-long')
    extracted = _make_node('RUSAL Holdings', uuid='new-rusal')

    state = _make_state(1)
    indexes = _make_indexes([existing_short, existing_long])

    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(return_value={
        'entity_resolutions': [
            # LLM returns "Company RUSAL" — contained in the long name
            # and also contains "RUSAL" (the short name).
            # Longest canon_lower wins.
            {'id': 0, 'name': 'RUSAL Holdings', 'duplicate_name': 'Company RUSAL'},
        ]
    })

    await _resolve_with_llm(
        llm_client, [extracted], indexes, state,
        episode=None, previous_episodes=None, entity_types=None,
    )

    # "company rusal" is in "company rusal international" (27 chars) > "rusal" (5 chars)
    assert state.resolved_nodes[0] == existing_long


@pytest.mark.asyncio
async def test_hallucinated_name_still_rejected():
    """Completely unrelated names should still be treated as no duplicate."""
    existing = _make_node('United Company RUSAL', uuid='existing-rusal')
    extracted = _make_node('Gazprom', uuid='new-gazprom')

    state = _make_state(1)
    indexes = _make_indexes([existing])

    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(return_value={
        'entity_resolutions': [
            {'id': 0, 'name': 'Gazprom', 'duplicate_name': 'Totally Fake Corp'},
        ]
    })

    await _resolve_with_llm(
        llm_client, [extracted], indexes, state,
        episode=None, previous_episodes=None, entity_types=None,
    )

    # No match — extracted node stays as-is
    assert state.resolved_nodes[0] == extracted
    assert state.uuid_map[extracted.uuid] == extracted.uuid


@pytest.mark.asyncio
async def test_empty_duplicate_name_means_no_duplicate():
    """Empty duplicate_name should keep the extracted node."""
    existing = _make_node('United Company RUSAL', uuid='existing-rusal')
    extracted = _make_node('Gazprom', uuid='new-gazprom')

    state = _make_state(1)
    indexes = _make_indexes([existing])

    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(return_value={
        'entity_resolutions': [
            {'id': 0, 'name': 'Gazprom', 'duplicate_name': ''},
        ]
    })

    await _resolve_with_llm(
        llm_client, [extracted], indexes, state,
        episode=None, previous_episodes=None, entity_types=None,
    )

    assert state.resolved_nodes[0] == extracted
