"""Tests for __identifier_name__ dedup bypass.

Entity types with ``__identifier_name__ = True`` on their Pydantic class should
only merge on exact normalized name match — never through fuzzy similarity or
LLM dedup.  This prevents data loss for entities named by formal identifiers
(report numbers, badge numbers, licence plates) where sequential IDs have high
n-gram similarity.
"""

from __future__ import annotations

from typing import ClassVar
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from graphiti_core.utils.maintenance.dedup_helpers import (
    DedupCandidateIndexes,
    DedupResolutionState,
    _build_candidate_indexes,
    _resolve_exact_only,
)
from graphiti_core.utils.maintenance.node_operations import (
    _is_identifier_name_type,
    resolve_extracted_nodes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class Report(BaseModel):
    """An investigation report identified by a formal code."""

    __identifier_name__: ClassVar[bool] = True


class Person(BaseModel):
    """A human person."""

    pass


def _make_node(name: str, uuid: str = '', labels: list[str] | None = None):
    """Create a minimal EntityNode mock for dedup testing."""
    node = MagicMock()
    node.name = name
    node.uuid = uuid or f'uuid-{name.lower().replace(" ", "-")}'
    node.labels = labels or ['Entity']
    node.group_id = 'test'
    node.attributes = {}
    node.summary = ''
    return node


# ---------------------------------------------------------------------------
# _is_identifier_name_type
# ---------------------------------------------------------------------------


class TestIsIdentifierNameType:
    def test_true_when_model_has_flag(self):
        node = _make_node('Report-001', labels=['Entity', 'Report'])
        assert _is_identifier_name_type(node, {'Report': Report}) is True

    def test_false_when_model_lacks_flag(self):
        node = _make_node('John', labels=['Entity', 'Person'])
        assert _is_identifier_name_type(node, {'Person': Person}) is False

    def test_false_when_entity_types_is_none(self):
        node = _make_node('Report-001', labels=['Entity', 'Report'])
        assert _is_identifier_name_type(node, None) is False

    def test_false_when_label_not_in_entity_types(self):
        node = _make_node('Report-001', labels=['Entity', 'Report'])
        assert _is_identifier_name_type(node, {'Person': Person}) is False

    def test_skips_entity_label(self):
        node = _make_node('Report-001', labels=['Entity'])
        assert _is_identifier_name_type(node, {'Report': Report}) is False


# ---------------------------------------------------------------------------
# _resolve_exact_only
# ---------------------------------------------------------------------------


class TestResolveExactOnly:
    def test_exact_match_merges(self):
        """Identical normalized names → merge."""
        existing = _make_node('Report-001', uuid='existing-uuid', labels=['Entity', 'Report'])
        extracted = _make_node('Report-001', uuid='new-uuid', labels=['Entity', 'Report'])

        indexes = _build_candidate_indexes([existing])
        state = DedupResolutionState(
            resolved_nodes=[None],
            uuid_map={},
            unresolved_indices=[],
        )

        _resolve_exact_only([extracted], indexes, state)

        assert state.resolved_nodes[0] is existing
        assert state.uuid_map['new-uuid'] == 'existing-uuid'
        assert len(state.duplicate_pairs) == 1

    def test_case_insensitive_match(self):
        """Exact match is case-insensitive."""
        existing = _make_node('REPORT-001', uuid='existing-uuid')
        extracted = _make_node('report-001', uuid='new-uuid')

        indexes = _build_candidate_indexes([existing])
        state = DedupResolutionState(
            resolved_nodes=[None],
            uuid_map={},
            unresolved_indices=[],
        )

        _resolve_exact_only([extracted], indexes, state)

        assert state.resolved_nodes[0] is existing

    def test_similar_names_do_not_merge(self):
        """Sequential identifiers that differ by one character → both survive."""
        existing = _make_node('Report-001', uuid='existing-uuid')
        extracted = _make_node('Report-002', uuid='new-uuid')

        indexes = _build_candidate_indexes([existing])
        state = DedupResolutionState(
            resolved_nodes=[None],
            uuid_map={},
            unresolved_indices=[],
        )

        _resolve_exact_only([extracted], indexes, state)

        # No match → treated as new node (maps to itself)
        assert state.resolved_nodes[0] is extracted
        assert state.uuid_map['new-uuid'] == 'new-uuid'
        assert len(state.duplicate_pairs) == 0

    def test_no_existing_candidates(self):
        """No candidates at all → new node."""
        extracted = _make_node('Report-001', uuid='new-uuid')

        indexes = _build_candidate_indexes([])
        state = DedupResolutionState(
            resolved_nodes=[None],
            uuid_map={},
            unresolved_indices=[],
        )

        _resolve_exact_only([extracted], indexes, state)

        assert state.resolved_nodes[0] is extracted
        assert state.uuid_map['new-uuid'] == 'new-uuid'

    def test_ambiguous_exact_match_merges_with_first(self):
        """Multiple candidates with same normalized name → merge with first (they're all equivalent for identifier types)."""
        existing_1 = _make_node('Report-001', uuid='uuid-1')
        existing_2 = _make_node('Report-001', uuid='uuid-2')
        extracted = _make_node('Report-001', uuid='new-uuid')

        indexes = _build_candidate_indexes([existing_1, existing_2])
        state = DedupResolutionState(
            resolved_nodes=[None],
            uuid_map={},
            unresolved_indices=[],
        )

        _resolve_exact_only([extracted], indexes, state)

        # Multiple exact matches → merge with first (all are equivalent for identifier-name types)
        assert state.resolved_nodes[0] is existing_1
        assert state.uuid_map['new-uuid'] == 'uuid-1'
        assert len(state.duplicate_pairs) == 1

    def test_offset_parameter(self):
        """The offset parameter writes to the correct slot in state.resolved_nodes."""
        existing = _make_node('Report-001', uuid='existing-uuid')
        extracted = _make_node('Report-001', uuid='new-uuid')

        indexes = _build_candidate_indexes([existing])
        state = DedupResolutionState(
            resolved_nodes=[None, None, None],
            uuid_map={},
            unresolved_indices=[],
        )

        _resolve_exact_only([extracted], indexes, state, offset=2)

        assert state.resolved_nodes[0] is None
        assert state.resolved_nodes[1] is None
        assert state.resolved_nodes[2] is existing

    def test_empty_name_treated_as_new(self):
        """Nodes with empty names are treated as new."""
        extracted = _make_node('', uuid='new-uuid')

        indexes = _build_candidate_indexes([])
        state = DedupResolutionState(
            resolved_nodes=[None],
            uuid_map={},
            unresolved_indices=[],
        )

        _resolve_exact_only([extracted], indexes, state)

        assert state.resolved_nodes[0] is extracted
        assert state.uuid_map['new-uuid'] == 'new-uuid'


# ---------------------------------------------------------------------------
# resolve_extracted_nodes (integration)
# ---------------------------------------------------------------------------


class TestResolveExtractedNodesPartitioning:
    """resolve_extracted_nodes routes identifier-name nodes through exact-only
    resolution and fuzzy nodes through similarity + LLM.

    graphiti-core v0.29.2 (#1224/#1432) collects candidates PER extracted node
    (``_collect_candidate_nodes -> list[list[EntityNode]]``) and passes the full
    extracted_nodes list to ``_resolve_with_llm`` (which selects unresolved ones
    internally via state.unresolved_indices), so the mocks below return a
    per-node candidate list.
    """

    @pytest.mark.asyncio
    async def test_identifier_nodes_skip_llm(self):
        """Identifier-name nodes with no exact match become new nodes without LLM call."""
        extracted_report = _make_node('Report-001', uuid='new-report', labels=['Entity', 'Report'])
        existing_report = _make_node('Report-002', uuid='existing-report', labels=['Entity', 'Report'])

        entity_types = {'Report': Report}

        mock_clients = MagicMock()
        # Candidates are per-extracted-node: one list for the one extracted node.
        with patch(
            'graphiti_core.utils.maintenance.node_operations._collect_candidate_nodes',
            new_callable=AsyncMock,
            return_value=[[existing_report]],
        ):
            with patch(
                'graphiti_core.utils.maintenance.node_operations._resolve_with_llm',
                new_callable=AsyncMock,
            ) as mock_llm:
                resolved, uuid_map, dup_pairs = await resolve_extracted_nodes(
                    clients=mock_clients,
                    extracted_nodes=[extracted_report],
                    entity_types=entity_types,
                )

                # LLM should NOT have been called (no fuzzy nodes)
                mock_llm.assert_not_called()

        # Report-001 and Report-002 are different → both survive
        assert len(resolved) == 1
        assert resolved[0] is extracted_report
        assert uuid_map['new-report'] == 'new-report'

    @pytest.mark.asyncio
    async def test_identifier_node_exact_match_merges(self):
        """Identifier-name node with exact match merges without LLM."""
        extracted_report = _make_node('Report-001', uuid='new-report', labels=['Entity', 'Report'])
        existing_report = _make_node('Report-001', uuid='existing-report', labels=['Entity', 'Report'])

        entity_types = {'Report': Report}

        mock_clients = MagicMock()
        with patch(
            'graphiti_core.utils.maintenance.node_operations._collect_candidate_nodes',
            new_callable=AsyncMock,
            return_value=[[existing_report]],
        ):
            with patch(
                'graphiti_core.utils.maintenance.node_operations._resolve_with_llm',
                new_callable=AsyncMock,
            ) as mock_llm:
                resolved, uuid_map, dup_pairs = await resolve_extracted_nodes(
                    clients=mock_clients,
                    extracted_nodes=[extracted_report],
                    entity_types=entity_types,
                )

                mock_llm.assert_not_called()

        assert len(resolved) == 1
        assert resolved[0] is existing_report
        assert uuid_map['new-report'] == 'existing-report'
        assert len(dup_pairs) == 1

    @pytest.mark.asyncio
    async def test_fuzzy_nodes_still_go_to_llm(self):
        """Non-identifier nodes with no deterministic match go to the LLM pipeline."""
        extracted_person = _make_node('John Smith', uuid='new-person', labels=['Entity', 'Person'])
        # A candidate that will NOT be a deterministic (exact/fuzzy) match, so the
        # node stays unresolved and _resolve_with_llm is invoked.
        existing_person = _make_node('Completely Different', uuid='existing-person', labels=['Entity', 'Person'])

        entity_types = {'Person': Person}

        mock_clients = MagicMock()
        with patch(
            'graphiti_core.utils.maintenance.node_operations._collect_candidate_nodes',
            new_callable=AsyncMock,
            return_value=[[existing_person]],
        ):
            with patch(
                'graphiti_core.utils.maintenance.node_operations._resolve_with_llm',
                new_callable=AsyncMock,
            ) as mock_llm:
                await resolve_extracted_nodes(
                    clients=mock_clients,
                    extracted_nodes=[extracted_person],
                    entity_types=entity_types,
                )

                # LLM SHOULD be called for fuzzy nodes
                mock_llm.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_entity_types_all_fuzzy(self):
        """When entity_types=None, all nodes go through fuzzy pipeline (backward compatible)."""
        extracted = _make_node('Report-001', uuid='new-uuid', labels=['Entity', 'Report'])
        existing = _make_node('Completely Different', uuid='existing-uuid', labels=['Entity'])

        mock_clients = MagicMock()
        with patch(
            'graphiti_core.utils.maintenance.node_operations._collect_candidate_nodes',
            new_callable=AsyncMock,
            return_value=[[existing]],
        ):
            with patch(
                'graphiti_core.utils.maintenance.node_operations._resolve_with_llm',
                new_callable=AsyncMock,
            ) as mock_llm:
                resolved, uuid_map, _ = await resolve_extracted_nodes(
                    clients=mock_clients,
                    extracted_nodes=[extracted],
                    entity_types=None,
                )

                # Should go through LLM pipeline
                mock_llm.assert_called_once()

    @pytest.mark.asyncio
    async def test_mixed_identifier_and_fuzzy(self):
        """Mixed batch: identifier nodes resolved exact-only, fuzzy nodes go to LLM."""
        extracted_report = _make_node('Report-001', uuid='new-report', labels=['Entity', 'Report'])
        extracted_person = _make_node('John Smith', uuid='new-person', labels=['Entity', 'Person'])
        existing_report = _make_node('Report-001', uuid='existing-report', labels=['Entity', 'Report'])
        existing_person = _make_node('Completely Different', uuid='existing-person', labels=['Entity', 'Person'])

        entity_types = {'Report': Report, 'Person': Person}

        mock_clients = MagicMock()
        # Per-node candidates: report matches its existing; person has a non-matching candidate.
        with patch(
            'graphiti_core.utils.maintenance.node_operations._collect_candidate_nodes',
            new_callable=AsyncMock,
            return_value=[[existing_report], [existing_person]],
        ):
            with patch(
                'graphiti_core.utils.maintenance.node_operations._resolve_with_llm',
                new_callable=AsyncMock,
            ) as mock_llm:
                resolved, uuid_map, dup_pairs = await resolve_extracted_nodes(
                    clients=mock_clients,
                    extracted_nodes=[extracted_report, extracted_person],
                    entity_types=entity_types,
                )

                # LLM is invoked once for the unresolved (fuzzy) node(s); the
                # identifier report was resolved exact-only and never sent.
                mock_llm.assert_called_once()

        # Report merged with existing via exact-only (no LLM)
        assert resolved[0] is existing_report
        assert uuid_map['new-report'] == 'existing-report'
        # Person resolved to itself (mock LLM does not resolve it)
        assert resolved[1] is extracted_person
