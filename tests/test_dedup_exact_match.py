"""Tests that exact name matches are resolved BEFORE the entropy gate.

The entropy gate in _resolve_with_similarity is designed to prevent false fuzzy
matches for short/low-entropy names. However, it must NOT prevent exact matches —
a node named "Spain" should always resolve to an existing "Spain" candidate,
regardless of name length or entropy.
"""

from unittest.mock import MagicMock

import pytest

from graphiti_core.utils.maintenance.dedup_helpers import (
    DedupCandidateIndexes,
    DedupResolutionState,
    _build_candidate_indexes,
    _has_high_entropy,
    _normalize_name_for_fuzzy,
    _resolve_with_similarity,
)


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


class TestEntropyGateDoesNotBlockExactMatch:
    """Verify that short/low-entropy names still get exact-matched."""

    @pytest.mark.parametrize(
        'name',
        ['Spain', 'Peru', 'Cuba', 'Iraq', 'Iran', 'Mali', 'Chad', 'Fiji', 'Laos', 'Oman'],
    )
    def test_short_country_names_have_low_entropy(self, name: str):
        """Confirm these names would be blocked by the entropy gate."""
        normalized = _normalize_name_for_fuzzy(name)
        assert not _has_high_entropy(normalized), f'{name!r} should have low entropy'

    def test_exact_match_resolves_despite_low_entropy(self):
        """A low-entropy name with an exact existing match should resolve deterministically."""
        existing_spain = _make_node('Spain', uuid='existing-spain-uuid')
        extracted_spain = _make_node('Spain', uuid='new-spain-uuid')

        indexes = _build_candidate_indexes([existing_spain])
        state = DedupResolutionState(
            resolved_nodes=[None],
            uuid_map={},
            unresolved_indices=[],
        )

        _resolve_with_similarity([extracted_spain], indexes, state)

        # Should be resolved (not deferred to LLM)
        assert state.resolved_nodes[0] is existing_spain
        assert state.uuid_map['new-spain-uuid'] == 'existing-spain-uuid'
        assert len(state.unresolved_indices) == 0
        assert len(state.duplicate_pairs) == 1

    def test_exact_match_case_insensitive(self):
        """Exact match is case-insensitive via _normalize_string_exact."""
        existing = _make_node('SPAIN', uuid='existing-uuid')
        extracted = _make_node('spain', uuid='new-uuid')

        indexes = _build_candidate_indexes([existing])
        state = DedupResolutionState(
            resolved_nodes=[None],
            uuid_map={},
            unresolved_indices=[],
        )

        _resolve_with_similarity([extracted], indexes, state)

        assert state.resolved_nodes[0] is existing
        assert len(state.unresolved_indices) == 0

    def test_ambiguous_exact_match_defers_to_llm(self):
        """When multiple existing nodes have the same normalized name, defer to LLM."""
        existing_1 = _make_node('Spain', uuid='spain-1')
        existing_2 = _make_node('Spain', uuid='spain-2')
        extracted = _make_node('Spain', uuid='new-spain')

        indexes = _build_candidate_indexes([existing_1, existing_2])
        state = DedupResolutionState(
            resolved_nodes=[None],
            uuid_map={},
            unresolved_indices=[],
        )

        _resolve_with_similarity([extracted], indexes, state)

        # Should be unresolved (deferred to LLM)
        assert state.resolved_nodes[0] is None
        assert 0 in state.unresolved_indices

    def test_no_existing_match_low_entropy_defers_to_llm(self):
        """A low-entropy name with NO existing match defers to LLM."""
        existing = _make_node('France', uuid='france-uuid')
        extracted = _make_node('Spain', uuid='new-spain')

        indexes = _build_candidate_indexes([existing])
        state = DedupResolutionState(
            resolved_nodes=[None],
            uuid_map={},
            unresolved_indices=[],
        )

        _resolve_with_similarity([extracted], indexes, state)

        # Should be unresolved (no exact match, low entropy blocks fuzzy)
        assert state.resolved_nodes[0] is None
        assert 0 in state.unresolved_indices


class TestHighEntropyExactMatchStillWorks:
    """Verify that high-entropy names also benefit from exact match."""

    def test_long_name_exact_match(self):
        """A high-entropy name with an exact existing match resolves deterministically."""
        existing = _make_node('Barcelona-El Prat Airport', uuid='existing-uuid')
        extracted = _make_node('Barcelona-El Prat Airport', uuid='new-uuid')

        indexes = _build_candidate_indexes([existing])
        state = DedupResolutionState(
            resolved_nodes=[None],
            uuid_map={},
            unresolved_indices=[],
        )

        _resolve_with_similarity([extracted], indexes, state)

        assert state.resolved_nodes[0] is existing
        assert state.uuid_map['new-uuid'] == 'existing-uuid'


class TestMixedBatch:
    """Test batches with both low- and high-entropy names."""

    def test_mixed_entropy_batch(self):
        """Both short and long names resolve correctly in the same batch."""
        existing_spain = _make_node('Spain', uuid='existing-spain')
        existing_airport = _make_node('Barcelona-El Prat Airport', uuid='existing-airport')

        extracted_spain = _make_node('Spain', uuid='new-spain')
        extracted_airport = _make_node('Barcelona-El Prat Airport', uuid='new-airport')
        extracted_new = _make_node('Vueling Airlines', uuid='new-vueling')

        indexes = _build_candidate_indexes([existing_spain, existing_airport])
        state = DedupResolutionState(
            resolved_nodes=[None, None, None],
            uuid_map={},
            unresolved_indices=[],
        )

        _resolve_with_similarity(
            [extracted_spain, extracted_airport, extracted_new],
            indexes,
            state,
        )

        # Spain: exact match (despite low entropy)
        assert state.resolved_nodes[0] is existing_spain
        # Airport: exact match (high entropy)
        assert state.resolved_nodes[1] is existing_airport
        # Vueling: no match → unresolved
        assert state.resolved_nodes[2] is None
        assert 2 in state.unresolved_indices
        assert len(state.unresolved_indices) == 1

    def test_same_node_resolves_to_itself(self):
        """If extracted node UUID matches an existing node, no duplicate pair is created."""
        node = _make_node('Spain', uuid='same-uuid')

        indexes = _build_candidate_indexes([node])
        state = DedupResolutionState(
            resolved_nodes=[None],
            uuid_map={},
            unresolved_indices=[],
        )

        _resolve_with_similarity([node], indexes, state)

        assert state.resolved_nodes[0] is node
        assert state.uuid_map['same-uuid'] == 'same-uuid'
        # Same UUID → not a duplicate pair
        assert len(state.duplicate_pairs) == 0
