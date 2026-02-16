"""Unit tests for graph_profiler module."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from graph_profiler import (
    _build_language_summary,
    _detect_languages_heuristic,
    _enrich_value_stats,
    _extract_property_profiles,
    _node_to_dict,
    _profile_entities,
    _profile_relationships,
    profile_graph,
)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_driver(query_results: dict[str, list[dict]]) -> AsyncMock:
    """Create a mock driver that returns preset results per Cypher pattern.

    Args:
        query_results: Maps a substring of the Cypher query to a list of records.
                       The first matching substring wins.
    """
    driver = AsyncMock()

    async def _execute(query: str, *args, **kwargs):
        for pattern, records in query_results.items():
            if pattern in query:
                return records, None, None
        return [], None, None

    driver.execute_query = _execute
    return driver


# ---------------------------------------------------------------------------
# _extract_property_profiles
# ---------------------------------------------------------------------------


class TestExtractPropertyProfiles:
    def test_basic_property_extraction(self):
        records = [
            {'n': {'name': 'Alice', 'age': 30, 'uuid': 'u1', 'group_id': 'g1'}},
            {'n': {'name': 'Bob', 'age': None, 'uuid': 'u2', 'group_id': 'g1'}},
        ]
        profiles = _extract_property_profiles(records, sample_size=5)

        assert 'name' in profiles
        assert profiles['name']['coverage'] == 1.0
        assert profiles['name']['sample_values'] == ['Alice', 'Bob']

        assert 'age' in profiles
        assert profiles['age']['coverage'] == 0.5  # Only Alice has a non-null age

        # uuid and group_id should be filtered out
        assert 'uuid' not in profiles
        assert 'group_id' not in profiles

    def test_empty_records(self):
        profiles = _extract_property_profiles([], sample_size=5)
        assert profiles == {}

    def test_long_values_truncated(self):
        long_text = 'x' * 300
        records = [{'n': {'description': long_text}}]
        profiles = _extract_property_profiles(records, sample_size=5)
        sample = profiles['description']['sample_values'][0]
        assert len(sample) == 203  # 200 + '...'
        assert sample.endswith('...')

    def test_embedding_properties_skipped(self):
        records = [{'n': {'name': 'Test', 'name_embedding': [0.1, 0.2], 'summary_embedding': [0.3]}}]
        profiles = _extract_property_profiles(records, sample_size=5)
        assert 'name' in profiles
        assert 'name_embedding' not in profiles
        assert 'summary_embedding' not in profiles

    def test_language_detection_for_long_text(self):
        records = [
            {'n': {'description': 'El avión aterrizó de emergencia en el aeropuerto de Barcelona después del incidente'}},
            {'n': {'description': 'The aircraft made an emergency landing at Barcelona airport after the incident'}},
        ]
        profiles = _extract_property_profiles(records, sample_size=5)
        desc = profiles['description']
        assert desc.get('avg_length', 0) > 30
        # Should detect at least some languages
        assert 'detected_languages' in desc


# ---------------------------------------------------------------------------
# _profile_entities
# ---------------------------------------------------------------------------


class TestProfileEntities:
    @pytest.mark.asyncio
    async def test_profiles_entity_labels(self):
        driver = _make_driver({
            'labels(n)': [
                {'lbls': ['Aircraft', 'Entity'], 'cnt': 5},
                {'lbls': ['Occurrence', 'Entity'], 'cnt': 10},
            ],
            'MATCH (n:`Aircraft`)': [
                {'n': {'name': 'Airbus A321neo', 'manufacturer': 'Airbus'}},
                {'n': {'name': 'Boeing 737-800', 'manufacturer': 'Boeing'}},
            ],
            'MATCH (n:`Occurrence`)': [
                {'n': {'name': 'Barcelona incident', 'description': 'Bird strike during approach'}},
            ],
        })

        profiles = await _profile_entities(driver, sample_size=5)

        assert 'Aircraft' in profiles
        assert profiles['Aircraft']['count'] == 5
        assert 'name' in profiles['Aircraft']['properties']
        assert 'manufacturer' in profiles['Aircraft']['properties']

        assert 'Occurrence' in profiles
        assert profiles['Occurrence']['count'] == 10

    @pytest.mark.asyncio
    async def test_skips_internal_labels(self):
        driver = _make_driver({
            'labels(n)': [
                {'lbls': ['Entity'], 'cnt': 100},
                {'lbls': ['Episodic'], 'cnt': 50},
                {'lbls': ['Community'], 'cnt': 5},
                {'lbls': ['Aircraft', 'Entity'], 'cnt': 10},
            ],
            'MATCH (n:`Aircraft`)': [
                {'n': {'name': 'Test'}},
            ],
        })

        profiles = await _profile_entities(driver, sample_size=5)

        assert 'Entity' not in profiles
        assert 'Episodic' not in profiles
        assert 'Community' not in profiles
        assert 'Aircraft' in profiles

    @pytest.mark.asyncio
    async def test_empty_graph(self):
        driver = _make_driver({
            'labels(n)': [],
        })

        profiles = await _profile_entities(driver, sample_size=5)
        assert profiles == {}

    @pytest.mark.asyncio
    async def test_enriches_with_distinct_counts(self):
        """Verify _enrich_value_stats adds distinct_count and replaces coverage with exact value."""
        property_profiles = {
            'ataChapter': {
                'coverage': 0.8,  # sample-based (will be replaced)
                'sample_values': ['28', '32', '72'],
            },
            'name': {
                'coverage': 1.0,
                'sample_values': ['Incident A', 'Incident B'],
            },
        }
        driver = _make_driver({
            'COUNT(DISTINCT n.`ataChapter`)': [
                {'distinct_count': 5, 'non_null_count': 8},
            ],
            'COUNT(DISTINCT n.`name`)': [
                {'distinct_count': 10, 'non_null_count': 10},
            ],
            # top-N queries (ataChapter is categorical: 5 < 20)
            'n.`ataChapter` AS val, COUNT(*)': [
                {'val': '28', 'freq': 3},
                {'val': '32', 'freq': 3},
                {'val': '72', 'freq': 2},
            ],
            # name: 10 distinct / 10 total = 1.0 >= 0.1 AND 10 < 20 → categorical
            'n.`name` AS val, COUNT(*)': [
                {'val': 'Incident A', 'freq': 1},
                {'val': 'Incident B', 'freq': 1},
            ],
        })

        await _enrich_value_stats(driver, 'Occurrence', 10, property_profiles)

        # distinct_count should be set
        assert property_profiles['ataChapter']['distinct_count'] == 5
        assert property_profiles['name']['distinct_count'] == 10

        # coverage should be exact (non_null_count / total_count)
        assert property_profiles['ataChapter']['coverage'] == 0.8  # 8/10
        assert property_profiles['name']['coverage'] == 1.0  # 10/10

    @pytest.mark.asyncio
    async def test_enriches_categorical_with_top_values(self):
        """For a property with distinct_count < 20, top_values should be populated."""
        property_profiles = {
            'severity': {
                'coverage': 0.9,
                'sample_values': ['High', 'Medium'],
            },
        }
        driver = _make_driver({
            'COUNT(DISTINCT n.`severity`)': [
                {'distinct_count': 3, 'non_null_count': 9},
            ],
            'n.`severity` AS val, COUNT(*)': [
                {'val': 'High', 'freq': 5},
                {'val': 'Medium', 'freq': 3},
                {'val': 'Low', 'freq': 1},
            ],
        })

        await _enrich_value_stats(driver, 'Occurrence', 10, property_profiles)

        assert property_profiles['severity']['distinct_count'] == 3
        assert 'top_values' in property_profiles['severity']
        top = property_profiles['severity']['top_values']
        assert len(top) == 3
        assert top[0] == {'value': 'High', 'count': 5}
        assert top[1] == {'value': 'Medium', 'count': 3}
        assert top[2] == {'value': 'Low', 'count': 1}

    @pytest.mark.asyncio
    async def test_no_top_values_for_high_cardinality(self):
        """For a property with distinct_count >= 20 AND ratio >= 0.1, no top_values."""
        property_profiles = {
            'description': {
                'coverage': 1.0,
                'sample_values': ['Some long text'],
            },
        }
        driver = _make_driver({
            'COUNT(DISTINCT n.`description`)': [
                {'distinct_count': 50, 'non_null_count': 50},
            ],
        })

        await _enrich_value_stats(driver, 'Occurrence', 50, property_profiles)

        assert property_profiles['description']['distinct_count'] == 50
        # 50 >= 20 AND 50/50 = 1.0 >= 0.1 → NOT categorical
        assert 'top_values' not in property_profiles['description']


# ---------------------------------------------------------------------------
# _profile_relationships
# ---------------------------------------------------------------------------


class TestProfileRelationships:
    @pytest.mark.asyncio
    async def test_profiles_relationship_types(self):
        driver = _make_driver({
            'type(r) AS rel_type': [
                {'rel_type': 'INVOLVED_AIRCRAFT', 'cnt': 12},
                {'rel_type': 'LOCATED_IN', 'cnt': 8},
            ],
            'MATCH (s)-[r:`INVOLVED_AIRCRAFT`]->(t) RETURN DISTINCT': [
                {'src': ['Occurrence', 'Entity'], 'tgt': ['Aircraft', 'Entity']},
            ],
            'MATCH (s)-[r:`INVOLVED_AIRCRAFT`]->(t) RETURN s.name': [
                {'source': 'Barcelona incident', 'target': 'Airbus A321neo'},
            ],
            'MATCH (s:`Occurrence`)-[r:`INVOLVED_AIRCRAFT`]->()': [
                {'avg_deg': 1.2},
            ],
            'MATCH (s)-[r:`LOCATED_IN`]->(t) RETURN DISTINCT': [
                {'src': ['Airport', 'Entity'], 'tgt': ['Country', 'Entity']},
            ],
            'MATCH (s)-[r:`LOCATED_IN`]->(t) RETURN s.name': [
                {'source': 'Barcelona-El Prat', 'target': 'Spain'},
            ],
            'MATCH (s:`Airport`)-[r:`LOCATED_IN`]->()': [
                {'avg_deg': 1.0},
            ],
        })

        profiles = await _profile_relationships(driver, sample_size=5)

        assert 'INVOLVED_AIRCRAFT' in profiles
        rel = profiles['INVOLVED_AIRCRAFT']
        assert rel['count'] == 12
        assert rel['source_target_patterns'] == [['Occurrence', 'Aircraft']]
        assert rel['avg_out_degree'] == 1.2
        assert len(rel['sample_paths']) == 1
        assert rel['sample_paths'][0]['source'] == 'Barcelona incident'

    @pytest.mark.asyncio
    async def test_empty_relationships(self):
        driver = _make_driver({
            'type(r) AS rel_type': [],
        })

        profiles = await _profile_relationships(driver, sample_size=5)
        assert profiles == {}


# ---------------------------------------------------------------------------
# _build_language_summary
# ---------------------------------------------------------------------------


class TestBuildLanguageSummary:
    def test_multilingual_detection(self):
        entity_profiles = {
            'Occurrence': {
                'count': 10,
                'properties': {
                    'description': {
                        'coverage': 1.0,
                        'detected_languages': ['en', 'es', 'fr'],
                    },
                    'name': {
                        'coverage': 1.0,
                    },
                },
            },
        }

        summary = _build_language_summary(entity_profiles)

        assert 'en' in summary['primary_languages']
        assert 'es' in summary['primary_languages']
        assert 'fr' in summary['primary_languages']
        assert 'Occurrence.description' in summary['multilingual_fields']

    def test_empty_profiles(self):
        summary = _build_language_summary({})
        assert summary['primary_languages'] == []
        assert summary['multilingual_fields'] == []


# ---------------------------------------------------------------------------
# _detect_languages_heuristic
# ---------------------------------------------------------------------------


class TestDetectLanguagesHeuristic:
    def test_detects_spanish(self):
        texts = ['El avión aterrizó de emergencia en el aeropuerto']
        langs = _detect_languages_heuristic(texts)
        assert 'es' in langs

    def test_detects_english(self):
        texts = ['The aircraft landed at the airport for an emergency']
        langs = _detect_languages_heuristic(texts)
        assert 'en' in langs

    def test_detects_french(self):
        texts = ["L'avion a atterri pour une urgence dans l'aéroport"]
        langs = _detect_languages_heuristic(texts)
        assert 'fr' in langs

    def test_mixed_languages(self):
        texts = [
            'El avión aterrizó de emergencia en el aeropuerto',
            'The aircraft landed at the airport for an emergency',
        ]
        langs = _detect_languages_heuristic(texts)
        assert 'es' in langs
        assert 'en' in langs

    def test_short_text_skipped(self):
        texts = ['hi']
        langs = _detect_languages_heuristic(texts)
        assert langs == []


# ---------------------------------------------------------------------------
# _node_to_dict
# ---------------------------------------------------------------------------


class TestNodeToDict:
    def test_dict_passthrough(self):
        d = {'name': 'Test', 'age': 42}
        assert _node_to_dict(d) == d

    def test_object_with_properties(self):
        class FakeNode:
            properties = {'name': 'Test', 'uuid': '123'}

        result = _node_to_dict(FakeNode())
        assert result == {'name': 'Test', 'uuid': '123'}


# ---------------------------------------------------------------------------
# profile_graph (integration of all parts)
# ---------------------------------------------------------------------------


class TestProfileGraph:
    @pytest.mark.asyncio
    async def test_full_profile(self):
        driver = _make_driver({
            'labels(n)': [
                {'lbls': ['Aircraft', 'Entity'], 'cnt': 5},
            ],
            'MATCH (n:`Aircraft`)': [
                {'n': {'name': 'Airbus A321neo', 'manufacturer': 'Airbus'}},
                {'n': {'name': 'Boeing 737-800', 'manufacturer': 'Boeing'}},
            ],
            'type(r) AS rel_type': [
                {'rel_type': 'OPERATES', 'cnt': 10},
            ],
            'MATCH (s)-[r:`OPERATES`]->(t) RETURN DISTINCT': [
                {'src': ['Airline', 'Entity'], 'tgt': ['Aircraft', 'Entity']},
            ],
            'MATCH (s)-[r:`OPERATES`]->(t) RETURN s.name': [
                {'source': 'KLM', 'target': 'Boeing 737-800'},
            ],
            'MATCH (s:`Airline`)-[r:`OPERATES`]->()': [
                {'avg_deg': 2.0},
            ],
        })

        result = await profile_graph(driver, sample_size=5)

        assert 'entity_profiles' in result
        assert 'relationship_profiles' in result
        assert 'language_summary' in result

        assert 'Aircraft' in result['entity_profiles']
        assert 'OPERATES' in result['relationship_profiles']
