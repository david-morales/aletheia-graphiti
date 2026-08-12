"""Unit tests for graph_profiler module."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from flavours.age import AgeFlavour
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
                {'source_labels': ['Occurrence', 'Entity'], 'target_labels': ['Aircraft', 'Entity']},
            ],
            'MATCH (s)-[r:`INVOLVED_AIRCRAFT`]->(t) RETURN s.name': [
                {'source': 'Barcelona incident', 'target': 'Airbus A321neo'},
            ],
            'MATCH (s:`Occurrence`)-[r:`INVOLVED_AIRCRAFT`]->()': [
                {'avg_deg': 1.2},
            ],
            'MATCH (s)-[r:`LOCATED_IN`]->(t) RETURN DISTINCT': [
                {'source_labels': ['Airport', 'Entity'], 'target_labels': ['Country', 'Entity']},
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


class TestSampleEpisodicLanguages:
    """Tests for episodic content language detection."""

    @pytest.mark.asyncio
    async def test_detects_french_in_episodic_content(self):
        """Episodic nodes with French content should contribute to language detection."""
        from graph_profiler import _sample_episodic_languages

        driver = _make_driver({
            'MATCH (e:Episodic)': [
                {'content': "L'avion a atterri pour une urgence dans l'aéroport de Toulouse"},
                {'content': "Le pilote a signalé une défaillance du moteur avec des dommages"},
            ],
        })
        langs = await _sample_episodic_languages(driver)
        assert 'fr' in langs

    @pytest.mark.asyncio
    async def test_detects_spanish_in_episodic_content(self):
        """Episodic nodes with Spanish content should contribute to language detection."""
        from graph_profiler import _sample_episodic_languages

        driver = _make_driver({
            'MATCH (e:Episodic)': [
                {'content': 'El avión aterrizó de emergencia en el aeropuerto de Barcelona después del incidente'},
            ],
        })
        langs = await _sample_episodic_languages(driver)
        assert 'es' in langs

    @pytest.mark.asyncio
    async def test_empty_episodic_returns_empty(self):
        """No Episodic nodes should return empty language set."""
        from graph_profiler import _sample_episodic_languages

        driver = _make_driver({
            'MATCH (e:Episodic)': [],
        })
        langs = await _sample_episodic_languages(driver)
        assert langs == []

    @pytest.mark.asyncio
    async def test_english_entity_with_french_episodic(self):
        """Language summary should include both entity and episodic languages."""
        driver = _make_driver({
            # Entity profiling returns English-only entities
            'labels(n)': [
                {'lbls': ['Aircraft', 'Entity'], 'cnt': 5},
            ],
            'MATCH (n:`Aircraft`)': [
                {'n': {'name': 'Airbus A321neo', 'summary': 'The aircraft was involved in a landing incident'}},
            ],
            # Episodic nodes have French content
            'MATCH (e:Episodic)': [
                {'content': "L'avion a atterri pour une urgence dans l'aéroport de Toulouse après une défaillance"},
                {'content': "Le commandant de bord a signalé une perte de puissance sur le moteur gauche"},
            ],
            # Relationship profiling (empty)
            'type(r) AS rel_type': [],
        })

        result = await profile_graph(driver, sample_size=5)
        langs = result['language_summary']['primary_languages']
        assert 'fr' in langs


# ---------------------------------------------------------------------------
# Flavour-awareness (the Overview tab's census must be the backend's own)
# ---------------------------------------------------------------------------


class _AgeStubDriver:
    """A driver with Apache AGE's label semantics, not FalkorDB's.

    On AGE a vertex carries exactly ONE stored label (its ontology leaf), so
    `labels(n)` answers `['Persona']`; the full hierarchy lives in the stored
    list property `n.labels` (`['Entity', 'Actor', 'Persona']`). The stub answers
    each census according to WHICH OF THE TWO the query asked for — that is what
    makes the abstract supertype visible or invisible, and what makes a test that
    asserts on the supertype fail for the real reason when the profiler issues the
    hardcoded FalkorDB-shaped census.

    `unmatchable` names labels that exist in the hierarchy but have NO label table
    on this backend — precisely the supertypes the census surfaces. A label-scoped
    probe against one of those either raises (AGE's own behaviour for an unknown
    label, `unmatchable_raises=True`) or returns no rows; both are reproduced,
    because they are different failure modes for the profiler.

    One census row carries `lbls: None`: AGE returns label-less vertices with a
    null labels list, and a `.get(k, [])` default does NOT cover an explicit null.

    The relationship-pattern census answers leaf columns ONLY when the query asks
    for them (`label(s) AS source_leaf`), so the same stub exercises both the
    leaf-preferring path and the positional fallback. `sibling_leaves` adds a
    second endpoint sharing the SAME supertype under a different leaf — two
    genuinely distinct patterns that a supertype-level pick collapses into one.
    """

    _LEAF = ['Persona']
    _HIERARCHY = ['Entity', 'Actor', 'Persona']
    _SIBLING_HIERARCHY = ['Entity', 'Actor', 'Empresa']

    def __init__(
        self,
        unmatchable: set[str] | None = None,
        unmatchable_raises: bool = True,
        sibling_leaves: bool = False,
    ):
        self.queries: list[str] = []
        self.unmatchable = unmatchable or set()
        self.unmatchable_raises = unmatchable_raises
        self.sibling_leaves = sibling_leaves

    def _probes_an_unmatchable_label(self, query: str) -> bool:
        # Both label-scoped shapes the profiler emits: the node sample / value
        # stats `(n:`X`)` and the out-degree probe `(s:`X`)`.
        return any(
            f'(n:`{label}`)' in query or f'(s:`{label}`)' in query
            for label in self.unmatchable
        )

    async def execute_query(self, query: str, *a, **k):
        self.queries.append(query)
        # Checked FIRST: a label-scoped probe is recognisable by its label, not by
        # what it projects, and this is what the real backend does to it.
        if self._probes_an_unmatchable_label(query):
            if self.unmatchable_raises:
                raise RuntimeError('label "Actor" does not exist')
            return [], None, None
        if 'count(n) AS cnt' in query:
            lbls = self._HIERARCHY if 'n.labels AS lbls' in query else self._LEAF
            return [{'lbls': lbls, 'cnt': 5}, {'lbls': None, 'cnt': 7}], None, None
        if 'COUNT(DISTINCT' in query or 'AS val, COUNT(*)' in query:
            # A domain field probed at the TOP level is valid Cypher on AGE that
            # matches nothing: 0/0, which silently overwrites the sample-based
            # coverage with 0.0. Only the nested path finds the value.
            if 'n.`documento`' in query:
                return [{'distinct_count': 0, 'non_null_count': 0}], None, None
            if 'COUNT(DISTINCT' in query:
                return [{'distinct_count': 2, 'non_null_count': 5}], None, None
            return [{'val': 'OMAR MOHAMED', 'freq': 3}], None, None
        if 'RETURN n LIMIT' in query or 'RETURN n AS n LIMIT' in query:
            # AGE hands back the whole VERTEX ENVELOPE, with Graphiti's own
            # columns inside `properties` and every DOMAIN field one level
            # deeper, in the queryable `attributes` agtype map (live driver repr).
            vertex = {
                'id': 2814749767106562,
                'label': 'Persona',
                'properties': {
                    'name': 'OMAR MOHAMED',
                    'uuid': '365fd6da',
                    'attributes': {'documento': 'X1234567L'},
                },
            }
            if 'RETURN n AS n' in query:
                return [{'n': vertex}], None, None
            # UNALIASED: AGE names the column `col0`, so a reader keyed on `n`
            # sees nothing — rows exist, so the label still reports sampled: true.
            return [{'col0': vertex}], None, None
        if 'type(r) AS rel_type' in query:
            return [{'rel_type': 'ES_DETENIDO', 'cnt': 2}], None, None
        if 'AS source_labels' in query:
            announces_leaf = 'AS source_leaf' in query
            hierarchies = [self._HIERARCHY]
            if self.sibling_leaves:
                hierarchies.append(self._SIBLING_HIERARCHY)
            rows = []
            for hierarchy in hierarchies:
                src = hierarchy if 's.labels AS source_labels' in query else self._LEAF
                row: dict = {'source_labels': src, 'target_labels': ['Detencion']}
                if announces_leaf:
                    row['source_leaf'] = hierarchy[-1]
                    row['target_leaf'] = 'Detencion'
                rows.append(row)
            return rows, None, None
        if 's.name AS source' in query:
            return [{'source': 'OMAR MOHAMED', 'target': 'Detencion 4/2026'}], None, None
        if 'avg(deg) AS avg_deg' in query:
            return [{'avg_deg': 1.5}], None, None
        return [], None, None


class TestFlavourAwareEntityCensus:
    @pytest.mark.asyncio
    async def test_profile_entities_uses_the_flavour_census(self):
        """AGE's census reads the stored hierarchy — `labels(n)` hides the supertypes."""
        driver = _AgeStubDriver()

        profiles = await _profile_entities(driver, sample_size=5, flavour=AgeFlavour())

        census = [q for q in driver.queries if 'count(n) AS cnt' in q]
        assert census, driver.queries
        assert all('n.labels AS lbls' in q for q in census), census
        assert all('labels(n) AS lbls' not in q for q in census), census
        # The abstract supertype is only visible through the flavour's census.
        assert 'Actor' in profiles, profiles
        assert 'Persona' in profiles, profiles
        assert 'Entity' not in profiles  # internal bookkeeping label, filtered as before
        # The label-less census row (lbls: None) contributes nothing and crashes nothing.
        assert set(profiles) == {'Actor', 'Persona'}, profiles

    @pytest.mark.asyncio
    async def test_no_flavour_keeps_the_historic_falkordb_census(self):
        """Back-compat pin: existing callers get today's literal, byte for byte."""
        driver = _AgeStubDriver()

        result = await profile_graph(driver, sample_size=5)

        assert 'MATCH (n) RETURN labels(n) AS lbls, count(n) AS cnt' in driver.queries, (
            driver.queries
        )
        # ...and with it, today's leaf-only reading of the graph.
        profiles = result['entity_profiles']
        assert set(profiles) == {'Persona'}, profiles

    @pytest.mark.asyncio
    async def test_a_raising_label_probe_degrades_one_label_not_the_profile(self):
        """AGE supertypes have no label table: `MATCH (n:`Actor`)` raises. One entry
        may degrade; the whole profile may not sink."""
        driver = _AgeStubDriver(unmatchable={'Actor'})

        result = await profile_graph(driver, sample_size=5, flavour=AgeFlavour())

        profiles = result['entity_profiles']
        assert set(profiles) == {'Actor', 'Persona'}, profiles
        assert profiles['Actor']['count'] == 5
        assert profiles['Actor']['properties'] == {}
        assert profiles['Actor']['sampled'] is False
        # The matchable label is unaffected and honestly reports its sample.
        assert profiles['Persona']['sampled'] is True
        assert 'documento' in profiles['Persona']['properties'], profiles['Persona']

    @pytest.mark.asyncio
    async def test_a_label_probe_returning_no_rows_is_reported_unsampled(self):
        driver = _AgeStubDriver(unmatchable={'Actor'}, unmatchable_raises=False)

        profiles = await _profile_entities(driver, sample_size=5, flavour=AgeFlavour())

        assert profiles['Actor'] == {'count': 5, 'properties': {}, 'sampled': False}
        assert profiles['Persona']['sampled'] is True


class TestFlavourAwareNodeProperties:
    """Overview parity: an AGE leaf flagged `sampled: true` must carry real
    properties, not an empty dict beside a positive assertion that it sampled."""

    @pytest.mark.asyncio
    async def test_age_shaped_samples_yield_real_property_profiles(self):
        driver = _AgeStubDriver()

        profiles = await _profile_entities(driver, sample_size=5, flavour=AgeFlavour())

        persona = profiles['Persona']
        assert persona['sampled'] is True
        # The domain field lives in the nested map; a top-level-only reader
        # returned `properties: {}` for all 23 AGE types live.
        assert 'documento' in persona['properties'], persona['properties']
        assert persona['properties']['documento']['sample_values'] == ['X1234567L']
        assert 'name' in persona['properties']
        # The transport container is not itself a domain property.
        assert 'attributes' not in persona['properties'], persona['properties']

    @pytest.mark.asyncio
    async def test_age_enrichment_reads_the_nested_path(self):
        """The full-scan probe must follow the same path as the sample, or it
        overwrites a real 1.0 coverage with a silent 0.0."""
        driver = _AgeStubDriver()

        profiles = await _profile_entities(driver, sample_size=5, flavour=AgeFlavour())

        probes = [q for q in driver.queries if 'COUNT(DISTINCT' in q and 'documento' in q]
        assert probes, driver.queries
        assert all('n.attributes.`documento`' in q for q in probes), probes
        documento = profiles['Persona']['properties']['documento']
        assert documento['distinct_count'] == 2, documento
        assert documento['coverage'] == 1.0, documento

    @pytest.mark.asyncio
    async def test_falkordb_shaped_samples_are_untouched(self):
        """No flavour → identity normalizer → today's behaviour byte for byte,
        including the enrichment query text."""
        driver = _make_driver({
            'labels(n) AS lbls': [{'lbls': ['Persona', 'Entity'], 'cnt': 5}],
            'MATCH (n:`Persona`) RETURN n LIMIT': [
                {'n': {'name': 'OMAR MOHAMED', 'documento': 'X1234567L'}},
            ],
            'COUNT(DISTINCT n.`documento`)': [{'distinct_count': 2, 'non_null_count': 5}],
            'n.`documento` AS val, COUNT(*)': [{'val': 'X1234567L', 'freq': 3}],
        })

        profiles = await _profile_entities(driver, sample_size=5)

        persona = profiles['Persona']
        assert persona['sampled'] is True
        assert persona['properties']['documento']['sample_values'] == ['X1234567L']
        assert persona['properties']['documento']['distinct_count'] == 2
        assert persona['properties']['documento']['coverage'] == 1.0

    @pytest.mark.asyncio
    async def test_the_base_enrichment_query_text_is_byte_identical(self):
        """The accessor seam must render the historic literals verbatim on the
        base path — this is the whole back-compat claim for FalkorDB."""
        seen: list[str] = []

        class _Recorder:
            async def execute_query(self, query: str, *a, **k):
                seen.append(query)
                return [{'distinct_count': 2, 'non_null_count': 5}], None, None

        await _enrich_value_stats(
            _Recorder(), 'Occurrence', 10, {'ataChapter': {'coverage': 1.0, 'sample_values': []}}
        )

        assert seen[0] == (
            'MATCH (n:`Occurrence`) WHERE n.`ataChapter` IS NOT NULL '
            'RETURN COUNT(DISTINCT n.`ataChapter`) AS distinct_count, '
            'COUNT(n) AS non_null_count'
        ), seen[0]
        assert seen[1] == (
            'MATCH (n:`Occurrence`) WHERE n.`ataChapter` IS NOT NULL '
            'RETURN n.`ataChapter` AS val, COUNT(*) AS freq '
            'ORDER BY freq DESC LIMIT 10'
        ), seen[1]

    def test_extract_property_profiles_without_a_flavour_is_the_historic_reader(self):
        """Back-compat pin: the nested map stays an opaque value on the base path."""
        records = [{'n': {'name': 'OMAR', 'attributes': {'documento': 'X1'}}}]

        profiles = _extract_property_profiles(records, sample_size=5)

        assert 'documento' not in profiles
        assert 'attributes' in profiles


class TestFlavourAwareRelationshipPatterns:
    @pytest.mark.asyncio
    async def test_pattern_census_comes_from_the_flavour(self):
        driver = _AgeStubDriver()

        profiles = await _profile_relationships(driver, sample_size=5, flavour=AgeFlavour())

        probes = [q for q in driver.queries if 'AS source_labels' in q]
        assert probes, driver.queries
        assert all('s.labels AS source_labels' in q for q in probes), probes
        assert all('labels(s) AS source_labels' not in q for q in probes), probes
        # EQUALITY, not membership: the announced leaf column is what makes the
        # endpoint deterministic. The hierarchy list itself is unordered, so a
        # positional pick over it could name any non-internal member.
        assert profiles['ES_DETENIDO']['source_target_patterns'] == [
            ['Persona', 'Detencion']
        ], profiles['ES_DETENIDO']

    @pytest.mark.asyncio
    async def test_the_out_degree_probe_receives_the_leaf_label(self):
        """The endpoint feeds a label-scoped probe. A supertype endpoint makes it
        MATCH a label with no label table — a silent 0.0 where AGE has a real
        number. `unmatchable={'Actor'}` is exactly that trap."""
        driver = _AgeStubDriver(unmatchable={'Actor'})

        profiles = await _profile_relationships(driver, sample_size=5, flavour=AgeFlavour())

        degree_probes = [q for q in driver.queries if 'avg(deg) AS avg_deg' in q]
        assert degree_probes, driver.queries
        assert all('(s:`Persona`)' in q for q in degree_probes), degree_probes
        assert profiles['ES_DETENIDO']['avg_out_degree'] == 1.5

    @pytest.mark.asyncio
    async def test_two_leaves_under_one_supertype_stay_two_patterns(self):
        """`if pair not in patterns` dedups on the announced endpoint, so a
        supertype-level pick merges Persona and Empresa into one `Actor` row and
        the Overview tab loses a relationship pattern outright."""
        driver = _AgeStubDriver(sibling_leaves=True)

        profiles = await _profile_relationships(driver, sample_size=5, flavour=AgeFlavour())

        assert profiles['ES_DETENIDO']['source_target_patterns'] == [
            ['Persona', 'Detencion'],
            ['Empresa', 'Detencion'],
        ], profiles['ES_DETENIDO']

    @pytest.mark.asyncio
    async def test_no_flavour_keeps_the_historic_pattern_endpoints(self):
        """Back-compat pin: the base census still reads `labels(s)`/`labels(t)`,
        announces NO leaf column, and so keeps the positional pick."""
        driver = _AgeStubDriver()

        profiles = await _profile_relationships(driver, sample_size=5)

        probes = [q for q in driver.queries if 'RETURN DISTINCT' in q]
        assert probes and all('labels(s)' in q and 'labels(t)' in q for q in probes), probes
        assert all('source_leaf' not in q for q in probes), probes
        assert profiles['ES_DETENIDO']['source_target_patterns'] == [['Persona', 'Detencion']]

    @pytest.mark.asyncio
    async def test_an_unmatchable_endpoint_does_not_sink_relationship_profiling(self):
        """The out-degree probe is label-scoped too — same AGE hazard, same rule."""
        driver = _AgeStubDriver(unmatchable={'Actor', 'Persona'})

        profiles = await _profile_relationships(driver, sample_size=5, flavour=AgeFlavour())

        assert 'ES_DETENIDO' in profiles, profiles
        rel = profiles['ES_DETENIDO']
        assert rel['count'] == 2
        assert rel['avg_out_degree'] == 0.0
        assert rel['sample_paths'] == [
            {'source': 'OMAR MOHAMED', 'target': 'Detencion 4/2026'}
        ]

    @pytest.mark.asyncio
    async def test_a_null_average_out_degree_reads_as_zero(self):
        """The other half of the unmatchable endpoint: a backend that answers with
        no rows rather than raising. `avg(deg)` over zero rows is one row holding a
        NULL, and `float(None)` would sink every relationship profile."""
        driver = _make_driver({
            'type(r) AS rel_type': [{'rel_type': 'ES_DETENIDO', 'cnt': 2}],
            'RETURN DISTINCT': [
                {'source_labels': ['Entity', 'Persona'], 'target_labels': ['Detencion']}
            ],
            'avg(deg) AS avg_deg': [{'avg_deg': None}],
        })

        profiles = await _profile_relationships(driver, sample_size=5, flavour=AgeFlavour())

        assert profiles['ES_DETENIDO']['avg_out_degree'] == 0.0


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
                {'source_labels': ['Airline', 'Entity'], 'target_labels': ['Aircraft', 'Entity']},
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


class TestServerWiring:
    """The seam only reaches the Overview tab if the tool passes the flavour."""

    @pytest.mark.asyncio
    async def test_profile_data_tool_passes_the_service_flavour(self, monkeypatch):
        import graphiti_mcp_server as srv

        driver = _AgeStubDriver()

        class _Client:
            def __init__(self, d):
                self.driver = d

        class _Service:
            flavour = AgeFlavour()

            async def get_client(self):
                return _Client(driver)

        monkeypatch.setattr(srv, 'graphiti_service', _Service())

        result = await srv.profile_data(sample_size=5)

        assert 'error' not in result, result
        census = [q for q in driver.queries if 'count(n) AS cnt' in q]
        assert census and all('n.labels AS lbls' in q for q in census), census
        assert 'Actor' in result['entity_profiles'], result['entity_profiles']
