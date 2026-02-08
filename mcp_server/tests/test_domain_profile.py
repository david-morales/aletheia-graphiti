from unittest.mock import AsyncMock, MagicMock

import pytest
from domain_profile import DomainProfile, EntityTypeInfo, EdgeTypeInfo, build_domain_profile


class TestDomainProfileDataclass:
    def test_create_entity_type_info(self):
        info = EntityTypeInfo(
            label='Aircraft',
            count=47,
            description='Aircraft involved in an occurrence',
            sample_names=['PH-KZB', 'EC-MYC', 'G-VKSS'],
        )
        assert info.label == 'Aircraft'
        assert info.count == 47
        assert len(info.sample_names) == 3

    def test_create_edge_type_info(self):
        info = EdgeTypeInfo(
            name='OPERATED_BY',
            count=31,
            description='Links aircraft to their operating airline',
            source_target_pattern='Aircraft -> Airline',
        )
        assert info.name == 'OPERATED_BY'
        assert info.count == 31

    def test_create_domain_profile(self):
        profile = DomainProfile(
            group_id='aviation_safety',
            entity_types={
                'Aircraft': EntityTypeInfo('Aircraft', 47, 'Aircraft entity', ['PH-KZB']),
            },
            edge_types={
                'OPERATED_BY': EdgeTypeInfo('OPERATED_BY', 31, 'Operator link', 'Aircraft -> Airline'),
            },
            time_range=('2019-03-10', '2024-11-22'),
        )
        assert profile.group_id == 'aviation_safety'
        assert len(profile.entity_types) == 1
        assert len(profile.edge_types) == 1

    def test_empty_profile(self):
        profile = DomainProfile(
            group_id='empty_graph',
            entity_types={},
            edge_types={},
            time_range=None,
        )
        assert profile.group_id == 'empty_graph'
        assert len(profile.entity_types) == 0


def make_mock_driver(label_records, edge_records, sample_records_by_label, time_records=None):
    """Create a mock driver that returns different results per query."""
    driver = AsyncMock()

    async def mock_execute(query, **kwargs):
        # FalkorDB returns (records, header, None)
        if 'labels(n)' in query and 'count' in query:
            header = ['entity_type', 'count']
            return (label_records, header, None)
        elif 'type(r)' in query and 'count' in query:
            header = ['relationship_type', 'count']
            return (edge_records, header, None)
        elif 'n.name' in query and 'LIMIT' in query:
            label = kwargs.get('label', '')
            records = sample_records_by_label.get(label, [])
            header = ['name']
            return (records, header, None)
        elif 'created_at' in query or 'valid_at' in query:
            header = ['earliest', 'latest']
            return (time_records or [], header, None)
        return ([], [], None)

    driver.execute_query = mock_execute
    return driver


class TestBuildDomainProfile:
    @pytest.mark.asyncio
    async def test_builds_profile_from_data_graph(self):
        label_records = [
            {'entity_type': ['Entity', 'Aircraft'], 'count': 47},
            {'entity_type': ['Entity', 'Occurrence'], 'count': 23},
        ]
        edge_records = [
            {'relationship_type': 'OPERATED_BY', 'count': 31},
            {'relationship_type': 'RELATES_TO', 'count': 50},
        ]
        sample_records = {
            'Aircraft': [{'name': 'PH-KZB'}, {'name': 'EC-MYC'}],
            'Occurrence': [{'name': 'Runway excursion LEMD'}],
        }

        driver = make_mock_driver(label_records, edge_records, sample_records)
        mock_client = MagicMock()
        mock_client.driver = driver

        profile = await build_domain_profile(mock_client, group_id='aviation_safety')

        assert profile.group_id == 'aviation_safety'
        assert 'Aircraft' in profile.entity_types
        assert profile.entity_types['Aircraft'].count == 47
        assert 'PH-KZB' in profile.entity_types['Aircraft'].sample_names
        assert 'Occurrence' in profile.entity_types
        assert 'OPERATED_BY' in profile.edge_types
        assert profile.edge_types['OPERATED_BY'].count == 31

    @pytest.mark.asyncio
    async def test_filters_out_entity_and_episodic_labels(self):
        """Labels like 'Entity' and 'Episodic' are internal -- should not appear as types."""
        label_records = [
            {'entity_type': ['Entity', 'Aircraft'], 'count': 10},
        ]
        driver = make_mock_driver(label_records, [], {})
        mock_client = MagicMock()
        mock_client.driver = driver

        profile = await build_domain_profile(mock_client, group_id='test')

        assert 'Entity' not in profile.entity_types
        assert 'Aircraft' in profile.entity_types

    @pytest.mark.asyncio
    async def test_handles_empty_graph(self):
        driver = make_mock_driver([], [], {})
        mock_client = MagicMock()
        mock_client.driver = driver

        profile = await build_domain_profile(mock_client, group_id='empty')

        assert len(profile.entity_types) == 0
        assert len(profile.edge_types) == 0


class TestOntologyEnrichment:
    @pytest.mark.asyncio
    async def test_enriches_entity_descriptions_from_ontology(self):
        """Entity types get descriptions from ontology node summaries."""
        label_records = [
            {'entity_type': ['Entity', 'Aircraft'], 'count': 10},
        ]
        driver = make_mock_driver(label_records, [], {'Aircraft': [{'name': 'PH-KZB'}]})
        mock_client = MagicMock()
        mock_client.driver = driver

        # Ontology client returns nodes with summaries
        ontology_client = MagicMock()
        ontology_driver = AsyncMock()

        async def ontology_execute(query, **kwargs):
            if 'n.name' in query and 'n.summary' in query:
                return (
                    [{'name': 'Aircraft', 'summary': 'Aircraft involved in an occurrence with type and registration'}],
                    ['name', 'summary'],
                    None,
                )
            return ([], [], None)

        ontology_driver.execute_query = ontology_execute
        ontology_client.driver = ontology_driver

        profile = await build_domain_profile(mock_client, 'test', ontology_client=ontology_client)

        assert 'Aircraft' in profile.entity_types
        assert 'registration' in profile.entity_types['Aircraft'].description

    @pytest.mark.asyncio
    async def test_works_without_ontology_client(self):
        """Profile builds fine with no ontology client — descriptions stay empty."""
        label_records = [
            {'entity_type': ['Entity', 'Aircraft'], 'count': 10},
        ]
        driver = make_mock_driver(label_records, [], {})
        mock_client = MagicMock()
        mock_client.driver = driver

        profile = await build_domain_profile(mock_client, 'test', ontology_client=None)

        assert profile.entity_types['Aircraft'].description == ''


class TestDomainProfileRendering:
    def _make_profile(self):
        return DomainProfile(
            group_id='aviation_safety',
            entity_types={
                'Aircraft': EntityTypeInfo('Aircraft', 47, 'Aircraft with type and registration', ['PH-KZB', 'EC-MYC']),
                'Occurrence': EntityTypeInfo('Occurrence', 23, 'Aviation safety occurrence', ['Runway excursion LEMD']),
            },
            edge_types={
                'OPERATED_BY': EdgeTypeInfo('OPERATED_BY', 31, 'Links aircraft to airline', 'Aircraft -> Airline'),
                'LOCATED_IN': EdgeTypeInfo('LOCATED_IN', 18, 'Links to country', 'Airport -> Country'),
            },
            time_range=('2019-03-10', '2024-11-22'),
        )

    def test_render_domain_summary(self):
        profile = self._make_profile()
        summary = profile.render_domain_summary()
        assert 'aviation_safety' in summary
        assert 'Aircraft' in summary
        assert '47' in summary
        assert 'OPERATED_BY' in summary
        assert '2019-03-10' in summary

    def test_render_entity_catalog(self):
        profile = self._make_profile()
        catalog = profile.render_entity_catalog()
        assert '## Aircraft' in catalog
        assert 'PH-KZB' in catalog
        assert 'Aircraft with type and registration' in catalog

    def test_render_relationship_types(self):
        profile = self._make_profile()
        rels = profile.render_relationship_types()
        assert 'OPERATED_BY' in rels
        assert 'Aircraft -> Airline' in rels

    def test_render_domain_summary_empty_graph(self):
        profile = DomainProfile(group_id='empty', entity_types={}, edge_types={}, time_range=None)
        summary = profile.render_domain_summary()
        assert 'empty' in summary
        assert 'no entities' in summary.lower() or '0' in summary

    def test_entity_type_names_list(self):
        profile = self._make_profile()
        names = profile.entity_type_names()
        assert 'Aircraft' in names
        assert 'Occurrence' in names

    def test_edge_type_names_list(self):
        profile = self._make_profile()
        names = profile.edge_type_names()
        assert 'OPERATED_BY' in names
        assert 'LOCATED_IN' in names
