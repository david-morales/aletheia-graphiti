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
