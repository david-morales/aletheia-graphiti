import pytest
from domain_profile import DomainProfile, EntityTypeInfo, EdgeTypeInfo


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
