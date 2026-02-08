# mcp_server/tests/test_tool_descriptions.py
import pytest
from domain_profile import DomainProfile, EntityTypeInfo, EdgeTypeInfo
from tool_descriptions import (
    build_instructions,
    build_search_description,
    build_explore_node_description,
    build_search_ontology_description,
    build_explore_ontology_description,
)


def make_test_profile():
    return DomainProfile(
        group_id='aviation_safety',
        entity_types={
            'Aircraft': EntityTypeInfo('Aircraft', 47, 'Aircraft with type and registration', ['PH-KZB', 'EC-MYC']),
            'Occurrence': EntityTypeInfo('Occurrence', 23, 'Aviation safety occurrence', ['Runway excursion LEMD']),
        },
        edge_types={
            'OPERATED_BY': EdgeTypeInfo('OPERATED_BY', 31, 'Links aircraft to airline', 'Aircraft -> Airline'),
        },
        time_range=('2019-03-10', '2024-11-22'),
    )


class TestBuildInstructions:
    def test_includes_domain_summary(self):
        instructions = build_instructions(make_test_profile())
        assert 'Aircraft' in instructions
        assert '47' in instructions
        assert 'OPERATED_BY' in instructions

    def test_includes_tool_guidance(self):
        instructions = build_instructions(make_test_profile())
        assert 'search' in instructions.lower()
        assert 'explore_node' in instructions
        assert 'Use when' in instructions or 'use when' in instructions

    def test_includes_entity_type_filter_hint(self):
        instructions = build_instructions(make_test_profile())
        assert 'Aircraft' in instructions
        assert 'Occurrence' in instructions

    def test_empty_profile(self):
        empty = DomainProfile(group_id='empty', entity_types={}, edge_types={}, time_range=None)
        instructions = build_instructions(empty)
        assert 'empty' in instructions


class TestSearchDescription:
    def test_includes_when_to_use(self):
        desc = build_search_description(make_test_profile())
        assert 'Use when' in desc or 'use when' in desc.lower()

    def test_includes_when_not_to_use(self):
        desc = build_search_description(make_test_profile())
        assert 'Do NOT use' in desc or 'do not use' in desc.lower() or 'NOT' in desc

    def test_includes_available_types(self):
        desc = build_search_description(make_test_profile())
        assert 'Aircraft' in desc
        assert 'OPERATED_BY' in desc

    def test_includes_example(self):
        desc = build_search_description(make_test_profile())
        assert 'Example' in desc or 'example' in desc


class TestExploreNodeDescription:
    def test_includes_sample_entity(self):
        desc = build_explore_node_description(make_test_profile())
        # Should include at least one sample entity name
        assert 'PH-KZB' in desc or 'Runway excursion' in desc

    def test_redirects_to_search(self):
        desc = build_explore_node_description(make_test_profile())
        assert 'search' in desc.lower()


class TestOntologyDescriptions:
    def test_search_ontology_description(self):
        desc = build_search_ontology_description(make_test_profile())
        assert 'ontology' in desc.lower()
        assert 'schema' in desc.lower() or 'type' in desc.lower()

    def test_explore_ontology_description(self):
        desc = build_explore_ontology_description(make_test_profile())
        assert 'ontology' in desc.lower()
        assert 'propert' in desc.lower()
