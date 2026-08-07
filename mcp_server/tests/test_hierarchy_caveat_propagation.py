"""A-D4: the hierarchy caveat must reach every surface built from the profile.

On the AGE backend the label census counts the FULL ontology hierarchy, so the
profile carries abstract supertypes that no vertex is stored under. The flavour
already states the consequence as data (`census_notes()`): `MATCH (n:Actor)`
matches ZERO rows, silently, which reads as "no Actors exist".

That caveat reached exactly one surface — `get_schema.analysis_notes`. The three
other surfaces built from the same `DomainProfile` advertised the hierarchy labels
as ordinary, usable types. Worst of them: `_build_example_queries` generated
`MATCH (s:Actor)-[:EJECUTADO_POR]->(t:Agente)` as the FIRST thing an agent reads
about `run_cypher` — three example queries that return nothing, on the server's
own account.
"""

from __future__ import annotations

import pytest

from domain_profile import DomainProfile, EdgeTypeInfo, EntityTypeInfo
from flavours.age import AgeFlavour
from flavours.falkordb import FalkorDbFlavour
from tool_descriptions import (
    _build_example_queries,
    build_instructions,
    build_run_cypher_description,
)

HIERARCHY_LABELS = ('Actor', 'Event')
STORAGE_LABELS = ('Agente', 'Persona')


def _mixed_profile() -> DomainProfile:
    """An AGE-shaped profile: two hierarchy-only labels, two storage labels.

    `Actor` sorts first, which is exactly how it reached the generated examples —
    the templates fall back to `entity_type_names()[0]`.
    """
    return DomainProfile(
        group_id='hierarchy_graph',
        entity_types={
            'Actor': EntityTypeInfo('Actor', 619, 'Abstract supertype', [], hierarchy=True),
            'Event': EntityTypeInfo('Event', 375, 'Abstract supertype', [], hierarchy=True),
            'Agente': EntityTypeInfo('Agente', 334, 'An officer', ['Z18']),
            'Persona': EntityTypeInfo('Persona', 197, 'A person', ['Ada']),
        },
        edge_types={
            'EJECUTADO_POR': EdgeTypeInfo('EJECUTADO_POR', 300, 'Carried out by', ''),
        },
        time_range=None,
    )


def _flat_profile() -> DomainProfile:
    """A FalkorDB-shaped profile: every censused label is a storage label."""
    return DomainProfile(
        group_id='flat_graph',
        entity_types={
            'Agente': EntityTypeInfo('Agente', 334, 'An officer', ['Z18']),
            'Persona': EntityTypeInfo('Persona', 197, 'A person', ['Ada']),
        },
        edge_types={
            'EJECUTADO_POR': EdgeTypeInfo('EJECUTADO_POR', 300, 'Carried out by', ''),
        },
        time_range=None,
    )


# --- the profile must know which labels are hierarchy-only ------------------

def test_entity_type_info_defaults_to_a_storage_label():
    assert EntityTypeInfo('Widget', 1, '').hierarchy is False


def test_the_profile_can_name_its_storage_labels_only():
    profile = _mixed_profile()
    assert profile.entity_type_names() == ['Actor', 'Agente', 'Event', 'Persona']
    assert profile.storage_entity_type_names() == ['Agente', 'Persona']


def test_a_flat_profile_reports_every_label_as_storage():
    profile = _flat_profile()
    assert profile.storage_entity_type_names() == profile.entity_type_names()


# --- the caveat reaches the instructions ------------------------------------

def test_the_instructions_carry_the_flavour_census_caveat():
    text = build_instructions(_mixed_profile(), AgeFlavour())
    caveat = AgeFlavour().census_notes()[0]
    assert caveat in text, 'the AGE hierarchy caveat never reached build_instructions'


def test_the_instructions_mark_each_hierarchy_label_where_it_is_listed():
    """A caveat at the bottom does not help an agent reading the type list."""
    text = build_instructions(_mixed_profile(), AgeFlavour())
    for line in text.split('\n'):
        for label in HIERARCHY_LABELS:
            if line.startswith(f'- {label} ('):
                assert 'hierarchy' in line.lower(), f'unmarked hierarchy label: {line!r}'


def test_a_flavour_without_caveats_adds_nothing():
    """FalkorDB stores every label it censuses — it needs no reading instructions.

    Scoped to the two things this change introduces; the catalog's own mention of
    `sample_subgraph`'s unordered label hierarchy is unrelated prose.
    """
    text = build_instructions(_flat_profile(), FalkorDbFlavour())
    assert "How to read this graph's labels and counts:" not in text
    assert '[hierarchy label' not in text


def test_instructions_without_a_flavour_still_build():
    """The no-flavour path (Neo4j / plain callers) must not regress."""
    text = build_instructions(_flat_profile())
    assert 'Agente' in text


# --- the examples must be runnable ------------------------------------------

@pytest.mark.parametrize('label', HIERARCHY_LABELS)
def test_generated_examples_never_match_on_a_hierarchy_label(label):
    for example in _build_example_queries(_mixed_profile()):
        assert f'(s:{label})' not in example, example
        assert f'(t:{label})' not in example, example
        assert f'(n:{label})' not in example, example


def test_generated_examples_still_exist_and_use_storage_labels():
    """Dropping the bad examples must not leave the tool with none."""
    examples = _build_example_queries(_mixed_profile())
    assert examples, 'the templates produced nothing at all'
    assert any(any(f':{s}' in ex for s in STORAGE_LABELS) for ex in examples)


def test_the_run_cypher_description_carries_only_runnable_examples():
    desc = build_run_cypher_description(_mixed_profile(), AgeFlavour())
    for label in HIERARCHY_LABELS:
        assert f'(s:{label})' not in desc
        assert f'(n:{label})' not in desc


def test_a_profile_of_nothing_but_hierarchy_labels_yields_no_examples():
    """Better no example than one that returns zero rows by construction."""
    profile = DomainProfile(
        group_id='all_abstract',
        entity_types={
            'Actor': EntityTypeInfo('Actor', 9, 'Abstract', ['x'], hierarchy=True),
        },
        edge_types={'REL': EdgeTypeInfo('REL', 1, 'rel', '')},
        time_range=None,
    )
    assert _build_example_queries(profile) == []


def test_an_explicit_pattern_naming_a_hierarchy_label_is_not_used():
    """The edge pattern is data too — it can name an unmatchable label."""
    profile = _mixed_profile()
    profile.edge_types['EJECUTADO_POR'].source_target_pattern = 'Actor -> Agente'
    for example in _build_example_queries(profile):
        assert '(s:Actor)' not in example, example
