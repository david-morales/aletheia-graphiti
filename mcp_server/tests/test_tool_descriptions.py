# mcp_server/tests/test_tool_descriptions.py
import pytest
from domain_profile import DomainProfile, EntityTypeInfo, EdgeTypeInfo
from tool_descriptions import (
    build_instructions,
    build_search_description,
    build_explore_entity_description,
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
        assert 'explore_entity' in instructions
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
        desc = build_explore_entity_description(make_test_profile())
        # Should include at least one sample entity name
        assert 'PH-KZB' in desc or 'Runway excursion' in desc

    def test_redirects_to_search(self):
        desc = build_explore_entity_description(make_test_profile())
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


class TestTheOntologyToolsClaimClassificationQuestions:
    """A tool answers only the questions it CLAIMS.

    How a dataset's categorical values are organised — which classification a
    value belongs to, what groups a scheme defines — lives in the companion
    ontology as individuals with memberships, and in the data graph nowhere at
    all. While these descriptions framed the ontology purely as schema
    documentation ("entity types, properties, formal schema"), a routing agent
    read classification questions as data questions and answered them from its
    own world knowledge instead of from the dataset's scheme.
    """

    _BUILDERS = (build_search_ontology_description, build_explore_ontology_description)

    @pytest.mark.parametrize('build', _BUILDERS)
    def test_claims_classification_questions(self, build):
        desc = build(make_test_profile()).lower()
        assert 'classification' in desc

    @pytest.mark.parametrize('build', _BUILDERS)
    def test_claims_grouping_and_scheme_questions(self, build):
        desc = build(make_test_profile()).lower()
        assert 'grouping' in desc or 'scheme' in desc

    @pytest.mark.parametrize('build', _BUILDERS)
    def test_says_the_memberships_are_held_as_individuals(self, build):
        desc = build(make_test_profile()).lower()
        assert 'individual' in desc
        assert 'membership' in desc

    @pytest.mark.parametrize('build', _BUILDERS)
    def test_says_they_are_absent_from_the_data_graph(self, build):
        desc = build(make_test_profile()).lower()
        assert 'data graph' in desc

    @pytest.mark.parametrize('build', _BUILDERS)
    def test_the_existing_schema_documentation_claims_survive(self, build):
        """The new claim is ADDITIVE — the tools still document the schema."""
        desc = build(make_test_profile()).lower()
        assert 'propert' in desc
        assert 'ontology' in desc

    def test_search_ontology_keeps_its_formal_schema_claim(self):
        desc = build_search_ontology_description(make_test_profile())
        assert 'formal schema behind the data' in desc

    def test_explore_ontology_keeps_its_hierarchy_claim(self):
        desc = build_explore_ontology_description(make_test_profile())
        assert 'class hierarchy' in desc

    @pytest.mark.parametrize('build', _BUILDERS)
    def test_the_builders_name_no_domain(self, build):
        """Domain names may only arrive interpolated FROM the profile."""
        import inspect

        import tool_descriptions

        source = inspect.getsource(getattr(tool_descriptions, build.__name__)).lower()
        for term in ('aircraft', 'aviation', 'crime', 'legal', 'penal', 'police'):
            assert term not in source, f'builder hardcodes the domain term {term!r}'


class TestGraphQueryDefersClassificationQuestions:
    """The other half of the same routing fix, on the tool that was absorbing them."""

    def test_has_a_do_not_use_bullet_for_classification_questions(self):
        from tool_descriptions import build_graph_query_description

        desc = build_graph_query_description(make_test_profile())
        block = desc.split('Do NOT use when:')[1].split('Guardrails:')[0].lower()
        assert 'classification' in block or 'taxonomy' in block

    def test_the_bullet_redirects_to_the_ontology_tools(self):
        from tool_descriptions import build_graph_query_description

        desc = build_graph_query_description(make_test_profile())
        block = desc.split('Do NOT use when:')[1].split('Guardrails:')[0].lower()
        line = next(l for l in block.splitlines() if 'classification' in l or 'taxonomy' in l)
        assert 'ontology' in line

    def test_the_existing_do_not_use_bullets_survive(self):
        from tool_descriptions import build_graph_query_description

        desc = build_graph_query_description(make_test_profile())
        block = desc.split('Do NOT use when:')[1].split('Guardrails:')[0]
        assert '- You need semantic similarity search -- use search instead' in block
        assert '- You need to discover entities by natural language -- use search instead' in block

    def test_the_aggregation_paragraph_is_byte_identical(self):
        """Step 3's measured routing win. It is not in scope for this change."""
        from tool_descriptions import build_graph_query_description

        desc = build_graph_query_description(make_test_profile())
        expected = (
            'THE AGGREGATION SURFACE. Cypher is the only tool here that computes over '
            'the graph rather than retrieving from it: counts, rankings and superlatives '
            'are evaluated against EVERY matching row in the whole graph, not against a '
            'retrieved sample -- unless your own query limits its input first, since a '
            'LIMIT placed before the aggregation truncates what it sees. Written without '
            'one, a count is exact and an ORDER BY ... LIMIT n is the real top n. search '
            'and explore_entity return relevance-ranked samples and can answer none of '
            'these, however often they are called.'
        )
        assert expected in desc

    def test_the_use_when_block_is_byte_identical(self):
        from tool_descriptions import build_graph_query_description

        desc = build_graph_query_description(make_test_profile())
        expected = (
            'Use when:\n'
            '- You need counts, aggregations, comparisons, or gap detection\n'
            '- You need a ranking or a superlative (most/least/largest/first/longest) '
            'over the whole graph\n'
            '- You need an exhaustive answer rather than the best few matches\n'
            '- You need path queries or pattern matching beyond what search provides\n'
            '- You want to compute metrics over the graph structure'
        )
        assert expected in desc

    def test_the_guardrail_text_is_byte_identical(self):
        from tool_descriptions import build_graph_query_description

        desc = build_graph_query_description(make_test_profile())
        expected = (
            'Guardrails: read-only (no CREATE/DELETE/SET), auto-limited to 200 rows '
            '(the cap bounds the rows RETURNED, not the rows aggregated over),\n'
            'LLM syntax auto-corrected (smart quotes, code blocks, missing RETURN).'
        )
        assert expected in desc


class TestFlavourAwareDescriptions:
    """ADR-019 R1/R6: dialect short-form comes from the flavour, not hardcoded."""

    def test_run_cypher_description_falkordb_dialect(self):
        from tool_descriptions import build_graph_query_description
        from flavours.falkordb import FalkorDbFlavour
        desc = build_graph_query_description(make_test_profile(), FalkorDbFlavour())
        assert 'Dialect notes:' in desc
        assert 'FalkorDB openCypher' in desc
        assert 'toLower' in desc

    def test_run_cypher_description_age_dialect(self):
        from tool_descriptions import build_graph_query_description
        from flavours.age import AgeFlavour
        desc = build_graph_query_description(make_test_profile(), AgeFlavour())
        assert 'Apache AGE openCypher' in desc
        assert 'n.attributes' in desc
        assert 'never name a variable `id`' in desc
        assert 'no $param bindings' in desc
        assert '(n:Label)' in desc
        # NOT the FalkorDB summary
        assert 'FalkorDB openCypher: no APOC' not in desc

    def test_instructions_surface_the_age_parameter_rule(self):
        from flavours.age import AgeFlavour
        instr = build_instructions(make_test_profile(), AgeFlavour())
        assert 'Cypher dialect' in instr
        assert 'no $param bindings' in instr

    def test_run_cypher_description_no_flavour_has_no_dialect_block(self):
        from tool_descriptions import build_graph_query_description
        desc = build_graph_query_description(make_test_profile())
        assert 'Dialect notes:' not in desc

    def test_instructions_include_flavour_dialect(self):
        from flavours.age import AgeFlavour
        instr = build_instructions(make_test_profile(), AgeFlavour())
        assert 'Cypher dialect' in instr
        assert 'Apache AGE openCypher' in instr

    def test_instructions_no_flavour_has_no_dialect(self):
        instr = build_instructions(make_test_profile())
        assert 'Cypher dialect:' not in instr


# ---------------------------------------------------------------------------
# Step 3: the served contract must say what `search` CANNOT do
# ---------------------------------------------------------------------------
#
# Measured on a graph-wide ranking question: the agent issued `search` seven
# times (server default limit 10) and `graph_query` zero times, missing the
# ranking fact on 8 of 8 runs across every arm. Nothing in the served contract
# said that `search` returns a relevance-ranked top-k SAMPLE, and nothing named
# `graph_query` as the surface a count, a ranking or a superlative belongs to —
# so an agent reading the announcement had no way to learn either. Repeating a
# sampling call is the rational move when the contract never says it is a sample.

FLAVOUR_ARMS = ['none', 'falkordb', 'age']


def _flavour(name):
    if name == 'falkordb':
        from flavours.falkordb import FalkorDbFlavour
        return FalkorDbFlavour()
    if name == 'age':
        from flavours.age import AgeFlavour
        return AgeFlavour()
    return None


def _sentence(text: str, token: str) -> str:
    """The ONE line of `text` containing `token`, or an assertion failure.

    Every guard below locates the sentence it is about by a token only that
    sentence contains, then asserts the rest of the claim INSIDE that line.

    This indirection is the whole point, and it was learned from a mutation.
    The first cut of these guards asserted 'sample' / 'rank' / 'exhaust' against
    the whole description — and deleting the entire lead paragraph left them
    green, because every one of those words was also supplied by the Do-NOT-use
    bullet. A pin that a DIFFERENT sentence can satisfy does not guard the
    sentence it names. Scoping to one line makes each claim independently
    deletable-and-caught.
    """
    lines = [ln for ln in text.split('\n') if token in ln]
    assert len(lines) == 1, (
        f'expected exactly one line carrying {token!r}, found {len(lines)} — the '
        f'token no longer identifies a single sentence, so the pin below is not '
        f'guarding what it claims to'
    )
    return lines[0]


class TestTheSearchDescriptionAnnouncesItIsASample:
    """The lead paragraph and the routing bullet are pinned SEPARATELY: they are
    two independent claims and either can be deleted without the other."""

    @pytest.mark.parametrize('flavour_name', FLAVOUR_ARMS)
    @pytest.mark.parametrize(
        'claim',
        [
            'relevance-ranked SAMPLE',
            'never an enumeration',
            'no offset or cursor',
            'returns the same sample rather than the next page',
        ],
    )
    def test_the_lead_paragraph_makes_its_own_claims(self, flavour_name, claim):
        desc = build_search_description(make_test_profile(), _flavour(flavour_name))
        lead = _sentence(desc, 'RANKED TOP-K RETRIEVAL')
        assert claim in lead, f'the lead paragraph no longer claims {claim!r}'

    @pytest.mark.parametrize('flavour_name', FLAVOUR_ARMS)
    @pytest.mark.parametrize('claim', ['count', 'ranking', 'superlative', 'graph_query'])
    def test_the_routing_bullet_makes_its_own_claims(self, flavour_name, claim):
        desc = build_search_description(make_test_profile(), _flavour(flavour_name))
        bullet = _sentence(desc, 'a ranked sample cannot answer these')
        assert claim in bullet, f'the routing bullet no longer names {claim!r}'

    @pytest.mark.parametrize('flavour_name', FLAVOUR_ARMS)
    def test_the_exhaustive_intent_is_disambiguated(self, flavour_name):
        """`intent="exhaustive"` is served in the inputSchema enum and collides
        head-on with "never an enumeration". Unresolved, the contract contains
        its own counter-argument."""
        desc = build_search_description(make_test_profile(), _flavour(flavour_name))
        line = _sentence(desc, 'intent="exhaustive"')
        assert 'not an exhaustive answer' in line
        assert 'graph_query' in line

    @pytest.mark.parametrize('flavour_name', FLAVOUR_ARMS)
    def test_the_announced_widened_limit_comes_from_the_code(self, flavour_name):
        """S4: an announcement carrying a hand-typed number drifts the first time
        the constant moves. `INTENT_STRATEGIES` lives in the server module, which
        imports THIS one — so it cannot be imported here without inverting the
        dependency. The guard does what the import would have: read the live
        value and require the served text to agree with it."""
        from graphiti_mcp_server import INTENT_STRATEGIES

        limit = INTENT_STRATEGIES['exhaustive']['limit']
        line = _sentence(
            build_search_description(make_test_profile(), _flavour(flavour_name)),
            'intent="exhaustive"',
        )
        assert f'(limit {limit})' in line, (
            f'the description announces a widened limit that is not {limit}, the '
            f'value INTENT_STRATEGIES actually applies (delimited pin: a bare '
            f'substring match let 50 -> 5 drift escape)'
        )

    @pytest.mark.parametrize('flavour_name', FLAVOUR_ARMS)
    def test_the_closing_example_is_not_an_enumeration_request(self, flavour_name):
        """The example sits in the highest-recency position of the string. One
        reading `Find all X entities` teaches the exact call the rest of the
        description spends four paragraphs arguing against."""
        desc = build_search_description(make_test_profile(), _flavour(flavour_name))
        assert 'Find all' not in desc, (
            'the served example still demonstrates an enumeration request'
        )
        assert 'all ' not in _sentence(desc, 'User:')


class TestTheExploreEntityDescriptionAnnouncesItIsASample:
    @pytest.mark.parametrize(
        'claim',
        ['ranked by proximity', 'not an exhaustive traversal', 'computes nothing'],
    )
    def test_the_lead_paragraph_makes_its_own_claims(self, claim):
        desc = build_explore_entity_description(make_test_profile())
        lead = _sentence(desc, 'NEIGHBORHOOD EXPANSION')
        assert claim in lead, f'the lead paragraph no longer claims {claim!r}'

    @pytest.mark.parametrize('claim', ['count', 'ranking', 'superlative'])
    def test_the_routing_bullet_makes_its_own_claims(self, claim):
        desc = build_explore_entity_description(make_test_profile())
        bullet = _sentence(desc, 'use graph_query instead')
        assert claim in bullet, f'the routing bullet no longer names {claim!r}'

    def test_it_promises_no_exhaustive_neighborhood(self):
        """"everything connected to it" is the same over-promise as "Find all"."""
        assert 'everything' not in build_explore_entity_description(make_test_profile())


class TestTheGraphQueryDescriptionClaimsTheAggregationSurface:
    @pytest.mark.parametrize('flavour_name', FLAVOUR_ARMS)
    @pytest.mark.parametrize(
        'claim',
        [
            'counts, rankings and superlatives',
            'whole graph',
            'not against a retrieved sample',
            'unless your own query limits its input first',
        ],
    )
    def test_the_lead_paragraph_makes_its_own_claims(self, flavour_name, claim):
        from tool_descriptions import build_graph_query_description

        desc = build_graph_query_description(make_test_profile(), _flavour(flavour_name))
        lead = _sentence(desc, 'THE AGGREGATION SURFACE')
        assert claim in lead, f'the lead paragraph no longer claims {claim!r}'

    @pytest.mark.parametrize('flavour_name', FLAVOUR_ARMS)
    def test_the_ranking_use_case_has_its_own_bullet(self, flavour_name):
        from tool_descriptions import build_graph_query_description

        desc = build_graph_query_description(make_test_profile(), _flavour(flavour_name))
        assert 'ranking or a superlative' in _sentence(desc, 'most/least/largest')

    @pytest.mark.parametrize('flavour_name', FLAVOUR_ARMS)
    def test_the_exhaustive_use_case_has_its_own_bullet(self, flavour_name):
        from tool_descriptions import build_graph_query_description

        desc = build_graph_query_description(make_test_profile(), _flavour(flavour_name))
        assert 'exhaustive answer' in _sentence(desc, 'rather than the best few matches')

    @pytest.mark.parametrize('flavour_name', FLAVOUR_ARMS)
    def test_the_row_cap_does_not_read_as_a_cap_on_the_aggregation(self, flavour_name):
        """Read naively, "200 rows are auto-limited" cancels the claim above it."""
        from tool_descriptions import build_graph_query_description

        desc = build_graph_query_description(make_test_profile(), _flavour(flavour_name))
        line = _sentence(desc, 'auto-limited to 200 rows')
        assert 'RETURNED' in line
        assert 'aggregated over' in line


class TestTheStaticDocstringsStayCoherentWithTheServedText:
    """The dynamic descriptions override these at registration — but the docstring
    IS the served description on the DEGRADED path, where `_register_degraded_tools`
    re-adds every dynamic tool with no description at all. A docstring that makes a
    weaker promise than the dynamic text is a second, quieter contract, served
    exactly when the agent is least equipped to notice.
    """

    def test_the_search_docstring_carries_the_sample_claim(self):
        from graphiti_mcp_server import search

        doc = (search.__doc__ or '')
        assert 'SAMPLE' in doc and 'never an enumeration' in doc
        assert 'no offset' in doc
        for word in ('Counts', 'rankings', 'superlatives'):
            assert word in doc, f'the search docstring never names {word!r}'
        assert 'graph_query' in doc

    def test_the_explore_entity_docstring_carries_the_sample_claim(self):
        from graphiti_mcp_server import explore_entity

        doc = (explore_entity.__doc__ or '')
        assert 'not an exhaustive traversal' in doc
        assert 'aggregates nothing' in doc
        for word in ('Counts', 'rankings', 'superlatives'):
            assert word in doc, f'the explore_entity docstring never names {word!r}'
        assert 'graph_query' in doc

    def test_the_explore_entity_docstring_promises_no_exhaustive_neighborhood(self):
        from graphiti_mcp_server import explore_entity

        assert 'everything' not in (explore_entity.__doc__ or ''), (
            'the docstring still promises "everything connected" — the exact claim '
            'the description now denies'
        )

    def test_the_graph_query_docstring_claims_the_aggregation_surface(self):
        from graphiti_mcp_server import graph_query

        doc = (graph_query.__doc__ or '')
        for word in ('aggregation', 'counts', 'rankings', 'superlatives'):
            assert word in doc, f'the graph_query docstring never claims {word!r}'
        assert 'unless your own query limits its input first' in doc
