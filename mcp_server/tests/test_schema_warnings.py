"""Census-validated property references in the graph_query response.

A property name the graph does not carry is not an error on either backend: the
query runs, the column comes back null for every row, and nothing says so. The
agent reads a clean-looking empty answer and concludes the fact is absent.

These tests pin the warning channel that closes that: it fires only from
MEASURED knowledge (a positive census), it never blocks or rewrites the query,
and it stays silent wherever the census cannot support a verdict.
"""

from __future__ import annotations

import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from utils.schema_warnings import build_schema_warnings


# ---------------------------------------------------------------------------
# Fixtures — census shapes as get_schema publishes them
# ---------------------------------------------------------------------------


def _flat_census(**overrides) -> dict:
    """A FLAT census: domain fields live at the top level, no container."""
    schema = {
        'attribute_container': None,
        'node_labels': {
            'Parte': {
                'count': 20,
                'properties': ['fecha_de_inicio', 'localidad', 'name', 'uuid'],
                'attribute_keys': ['fecha_de_inicio', 'localidad'],
                'sampled': True,
            },
            'Persona': {
                'count': 40,
                'properties': ['fecha_de_nacimiento', 'name', 'nombre_completo', 'uuid'],
                'attribute_keys': ['fecha_de_nacimiento', 'nombre_completo'],
                'sampled': True,
            },
        },
    }
    schema.update(overrides)
    return schema


def _nesting_census(**overrides) -> dict:
    """A NESTING census, shaped as the real one is.

    `properties` is what `keys(n)` returns, and on a nesting backend that is the
    bookkeeping columns PLUS the container itself — so `attributes` IS in the
    property union. An earlier version of this fixture omitted it, which made
    the union look flat-shaped and hid a regression that killed the whole
    unknown-property check on this arm. A fixture that is not honest about the
    shape it stands for tests nothing.
    """
    schema = {
        'attribute_container': 'attributes',
        'node_labels': {
            'Parte': {
                'count': 20,
                'properties': ['attributes', 'created_at', 'group_id', 'name', 'uuid'],
                'attribute_keys': ['fecha_de_inicio', 'localidad'],
                'sampled': True,
            },
        },
    }
    schema.update(overrides)
    return schema


def _joined(warnings: list[str]) -> str:
    return '\n'.join(warnings)


# ---------------------------------------------------------------------------
# Unknown property references
# ---------------------------------------------------------------------------


class TestUnknownProperties:
    def test_property_on_no_censused_label_warns(self):
        w = build_schema_warnings(
            'MATCH (p:Parte) RETURN p.fecha_inicio', _flat_census()
        )
        assert w, 'a property no censused label carries must be reported'
        assert 'fecha_inicio' in _joined(w)

    def test_warning_offers_nearest_census_matches(self):
        w = build_schema_warnings(
            'MATCH (p:Parte) RETURN p.fecha_inicio', _flat_census()
        )
        assert 'fecha_de_inicio' in _joined(w), 'did-you-mean must name the real property'

    def test_property_known_on_another_label_does_not_warn(self):
        """The check is a label-agnostic UNION: no alias->label resolution."""
        w = build_schema_warnings(
            'MATCH (p:Parte) RETURN p.nombre_completo', _flat_census()
        )
        assert w == []

    def test_known_property_does_not_warn(self):
        w = build_schema_warnings(
            'MATCH (p:Parte) RETURN p.fecha_de_inicio', _flat_census()
        )
        assert w == []

    def test_where_clause_reference_is_checked(self):
        w = build_schema_warnings(
            "MATCH (p:Parte) WHERE p.fecha_inicio = '2026-01-23' RETURN p.name",
            _flat_census(),
        )
        assert 'fecha_inicio' in _joined(w)

    def test_reference_after_with_rebinding_is_checked(self):
        """No label binding survives a WITH — the union check does not need one."""
        w = build_schema_warnings(
            'MATCH (p:Parte) WITH p AS q RETURN q.fecha_inicio', _flat_census()
        )
        assert 'fecha_inicio' in _joined(w)

    def test_each_unknown_property_reported_once(self):
        w = build_schema_warnings(
            'MATCH (p:Parte) WHERE p.fecha_inicio IS NOT NULL RETURN p.fecha_inicio',
            _flat_census(),
        )
        assert len([x for x in w if 'fecha_inicio' in x]) == 1

    def test_output_is_deterministic(self):
        query = 'MATCH (p:Parte) RETURN p.zzz_unknown, p.aaa_unknown'
        assert build_schema_warnings(query, _flat_census()) == build_schema_warnings(
            query, _flat_census()
        )


# ---------------------------------------------------------------------------
# Always-allowed properties
# ---------------------------------------------------------------------------


class TestInternalProperties:
    def test_internal_node_properties_never_warn(self):
        for prop in ('uuid', 'name', 'group_id', 'summary', 'created_at', 'labels'):
            w = build_schema_warnings(f'MATCH (n) RETURN n.{prop}', _flat_census())
            assert w == [], f'internal property {prop} must never warn'

    def test_embedding_properties_never_warn(self):
        w = build_schema_warnings('MATCH (n) RETURN n.name_embedding', _flat_census())
        assert w == []

    def test_episodic_properties_never_warn(self):
        for prop in ('content', 'source', 'source_description', 'valid_at'):
            w = build_schema_warnings(f'MATCH (e:Episodic) RETURN e.{prop}', _flat_census())
            assert w == [], f'episodic property {prop} must never warn'

    def test_edge_internal_properties_never_warn(self):
        for prop in ('fact', 'valid_at', 'invalid_at', 'episodes', 'expired_at'):
            w = build_schema_warnings(
                f'MATCH (a)-[r:REL]->(b) RETURN r.{prop}', _flat_census()
            )
            assert w == [], f'edge property {prop} must never warn'

    def test_edge_bound_variable_is_not_judged_against_a_node_census(self):
        """The census describes NODE labels; an edge property is out of its scope."""
        w = build_schema_warnings(
            'MATCH (a)-[r:REL]->(b) RETURN r.peso_relativo', _flat_census()
        )
        assert w == []

    def test_alias_of_an_edge_variable_is_not_judged_either(self):
        w = build_schema_warnings(
            'MATCH (a)-[r:REL]->(b) WITH r AS rel RETURN rel.rol', _flat_census()
        )
        assert w == []

    def test_node_reference_still_checked_alongside_an_edge_one(self):
        w = build_schema_warnings(
            'MATCH (a:Parte)-[r:REL]->(b) RETURN r.peso_relativo, a.fecha_inicio',
            _flat_census(),
        )
        assert 'fecha_inicio' in _joined(w)
        assert 'peso_relativo' not in _joined(w)


# ---------------------------------------------------------------------------
# Only NODE-bound references are judged
# ---------------------------------------------------------------------------


class TestOnlyNodeBoundReferencesAreJudged:
    """A node-label census can only speak about nodes.

    Every probe here is a VALID query. Judging its non-node atoms against a
    node census told the model to edit working Cypher on the connector's
    authority — the worst failure this channel can have, because the advice
    is both wrong and confidently sourced.
    """

    def test_map_projection_alias_is_not_judged(self):
        """This is how a model writes top-N-per-group — it lands on the guard cells."""
        w = build_schema_warnings(
            'MATCH (p:Parte) WITH {mun: p.localidad, n: count(*)} AS agg '
            'RETURN agg.mun, agg.n',
            _flat_census(),
        )
        assert w == []

    def test_unwound_row_alias_is_not_judged(self):
        w = build_schema_warnings(
            'MATCH (p:Parte) '
            'WITH collect({localidad: p.localidad, total: 1}) AS rows '
            'UNWIND rows AS row RETURN row.localidad, row.total',
            _flat_census(),
        )
        assert w == []

    def test_function_result_atom_is_not_judged(self):
        w = build_schema_warnings(
            'MATCH (a)-[r:REL]->(b) RETURN properties(r).rol', _flat_census()
        )
        assert w == []

    def test_constructed_value_atom_is_not_judged(self):
        w = build_schema_warnings(
            'MATCH (p:Parte) RETURN point({latitude: 1, longitude: 2}).latitude',
            _flat_census(),
        )
        assert w == []

    def test_a_node_alias_is_still_judged(self):
        """The allow-list must not silence the case this channel exists for."""
        w = build_schema_warnings(
            'MATCH (p:Parte) WITH p AS q RETURN q.fecha_inicio', _flat_census()
        )
        assert 'fecha_inicio' in _joined(w)


class TestAReturnAliasIsNotARebinding:
    """`RETURN parte.name AS parte` must not silence the rest of the query.

    Measured on a live judged run: this shape — projecting a node property
    under an alias equal to the node's own variable — is the agent's bread and
    butter, and it took the whole cure down. A RETURN alias creates no binding
    that any same-clause reference can see: every `parte.<prop>` in the same
    RETURN and its ORDER BY is evaluated in MATCH scope, where `parte` is the
    node. Treating the alias as a rebinding revoked node-ness for the WHOLE
    query and retro-silenced references that had already been read correctly.
    """

    _CENSUS = {
        'attribute_container': None,
        'node_labels': {
            'ParteDeIntervencion': {
                'count': 20,
                'properties': ['clase_de_actuacion', 'fecha_de_inicio', 'name', 'uuid'],
                'attribute_keys': ['clase_de_actuacion', 'fecha_de_inicio'],
                'sampled': True,
            },
            'Persona': {
                'count': 40,
                'properties': ['name', 'uuid'],
                'attribute_keys': [],
                'sampled': True,
            },
        },
    }

    # The exact query the benchmark agent ran, verbatim.
    _AGENT_QUERY = (
        'MATCH (p:Persona {name: "KHADIJA DAOUD"})-[r1]->(id)'
        '-[r2:EN_PARTE]->(parte:ParteDeIntervencion)\n'
        'RETURN parte.name AS parte, type(r1) AS rol, '
        'parte.fecha_inicio AS fecha, parte.clase_de_actuacion AS clase\n'
        'ORDER BY parte.fecha_inicio'
    )

    def test_the_agent_query_still_gets_its_warning(self):
        w = build_schema_warnings(self._AGENT_QUERY, self._CENSUS)
        assert w, 'the cure went silent on the shape it exists for'
        assert 'fecha_inicio' in _joined(w)

    def test_the_agent_query_warning_carries_did_you_mean(self):
        w = build_schema_warnings(self._AGENT_QUERY, self._CENSUS)
        assert 'fecha_de_inicio' in _joined(w)

    def test_the_correctly_named_columns_are_not_reported(self):
        w = build_schema_warnings(self._AGENT_QUERY, self._CENSUS)
        assert 'clase_de_actuacion' not in _joined(w)

    def test_self_shadow_with_an_alias_equal_to_the_bad_property(self):
        """Run 1's shape: self-shadow AND an alias named like the misspelling."""
        w = build_schema_warnings(
            'MATCH (parte:ParteDeIntervencion) WHERE parte.name IN ["a", "b"] '
            'RETURN parte.name AS parte, parte.fecha_inicio AS fecha_inicio',
            self._CENSUS,
        )
        assert 'fecha_inicio' in _joined(w)

    def test_a_plain_return_alias_does_not_revoke(self):
        w = build_schema_warnings(
            'MATCH (p:ParteDeIntervencion) RETURN count(*) AS p, p.fecha_inicio',
            self._CENSUS,
        )
        assert 'fecha_inicio' in _joined(w)


class TestARebindThatShadowsANodeName:
    """Binding is not add-only: a name can STOP being a node.

    The allow-list closed B1 for fresh names, but `WITH <non-node> AS p` where
    `p` was a node left the old entry standing, so every shape B1 covered came
    back the moment it reused a node's name — and reusing `p` is the natural
    thing to write. The rule has to be symmetric: a rebinding grants node-ness
    or revokes it, never only grants.
    """

    def test_relationship_rebound_onto_a_node_name(self):
        w = build_schema_warnings(
            'MATCH (p:Parte)-[r:REL]->(b) WITH r AS p RETURN p.rol', _flat_census()
        )
        assert w == []

    def test_map_literal_rebound_onto_a_node_name(self):
        w = build_schema_warnings(
            'MATCH (p:Parte) WITH {k: 1} AS p RETURN p.k', _flat_census()
        )
        assert w == []

    def test_unwound_list_rebound_onto_a_node_name(self):
        w = build_schema_warnings(
            'MATCH (p:Parte) WITH [1, 2] AS l UNWIND l AS p RETURN p.zzz', _flat_census()
        )
        assert w == []

    def test_aggregate_rebound_onto_a_node_name(self):
        w = build_schema_warnings(
            'MATCH (p:Parte) WITH count(*) AS p RETURN p.zzz', _flat_census()
        )
        assert w == []

    def test_a_node_to_node_rebinding_still_grants(self):
        """Revoking must not overreach — this is the pinned cure case."""
        w = build_schema_warnings(
            'MATCH (p:Parte) WITH p AS x RETURN x.fecha_inicio', _flat_census()
        )
        assert 'fecha_inicio' in _joined(w)


# ---------------------------------------------------------------------------
# Component access on a real property
# ---------------------------------------------------------------------------


class TestComponentAccess:
    """`p.created_at.year` reads a COMPONENT of a real property, not a bad path.

    Warned about naively it produced four findings and advised `p.year` — a
    property that does not exist, so following the advice makes the query worse
    and invites the model to loop.
    """

    def test_internal_property_component_is_silent(self):
        w = build_schema_warnings(
            'MATCH (p:Parte) RETURN p.created_at.year, p.created_at.month',
            _flat_census(),
        )
        assert w == []

    def test_census_property_component_is_silent(self):
        w = build_schema_warnings(
            'MATCH (p:Parte) RETURN p.fecha_de_inicio.year', _flat_census()
        )
        assert w == []

    def test_component_of_an_unknown_property_still_warns(self):
        """The FIRST segment is what is being read; an unknown one is still unknown."""
        w = build_schema_warnings(
            'MATCH (p:Parte) RETURN p.fecha_inicio.year', _flat_census()
        )
        assert 'fecha_inicio' in _joined(w)


# ---------------------------------------------------------------------------
# Nested-chain references
# ---------------------------------------------------------------------------


class TestNestedChains:
    def test_nested_chain_warns_on_a_flat_census(self):
        w = build_schema_warnings(
            'MATCH (p:Parte) RETURN p.attributes.fecha_de_inicio', _flat_census()
        )
        assert w, 'a nested read against flat storage must be reported'
        assert 'attributes' in _joined(w)

    def test_nested_chain_warns_even_when_the_leaf_is_real(self):
        """The chain itself is the defect — the leaf being censused does not save it."""
        w = build_schema_warnings(
            'MATCH (p:Parte) RETURN p.attributes.fecha_de_inicio', _flat_census()
        )
        assert any('fecha_de_inicio' in x and 'attributes' in x for x in w)

    def test_nested_chain_names_the_flat_form_to_use_instead(self):
        w = build_schema_warnings(
            'MATCH (p:Parte) RETURN p.attributes.fecha_de_inicio', _flat_census()
        )
        assert 'p.fecha_de_inicio' in _joined(w)

    def test_nested_chain_with_an_unknown_leaf_reports_both_defects(self):
        """The measured failure: the shape AND the name were both guessed."""
        w = build_schema_warnings(
            'MATCH (p:Parte) RETURN p.attributes.fecha_inicio', _flat_census()
        )
        joined = _joined(w)
        assert 'attributes' in joined, 'the nested shape must be reported'
        assert 'fecha_de_inicio' in joined, 'the unknown leaf must still get did-you-mean'

    def test_announced_container_chain_does_not_warn(self):
        """Where the backend ANNOUNCES the container, the nested form is correct."""
        w = build_schema_warnings(
            'MATCH (p:Parte) RETURN p.attributes.fecha_de_inicio', _nesting_census()
        )
        assert w == []

    def test_wrong_container_warns_even_on_a_nesting_census(self):
        w = build_schema_warnings(
            'MATCH (p:Parte) RETURN p.props.fecha_de_inicio', _nesting_census()
        )
        assert w, 'a map the backend never announced is still a defect'
        assert 'attributes' in _joined(w), 'the announced container must be named'

    def test_deeper_chain_warns(self):
        w = build_schema_warnings(
            'MATCH (p:Parte) RETURN p.a.b.fecha_de_inicio', _flat_census()
        )
        assert w


class TestTheNestingArmIsStillChecked:
    """On a nesting backend the container is IN the property union.

    `properties` is `keys(n)`, which on that backend returns the bookkeeping
    columns plus the container — so a component-access rule keyed on "is the
    first segment a known property?" short-circuits `n.attributes.<key>`, the
    only correct reference form there. That silently killed the entire
    unknown-property check on this arm while every flat-arm test stayed green.

    Through the announced container the DOMAIN reference is `path[1]`, not the
    head and not the last segment. These pin that reading.
    """

    def test_misspelled_leaf_under_the_right_container_warns(self):
        w = build_schema_warnings(
            'MATCH (p:Parte) RETURN p.attributes.fecha_inicio', _nesting_census()
        )
        assert w, 'the measured defect must be caught on this arm too'
        assert 'fecha_inicio' in _joined(w)

    def test_that_warning_carries_did_you_mean(self):
        w = build_schema_warnings(
            'MATCH (p:Parte) RETURN p.attributes.fecha_inicio', _nesting_census()
        )
        assert 'fecha_de_inicio' in _joined(w)

    def test_the_correct_form_stays_silent(self):
        w = build_schema_warnings(
            'MATCH (p:Parte) RETURN p.attributes.fecha_de_inicio', _nesting_census()
        )
        assert w == []

    def test_component_of_a_real_nested_property_stays_silent(self):
        w = build_schema_warnings(
            'MATCH (p:Parte) RETURN p.attributes.fecha_de_inicio.year', _nesting_census()
        )
        assert w == []

    def test_component_of_a_misspelled_nested_property_warns(self):
        w = build_schema_warnings(
            'MATCH (p:Parte) RETURN p.attributes.fecha_inicio.year', _nesting_census()
        )
        assert 'fecha_inicio' in _joined(w)
        assert 'year' not in _joined(w), 'the component is not the reference being made'

    def test_the_container_name_alone_is_not_an_unknown_property(self):
        w = build_schema_warnings(
            'MATCH (p:Parte) RETURN p.attributes', _nesting_census()
        )
        assert w == []


# ---------------------------------------------------------------------------
# The positive-census gate — a warning may only fire from measured knowledge
# ---------------------------------------------------------------------------


class TestPositiveCensusGate:
    def test_no_schema_means_no_warnings_at_all(self):
        w = build_schema_warnings('MATCH (p:Parte) RETURN p.attributes.whatever', None)
        assert w == []

    def test_an_unsampled_label_silences_unknown_property_warnings(self):
        census = _flat_census()
        census['node_labels']['Persona']['sampled'] = False
        w = build_schema_warnings('MATCH (p:Parte) RETURN p.fecha_inicio', census)
        assert w == [], 'one unsampled label makes the union incomplete evidence'

    def test_an_empty_property_union_silences_unknown_property_warnings(self):
        census = _flat_census(
            node_labels={
                'Parte': {'count': 20, 'properties': [], 'attribute_keys': [], 'sampled': True}
            }
        )
        w = build_schema_warnings('MATCH (p:Parte) RETURN p.fecha_inicio', census)
        assert w == []

    def test_no_censused_labels_silences_unknown_property_warnings(self):
        w = build_schema_warnings(
            'MATCH (p:Parte) RETURN p.fecha_inicio', _flat_census(node_labels={})
        )
        assert w == []

    def test_the_chain_warning_survives_a_failed_census_gate(self):
        """The nested-shape defect is schema-independent, so the gate does not bind it."""
        census = _flat_census()
        census['node_labels']['Persona']['sampled'] = False
        w = build_schema_warnings('MATCH (p:Parte) RETURN p.attributes.x', census)
        assert w, 'the chain warning does not depend on the property union'
        assert 'attributes' in _joined(w)

    def test_unparseable_query_yields_no_warnings(self):
        w = build_schema_warnings('this is not cypher at all {{{', _flat_census())
        assert w == []


# ---------------------------------------------------------------------------
# False-positive guards
# ---------------------------------------------------------------------------


class TestNoFalsePositives:
    def test_string_literal_containing_dots_is_not_a_reference(self):
        w = build_schema_warnings(
            "MATCH (p:Parte) WHERE p.localidad = 'a.bogus.value' RETURN p", _flat_census()
        )
        assert w == []

    def test_function_calls_are_not_references(self):
        w = build_schema_warnings(
            'MATCH (n)-[r]->(m) RETURN type(r), keys(n), labels(m), count(*)', _flat_census()
        )
        assert w == []

    def test_map_literal_keys_are_not_references(self):
        w = build_schema_warnings(
            'MATCH (p:Parte {localidad: "x"}) RETURN p.name', _flat_census()
        )
        assert w == []

    def test_parameters_are_not_references(self):
        w = build_schema_warnings(
            'MATCH (p:Parte) WHERE p.localidad = $localidad RETURN p.name', _flat_census()
        )
        assert w == []

    def test_label_predicate_is_not_a_reference(self):
        w = build_schema_warnings(
            'MATCH (n) WHERE n:Parte RETURN n.name', _flat_census()
        )
        assert w == []

    def test_aggregation_alias_is_not_a_reference(self):
        w = build_schema_warnings(
            'MATCH (p:Parte) RETURN count(p) AS total ORDER BY total DESC', _flat_census()
        )
        assert w == []


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


class TestTheClaimMatchesTheEvidence:
    """The census SAMPLES; it does not enumerate.

    Measured on the live bench graph: a full `MATCH (n) UNWIND keys(n)` scan
    turned up a real domain property the 50-node-per-label sample never saw.
    `sampled: True` means "the probe returned rows", not "every key was
    observed" — so an absolute claim of non-existence is a claim the connector
    cannot support, and acting on it means editing a WORKING query.
    """

    def test_no_absolute_nonexistence_claim(self):
        w = build_schema_warnings(
            'MATCH (p:Parte) RETURN p.situacion_especial', _flat_census()
        )
        joined = _joined(w)
        assert joined, 'the reference is still worth reporting'
        for absolute in (
            'not carried by any node type',
            'does not exist',
            'no node type',
            'nowhere in',
        ):
            assert absolute not in joined.lower(), (
                f'the warning still claims {absolute!r}, which the census cannot prove'
            )

    def test_the_claim_is_scoped_to_the_census(self):
        w = build_schema_warnings(
            'MATCH (p:Parte) RETURN p.situacion_especial', _flat_census()
        )
        assert 'census' in _joined(w).lower()

    def test_the_did_you_mean_survives_the_rewording(self):
        w = build_schema_warnings(
            'MATCH (p:Parte) RETURN p.fecha_inicio', _flat_census()
        )
        assert 'fecha_de_inicio' in _joined(w)

    def test_the_get_schema_pointer_survives(self):
        w = build_schema_warnings(
            'MATCH (p:Parte) RETURN p.situacion_especial', _flat_census()
        )
        assert 'get_schema' in _joined(w)

    def test_the_silent_null_column_explanation_survives(self):
        """Unconditionally true regardless of how complete the census is."""
        w = build_schema_warnings(
            'MATCH (p:Parte) RETURN p.situacion_especial', _flat_census()
        )
        assert 'null' in _joined(w).lower()

    def test_the_shared_note_makes_no_absolute_claim_either(self):
        """The note is where the phrasing B2 removed crept back in."""
        w = build_schema_warnings(
            'MATCH (p:Parte) RETURN p.situacion_especial', _flat_census()
        )
        assert 'the graph does not carry' not in _joined(w).lower()

    def test_the_shared_note_is_absent_when_no_name_is_in_question(self):
        """A chain-only finding disputes a SHAPE; no property name is doubted."""
        w = build_schema_warnings(
            'MATCH (p:Parte) RETURN p.attributes.fecha_de_inicio', _flat_census()
        )
        assert w, 'the chain itself is still reported'
        assert 'null' not in _joined(w).lower()


class TestOutputIsBounded:
    """A warning list is read by a model with a context budget."""

    def _many_bad_props(self, n: int) -> str:
        refs = ', '.join(f'p.bogus_prop_{i}' for i in range(n))
        return f'MATCH (p:Parte) RETURN {refs}'

    def test_one_hundred_bad_references_stay_bounded(self):
        w = build_schema_warnings(self._many_bad_props(100), _flat_census())
        assert len(w) <= 8, f'emitted {len(w)} warnings'
        assert len(_joined(w)) < 4000, f'emitted {len(_joined(w))} chars'

    def test_the_overflow_is_counted_not_dropped_silently(self):
        w = build_schema_warnings(self._many_bad_props(100), _flat_census())
        assert 'more' in _joined(w).lower()

    def test_a_small_number_is_not_truncated(self):
        w = build_schema_warnings(
            'MATCH (p:Parte) RETURN p.bogus_one, p.bogus_two', _flat_census()
        )
        assert 'more' not in _joined(w).lower()

    def test_the_shared_explanation_is_stated_once(self):
        w = build_schema_warnings(
            'MATCH (p:Parte) RETURN p.bogus_one, p.bogus_two, p.bogus_three',
            _flat_census(),
        )
        assert _joined(w).lower().count('null for every row') == 1

    def test_many_bad_chains_stay_bounded(self):
        refs = ', '.join(f'p.attributes.bogus_{i}' for i in range(100))
        w = build_schema_warnings(f'MATCH (p:Parte) RETURN {refs}', _flat_census())
        assert len(_joined(w)) < 4000, f'emitted {len(_joined(w))} chars'


class TestContract:
    def test_returns_plain_strings(self):
        w = build_schema_warnings('MATCH (p:Parte) RETURN p.fecha_inicio', _flat_census())
        assert all(isinstance(x, str) for x in w)

    def test_warnings_are_domain_agnostic(self):
        """The wording is produced by the module; only the NAMES come from the graph."""
        w = build_schema_warnings(
            'MATCH (p:Parte) RETURN p.attributes.fecha_inicio', _flat_census()
        )
        joined = _joined(w).lower()
        for banned in ('aircraft', 'aviation', 'police', 'crime', 'delito', 'parte'):
            assert banned not in joined.replace('p.fecha_inicio', ''), (
                f'warning text must not name a domain: {banned}'
            )
