"""The parser must read the queries agents actually write.

Every fixture in this suite used to be hand-written, and hand-written Cypher
does not look like generated Cypher. Agents open with a newline: 166 of 182
real `graph_query` calls across three benchmark artifacts (91%) start with
whitespace. This grammar wants a clause keyword as its first token, so all 166
reported a syntax error — and `parse_errors > 0` suppresses the schema-warning
channel entirely and collapses `cypher_quality` to `parse_failed`.

The result was a feature that passed every test and did nothing in production.
Three review rounds hardened its semantics while it was inert on ~all traffic,
because no test ever fed it a real query.

So this module tests against real queries, captured verbatim from live runs and
committed alongside. The load-bearing assertion is the cheapest one here: every
corpus query PARSES. That is what would have caught this three rounds ago, and
it is what will catch the next thing that only breaks on input we did not think
to imagine.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from utils.cypher_extractor import extract_elements
from utils.cypher_quality import assess_quality
from utils.schema_warnings import build_schema_warnings

_CORPUS_PATH = Path(__file__).parent / 'data' / 'real_graph_query_corpus.json'


def _corpus() -> dict:
    return json.loads(_CORPUS_PATH.read_text())


CORPUS_QUERIES: list[str] = _corpus()['queries']

# Queries the vendored grammar genuinely cannot read, enumerated in the corpus
# file beside the reason rather than absorbed into a weaker assertion. Today
# that is the pre-existing consecutive-WITH gap (6 of the 182 captured calls);
# `exists()` and `CALL {}` appear nowhere in the traffic, and UNION parses
# cleanly while silently dropping its later arms — a false negative, not a
# parse error. The list is itself under test below, so it cannot rot into an
# excuse: every entry must still be in the corpus and must still fail.
KNOWN_UNPARSEABLE: frozenset[str] = frozenset(_corpus().get('_known_unparseable', []))


class TestTheCorpusIsWhatItClaims:
    def test_the_corpus_is_committed_and_non_trivial(self):
        assert _CORPUS_PATH.exists()
        assert len(CORPUS_QUERIES) >= 30, 'too few queries to represent real traffic'

    def test_the_corpus_preserves_the_leading_whitespace(self):
        """Stripped on capture, the corpus could not reproduce the defect."""
        leading = [q for q in CORPUS_QUERIES if q[:1].isspace()]
        assert len(leading) > len(CORPUS_QUERIES) // 2, (
            'the corpus no longer reflects how agents write queries'
        )

    def test_every_query_is_non_empty(self):
        assert all(q.strip() for q in CORPUS_QUERIES)


class TestEveryRealQueryParses:
    """The assertion that would have caught this class three rounds ago."""

    @pytest.mark.parametrize('query', CORPUS_QUERIES, ids=range(len(CORPUS_QUERIES)))
    def test_query_parses_without_error(self, query):
        if query in KNOWN_UNPARSEABLE:
            pytest.skip('recorded as a known grammar limit')
        elements = extract_elements(query)
        assert elements.parse_errors == 0, (
            f'parse_errors={elements.parse_errors} for a query taken verbatim '
            f'from a live run:\n{query!r}'
        )

    def test_no_query_is_silently_unreadable(self):
        broken = [
            q for q in CORPUS_QUERIES
            if q not in KNOWN_UNPARSEABLE and extract_elements(q).parse_errors
        ]
        assert broken == [], f'{len(broken)} real queries do not parse'

    def test_the_overwhelming_majority_of_real_queries_parse(self):
        """The headline number, asserted so a regression is loud."""
        clean = sum(1 for q in CORPUS_QUERIES if not extract_elements(q).parse_errors)
        assert clean / len(CORPUS_QUERIES) > 0.9


class TestTheKnownUnparseableListCannotRot:
    """An exemption list is only honest while every entry still earns its place."""

    def test_the_exemptions_carry_a_reason(self):
        if KNOWN_UNPARSEABLE:
            assert _corpus().get('_known_unparseable_reason', '').strip()

    def test_every_exemption_is_actually_in_the_corpus(self):
        assert set(CORPUS_QUERIES) >= KNOWN_UNPARSEABLE

    @pytest.mark.parametrize('query', sorted(KNOWN_UNPARSEABLE), ids=range(len(KNOWN_UNPARSEABLE)))
    def test_every_exemption_still_fails_to_parse(self, query):
        """If the grammar learns one of these, delete the entry — do not keep it."""
        assert extract_elements(query).parse_errors > 0, (
            'this query parses now; remove it from _known_unparseable'
        )

    def test_the_exemptions_are_the_recorded_grammar_gap(self):
        """Not a dumping ground: each entry matches a limit already documented."""
        for query in KNOWN_UNPARSEABLE:
            assert len(re.findall(r'\bWITH\b', query, re.I)) > 1, (
                'an exemption that is not the recorded consecutive-WITH gap'
            )

    @pytest.mark.parametrize('query', CORPUS_QUERIES, ids=range(len(CORPUS_QUERIES)))
    def test_parsing_a_real_query_binds_at_least_one_node(self, query):
        """A clean parse that extracted nothing would be inertness by another route."""
        if query in KNOWN_UNPARSEABLE:
            pytest.skip('recorded as a known grammar limit')
        assert extract_elements(query).node_vars, f'no node variable found in:\n{query!r}'


class TestTheQualityVerdictIsAlsoRestored:
    """`cypher_quality` shares the extractor, so it was degraded on the same 91%.

    A parse error there means verdict `parse_failed` and an EMPTY `schema_match`
    — every label, relationship and property diagnostic gone — and for a
    zero-row result `refine_verdict` leaves it that way.
    """

    _CENSUS = {
        'attribute_container': None,
        'node_labels': {
            'TipoHecho': {
                'count': 20,
                'properties': ['name', 'summary', 'uuid'],
                'attribute_keys': ['summary'],
                'sampled': True,
            },
        },
        'relationship_types': {},
    }

    def test_a_leading_newline_no_longer_forces_parse_failed(self):
        quality = assess_quality(
            '\nMATCH (t:TipoHecho)\nRETURN t.name, t.summary\n', schema=self._CENSUS
        )
        assert quality.verdict != 'parse_failed'

    def test_the_schema_match_is_populated_again(self):
        quality = assess_quality(
            '\nMATCH (t:TipoHecho)\nRETURN t.name, t.summary\n', schema=self._CENSUS
        )
        assert quality.schema_match is not None
        assert 'TipoHecho' in quality.schema_match.labels.found


class TestTheCureFiresOnTheShapeAgentsWrite:
    """The guard against silent inertness, in one test.

    Every ingredient of the real thing at once: the leading newline, the
    `RETURN x.name AS x` self-alias, the multi-hop pattern, `type(r)`, and an
    ORDER BY on the misspelled column. If this goes quiet, the feature is off
    in production however green the rest of the suite is.
    """

    # Verbatim from 2026-08-23_031124_14602bef_policia.json, multihop-01.
    _REAL_QUERY = (
        '\nMATCH (p:Persona {name: "KHADIJA DAOUD"})-[r1]->(id)'
        '-[r2:EN_PARTE]->(parte:ParteDeIntervencion)\n'
        'RETURN parte.name AS parte, type(r1) AS rol, '
        'parte.fecha_inicio AS fecha, parte.clase_de_actuacion AS clase\n'
        'ORDER BY parte.fecha_inicio\n'
    )

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
        'relationship_types': {},
    }

    def test_the_real_query_parses(self):
        assert extract_elements(self._REAL_QUERY).parse_errors == 0

    def test_the_misspelled_property_is_reported(self):
        warnings = build_schema_warnings(self._REAL_QUERY, self._CENSUS)
        assert warnings, 'the cure is inert on the shape agents actually write'
        assert 'fecha_inicio' in '\n'.join(warnings)

    def test_the_report_carries_did_you_mean(self):
        warnings = build_schema_warnings(self._REAL_QUERY, self._CENSUS)
        assert 'fecha_de_inicio' in '\n'.join(warnings)

    def test_the_correctly_named_column_is_not_reported(self):
        warnings = build_schema_warnings(self._REAL_QUERY, self._CENSUS)
        assert 'clase_de_actuacion' not in '\n'.join(warnings)
