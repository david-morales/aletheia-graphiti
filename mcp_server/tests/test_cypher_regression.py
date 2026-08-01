"""Regression corpus for the Cypher sanitization pipeline — BOTH flavours.

Each case is a real query shape that reached a live backend, with the verdict the pipeline
must produce. Add a case whenever a query is handled wrongly in production.

Fields:
    flavour_name   'falkordb' | 'age'
    raw_query      the query as the agent wrote it
    expected       'pass' (unchanged apart from LIMIT) | 'fix' | 'reject'
    expected_reason for 'reject', the CypherError.reason
    expected_in    substrings the sanitized query must contain (for 'fix')
    forbidden_in   substrings the sanitized query must NOT contain (for 'fix')
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from flavours.age import AgeFlavour  # noqa: E402
from flavours.falkordb import FalkorDbFlavour  # noqa: E402
from utils.cypher import CypherError, SanitizedQuery, validate_and_sanitize  # noqa: E402

_FLAVOURS = {'falkordb': FalkorDbFlavour(), 'age': AgeFlavour()}


@dataclass(frozen=True)
class Case:
    name: str
    flavour_name: str
    raw_query: str
    expected: str
    expected_reason: str | None = None
    expected_in: tuple[str, ...] = ()
    forbidden_in: tuple[str, ...] = ()
    notes: str = ''


_CASES: list[Case] = [
    # --- FalkorDB: the no-regression arm ---
    Case('falkor_plain_count', 'falkordb', 'MATCH (n) RETURN count(n)', 'pass',
         notes='the former placeholder case, kept'),
    Case('falkor_lower_to_tolower', 'falkordb', 'MATCH (n) RETURN lower(n.name)', 'fix',
         expected_in=('toLower(',), forbidden_in=('lower(n.name)',)),
    Case('falkor_neq_to_diamond', 'falkordb', "MATCH (n) WHERE n.x != 'a' RETURN n", 'fix',
         expected_in=('<>',), forbidden_in=('!=',)),
    Case('falkor_apoc_rejected', 'falkordb', 'MATCH (n) CALL apoc.path.expand(n) RETURN n',
         'reject', expected_reason='apoc_unsupported'),
    Case('falkor_write_blocked', 'falkordb', 'MATCH (n) DELETE n', 'reject',
         expected_reason='write_operation'),
    Case('falkor_reltype_disjunction_allowed', 'falkordb',
         'MATCH (a)-[:A|B]->(b) RETURN b', 'pass',
         notes='FalkorDB supports [:A|B]; AGE does not — the divergence this corpus pins'),

    # --- AGE ---
    Case('age_plain_count', 'age', 'MATCH (n) RETURN count(n)', 'pass'),
    Case('age_param_rejected', 'age', 'MATCH (n) WHERE n.name = $nm RETURN n', 'reject',
         expected_reason='unbound_parameter',
         notes='8 of 11 AGE bench failures'),
    Case('age_dollar_quote_rejected', 'age', 'MATCH (n) WHERE n.x = $$ RETURN n', 'reject',
         expected_reason='dollar_quote_unsupported'),
    Case('age_id_variable_rejected', 'age', 'MATCH (id) RETURN id', 'reject',
         expected_reason='reserved_id_variable'),
    Case('age_reltype_disjunction_rejected', 'age', 'MATCH (a)-[:A|B]->(b) RETURN b', 'reject',
         expected_reason='reltype_disjunction_unsupported'),
    Case('age_write_blocked', 'age', 'MATCH (n) DELETE n', 'reject',
         expected_reason='write_operation'),
    Case('age_neq_to_diamond', 'age', "MATCH (n) WHERE n.x != 'a' RETURN n.name AS name",
         'fix', expected_in=('<>',), forbidden_in=('!=',)),
    Case('age_profile_stripped', 'age', 'PROFILE MATCH (n) RETURN n.name AS name', 'fix',
         expected_in=('MATCH (n)',), forbidden_in=('PROFILE',)),
    Case('age_count_alias_renamed', 'age',
         'MATCH (n) RETURN label(n) AS type, count(n) AS count ORDER BY count DESC', 'fix',
         expected_in=('AS count_', 'ORDER BY count_ DESC'),
         notes='the domain_profile startup failure shape'),
    Case('age_dollar_in_string_allowed', 'age',
         "MATCH (n) WHERE n.note = 'costs $50' RETURN n.name AS name", 'pass',
         notes='masked-span negative: a $ inside a literal must not reject'),
]


@pytest.mark.parametrize('case', _CASES, ids=lambda c: c.name)
def test_regression_case(case: Case):
    flavour = _FLAVOURS[case.flavour_name]
    result = validate_and_sanitize(case.raw_query, flavour)

    if case.expected == 'reject':
        assert isinstance(result, CypherError), f'{case.name}: expected a rejection'
        assert result.reason == case.expected_reason, case.name
        assert result.suggestion, f'{case.name}: a rejection must carry a suggestion'
        return

    assert isinstance(result, SanitizedQuery), f'{case.name}: {result}'
    for needle in case.expected_in:
        assert needle in result.query, f'{case.name}: missing {needle!r} in {result.query!r}'
    for needle in case.forbidden_in:
        assert needle not in result.query, f'{case.name}: {needle!r} survived'
    if case.expected == 'fix':
        non_limit = [f for f in result.auto_fixes if not f.startswith('Injected LIMIT')]
        assert non_limit, f'{case.name}: expected an auto_fixes note'
    else:
        non_limit = [f for f in result.auto_fixes if not f.startswith('Injected LIMIT')]
        assert not non_limit, f'{case.name}: unexpected rewrite {non_limit}'


def test_corpus_covers_both_flavours():
    covered = {c.flavour_name for c in _CASES}
    assert covered == {'falkordb', 'age'}
    assert len([c for c in _CASES if c.flavour_name == 'age']) >= 8
