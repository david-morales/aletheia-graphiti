"""Flavour-aware domain_profile probes (offline — stubbed drivers)."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from domain_profile import _query_entity_types, build_domain_profile  # noqa: E402
from flavours.age import AgeFlavour  # noqa: E402
from flavours.base import BaseFlavour  # noqa: E402
from flavours.falkordb import FalkorDbFlavour  # noqa: E402

_KEYS = {'entity_types', 'edge_types', 'sample_names', 'time_range'}


class _RecordingDriver:
    """Captures every query text and replays canned records per probe."""

    def __init__(self, records_by_marker: dict[str, list[dict]]):
        self.queries: list[str] = []
        self._records = records_by_marker

    async def execute_query(self, query, **kwargs):
        self.queries.append(query)
        for marker, records in self._records.items():
            if marker in query:
                return (records, [], None)
        return ([], [], None)


class TestProfileQueryContract:
    def test_every_flavour_exposes_the_four_probes(self):
        for flavour in (BaseFlavour(), FalkorDbFlavour(), AgeFlavour()):
            assert set(flavour.profile_queries()) == _KEYS, flavour.name

    def test_falkordb_inherits_the_base_queries(self):
        assert FalkorDbFlavour().profile_queries() == BaseFlavour().profile_queries()

    def test_no_flavour_aliases_a_count_column_as_count(self):
        # `count` is reserved on AGE; the shared reader keys on `cnt` for every flavour.
        for flavour in (BaseFlavour(), AgeFlavour()):
            for name, q in flavour.profile_queries().items():
                assert 'AS count' not in q, f'{flavour.name}/{name}'
            assert 'AS cnt' in flavour.profile_queries()['entity_types']

    def test_profile_queries_are_a_defensive_copy(self):
        q = BaseFlavour().profile_queries()
        q['entity_types'] = 'MUTATED'
        assert BaseFlavour().profile_queries()['entity_types'] != 'MUTATED'


class TestAgeProfileQueries:
    def test_age_drops_entity_scoping(self):
        # Live: MATCH (n:Entity) sees 6 of 2228 vertices on the AGE bench graph.
        for name, q in AgeFlavour().profile_queries().items():
            assert ':Entity' not in q, name

    def test_age_entity_types_uses_the_stored_labels_property(self):
        q = AgeFlavour().profile_queries()['entity_types']
        assert 'n.labels AS entity_type' in q
        assert 'labels(n)' not in q          # AGE's labels() returns only the leaf
        assert 'n.labels IS NOT NULL' in q   # Episodic vertices carry no labels list

    def test_age_sample_names_matches_on_the_stored_labels_property(self):
        assert '$label IN n.labels' in AgeFlavour().profile_queries()['sample_names']

    def test_age_time_range_keeps_the_earliest_latest_contract(self):
        q = AgeFlavour().profile_queries()['time_range']
        assert 'AS earliest' in q and 'AS latest' in q


class TestDomainProfileUsesTheFlavour:
    @pytest.mark.asyncio
    async def test_entity_types_probe_sends_the_flavour_query(self):
        driver = _RecordingDriver({'n.labels AS entity_type': [
            {'entity_type': ['Entity', 'Actor', 'Persona'], 'cnt': 196},
        ]})
        types = await _query_entity_types(driver, 'g1', AgeFlavour())
        assert driver.queries == [AgeFlavour().profile_queries()['entity_types']]
        assert types['Persona'].count == 196
        assert 'Entity' not in types

    @pytest.mark.asyncio
    async def test_entity_types_probe_tolerates_a_null_label_list(self):
        # AGE returns NULL for entity_type on vertices without a labels property.
        driver = _RecordingDriver({'n.labels AS entity_type': [
            {'entity_type': None, 'cnt': 101},
            {'entity_type': ['Entity', 'Ubicacion'], 'cnt': 287},
        ]})
        types = await _query_entity_types(driver, 'g1', AgeFlavour())
        assert list(types) == ['Ubicacion']

    @pytest.mark.asyncio
    async def test_build_domain_profile_end_to_end_on_the_age_flavour(self):
        driver = _RecordingDriver({
            'n.labels AS entity_type': [{'entity_type': ['Entity', 'Persona'], 'cnt': 196}],
            'type(r) AS relationship_type': [{'relationship_type': 'EN_PARTE', 'cnt': 936}],
            'n.name AS name': [{'name': 'OMAR MOHAMED'}],
            'AS earliest': [{'earliest': '2026-07-22T08:24:44+00:00',
                             'latest': '2026-07-22T10:56:35+00:00'}],
        })
        client = MagicMock()
        client.driver = driver
        profile = await build_domain_profile(client, 'g1', flavour=AgeFlavour())
        assert profile.entity_types['Persona'].count == 196
        assert profile.entity_types['Persona'].sample_names == ['OMAR MOHAMED']
        assert profile.edge_types['EN_PARTE'].count == 936
        assert profile.time_range == ('2026-07-22', '2026-07-22')

    @pytest.mark.asyncio
    async def test_a_failing_probe_logs_the_classified_reason_and_returns_empty(self, caplog):
        class _Boom:
            async def execute_query(self, query, **kwargs):
                raise RuntimeError('syntax error at or near "DESC"')

        with caplog.at_level('WARNING'):
            types = await _query_entity_types(_Boom(), 'g1', AgeFlavour())
        assert types == {}
        # The probe query itself contains `AS cnt`, not a reserved alias, so the reserved_alias
        # pattern cannot corroborate — the reason is the generic one, and that is the point:
        # the WARNING carries a classified reason and an actionable hint either way.
        messages = [r.getMessage() for r in caplog.records]
        assert any('Domain-profile probe entity_types failed [query_failed]' in m
                   for m in messages), messages
        assert any('get_schema' in m for m in messages), messages


class TestAgeEdgeTypesExcludesBookkeeping:
    """F6 — the AGE probe must scope its endpoints the way the FalkorDB shape does.

    The base query scopes both endpoints to `:Entity`, which excludes Graphiti's bookkeeping
    edges by construction. The AGE variant dropped that (`:Entity` is useless on AGE), so the
    Episodic->Entity `MENTIONS` edge was advertised as a domain relationship: live on the
    :5433 bed the probe returned `MENTIONS: 11`. Episodic vertices carry no `labels` list, so
    testing that property is the AGE-correct equivalent of the endpoint scoping.
    """

    def test_age_edge_types_scopes_both_endpoints(self):
        q = AgeFlavour().profile_queries()['edge_types']
        assert 's.labels IS NOT NULL' in q, q
        assert 't.labels IS NOT NULL' in q, q

    def test_age_time_range_scopes_its_source_endpoint_too(self):
        # Same bookkeeping-edge exposure, same fix.
        q = AgeFlavour().profile_queries()['time_range']
        assert 's.labels IS NOT NULL' in q, q
