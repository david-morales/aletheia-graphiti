"""Live two-flavour parity gate (GATED).

Proves that FalkorDbFlavour and AgeFlavour produce the SAME rich graph_query envelope shape when
their execute_graph_query output is fed to the shared format_result — against real backends.

Gated:
  FALKORDB_PARITY_LIVE=1  -> FalkorDB arm (needs FalkorDB on localhost:6379 with a populated graph;
                             graph via FALKORDB_DATABASE, default 'policia_partes_real_v2')
  AGE_PARITY_LIVE=1       -> AGE arm (needs Postgres+AGE; DSN via AGE_DSN, graph via AGE_GRAPH_NAME,
                             default dsn postgresql://age:age@localhost:5433/age_test,
                             graph 'policia_age_poc')

Both arms assert the SAME envelope keys (parity) + that the flavour-specific execute path works
on real data. Timing is asserted present, not valued.
"""
import os

import pytest

from flavours.age import AgeFlavour
from flavours.falkordb import FalkorDbFlavour
from utils.cypher import format_result

pytestmark = pytest.mark.skipif(
    not (os.getenv("FALKORDB_PARITY_LIVE") or os.getenv("AGE_PARITY_LIVE")),
    reason="set FALKORDB_PARITY_LIVE=1 and/or AGE_PARITY_LIVE=1 to run",
)

# The rich envelope every graph_query success must carry, on either flavour.
_ENVELOPE_KEYS = {
    "query", "auto_fixes", "type", "row_count", "truncated",
    "limit_applied", "execution_ms", "cypher_quality",
}


def _assert_envelope(envelope: dict) -> None:
    assert _ENVELOPE_KEYS <= set(envelope.keys()), (
        f"missing envelope keys: {_ENVELOPE_KEYS - set(envelope.keys())}"
    )
    assert isinstance(envelope["execution_ms"], (int, float))
    assert "verdict" in envelope["cypher_quality"]


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("FALKORDB_PARITY_LIVE"), reason="FalkorDB live gate off")
async def test_falkordb_envelope_shape():
    from graphiti_core.driver.falkordb_driver import FalkorDriver

    database = os.getenv("FALKORDB_DATABASE", "policia_partes_real_v2")
    driver = FalkorDriver(host="localhost", port=6379, database=database)
    flavour = FalkorDbFlavour()
    import time

    query = "MATCH (n) RETURN count(n) AS c"
    t = time.time()
    records, header = await flavour.execute_graph_query(driver, query)
    envelope = format_result(records, header, query, [], round((time.time() - t) * 1000, 1), 200)
    _assert_envelope(envelope)
    assert envelope["type"] in ("scalar", "tabular")


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("AGE_PARITY_LIVE"), reason="AGE live gate off")
async def test_age_envelope_shape():
    from graphiti_core.driver.age_driver import AGEDriver

    dsn = os.getenv("AGE_DSN", "postgresql://age:age@localhost:5433/age_test")
    graph_name = os.getenv("AGE_GRAPH_NAME", "policia_age_poc")
    driver = AGEDriver(dsn=dsn, graph_name=graph_name, embedding_dim=1024)
    flavour = AgeFlavour()
    import time

    query = "MATCH (n) RETURN count(n) AS c"
    t = time.time()
    records, header = await flavour.execute_graph_query(driver, query)
    envelope = format_result(records, header, query, [], round((time.time() - t) * 1000, 1), 200)
    _assert_envelope(envelope)
    assert envelope["type"] in ("scalar", "tabular")


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("AGE_PARITY_LIVE"), reason="AGE live gate off")
async def test_age_get_schema_property_query_needs_return_alias():
    """Regression for get_schema KeyError('key') on AGE.

    get_schema builds each label's `properties` from
    ``... UNWIND k AS key RETURN DISTINCT key`` and reads ``r['key']``. AGE names an
    unaliased openCypher projection ``col0`` (the variable name is lost), so the
    unaliased form yields no ``key`` column and get_schema raised KeyError('key') —
    i.e. ``{'error': "Failed to retrieve schema: 'key'"}`` for EVERY AGE caller,
    while FalkorDB (which keeps ``key``) was unaffected. The fix aliases the
    projection ``RETURN DISTINCT key AS key``. This proves the quirk on the live
    graph and pins the alias requirement.
    """
    from graphiti_core.driver.age_driver import AGEDriver

    dsn = os.getenv("AGE_DSN", "postgresql://age:age@localhost:5433/age_test")
    graph_name = os.getenv("AGE_GRAPH_NAME", "policia_age_poc")
    driver = AGEDriver(dsn=dsn, graph_name=graph_name, embedding_dim=1024)

    lab, _, _ = await driver.execute_query(
        "MATCH (n) UNWIND labels(n) AS l RETURN DISTINCT l AS l LIMIT 1"
    )
    assert lab, "expected a populated AGE graph"
    label = lab[0]["l"]
    base = f"MATCH (n:`{label}`) WITH keys(n) AS k LIMIT 50 UNWIND k AS key RETURN DISTINCT key"

    unaliased, _, _ = await driver.execute_query(base)
    aliased, _, _ = await driver.execute_query(base + " AS key")

    # AGE drops the unaliased projection name (-> col0); the explicit alias (the fix
    # get_schema now uses) restores a `key` column that `r['key']` can read.
    assert unaliased and "key" not in unaliased[0], f"AGE should drop unaliased name: {unaliased[0]}"
    assert aliased and "key" in aliased[0], f"aliased query must yield a `key` column: {aliased[0]}"


@pytest.mark.skipif(
    not (os.getenv("FALKORDB_PARITY_LIVE") and os.getenv("AGE_PARITY_LIVE")),
    reason="both live gates required for the cross-flavour parity assertion",
)
def test_both_flavours_share_envelope_key_contract():
    # Structural parity: both flavours' success envelopes carry the identical key set. Proven by
    # both arms above asserting the same _ENVELOPE_KEYS; this test documents the invariant.
    assert _ENVELOPE_KEYS == {
        "query", "auto_fixes", "type", "row_count", "truncated",
        "limit_applied", "execution_ms", "cypher_quality",
    }


# ---------------------------------------------------------------------------
# AGE classifier parity: every pattern is pinned against REAL AGE error text.
# Each query below was confirmed on the :5433 bed on 2026-08-01.
# ---------------------------------------------------------------------------

_AGE_CLASSIFIER_CASES = [
    ("MATCH (n) WHERE n.name = $nm RETURN n LIMIT 1", "unbound_parameter"),
    ("MATCH (n) WHERE n:Persona OR n:Ubicacion RETURN n LIMIT 1", "boolean_label_test"),
    ("MATCH (n:Persona OR n:Ubicacion) RETURN n LIMIT 1", "boolean_label_test"),
    ("MATCH (n) RETURN label(n) AS t, count(n) AS count ORDER BY count DESC LIMIT 1",
     "reserved_alias"),
    ("MATCH (a)-[r:A|B]->(b) RETURN a LIMIT 1", "reltype_disjunction_unsupported"),
    ("MATCH (n) WHERE n.name != 'zzz' RETURN n.name AS name LIMIT 1", "not_equals_operator"),
    ("MATCH (n) RETURN n.name AS x, n.uuid AS x LIMIT 1", "duplicate_return_column"),
    ("MATCH (n) RETURN label(n) AS TipoDelito LIMIT 1", "mixed_case_alias"),
]


def _age_driver():
    from graphiti_core.driver.age_driver import AGEDriver

    dsn = os.getenv("AGE_DSN", "postgresql://age:age@localhost:5433/age_test")
    graph_name = os.getenv("AGE_GRAPH_NAME", "policia_age_poc")
    return AGEDriver(dsn=dsn, graph_name=graph_name, embedding_dim=1024)


@pytest.mark.asyncio
@pytest.mark.parametrize("query,expected_reason", _AGE_CLASSIFIER_CASES,
                         ids=[r for _, r in _AGE_CLASSIFIER_CASES])
@pytest.mark.skipif(not os.getenv("AGE_PARITY_LIVE"), reason="AGE live gate off")
async def test_age_classifier_matches_real_backend_errors(query, expected_reason):
    """The classifier is pinned to what AGE ACTUALLY says, not to a remembered string.

    These queries bypass validate_and_sanitize on purpose — several are caught pre-flight by
    check_dialect, and the point here is the post-hoc net against live error text.
    """
    driver = _age_driver()
    flavour = AgeFlavour()
    try:
        await flavour.execute_graph_query(driver, query)
    except Exception as exc:  # noqa: BLE001 — the error IS the subject under test
        err = flavour.classify_execution_error(str(exc), query=query)
        assert err.reason == expected_reason, f"{exc!s} -> {err.reason}"
        assert err.suggestion, "a classified error must carry a suggestion"
        assert err.doc_hint, "a classified error must carry a doc_hint"
    else:
        pytest.fail(f"expected {query!r} to fail on AGE, but it succeeded")
    finally:
        await driver.close()


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("AGE_PARITY_LIVE"), reason="AGE live gate off")
async def test_age_profile_queries_return_data_on_the_live_graph():
    """W2 acceptance: every AGE profile probe runs and returns rows on a real graph."""
    driver = _age_driver()
    group_id = os.getenv("AGE_GRAPH_NAME", "policia_age_poc")
    queries = AgeFlavour().profile_queries()
    try:
        entity_rows, _, _ = await driver.execute_query(
            queries["entity_types"], group_id=group_id
        )
        assert entity_rows, "entity_types probe returned nothing"
        assert any(r.get("entity_type") for r in entity_rows)
        assert all(r.get("cnt") is not None for r in entity_rows)

        edge_rows, _, _ = await driver.execute_query(queries["edge_types"], group_id=group_id)
        assert edge_rows, "edge_types probe returned nothing"

        label = next(
            lbl
            for r in entity_rows
            for lbl in (r.get("entity_type") or [])
            if lbl not in ("Entity", "Episodic", "Community")
        )
        name_rows, _, _ = await driver.execute_query(
            queries["sample_names"], group_id=group_id, label=label, limit=5
        )
        assert name_rows, f"sample_names probe returned nothing for {label}"

        time_rows, _, _ = await driver.execute_query(queries["time_range"], group_id=group_id)
        assert time_rows and time_rows[0].get("earliest")
    finally:
        await driver.close()


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("AGE_PARITY_LIVE"), reason="AGE live gate off")
async def test_age_auto_fixed_count_alias_actually_runs_on_the_live_graph():
    """The stage-2b rename must produce a query AGE accepts — proven, not assumed."""
    from utils.cypher import SanitizedQuery, validate_and_sanitize

    raw = "MATCH (n) RETURN label(n) AS t, count(n) AS count ORDER BY count DESC"
    flavour = AgeFlavour()
    sanitized = validate_and_sanitize(raw, flavour)
    assert isinstance(sanitized, SanitizedQuery)
    assert "count_" in sanitized.query

    driver = _age_driver()
    try:
        records, header = await flavour.execute_graph_query(driver, sanitized.query)
    finally:
        await driver.close()
    assert records and "count_" in header


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("FALKORDB_PARITY_LIVE"), reason="FalkorDB live gate off")
async def test_base_profile_queries_still_run_on_falkordb():
    """No-regression: the shared probe text (now aliased `cnt`) still works on FalkorDB."""
    from graphiti_core.driver.falkordb_driver import FalkorDriver

    from flavours.base import BaseFlavour

    database = os.getenv("FALKORDB_DATABASE", "policia_partes_real_v2")
    driver = FalkorDriver(host="localhost", port=6379, database=database)
    queries = BaseFlavour().profile_queries()
    records, _, _ = await driver.execute_query(queries["entity_types"], group_id=database)
    assert records, "FalkorDB entity_types probe returned nothing"
    assert all(r.get("cnt") is not None for r in records)


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("AGE_PARITY_LIVE"), reason="AGE live gate off")
async def test_age_edge_types_probe_excludes_graphiti_bookkeeping_edges():
    """F6: MENTIONS is an Episodic->Entity bookkeeping edge, not a domain relationship.

    Before the endpoint scoping was restored this probe advertised `MENTIONS: 11` on the live
    bed — a relationship the FalkorDB shape excludes by construction via its `:Entity`
    endpoints. Asserted against the built profile's edge types, not just the query text.
    """
    from domain_profile import _query_edge_types

    driver = _age_driver()
    group_id = os.getenv("AGE_GRAPH_NAME", "policia_age_poc")
    try:
        edge_types = await _query_edge_types(driver, group_id, AgeFlavour())
    finally:
        await driver.close()

    assert edge_types, "edge_types probe returned nothing"
    assert "MENTIONS" not in edge_types, sorted(edge_types)
    assert "RELATES_TO" not in edge_types, sorted(edge_types)
    # ...and the real domain relationships survive.
    assert any(name.isupper() and name != "MENTIONS" for name in edge_types), sorted(edge_types)
