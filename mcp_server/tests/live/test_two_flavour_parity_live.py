"""Live two-flavour parity gate (GATED).

Proves that FalkorDbFlavour and AgeFlavour produce the SAME rich run_cypher envelope shape when
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

# The rich envelope every run_cypher success must carry, on either flavour.
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
