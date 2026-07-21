"""Tests for the PostgreSQL + Apache AGE + pgvector Graphiti driver (Phase 0 spike).

Requires the docker-compose.age.yml service running and reachable at AGE_TEST_DSN
(default postgresql://age:age@localhost:5433/age_test).
"""

import os

import asyncpg
import pytest

AGE_DSN = os.environ.get('AGE_TEST_DSN', 'postgresql://age:age@localhost:5433/age_test')


@pytest.mark.asyncio
async def test_extensions_available():
    """Both extensions load and AGE's agtype is usable."""
    conn = await asyncpg.connect(AGE_DSN)
    try:
        exts = {
            r['extname']
            for r in await conn.fetch(
                "SELECT extname FROM pg_extension WHERE extname IN ('age','vector')"
            )
        }
        assert exts == {'age', 'vector'}, f'missing extensions, found: {exts}'

        # AGE requires ag_catalog on search_path and its library loaded per session.
        await conn.execute("LOAD 'age'; SET search_path = ag_catalog, '$user', public;")
        val = await conn.fetchval("SELECT '1'::agtype")
        assert str(val) == '1'
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_driver_connects_and_runs_cypher(age_driver):
    """AGEDriver connects, declares provider AGE, and runs a Cypher aggregation."""
    from graphiti_core.driver.driver import GraphProvider

    assert age_driver.provider == GraphProvider.AGE
    records, header, _ = await age_driver.execute_query('MATCH (n) RETURN count(n) AS n')
    assert records == [{'n': 0}]
    assert header == ['n']
