"""Live proof that a `$$`-bearing query is data, not a SQL breakout.

Companion to `test_age_dollar_quote_offline.py`, which pins the SQL string the
driver builds. These two ask the question the string cannot answer: what does
Postgres actually do with it.

Live-gated through the shared `age_driver` fixture (tests/driver/conftest.py):
a per-test `g_<hex>` graph on AGE_TEST_DSN, dropped in teardown, skipped only
when the store is unreachable.
"""

import asyncpg
import pytest

BREAKOUT = 'MATCH (n) RETURN 1 $$) AS (result agtype) LIMIT 1 --'


@pytest.mark.asyncio
async def test_breakout_payload_does_not_reach_sql(age_driver):
    """With the fixed `$$` tag this returned a row: the payload closed the quote,
    its own `AS (result agtype)` completed the statement and `--` commented the
    rest away. With a unique tag the whole payload is Cypher, and Cypher cannot
    parse it — so the call must RAISE rather than succeed."""
    with pytest.raises(asyncpg.PostgresError):
        await age_driver.execute_query(BREAKOUT, columns=['result'])


@pytest.mark.asyncio
async def test_dollar_dollar_inside_a_string_literal_round_trips_as_data(age_driver):
    """The positive control: `$$` inside a Cypher string literal used to destroy
    the statement; now it is ordinary text and comes back unchanged."""
    records, _, _ = await age_driver.execute_query("RETURN 'a $$ b' AS result")
    assert records == [{'result': 'a $$ b'}]


@pytest.mark.asyncio
async def test_the_connection_survives_a_rejected_breakout(age_driver):
    """A parse error must not poison the pooled connection for the next caller."""
    with pytest.raises(asyncpg.PostgresError):
        await age_driver.execute_query(BREAKOUT, columns=['result'])
    records, _, _ = await age_driver.execute_query(
        "CREATE (n:Entity {uuid: 'dollarquote-1', group_id: 'dq'}) RETURN n.uuid AS uuid"
    )
    assert records == [{'uuid': 'dollarquote-1'}]
