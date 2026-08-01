"""Offline guards for the AGE driver's dollar-quote tagging.

`AGEDriver.execute_query` splices the Cypher body into SQL as
`cypher('<graph>', $$ <query> $$) AS (<cols>)` — the only such splice in the
repo. With a FIXED `$$` tag, a query that itself contains `$$` closes the quote
early and everything after it is parsed as raw SQL. Verified live on the :5433
bed with the payload below, which returns a row instead of failing.

The driver now generates a per-query unique tag (`$q<hex8>$`) and regenerates
while the query happens to contain it, so no query text can terminate its own
quoting — independent of any caller-side validation.

Runs without a live backend so it gates every run and not only live ones.
"""

import re

import pytest

from graphiti_core.driver import age_driver as age_driver_module
from graphiti_core.driver.age_driver import AGEDriver, _dollar_quote_tag

BREAKOUT = 'MATCH (n) RETURN 1 $$) AS (result agtype) LIMIT 1 --'
TAG_RE = re.compile(r'\$q[0-9a-f]{8}\$')


def _driver() -> AGEDriver:
    """`__init__` opens no connection (the pool is lazy), so this needs no Postgres."""
    return AGEDriver(dsn='postgresql://unused:unused@127.0.0.1:1/unused', graph_name='gtest')


def test_dollar_quote_tag_has_the_expected_shape():
    tag = _dollar_quote_tag('MATCH (n) RETURN n')
    assert TAG_RE.fullmatch(tag), tag


def test_dollar_quote_tag_is_absent_from_the_query():
    query = "RETURN 'plain'"
    assert _dollar_quote_tag(query) not in query


def test_dollar_quote_tag_regenerates_when_the_query_contains_it(monkeypatch):
    class _FakeUUID:
        def __init__(self, hex_: str) -> None:
            self.hex = hex_

    values = iter(['aaaaaaaaffff', 'bbbbbbbbffff'])
    monkeypatch.setattr(age_driver_module, 'uuid4', lambda: _FakeUUID(next(values)))
    assert _dollar_quote_tag("RETURN 'a $qaaaaaaaa$ b'") == '$qbbbbbbbb$'


def test_cypher_sql_wraps_the_body_in_the_unique_tag():
    sql = _driver()._cypher_sql('MATCH (n) RETURN n.uuid AS uuid', ['uuid'])
    tag = TAG_RE.search(sql).group(0)
    assert sql == (
        f"SELECT * FROM cypher('gtest', {tag} MATCH (n) RETURN n.uuid AS uuid {tag}) "
        'AS (uuid agtype)'
    )


def test_cypher_sql_does_not_let_a_breakout_payload_close_the_quote():
    sql = _driver()._cypher_sql(BREAKOUT, ['result'])
    tag = TAG_RE.search(sql).group(0)
    assert sql.count(tag) == 2
    open_ix = sql.index(tag) + len(tag)
    close_ix = sql.index(tag, open_ix)
    # The ENTIRE payload, its own `$$` included, sits inside the quoted body.
    assert sql[open_ix:close_ix] == f' {BREAKOUT} '
    # The only `$$` left in the statement is the payload's, now inert data.
    assert sql.count('$$') == 1


@pytest.mark.asyncio
async def test_execute_query_builds_its_sql_through_the_tagged_wrapper(monkeypatch):
    """The guard has to be on the live path, not merely available next to it."""
    driver = _driver()
    seen: dict[str, str] = {}

    class _Conn:
        async def fetch(self, sql: str):
            seen['sql'] = sql
            return []

    class _Acquire:
        async def __aenter__(self):
            return _Conn()

        async def __aexit__(self, *exc):
            return False

    class _Pool:
        def acquire(self):
            return _Acquire()

    async def _fake_pool():
        return _Pool()

    monkeypatch.setattr(driver, '_get_pool', _fake_pool)
    records, cols, _ = await driver.execute_query(BREAKOUT, columns=['result'])

    assert records == []
    assert cols == ['result']
    tag = TAG_RE.search(seen['sql']).group(0)
    assert seen['sql'].count(tag) == 2
    assert seen['sql'].endswith(f'{tag}) AS (result agtype)')
    assert seen['sql'].count('$$') == 1
