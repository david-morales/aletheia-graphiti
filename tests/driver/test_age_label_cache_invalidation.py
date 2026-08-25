"""BUG-48: the per-instance label cache survives a FOREIGN graph rebuild.

`AGEDriver._known_labels` records the (kind, label) pairs this driver instance has
materialised, and `_ensure_label` short-circuits on a hit — that cache is what
keeps the BUG-38 guard cheap. It is cleared only by the instance's OWN
`build_indices_and_constraints(delete_existing=True)` and `drop_graph`.

So when a DIFFERENT instance or process rebuilds the graph, this instance's cache
goes on asserting labels that no longer exist. `ensure_vertex_label` /
`ensure_edge_label` return without doing anything, the MERGE is the label's first
real use again, and the write is back on AGE's unsynchronised implicit DDL — the
exact path BUG-38 exists to keep it off. Measured after a foreign rebuild: 7 of
24 concurrent saves failed with `DuplicateTableError`, the BUG-38 signature.

Fix under test: recover at the write. When a label-bearing write fails with the
DDL-race error class, the declared (kind, label) pairs are DROPPED from the
cache, re-ensured ONCE against the real catalogue, and the write is retried ONCE.
A second consecutive failure propagates as the real error it is — there is no
loop, and no error-text matching: the decision is the typed asyncpg error class
and nothing else.

Two properties this pins beyond "it recovers":

  * the HOT PATH is untouched. A genuine cache hit still costs zero round-trips
    and the happy path still issues exactly one write. A fix that re-verified
    against `ag_catalog` on every write would also close the bug, and would put
    a query in front of every single save.
  * the retry is BOUNDED at one. Every write that goes through `_write` is a
    uuid-keyed MERGE, so replaying one is idempotent; replaying it forever on a
    genuinely broken write is not a recovery, it is a hang.

NOT PROVEN HERE: the live two-driver-instance race — writer A holding a warm
cache while rebuilder B drops the graph underneath it, then N concurrent saves.
That needs the :5433 AGE bed, which this lane does not have, so it is deferred to
the AGE-bed decision / Lane C. What is proven here is the recovery mechanism
itself, driven through the real `_write` funnel and the real `AGEDriver` cache
with a stub connection.
"""

import logging

import asyncpg
import pytest

from graphiti_core.driver.age_driver import AGEDriver
from graphiti_core.driver.graph_operations.age_graph_operations import AGEGraphOperations

# Never dialled: every test either drives `_write` against a double or hands the
# real driver a pre-built fake pool, so `_get_pool` never opens a connection.
# A closed loopback port rather than 5433 so a mistake here fails, not connects.
UNREACHABLE_DSN = 'postgresql://age:age@127.0.0.1:1/age_test'


def _duplicate_table() -> Exception:
    """The measured BUG-48/38 signature: 42P07 on the label's backing relation."""
    return asyncpg.exceptions.DuplicateTableError('relation "BROADER" already exists')


def _duplicate_sequence() -> Exception:
    """The same race landing on the label's SEQUENCE instead of its table: 23505
    on `pg_class_relname_nsp_index`. Recorded alongside 42P07 in the BUG-38
    comment because AGE creates a table AND a sequence per label."""
    return asyncpg.exceptions.UniqueViolationError(
        'duplicate key value violates unique constraint "pg_class_relname_nsp_index"'
    )


# --------------------------------------------------------------- the `_write` double


class _StubDriver:
    """A driver double with the real cache semantics and a scripted write.

    `failures` is a list of exceptions consumed one per MUTATING query, so a test
    scripts exactly how many times the write fails and in what way. Reads are
    exempt on purpose: a real writer like `node_save` issues a MATCH of its own
    before the MERGE, and that read is not a label-bearing write — scripting it
    as one would test the wrong statement.
    """

    def __init__(self, failures: list[BaseException] | None = None) -> None:
        self.known_labels: set[tuple[str, str]] = set()
        self.ensured: list[tuple[str, str]] = []
        self.forgotten: list[tuple[str, str]] = []
        self.queries: list[str] = []
        self._failures = list(failures or [])
        self._node_tbl = 'age_nodes'
        self._edge_tbl = 'age_edges'

    @property
    def writes(self) -> list[str]:
        return [q for q in self.queries if 'MERGE' in q or 'SET ' in q]

    async def execute_query(self, cypher: str, **_kwargs):
        self.queries.append(cypher)
        if self._failures and ('MERGE' in cypher or 'SET ' in cypher):
            raise self._failures.pop(0)
        return [{'uuid': 'merged'}], ['uuid'], None

    async def execute_sql(self, _sql: str, *_args):
        return []

    async def _ensure(self, kind: str, label: str) -> None:
        self.ensured.append((kind, label))
        self.known_labels.add((kind, label))

    async def ensure_vertex_label(self, label: str) -> None:
        await self._ensure('v', label)

    async def ensure_edge_label(self, label: str) -> None:
        await self._ensure('e', label)

    def forget_label(self, kind: str, label: str) -> None:
        self.forgotten.append((kind, label))
        self.known_labels.discard((kind, label))


# ------------------------------------------------------------------- hot path


@pytest.mark.asyncio
async def test_a_write_that_succeeds_issues_exactly_one_query_and_forgets_nothing():
    driver = _StubDriver()
    await AGEGraphOperations._write(
        driver, 'MERGE (n:Persona {uuid: 1})', vertex_labels=('Persona',)
    )

    assert len(driver.queries) == 1
    assert driver.ensured == [('v', 'Persona')]
    assert driver.forgotten == []


@pytest.mark.asyncio
async def test_the_cached_label_hot_path_costs_no_round_trips():
    """A genuine cache hit must not reach the pool at all — that is the whole
    reason the cache exists, and a re-verify-every-time fix would lose it."""
    driver, pool = _driver_with_fake_pool(catalogue={('v', 'Persona')})
    driver._known_labels.add(('v', 'Persona'))

    await driver.ensure_vertex_label('Persona')

    assert pool.acquisitions == 0


# ------------------------------------------------------------------- recovery


@pytest.mark.asyncio
@pytest.mark.parametrize('error', [_duplicate_table, _duplicate_sequence], ids=['42P07', '23505'])
async def test_a_ddl_race_drops_the_stale_pair_re_ensures_once_and_retries_once(error):
    """The bug's exact shape: the cache says the label is there, it is not, and
    the MERGE lands back on the implicit DDL. One recovery round is enough
    because the re-ensure goes to the real catalogue, not to the cache."""
    driver = _StubDriver(failures=[error()])
    driver.known_labels.add(('e', 'BROADER'))  # stale: a foreign rebuild removed it

    records, _, _ = await AGEGraphOperations._write(
        driver, 'MERGE (a)-[r:BROADER]->(b)', edge_labels=('BROADER',)
    )

    assert driver.forgotten == [('e', 'BROADER')]
    assert driver.ensured == [('e', 'BROADER'), ('e', 'BROADER')]  # once, then once more
    assert len(driver.queries) == 2  # retried exactly once
    assert records == [{'uuid': 'merged'}]  # the retry's result is returned


@pytest.mark.asyncio
async def test_every_label_the_write_declared_is_invalidated():
    """Which of the declared labels raced is not knowable without parsing the
    error text, so all of them are dropped and re-ensured. Decision-free, and
    bounded by the one or two labels a single write names."""
    driver = _StubDriver(failures=[_duplicate_table()])
    await AGEGraphOperations._write(
        driver,
        'MERGE (e:Episodic)-[r:MENTIONS]->(n)',
        vertex_labels=('Episodic',),
        edge_labels=('MENTIONS',),
    )

    assert sorted(driver.forgotten) == [('e', 'MENTIONS'), ('v', 'Episodic')]
    assert driver.known_labels == {('v', 'Episodic'), ('e', 'MENTIONS')}  # re-ensured


@pytest.mark.asyncio
async def test_the_recovery_warning_carries_the_original_error(caplog):
    """A recovery that succeeds swallows the only evidence it happened unless the
    caught error is logged with it: which SQLSTATE fired, and on which relation.
    Without it a recovered race is indistinguishable in the log from a race that
    never happened, and the 42P07-vs-23505 arm is unknowable."""
    driver = _StubDriver(failures=[_duplicate_table()])

    with caplog.at_level(logging.WARNING, logger='graphiti_core.driver.graph_operations'):
        await AGEGraphOperations._write(
            driver, 'MERGE (a)-[r:BROADER]->(b)', edge_labels=('BROADER',)
        )

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert warnings[0].exc_info is not None, 'the caught error was dropped'
    original = warnings[0].exc_info[1]
    assert isinstance(original, asyncpg.exceptions.DuplicateTableError)
    assert 'BROADER' in str(original)


@pytest.mark.asyncio
async def test_a_second_consecutive_failure_propagates_and_does_not_loop():
    """A retry that keeps failing is not a stale cache — it is a real error, and
    burying it in a loop would turn a raised save into a hang."""
    driver = _StubDriver(failures=[_duplicate_table(), _duplicate_table()])

    with pytest.raises(asyncpg.exceptions.DuplicateTableError):
        await AGEGraphOperations._write(
            driver, 'MERGE (a)-[r:BROADER]->(b)', edge_labels=('BROADER',)
        )

    assert len(driver.queries) == 2  # the original and ONE retry, no more
    assert driver.forgotten == [('e', 'BROADER')]  # invalidated once, not per attempt


@pytest.mark.asyncio
async def test_an_unrelated_error_is_not_retried():
    """Only the DDL-race class is recoverable. Anything else must surface on the
    first attempt, uncached-over and unretried."""
    driver = _StubDriver(failures=[RuntimeError('something else entirely')])

    with pytest.raises(RuntimeError):
        await AGEGraphOperations._write(
            driver, 'MERGE (n:Persona {uuid: 1})', vertex_labels=('Persona',)
        )

    assert len(driver.queries) == 1
    assert driver.forgotten == []


@pytest.mark.asyncio
async def test_the_wrong_kind_error_still_propagates():
    """AGE raises 3F000 when a name is already taken by the other kind. That is a
    true, permanent error the driver already surfaces deliberately, and the
    recovery must not swallow it into a retry."""
    driver = _StubDriver(
        failures=[asyncpg.exceptions.InvalidSchemaNameError('label "Homonima" already exists')]
    )

    with pytest.raises(asyncpg.exceptions.InvalidSchemaNameError):
        await AGEGraphOperations._write(
            driver, 'MERGE (n:Homonima {uuid: 1})', vertex_labels=('Homonima',)
        )

    assert len(driver.queries) == 1
    assert driver.forgotten == []


@pytest.mark.asyncio
async def test_a_write_that_declared_no_labels_is_not_retried():
    """With nothing declared there is no cached pair to be stale, so a retry
    would just re-run a genuinely failing statement."""
    driver = _StubDriver(failures=[_duplicate_table()])

    with pytest.raises(asyncpg.exceptions.DuplicateTableError):
        await AGEGraphOperations._write(driver, 'MATCH (n) SET n.x = 1')

    assert len(driver.queries) == 1
    assert driver.forgotten == []


@pytest.mark.asyncio
async def test_a_real_writer_recovers_a_stale_cached_label():
    """End to end through a real writer, which is what a re-ingest actually runs:
    `node_save` on a driver whose cache is stale for the node's label."""
    from datetime import datetime, timezone

    from graphiti_core.nodes import EntityNode

    driver = _StubDriver(failures=[_duplicate_table()])
    driver.known_labels.add(('v', 'Persona'))  # stale
    node = EntityNode(
        name='N',
        group_id='bug48',
        labels=['Entity', 'Persona'],
        created_at=datetime.now(timezone.utc),
        summary='',
        attributes={},
    )
    node.name_embedding = None

    await AGEGraphOperations().node_save(node, driver)  # must not raise

    assert ('v', 'Persona') in driver.forgotten
    assert ('v', 'Persona') in driver.known_labels  # re-ensured against the catalogue
    assert len(driver.writes) == 2  # the MERGE and its one retry; the read is not scripted


# ------------------------------------------------- the real driver's cache API


class _FakeTx:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False


class _FakeConn:
    """Stands in for an asyncpg connection over `ag_catalog`. The catalogue is a
    plain set of (kind, label), so a test can make the store disagree with the
    driver's cache — which is precisely the foreign-rebuild condition."""

    def __init__(self, catalogue: set[tuple[str, str]]) -> None:
        self.catalogue = catalogue
        self.executed: list[str] = []

    async def fetchval(self, sql: str, *args):
        if 'ag_label' in sql:
            _graph, label, kind = args
            return 1 if (kind, label) in self.catalogue else None
        return None

    async def execute(self, sql: str, *args):
        self.executed.append(sql)
        if 'create_vlabel' in sql:
            self.catalogue.add(('v', args[1]))
        elif 'create_elabel' in sql:
            self.catalogue.add(('e', args[1]))

    def transaction(self):
        return _FakeTx()


class _FakePool:
    def __init__(self, conn: _FakeConn) -> None:
        self.conn = conn
        self.acquisitions = 0

    def acquire(self):
        pool = self

        class _Acquire:
            async def __aenter__(self):
                pool.acquisitions += 1
                return pool.conn

            async def __aexit__(self, *_exc):
                return False

        return _Acquire()


def _driver_with_fake_pool(catalogue: set[tuple[str, str]]) -> tuple[AGEDriver, _FakePool]:
    driver = AGEDriver(dsn=UNREACHABLE_DSN, graph_name='g_bug48')
    pool = _FakePool(_FakeConn(set(catalogue)))
    driver._pool = pool  # `_get_pool` returns it as-is; nothing ever connects
    return driver, pool


@pytest.mark.asyncio
async def test_forget_label_drops_the_pair_and_the_next_ensure_re_checks_the_store():
    """The recovery's other half, on the real cache: after forgetting, the next
    ensure must go back to `ag_catalog` rather than trust the cache again."""
    driver, pool = _driver_with_fake_pool(catalogue=set())
    await driver.ensure_vertex_label('Persona')
    assert ('v', 'Persona') in driver._known_labels
    acquisitions_after_create = pool.acquisitions

    await driver.ensure_vertex_label('Persona')  # cache hit: free
    assert pool.acquisitions == acquisitions_after_create

    driver.forget_label('v', 'Persona')
    assert ('v', 'Persona') not in driver._known_labels

    await driver.ensure_vertex_label('Persona')  # must consult the store again
    assert pool.acquisitions > acquisitions_after_create
    assert ('v', 'Persona') in driver._known_labels


@pytest.mark.asyncio
async def test_forgetting_a_pair_that_was_never_cached_is_a_no_op():
    """The recovery drops every declared pair without checking first, so a pair
    that was never cached must not raise."""
    driver, _pool = _driver_with_fake_pool(catalogue=set())
    driver.forget_label('e', 'NUNCA_VISTA')  # must not raise
    assert driver._known_labels == set()


@pytest.mark.asyncio
async def test_forgetting_one_kind_leaves_the_other_kind_cached():
    """The cache is keyed on (kind, label) and the invalidation has to respect
    that, or recovering an edge label would silently uncache the vertex one."""
    driver, _pool = _driver_with_fake_pool(catalogue=set())
    driver._known_labels.update({('v', 'Cosa'), ('e', 'COSA')})

    driver.forget_label('e', 'COSA')

    assert driver._known_labels == {('v', 'Cosa')}


@pytest.mark.asyncio
async def test_a_stale_cached_label_is_re_created_after_it_is_forgotten():
    """The foreign-rebuild condition itself, at the driver: the cache says the
    label is present, the store says it is not. Forgetting is what lets the next
    ensure notice and re-create it."""
    driver, _pool = _driver_with_fake_pool(catalogue=set())  # store: empty
    driver._known_labels.add(('e', 'BROADER'))  # cache: stale

    await driver.ensure_edge_label('BROADER')
    assert _pool.conn.catalogue == set()  # short-circuited: nothing created

    driver.forget_label('e', 'BROADER')
    await driver.ensure_edge_label('BROADER')
    assert ('e', 'BROADER') in _pool.conn.catalogue
