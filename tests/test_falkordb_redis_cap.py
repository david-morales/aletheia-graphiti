"""Regression: the `[falkordb]` extra must cap redis below 8.1 (aletheia BUG-35).

redis 8.1.0's async ConnectionPool injects internal bookkeeping keys
(`himport_registry`, `maint_notifications_*`, `orig_*`) into
`connection_kwargs` (redis/asyncio/connection.py:2632-2642). falkordb<=1.6.2
`Is_Cluster` (falkordb/asyncio/cluster.py:27) forwards those kwargs verbatim
to a SYNC `redis.Redis(**kwargs)` probe, which rejects them:

    TypeError: Redis.__init__() got an unexpected keyword argument
    'himport_registry'

`Is_Cluster` runs inside `falkordb.asyncio.FalkorDB.__init__`, so EVERY async
client construction dies — graphiti's FalkorDriver cannot start. falkordb
declares `redis>=7.1.0` unbounded, so without a cap of our own any fresh
resolve is free to pick the broken combo.

Two tests:
  * packaging — the falkordb extra carries a redis requirement whose
    specifier excludes 8.1.0 while keeping the known-good 8.0.x line;
  * live canary — constructing `falkordb.asyncio.FalkorDB` exercises the
    exact poisoned path (Is_Cluster -> sync probe -> INFO). Under the broken
    combo the TypeError fires BEFORE any I/O, so the test FAILS (not skips)
    even with no server; it skips on redis connection-level errors (no
    FalkorDB reachable) and honours DISABLE_FALKORDB. Any other exception
    propagates as a genuine failure.
"""

import os
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

PYPROJECT = Path(__file__).parents[1] / 'pyproject.toml'


def test_every_extra_that_pulls_falkordb_also_caps_redis():
    """Any extra requiring falkordb must carry its own redis cap — falkordb
    declares redis>=7.1.0 unbounded, so each such group independently resolves
    the broken 8.1 line unless capped (the dev extra duplicates the falkordb
    requirement rather than referencing the extra, so the cap must follow it).
    """
    with PYPROJECT.open('rb') as f:
        data = tomllib.load(f)
    extras = data['project']['optional-dependencies']
    falkordb_extras = [
        name
        for name, group in extras.items()
        if any(req.name == 'falkordb' for req in map(Requirement, group))
    ]
    assert falkordb_extras, 'expected at least the falkordb extra to require falkordb'
    for name in falkordb_extras:
        redis_reqs = [req for req in map(Requirement, extras[name]) if req.name == 'redis']
        assert redis_reqs, (
            f'extra "{name}" requires falkordb but carries no redis requirement — '
            f'falkordb<=1.6.2 declares redis>=7.1.0 unbounded, so this group '
            f'resolves the broken 8.1 line (aletheia BUG-35)'
        )
        spec = redis_reqs[0].specifier
        assert not spec.contains('8.1.0'), (
            f'extra "{name}" redis specifier "{spec}" admits the broken 8.1.0'
        )
        assert spec.contains('8.0.1'), (
            f'extra "{name}" redis specifier "{spec}" must keep the known-good 8.0.x line'
        )


@pytest.mark.skipif(
    os.getenv('DISABLE_FALKORDB') is not None, reason='FalkorDB tests disabled'
)
def test_async_falkordb_client_construction_survives_is_cluster_probe():
    falkordb_asyncio = pytest.importorskip('falkordb.asyncio')
    redis_exceptions = pytest.importorskip('redis.exceptions')
    host = os.getenv('FALKORDB_HOST', 'localhost')
    port = int(os.getenv('FALKORDB_PORT', '6379'))
    try:
        falkordb_asyncio.FalkorDB(host=host, port=port)
    except TypeError as exc:
        # The BUG-35 signature: the async pool's internal kwargs leaked into
        # the sync probe constructor. Never skip on this — it is the bug.
        pytest.fail(f'async FalkorDB construction died in the sync Is_Cluster probe: {exc}')
    except (redis_exceptions.RedisError, OSError) as exc:
        # Connection-level only: no server reachable. Anything else propagates.
        pytest.skip(f'no FalkorDB reachable at {host}:{port}: {exc}')
