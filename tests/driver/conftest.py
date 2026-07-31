"""Fixtures for the AGE driver spike tests.

Requires the docker-compose.age.yml service running and reachable at AGE_TEST_DSN
(default postgresql://age:age@localhost:5433/age_test).
"""

import os
import socket
import uuid

import asyncpg
import pytest
import pytest_asyncio

from graphiti_core.driver.age_driver import AGEDriver

AGE_DSN = os.environ.get('AGE_TEST_DSN', 'postgresql://age:age@localhost:5433/age_test')

# Connectivity-class failures only: anything else (e.g. the driver's own DDL
# breaking, AGE missing from a reachable Postgres) must FAIL, not skip — a bare
# except here once turned the whole live AGE suite silently green.
_UNREACHABLE_ERRORS = (
    OSError,
    socket.gaierror,
    asyncpg.CannotConnectNowError,
    asyncpg.InvalidCatalogNameError,
    asyncpg.InvalidPasswordError,
)


@pytest_asyncio.fixture
async def age_driver():
    """A connected AGEDriver on a fresh, uniquely-named graph, dropped after the test.

    Skips (never fails) when the store is unreachable, so the offline suite stays
    green without a live AGE bed. The graph is created inside the try that guards
    connectivity and dropped in the finally that follows it, so no path can leave
    a graph behind.
    """
    graph = 'g_' + uuid.uuid4().hex[:10]
    driver = AGEDriver(dsn=AGE_DSN, graph_name=graph, embedding_dim=1536)
    try:
        await driver.build_indices_and_constraints(delete_existing=True)
    except _UNREACHABLE_ERRORS as exc:  # store down => skip; real failures propagate
        await driver.close()
        pytest.skip(f'AGE store not reachable at {AGE_DSN}: {exc}')
    try:
        yield driver
    finally:
        await driver.drop_graph()
        await driver.close()
