"""Fixtures for the AGE driver spike tests.

Requires the docker-compose.age.yml service running and reachable at AGE_TEST_DSN
(default postgresql://age:age@localhost:5433/age_test).
"""

import os
import uuid

import pytest_asyncio

from graphiti_core.driver.age_driver import AGEDriver

AGE_DSN = os.environ.get('AGE_TEST_DSN', 'postgresql://age:age@localhost:5433/age_test')


@pytest_asyncio.fixture
async def age_driver():
    """A connected AGEDriver on a fresh, uniquely-named graph, dropped after the test."""
    graph = 'g_' + uuid.uuid4().hex[:10]
    driver = AGEDriver(dsn=AGE_DSN, graph_name=graph, embedding_dim=1536)
    await driver.build_indices_and_constraints(delete_existing=True)
    try:
        yield driver
    finally:
        await driver.drop_graph()
        await driver.close()
