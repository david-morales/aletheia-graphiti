"""PostgreSQL + Apache AGE + pgvector graph driver for Graphiti (Phase 0 spike).

This driver uses Graphiti's escape-hatch interfaces (`search_interface`,
`graph_operations_interface`) so node/edge/search operations are implemented in
native PostgreSQL SQL rather than through Graphiti's Cypher generation. AGE
provides graph storage/traversal, pgvector provides similarity search, and
Postgres tsvector provides keyword search (both via uuid-keyed shadow tables).

`execute_query` remains for the queries the driver itself issues (graph DDL,
maintenance) and for any core path not covered by the interfaces. It wraps
Cypher in AGE's `cypher()` SQL function and decodes agtype results.
"""

import json
import logging
import re
from typing import Any

import asyncpg

from graphiti_core.driver.driver import GraphDriver, GraphDriverSession, GraphProvider

logger = logging.getLogger(__name__)

# AGE must be loaded and ag_catalog put on the search_path. This runs via the
# asyncpg pool `setup` hook on EVERY acquire (not `init`, which runs once per
# physical connection): the pool issues RESET ALL on release, which would wipe a
# once-set search_path and leave cypher() unresolvable on the next acquire.
_SESSION_INIT = "LOAD 'age'; SET search_path = ag_catalog, \"$user\", public;"


class AGEDriverSession(GraphDriverSession):
    provider = GraphProvider.AGE

    def __init__(self, driver: 'AGEDriver'):
        self._driver = driver

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def close(self):
        pass

    async def execute_write(self, func, *args, **kwargs):
        return await func(self, *args, **kwargs)

    async def run(self, query: str | list, **kwargs: Any) -> Any:
        if isinstance(query, list):
            for cypher, params in query:
                await self._driver.execute_query(cypher, **params)
        else:
            await self._driver.execute_query(query, **kwargs)
        return None


class AGEDriver(GraphDriver):
    provider = GraphProvider.AGE
    default_group_id: str = ''
    fulltext_syntax: str = ''  # Postgres tsquery needs no prefix
    aoss_client: None = None

    def __init__(self, dsn: str, graph_name: str = 'graphiti', embedding_dim: int = 1536):
        super().__init__()
        self._dsn = dsn
        self._database = graph_name
        self.embedding_dim = embedding_dim
        self._pool: asyncpg.Pool | None = None
        # Escape-hatch: assign the interface implementations. Imported lazily to
        # avoid a circular import at module load.
        from graphiti_core.driver.graph_operations.age_graph_operations import (
            AGEGraphOperations,
        )
        from graphiti_core.driver.search_interface.age_search import AGESearch

        self.graph_operations_interface = AGEGraphOperations()
        self.search_interface = AGESearch()

    async def _get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            async def _setup(conn: asyncpg.Connection) -> None:
                await conn.execute(_SESSION_INIT)

            self._pool = await asyncpg.create_pool(
                self._dsn, setup=_setup, min_size=1, max_size=8
            )
        return self._pool

    # ---- agtype / RETURN-clause helpers (Phase 0: simple queries only) ----

    @staticmethod
    def _split_top_commas(s: str) -> list[str]:
        parts: list[str] = []
        depth = 0
        buf: list[str] = []
        for ch in s:
            if ch in '([{':
                depth += 1
            elif ch in ')]}':
                depth -= 1
            if ch == ',' and depth == 0:
                parts.append(''.join(buf))
                buf = []
            else:
                buf.append(ch)
        if buf:
            parts.append(''.join(buf))
        return parts

    @classmethod
    def _columns_from_return(cls, cypher: str) -> list[str] | None:
        """Best-effort extraction of output column names from a RETURN clause.

        Phase 0: handles the simple, controlled queries this driver issues.
        Robust parsing for arbitrary user Cypher is deferred to the run_cypher
        MCP phase.
        """
        m = re.search(r'\breturn\b(.*)$', cypher, re.IGNORECASE | re.DOTALL)
        if not m:
            return None
        clause = re.split(
            r'\b(order\s+by|limit|skip)\b', m.group(1), flags=re.IGNORECASE
        )[0]
        cols: list[str] = []
        for i, part in enumerate(cls._split_top_commas(clause)):
            alias = re.search(r'\bas\b\s+([A-Za-z_]\w*)\s*$', part.strip(), re.IGNORECASE)
            cols.append(alias.group(1) if alias else f'col{i}')
        return cols or None

    @staticmethod
    def _decode_agtype(val: Any) -> Any:
        if val is None or not isinstance(val, str):
            return val
        # Strip a trailing ::type annotation (e.g. ::vertex, ::edge, ::numeric).
        stripped = re.sub(r'::[a-zA-Z_]+\s*$', '', val)
        try:
            return json.loads(stripped)
        except (ValueError, TypeError):
            return stripped

    async def execute_query(self, cypher_query_: str, columns: list[str] | None = None, **kwargs: Any):
        if kwargs:
            # Passing Cypher params through AGE's cypher() agtype 3rd argument is
            # a known friction point (must be a bound parameter). Phase 0's
            # controlled queries pass none; implement when a task first needs it.
            raise NotImplementedError(
                'AGE execute_query with cypher params is not implemented yet'
            )
        cols = columns or self._columns_from_return(cypher_query_) or ['result']
        col_def = ', '.join(f'{c} agtype' for c in cols)
        sql = f"SELECT * FROM cypher('{self._database}', $$ {cypher_query_} $$) AS ({col_def})"
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql)
        records = [{c: self._decode_agtype(r[c]) for c in cols} for r in rows]
        return records, list(cols), None

    def session(self, database: str | None = None) -> GraphDriverSession:
        return AGEDriverSession(self)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def _graph_exists(self, conn: asyncpg.Connection) -> bool:
        count = await conn.fetchval(
            'SELECT count(*) FROM ag_catalog.ag_graph WHERE name = $1', self._database
        )
        return bool(count)

    async def build_indices_and_constraints(self, delete_existing: bool = False):
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            if delete_existing and await self._graph_exists(conn):
                await conn.execute('SELECT drop_graph($1::name, true)', self._database)
            if not await self._graph_exists(conn):
                await conn.execute('SELECT create_graph($1::name)', self._database)
        # Task 3 adds the pgvector/tsvector shadow tables + indexes here.

    async def delete_all_indexes(self) -> None:
        # Shadow-table indexes are created in Task 3; nothing to drop yet.
        return None

    async def drop_graph(self) -> None:
        """Drop this driver's AGE graph (test teardown helper)."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            if await self._graph_exists(conn):
                await conn.execute('SELECT drop_graph($1::name, true)', self._database)
