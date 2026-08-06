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
from uuid import uuid4

import asyncpg

from graphiti_core.driver.driver import GraphDriver, GraphDriverSession, GraphProvider

logger = logging.getLogger(__name__)

# AGE must be loaded and ag_catalog put on the search_path. This runs via the
# asyncpg pool `setup` hook on EVERY acquire (not `init`, which runs once per
# physical connection): the pool issues RESET ALL on release, which would wipe a
# once-set search_path and leave cypher() unresolvable on the next acquire.
_SESSION_INIT = "LOAD 'age'; SET search_path = ag_catalog, \"$user\", public;"

_PARAM_RE = re.compile(r'\$([A-Za-z_][A-Za-z0-9_]*)')


def _inline_cypher_params(query: str, params: dict[str, Any]) -> str:
    """Substitute openCypher ``$name`` parameters into the query as escaped
    literals.

    AGE's ``cypher()`` SQL function does not accept openCypher ``$name`` bound
    parameters (they must go through the awkward agtype 3rd argument), so core
    callers that pass params — e.g. ``explore_node`` looking a node up by uuid or
    name — would otherwise be unusable. Reuse the write-path ``_cy`` serializer
    (the same one the graph-operations layer uses for MERGE literals) to turn each
    value into a safe Cypher literal, replacing every ``$name`` token whose name
    is present in ``params``. Unknown ``$name`` tokens are left untouched. The
    regex matches whole identifier tokens, so ``$group`` never partially matches
    inside ``$group_ids``.
    """
    from graphiti_core.driver.graph_operations.age_graph_operations import _cy

    def _repl(m: 're.Match[str]') -> str:
        name = m.group(1)
        return _cy(params[name]) if name in params else m.group(0)

    return _PARAM_RE.sub(_repl, query)


def _is_word_char(ch: str) -> bool:
    """True for identifier characters — used for keyword word-boundary checks."""
    return bool(ch) and (ch.isalnum() or ch == '_')


def _iter_code_chars(s: str):
    """Yield ``(index, char, depth)`` for every CODE character of ``s``.

    Skipped whole: string literals (``'…'``, ``"…"``), backtick-quoted identifiers
    (``n.`credit union```) and comments (``//…``, ``/*…*/``). ``depth`` is the
    bracket nesting level counted over code characters only, reported with an
    opening bracket already counted and a closing bracket already discounted.

    This is the ONE lexer every structural scan below runs on — the RETURN and
    UNION and ORDER BY/LIMIT/SKIP keyword searches and the projection-splitting
    comma search. It exists because this driver builds every write by inlining
    its property values as Cypher literals (``SET n += {summary: '…'}``), so the
    query text is part query and part arbitrary user DATA. A scan over the raw
    text cannot tell the two apart, and any keyword or comma inside a value then
    steers the generated ``AS (col agtype, …)`` list — which is how a summary
    containing ``… RETURN p.name, count(d)`` made AGE reject the whole write with
    ``column definition list for CREATE clause must contain a single agtype
    attribute`` (BUG-42). Deriving structure only from code positions makes that
    impossible for ANY value, not just the ones seen so far.

    An unterminated literal or comment runs to the end of the string, which ends
    the scan rather than hanging it.
    """
    depth = 0
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch in '\'"`':
            quote = ch
            i += 1
            while i < n:
                c = s[i]
                # Backtick-quoted identifiers take no backslash escapes (a literal
                # backtick is doubled, which closes and immediately reopens here —
                # harmless, since the span still ends where the identifier does).
                if quote != '`' and c == '\\':
                    i += 2
                    continue
                i += 1
                if c == quote:
                    break
            continue
        if ch == '/' and s[i + 1 : i + 2] == '/':
            end = s.find('\n', i + 2)
            i = n if end == -1 else end + 1
            continue
        if ch == '/' and s[i + 1 : i + 2] == '*':
            end = s.find('*/', i + 2)
            i = n if end == -1 else end + 2
            continue
        if ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
        yield i, ch, depth
        i += 1


def _iter_code_keyword(s: str, keyword: str, top_level_only: bool = False):
    """Yield the index of every whole-word, code-position ``keyword`` occurrence.

    Case-insensitive. ``top_level_only`` additionally restricts matches to bracket
    depth 0. Occurrences inside literals, backticked identifiers or comments are
    not query structure and are never yielded — see ``_iter_code_chars``.
    """
    k = len(keyword)
    heads = (keyword[0].lower(), keyword[0].upper())
    for i, ch, depth in _iter_code_chars(s):
        if ch not in heads or (top_level_only and depth != 0):
            continue
        if s[i : i + k].lower() != keyword:
            continue
        before = s[i - 1] if i else ''
        if not _is_word_char(before) and not _is_word_char(s[i + k : i + k + 1]):
            yield i


def _find_code_keyword(s: str, keyword: str, top_level_only: bool = False) -> int:
    """Index of the first whole-word, code-position ``keyword``; -1 when absent."""
    return next(_iter_code_keyword(s, keyword, top_level_only), -1)


def _dollar_quote_tag(query: str) -> str:
    """A dollar-quote tag (``$q<hex8>$``) that does not occur inside ``query``.

    The Cypher body is spliced into SQL as ``cypher('<graph>', <tag> … <tag>)``.
    A FIXED ``$$`` tag is a breakout: a query containing ``$$`` closes the quote
    early and the remainder is parsed as raw SQL — e.g.
    ``MATCH (n) RETURN 1 $$) AS (result agtype) LIMIT 1 --``, which the Cypher
    write-verb whitelist upstream of here does not stop. Generating the tag per
    query, and regenerating while the query happens to contain it, makes that
    structurally impossible regardless of any caller-side validation.
    """
    while True:
        tag = f'$q{uuid4().hex[:8]}$'
        if tag not in query:
            return tag


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
        # (kind, label) pairs this driver has already materialised — see
        # `_ensure_label`. Positive entries only, so a label another process
        # created is simply re-verified once and then cached too.
        self._known_labels: set[tuple[str, str]] = set()
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

    async def execute_sql(self, sql: str, *args: Any) -> list:
        """Run plain SQL against the pool (used by the interface implementations
        for pgvector/tsvector shadow-table access). Returns fetched rows."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            return await conn.fetch(sql, *args)

    # ---- label materialisation (BUG-38: AGE's implicit label DDL is not safe
    # ---- under concurrency) --------------------------------------------------
    #
    # AGE creates a label's backing relations the first time a MERGE/CREATE names
    # it, implicitly, from inside whatever statement gets there first — with no
    # lock and no IF NOT EXISTS. Concurrent writers sharing a brand-new label
    # therefore run the same DDL at once and the losers abort their whole save:
    #   DuplicateTableError  42P07  relation "BROADER" already exists
    #   UniqueViolationError 23505  pg_class_relname_nsp_index / "BROADER_id_seq"
    # (5 of 428 skos:broader edges lost on a live ontology load, 2026-08-05.)
    #
    # Catching those two SQLSTATEs and retrying would be a guess at a list AGE
    # does not publish — it creates a table AND a sequence per label today, and
    # the collision can land on either. Instead the DDL is made EXPLICIT and
    # serialized: one advisory lock per (graph, label), taken only on the path
    # that would otherwise create the label, released at commit. Writers declare
    # the labels they name (`AGEGraphOperations._write`), so by the time any
    # MERGE runs its labels already exist and the implicit path is never reached
    # concurrently. Per-label keying keeps unrelated labels parallel.

    async def _label_exists(self, conn: asyncpg.Connection, label: str) -> bool:
        return bool(
            await conn.fetchval(
                'SELECT 1 FROM ag_catalog.ag_label l '
                'JOIN ag_catalog.ag_graph g ON g.graphid = l.graph '
                'WHERE g.name = $1 AND l.name = $2',
                self._database,
                label,
            )
        )

    async def _ensure_label(self, label: str, kind: str) -> None:
        if (kind, label) in self._known_labels:
            return
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            if not await self._label_exists(conn, label):
                async with conn.transaction():
                    # Transaction-scoped: the winner holds it across its DDL, the
                    # losers block here and then find the label already present.
                    await conn.execute(
                        'SELECT pg_advisory_xact_lock(hashtext($1), hashtext($2))',
                        self._database,
                        label,
                    )
                    if not await self._label_exists(conn, label):
                        create = 'create_vlabel' if kind == 'v' else 'create_elabel'
                        await conn.execute(
                            f'SELECT ag_catalog.{create}($1::name::cstring, $2::name::cstring)',
                            self._database,
                            label,
                        )
        self._known_labels.add((kind, label))

    async def ensure_vertex_label(self, label: str) -> None:
        """Materialise a vertex label before any write MERGEs on it."""
        await self._ensure_label(label, 'v')

    async def ensure_edge_label(self, label: str) -> None:
        """Materialise an edge label before any write MERGEs on it."""
        await self._ensure_label(label, 'e')

    # Shadow tables (pgvector + tsvector) keyed by node/edge uuid. AGE has no
    # native vector or fulltext, so search runs against these, kept in sync by
    # the graph_operations save methods. One pair per graph, in `public`.
    @property
    def _node_tbl(self) -> str:
        return f'"{self._database}__node_search"'

    @property
    def _edge_tbl(self) -> str:
        return f'"{self._database}__edge_search"'

    @property
    def _ix(self) -> str:
        # index-name prefix; graph names use only safe identifier chars
        return self._database

    # ---- agtype / RETURN-clause helpers (Phase 0: simple queries only) ----

    @staticmethod
    def _split_top_commas(s: str) -> list[str]:
        """Split on the commas that separate projections — bracket depth 0 AND a
        code position, so a comma inside a string literal (``RETURN 'a, b' AS x``)
        never invents a column."""
        parts: list[str] = []
        start = 0
        for i, ch, depth in _iter_code_chars(s):
            if ch == ',' and depth == 0:
                parts.append(s[start:i])
                start = i + 1
        if s[start:]:
            parts.append(s[start:])
        return parts

    @staticmethod
    def _cut_at_top_level_union(clause: str) -> str:
        """Truncate ``clause`` at the first top-level ``UNION`` keyword.

        A UNION query has one RETURN per branch, and the branches MUST share their
        aliases — that is what makes a UNION well-formed. Capturing past the first
        branch would therefore repeat every alias in the generated
        ``AS (col agtype, …)`` list, which PostgreSQL rejects with
        ``column name "…" specified more than once``. PostgreSQL itself names the
        columns of a set operation after its FIRST arm, so the first branch's
        RETURN is the correct source.

        Runs on the shared ``_iter_code_chars`` lexer — bracket depth, plus the
        three spans where a ``UNION`` is only text and never a branch boundary:
        string literals (``'…'``, ``"…"``), backtick-quoted identifiers
        (``n.`credit union```), and comments (``//…``, ``/*…*/``). Cutting inside
        one of those drops the projections that follow it, and the short column
        list then fails AGE's own arity check — a regression on queries that
        worked before, so all three are skipped whole.
        """
        cut = _find_code_keyword(clause, 'union', top_level_only=True)
        return clause if cut < 0 else clause[:cut]

    @staticmethod
    def _strip_trailing_subclauses(clause: str) -> str:
        """Drop everything from the first ``ORDER BY`` / ``LIMIT`` / ``SKIP`` on.

        Code positions only, for the same reason the rest of the scanning is:
        ``RETURN 'no limit' AS nota, n.x AS y`` must keep both projections. A raw
        text split truncated the clause at the word inside the literal and
        returned a column list one short, which AGE rejects for arity.
        """
        cut = len(clause)
        for keyword in ('limit', 'skip'):
            found = _find_code_keyword(clause, keyword)
            if 0 <= found < cut:
                cut = found
        # ORDER BY is two words with arbitrary whitespace between them; a bare
        # `order` (a property named order, say) is not a subclause.
        for found in _iter_code_keyword(clause, 'order'):
            if re.match(r'\s+by\b', clause[found + 5 :], re.IGNORECASE):
                cut = min(cut, found)
                break
        return clause[:cut]

    @classmethod
    def _columns_from_return(cls, cypher: str) -> list[str] | None:
        """Best-effort extraction of output column names from a RETURN clause.

        Phase 0: handles the simple, controlled queries this driver issues, plus
        UNION queries (columns come from the first branch — see
        ``_cut_at_top_level_union``). Robust parsing for arbitrary user Cypher is
        deferred to the run_cypher MCP phase.

        The RETURN keyword is located over code positions only, so a property
        value that merely contains the word — the write path inlines every value
        as a literal — is not mistaken for a clause. ``None`` (no RETURN at all)
        is what makes a write query declare the single agtype column AGE demands.
        """
        start = _find_code_keyword(cypher, 'return')
        if start < 0:
            return None
        # Cut at UNION *before* stripping ORDER BY / LIMIT / SKIP: those may sit on a
        # later branch, in which case stripping first would leave the branch text in.
        clause = cls._strip_trailing_subclauses(
            cls._cut_at_top_level_union(cypher[start + len('return') :])
        )
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

    def _cypher_sql(self, cypher_query_: str, cols: list[str]) -> str:
        """The ``cypher()`` SQL wrapper for one Cypher body, dollar-quoted with a
        per-query unique tag (see ``_dollar_quote_tag``)."""
        tag = _dollar_quote_tag(cypher_query_)
        col_def = ', '.join(f'{c} agtype' for c in cols)
        return (
            f"SELECT * FROM cypher('{self._database}', {tag} {cypher_query_} {tag}) AS ({col_def})"
        )

    async def execute_query(self, cypher_query_: str, columns: list[str] | None = None, **kwargs: Any):
        if kwargs:
            # AGE's cypher() takes no openCypher `$name` params, so inline them as
            # escaped literals (reusing the write-path serializer). Unblocks core
            # paths that pass params — e.g. explore_node's uuid/name lookups.
            cypher_query_ = _inline_cypher_params(cypher_query_, kwargs)
        cols = columns or self._columns_from_return(cypher_query_) or ['result']
        sql = self._cypher_sql(cypher_query_, cols)
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
        dim = self.embedding_dim
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            if delete_existing:
                if await self._graph_exists(conn):
                    await conn.execute('SELECT drop_graph($1::name, true)', self._database)
                # Dropping the graph drops every label with it; a cache that
                # survived would let the first write after a rebuild fall back
                # into the unsynchronised implicit-DDL path.
                self._known_labels.clear()
                await conn.execute(f'DROP TABLE IF EXISTS {self._node_tbl} CASCADE')
                await conn.execute(f'DROP TABLE IF EXISTS {self._edge_tbl} CASCADE')
            if not await self._graph_exists(conn):
                await conn.execute('SELECT create_graph($1::name)', self._database)
            await conn.execute(
                f"""CREATE TABLE IF NOT EXISTS {self._node_tbl} (
                    uuid text PRIMARY KEY,
                    group_id text NOT NULL,
                    content text,
                    name_embedding vector({dim}),
                    tsv tsvector GENERATED ALWAYS AS
                        (to_tsvector('simple', coalesce(content, ''))) STORED
                )"""
            )
            await conn.execute(
                f"""CREATE TABLE IF NOT EXISTS {self._edge_tbl} (
                    uuid text PRIMARY KEY,
                    group_id text NOT NULL,
                    source_node_uuid text,
                    target_node_uuid text,
                    content text,
                    fact_embedding vector({dim}),
                    tsv tsvector GENERATED ALWAYS AS
                        (to_tsvector('simple', coalesce(content, ''))) STORED
                )"""
            )
            await conn.execute(
                f'CREATE INDEX IF NOT EXISTS {self._ix}_node_emb ON {self._node_tbl} '
                f'USING hnsw (name_embedding vector_cosine_ops)'
            )
            await conn.execute(
                f'CREATE INDEX IF NOT EXISTS {self._ix}_node_tsv ON {self._node_tbl} USING gin (tsv)'
            )
            await conn.execute(
                f'CREATE INDEX IF NOT EXISTS {self._ix}_node_gid ON {self._node_tbl} (group_id)'
            )
            await conn.execute(
                f'CREATE INDEX IF NOT EXISTS {self._ix}_edge_emb ON {self._edge_tbl} '
                f'USING hnsw (fact_embedding vector_cosine_ops)'
            )
            await conn.execute(
                f'CREATE INDEX IF NOT EXISTS {self._ix}_edge_tsv ON {self._edge_tbl} USING gin (tsv)'
            )
            await conn.execute(
                f'CREATE INDEX IF NOT EXISTS {self._ix}_edge_gid ON {self._edge_tbl} (group_id)'
            )

    async def delete_all_indexes(self) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            for suffix in ('node_emb', 'node_tsv', 'node_gid', 'edge_emb', 'edge_tsv', 'edge_gid'):
                await conn.execute(f'DROP INDEX IF EXISTS {self._ix}_{suffix}')

    async def drop_graph(self) -> None:
        """Drop this driver's AGE graph + shadow tables (test teardown helper)."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            if await self._graph_exists(conn):
                await conn.execute('SELECT drop_graph($1::name, true)', self._database)
            self._known_labels.clear()
            await conn.execute(f'DROP TABLE IF EXISTS {self._node_tbl} CASCADE')
            await conn.execute(f'DROP TABLE IF EXISTS {self._edge_tbl} CASCADE')
