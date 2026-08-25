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

# The error class a write raises when it reached AGE's implicit label DDL and
# lost the race — i.e. when the explicit-DDL guard below did NOT run first.
# Declared once, here, next to the SQLSTATEs it names; imported by
# `AGEGraphOperations._write`, which is the only place that recovers from it.
#
#   DuplicateTableError  42P07  the label's backing relation
#   UniqueViolationError 23505  pg_class_relname_nsp_index, i.e. the same race
#                               landing on the label's SEQUENCE instead
#
# This is NOT a second guard against BUG-38 — the advisory-locked explicit DDL is
# the guard, and catching SQLSTATEs was rejected for that job precisely because
# AGE publishes no bounded list of the relations it creates per label. This is
# the narrower recovery for BUG-48: the guard was SKIPPED because the cache
# claimed a label that a foreign rebuild had removed. So the response is not
# "retry until it works", it is "the cache lied — drop it, re-ensure against the
# real catalogue, try once more". A second failure means the cache was not the
# problem and the error is real.
#
# `InvalidSchemaNameError` (3F000, "label X already exists" for the other kind)
# is deliberately absent: that is a true, permanent error the driver surfaces on
# purpose, and retrying it would only bury it.
LABEL_DDL_RACE_ERRORS: tuple[type[BaseException], ...] = (
    asyncpg.exceptions.DuplicateTableError,
    asyncpg.exceptions.UniqueViolationError,
)

# The text-search configuration used when none is named. `simple` folds case and
# does nothing else — no stemming, no stopwords — which is the only defensible
# DEFAULT for a language-agnostic engine. A deployment that knows its corpus
# language names it (`spanish`, `english`, …) and gets stemming + stopwords.
DEFAULT_TEXT_SEARCH_CONFIG = 'simple'

# An optionally schema-qualified SQL identifier — the shape `regconfig` accepts.
#
# The configuration name reaches SQL by two routes. The QUERY side takes it as a
# bind parameter (`age_search._OR_TSQUERY`) and needs no validation at all. The
# INDEX side cannot: a generated-column expression takes no parameters, so the
# name has to be inlined as a literal. This regex is what makes that literal
# safe — it admits no quote, no semicolon, no whitespace, nothing that could
# break out of the single quotes it is emitted inside.
_TEXT_SEARCH_CONFIG_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$')

# Pulls the configuration back out of a stored generated-column expression, which
# Postgres renders as e.g. `to_tsvector('simple'::regconfig, COALESCE(content, ''::text))`.
_STORED_TSV_CONFIG_RE = re.compile(r"to_tsvector\(\s*'([^']*)'")


def validate_text_search_config(name: str) -> str:
    """Check a text-search configuration name is safe to inline into DDL.

    Raised at driver CONSTRUCTION rather than at DDL time so a bad value fails
    before it can reach a database — and so the check is testable with no
    database at all.
    """
    if not isinstance(name, str) or not _TEXT_SEARCH_CONFIG_RE.match(name):
        raise ValueError(
            f'Invalid text_search_config {name!r}: expected a PostgreSQL text-search '
            'configuration name (an optionally schema-qualified identifier, e.g. '
            "'simple', 'spanish', 'pg_catalog.english')."
        )
    return name


def stored_tsv_config(generated_expr: str | None) -> str | None:
    """The configuration a stored `tsv` column was generated with, if readable."""
    if not generated_expr:
        return None
    match = _STORED_TSV_CONFIG_RE.search(generated_expr)
    return match.group(1) if match else None


def text_search_configs_agree(configured: str, stored: str | None) -> bool:
    """Whether a query-time configuration matches the one a `tsv` column holds.

    A tsquery only matches a tsvector when both were produced by the same
    configuration — stemmed query lexemes cannot match unstemmed stored ones — so
    a disagreement here means fulltext search silently returns nothing.

    An unreadable stored expression is NOT reported as a disagreement: absence of
    evidence would otherwise turn into a false alarm on every start-up.
    Schema-qualified names compare on their last segment, because Postgres
    resolves `spanish` and `pg_catalog.spanish` to the same configuration and
    renders whichever the search_path yields.
    """
    if stored is None:
        return True
    return configured.rsplit('.', 1)[-1].lower() == stored.rsplit('.', 1)[-1].lower()


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

    def __init__(
        self,
        dsn: str,
        graph_name: str = 'graphiti',
        embedding_dim: int = 1536,
        text_search_config: str = DEFAULT_TEXT_SEARCH_CONFIG,
    ):
        super().__init__()
        self._dsn = dsn
        self._database = graph_name
        self.embedding_dim = embedding_dim
        # Language is a property of the CORPUS, never of the engine (ADR-003's
        # rule in the fork's own terms), so it arrives the same way
        # `embedding_dim` does: as a driver parameter. It is read back by
        # `AGESearch` off the driver, and inlined into the `tsv` generated column
        # below — the two must name the SAME configuration or fulltext matches
        # nothing. See DESIGN-age-fulltext.md.
        self.text_search_config = validate_text_search_config(text_search_config)
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

    async def _label_exists(self, conn: asyncpg.Connection, label: str, kind: str) -> bool:
        """Whether this graph already has ``label`` AS ``kind`` ('v' or 'e').

        The kind belongs in the filter because the cache this feeds is keyed on
        (kind, label). Matching on the name alone would report an existing VERTEX
        label as satisfying a request for an EDGE label of that name: the miss
        path would be skipped, nothing created, and ('e', name) cached as present
        — a recorded fact that is not true. AGE forbids the two anyway
        (``create_elabel`` then raises ``label "X" already exists``, SQLSTATE
        3F000), so filtering by kind costs nothing and lets that clear error
        surface instead of being cached over.
        """
        return bool(
            await conn.fetchval(
                'SELECT 1 FROM ag_catalog.ag_label l '
                'JOIN ag_catalog.ag_graph g ON g.graphid = l.graph '
                'WHERE g.name = $1 AND l.name = $2 AND l.kind::text = $3',
                self._database,
                label,
                kind,
            )
        )

    async def _ensure_label(self, label: str, kind: str) -> None:
        if (kind, label) in self._known_labels:
            return
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            if not await self._label_exists(conn, label, kind):
                async with conn.transaction():
                    # Transaction-scoped: the winner holds it across its DDL, the
                    # losers block here and then find the label already present.
                    await conn.execute(
                        'SELECT pg_advisory_xact_lock(hashtext($1), hashtext($2))',
                        self._database,
                        label,
                    )
                    # This re-check is what makes the losers no-op, and it is
                    # correct only under READ COMMITTED — where each statement
                    # takes a fresh snapshot and therefore sees the winner's
                    # committed label. Under REPEATABLE READ or SERIALIZABLE the
                    # loser's snapshot predates that commit, the re-check misses,
                    # and create_vlabel/create_elabel raises 3F000 ("label …
                    # already exists" — AGE checks its own catalog before any
                    # CREATE TABLE, so the implicit-DDL 42P07 never appears on
                    # this path; measured at 24-way concurrency) — i.e. the very
                    # failure this guard exists to prevent. Postgres ships READ
                    # COMMITTED by default and the AGE bed runs it; a deployment
                    # that raises default_transaction_isolation must set this
                    # connection back to READ COMMITTED.
                    if not await self._label_exists(conn, label, kind):
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

    def forget_label(self, kind: str, label: str) -> None:
        """Discard a cached (kind, label) pair so the next ensure re-checks the store.

        BUG-48. `_known_labels` is per-INSTANCE and is cleared only by this
        instance's own `build_indices_and_constraints(delete_existing=True)` and
        `drop_graph`. A different instance or process rebuilding the graph
        therefore leaves this cache asserting labels that no longer exist:
        `_ensure_label` short-circuits, nothing is created, and the MERGE becomes
        the label's first real use again — back on AGE's unsynchronised implicit
        DDL, which is the failure the guard exists to prevent (measured: 7 of 24
        concurrent saves lost after a foreign rebuild).

        Nothing here polls or re-verifies on a cadence: a periodic `ag_catalog`
        check would cost a query on writes that are fine, and would still have a
        window. The cache is dropped only where the lie has already shown itself
        — `AGEGraphOperations._write`, on `LABEL_DDL_RACE_ERRORS` — so the hot
        path keeps costing nothing.

        Idempotent by construction: the recovery drops every label the failed
        write declared without first asking which of them raced, so forgetting a
        pair that was never cached must be a no-op.
        """
        self._known_labels.discard((kind, label))

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

    @property
    def _tsv_generated_expr(self) -> str:
        """The generated-column expression behind every shadow table's `tsv`.

        The configuration is INLINED, not parameterised: a generated column takes
        no parameters, and the two-argument `to_tsvector` is the immutable form a
        generated column requires. `validate_text_search_config` in `__init__` is
        what makes inlining safe — the accepted shape contains no quote to break
        out of.
        """
        return f"to_tsvector('{self.text_search_config}', coalesce(content, ''))"

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
                        ({self._tsv_generated_expr}) STORED
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
                        ({self._tsv_generated_expr}) STORED
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
            await self._warn_on_text_search_config_drift(conn)

    async def _warn_on_text_search_config_drift(self, conn: Any) -> None:
        """Say so when the tables on disk were lexized differently than we query.

        `CREATE TABLE IF NOT EXISTS` does not alter a table that already exists,
        so pointing a driver with `text_search_config='spanish'` at a graph whose
        `tsv` columns were generated with `simple` leaves the two disagreeing —
        and a tsquery that disagrees with its tsvector does not error, it just
        stops matching. That is the same silence BUG-98 cost a benchmark wave to
        find, so it is detected rather than documented.

        Detected, NOT repaired: rewriting a generated column on a live graph is a
        migration (the whole table is rewritten and every row re-lexized), not
        something a start-up path may do behind the operator's back. The cure is
        `build_indices_and_constraints(delete_existing=True)` plus a re-ingest.
        """
        for table in (self._node_tbl, self._edge_tbl):
            try:
                expr = await conn.fetchval(
                    'SELECT pg_get_expr(ad.adbin, ad.adrelid) '
                    'FROM pg_attribute a '
                    'JOIN pg_attrdef ad ON ad.adrelid = a.attrelid AND ad.adnum = a.attnum '
                    'WHERE a.attrelid = $1::regclass AND a.attname = $2',
                    table,
                    'tsv',
                )
            except Exception as e:
                # The check is advisory; never let it fail a start-up that would
                # otherwise have worked.
                logger.debug(f'Could not read the tsv expression for {table}: {e}')
                continue
            stored = stored_tsv_config(expr)
            if not text_search_configs_agree(self.text_search_config, stored):
                logger.warning(
                    f'{table}.tsv was generated with text-search configuration {stored!r} but '
                    f'this driver queries with {self.text_search_config!r}. Fulltext search will '
                    f'silently return nothing for stemmed terms. Rebuild the shadow tables '
                    f'(build_indices_and_constraints(delete_existing=True) + re-ingest) or set '
                    f'text_search_config={stored!r}.'
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
