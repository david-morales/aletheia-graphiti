"""AGE flavour — Apache AGE (openCypher over PostgreSQL).

Dialect data (reference text + error patterns) is carried over from the aletheia GraphRAG
scaffold's age.py; the method shapes match the fork's CypherError protocol. AGE has no
ro_query, so execute_graph_query (inherited from BaseFlavour) relies on the pipeline's
whitelist as the read-only guard.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from flavours.base import BaseFlavour, RESERVED_KEYS, _uuid_list_literal  # noqa: F401
from utils.cypher import (
    _BACKTICK_IDENT_RE,
    _NON_CODE_SPAN_RE,
    CypherError,
    _apply_to_code_spans,
    _strip_non_code_spans,
)

# AGE Cypher dialect reference (single source). Every statement here was verified against a
# live Apache AGE graph. get_schema surfaces this via flavour.dialect_reference (canonical)
# plus a cypher_reference alias; the run_cypher description and the server instructions
# surface the short form (dialect_summary).
_AGE_DIALECT = (
    "## Cypher Quick Reference (Apache AGE)\n\n"
    "Backend: Apache AGE (openCypher over PostgreSQL) — NOT FalkorDB and NOT Neo4j. "
    "APOC is unavailable; string helpers such as indexOf/split/replace are absent or behave "
    "differently. Always include a LIMIT.\n\n"
    "### Labels\n"
    "- A vertex carries exactly ONE stored label (its ontology leaf). `label(n)` returns that "
    "leaf as a string: `RETURN label(n) AS tipo`.\n"
    "- The FULL ontology hierarchy is kept in a list property `n.labels`, e.g. "
    "`['Entity', 'Event', 'ParteDeIntervencion']`. Use it to match a supertype: "
    "`WHERE 'Actor' IN n.labels`.\n"
    "- `labels(n)` returns only `[leafLabel]` here — it is NOT the hierarchy.\n"
    "- Do not scope by `:Entity`: only a handful of vertices carry it. Match `(n)` and filter.\n\n"
    "### Parameters\n"
    "- There are NO `$param` bindings. `$name` reaches the parser unbound and fails with "
    "'parameters argument is missing from cypher() function call'.\n"
    "- Inline literals instead: `WHERE n.name = 'OMAR MOHAMED'`, `WHERE n.edad > 30`.\n"
    "- A `$$` (or a stray `$`) in code position — i.e. outside a string literal or comment — "
    "is rejected before execution, because the query is spliced into a dollar-quoted SQL "
    "literal. Inside a quoted string a `$$` is passed through to the driver, which guards the "
    "splice itself; do not rely on this tool to catch it there.\n\n"
    "### Aliases\n"
    "- These words are RESERVED and cannot be used as an alias: count, exists, all, any, none, "
    "single, distinct, end, contains, starts, ends, null, true, false, coalesce. "
    "Use `count(n) AS cnt ... ORDER BY cnt DESC`.\n"
    "- Aliases are folded to lower case. Use snake_case (`AS tipo_delito`), never camelCase "
    "(`AS TipoDelito` comes back as `tipodelito` and the lookup fails).\n"
    "- Every RETURN alias must be UNIQUE — they become SQL column names.\n\n"
    "### Label Tests\n"
    "- A bare label test in an expression is a syntax error: `WHERE n:Persona` fails.\n"
    "- Parenthesise it: `WHERE (n:Persona) OR (n:Ubicacion)`.\n"
    "- Or compare the leaf: `WHERE label(n) IN ['Persona', 'Ubicacion']`.\n"
    "- A disjunction inside the pattern (`MATCH (n:A OR n:B)`) is a syntax error.\n\n"
    "### Relationship Types\n"
    "- `[:A|B|C]` disjunction is NOT supported. Match a generic edge and filter: "
    "`MATCH (a)-[r]->(b) WHERE type(r) IN ['A','B','C']`.\n\n"
    "### Attributes & agtype\n"
    "- Descriptive fields live in a queryable MAP property `attributes`. Address members "
    "directly: `n.attributes.<field>`. NEVER string-parse it. See `attribute_keys` in "
    "get_schema, or `RETURN keys(n.attributes)`.\n"
    "- Dates are ISO 'YYYY-MM-DD' strings: `toInteger(substring(n.attributes.<field>, 0, 4))` "
    "for the year; derive ranges with a CASE.\n"
    "- The not-equals operator is `<>`; `!=` does not exist.\n\n"
    "### Projections\n"
    "- Alias EVERY returned expression. An unaliased projection loses its name and comes back "
    "as `col0`.\n"
    "- Prefer returning scalars (`n.name AS name`) over whole nodes when you only need fields.\n\n"
    "### Known Limitations\n"
    "- No APOC, no `$params`, no `[:A|B|C]`, no bare label tests, no `!=`\n"
    "- NEVER name a variable `id` — it collides with AGE's built-in id()/graphid and fails "
    "with 'column notation .id applied to type graphid'; use `ident` or `x`\n"
    "- `PROFILE` is a syntax error and `EXPLAIN` returns a query plan, not rows (both are "
    "stripped automatically)\n"
    "- LIMIT auto-injected (200) if not specified"
)

# `id` as a node/relationship PATTERN variable — `(id` / `[id` not followed by `.`(property)
# or `(`(function). Brace maps `{id:` are intentionally NOT matched (map key, not a variable).
_ID_PATTERN_VAR_RE = re.compile(r"[\(\[]\s*id\b(?!\s*[.(])", re.IGNORECASE)
# `id` as an alias — `AS id` (an aliased column is a variable named id).
_ID_ALIAS_RE = re.compile(r"\bAS\s+id\b", re.IGNORECASE)

# Relationship-type disjunction inside a [...] pattern: [:A|B|C] or [r:A|B|C]. Used for
# DETECTION only (check_dialect rejects it). A reliable structural REWRITE of arbitrary
# openCypher is not feasible with pattern matching — a wrong rewrite silently returns wrong
# data — so we reject-with-hint and let the agent write the correct `WHERE type(r) IN [...]`
# form (which it does in one step from the hint / dialect_reference).
_REL_DISJUNCTION_RE = re.compile(
    r"\[\s*(?P<var>[A-Za-z_]\w*)?\s*:\s*"
    r"(?P<types>`?\w[\w`]*`?(?:\s*\|\s*`?\w[\w`]*`?)+)\s*\]"
)


def check_age_dialect(query: str) -> CypherError | None:
    """Reject track — return a CypherError for AGE-incompatible constructs, else None.

    Every check runs on the code-only view (`_strip_non_code_spans`, the same primitive the
    FalkorDB flavour and the security whitelist use), so a construct that appears only inside
    a string literal or a comment can never trigger a rejection. Checks are ordered by
    severity; the first match wins.
    """
    code_only = _strip_non_code_spans(query)

    # (0) `$$` / a bare `$` in code position. The driver splices the query into a
    # dollar-quoted SQL literal, so `$$` would break out of it. Reject, never rewrite.
    m = _DOLLAR_QUOTE_RE.search(code_only)
    if m:
        return CypherError(
            stage="age_dialect",
            reason="dollar_quote_unsupported",
            found=code_only[m.start():m.start() + 2],
            explanation="A `$` that is not part of a `$name` parameter is not accepted. "
            "Apache AGE queries are submitted inside a dollar-quoted SQL literal, so a `$$` "
            "or a stray `$` corrupts the statement.",
            suggestion="Remove the `$` and inline the value you meant as a literal "
            "(single-quoted for strings, bare for numbers). If the `$` is part of text you "
            "want to match, put it inside a quoted string literal.",
            doc_hint="Apache AGE: `$` is only valid as `$name`, and even those cannot be bound "
            "here — inline literals instead.",
        )

    # (1) openCypher `$name` parameters. AGE's cypher() cannot bind them and run_cypher
    # supplies no parameter map, so they would reach the parser unbound.
    params = _find_parameters(query)
    if params:
        named = ", ".join(params)
        return CypherError(
            stage="age_dialect",
            reason="unbound_parameter",
            found=params[0],
            explanation=f"This query uses openCypher parameters ({named}). Apache AGE's "
            "cypher() function cannot bind them and this tool supplies no parameter map, so "
            "they would fail at execution with 'parameters argument is missing from cypher() "
            "function call'.",
            suggestion=f"Inline the literal values directly into the query instead of "
            f"{named} — e.g. `WHERE n.name = 'OMAR MOHAMED'` rather than "
            "`WHERE n.name = $name`. Quote strings with single quotes; leave numbers bare.",
            doc_hint="Apache AGE: no `$param` binding; inline literals into the Cypher text.",
        )

    # (2) `id` used as a VARIABLE — a pattern variable ((id / [id) or an alias (AS id).
    # Precise on purpose: property access (n.id), the id() function and map keys ({id: ...})
    # are allowed; ambiguous bare references fall through to the execution-error net.
    if _ID_PATTERN_VAR_RE.search(code_only) or _ID_ALIAS_RE.search(code_only):
        return CypherError(
            stage="age_dialect",
            reason="reserved_id_variable",
            found="id",
            explanation="A variable named `id` collides with AGE's built-in id()/graphid "
            "and fails at execution ('column notation .id applied to type graphid').",
            suggestion="Rename the variable (e.g. `ident`, `x`) and retry.",
            doc_hint="AGE reserves id()/graphid; never name a variable `id`.",
        )

    # (3) relationship-type disjunction [:A|B|C] — unsupported by AGE.
    if _REL_DISJUNCTION_RE.search(code_only):
        return CypherError(
            stage="age_dialect",
            reason="reltype_disjunction_unsupported",
            found="[:A|B|C]",
            explanation="AGE does not support relationship-type disjunction like [:A|B|C].",
            suggestion="Match a generic edge and filter by type: "
            "MATCH (a)-[r]->(b) WHERE type(r) IN ['A','B','C'].",
            doc_hint="AGE has no [:A|B|C]; use WHERE type(r) IN [...].",
        )
    return None


# ---------------------------------------------------------------------------
# Execution-error classification (Layer 3)
# ---------------------------------------------------------------------------

@dataclass
class ExecutionErrorPattern:
    """Maps an AGE/Postgres error string to an actionable CypherError.

    Walked in order by :func:`classify_age_execution_error`; first match wins.

    ``matcher`` is tested against the error MESSAGE. ``query_check``, when present, is tested
    against the SUBMITTED QUERY and must also match — Postgres messages are far less specific
    than FalkorDB's, so several AGE patterns are only distinguishable by corroborating what the
    agent actually wrote. A pattern carrying a ``query_check`` never fires when no query was
    supplied. ``suggestion_fn(message, query)`` overrides the static ``suggestion`` when the
    hint must name concrete tokens from the query.

    ``query_view`` is the masking applied to the query before ``query_check`` runs. The default
    (``_strip_non_code_spans``) hides string literals and comments; a pattern whose keyword can
    also appear as a quoted IDENTIFIER — ``n.`credit union``` — sets ``_code_view_no_idents``
    instead. It is per-pattern because the masking that removes one pattern's false positives
    removes another's true ones (``_LABEL_TEST_RE`` and ``_RESERVED_ALIAS_RE`` both read
    backticked identifiers on purpose).
    """

    name: str
    matcher: re.Pattern
    suggestion: str
    doc_hint: str
    example_fix: str | None = None
    query_check: re.Pattern | None = None
    suggestion_fn: Callable[[str, str], str] | None = None
    query_view: Callable[[str], str] = _strip_non_code_spans


# Postgres reports the offending token as `at or near "X"` but never echoes the query, so the
# AGE classifier reconstructs FalkorDB's `errCtx` itself: find X in the submitted query and
# quote the surrounding fragment. That echo is what let the FalkorDB arm of the v0.31.0 bench
# self-correct; without it the agent has nothing to diff against.
_AT_OR_NEAR_RE = re.compile(r'at or near "([^"]+)"')

_WORD_CHAR_RE = re.compile(r"\w")


def _token_occurrences(flat: str, token: str) -> list[re.Match]:
    """Occurrences of ``token`` in ``flat`` that are in CODE position and stand alone.

    Two filters, both needed to land the window on the real failing site:

    * **Word boundaries** — Postgres names the offending token (`DESC`), not its offset, and a
      naive substring search happily matches the `desc` inside `n.attributes.descripcion`.
      The guards are applied only on the sides where the token is word-like, so punctuation
      tokens (`|`, `:`) still match.
    * **Code position** — an occurrence inside a string literal or a comment is data, not the
      failing site. Spans come from the same regex the shared pipeline masks with.
    """
    if not token:
        return []
    prefix_guard = r"(?<!\w)" if _WORD_CHAR_RE.match(token[0]) else ""
    suffix_guard = r"(?!\w)" if _WORD_CHAR_RE.match(token[-1]) else ""
    pattern = re.compile(prefix_guard + re.escape(token) + suffix_guard, re.IGNORECASE)
    non_code = [(s.start(), s.end()) for s in _NON_CODE_SPAN_RE.finditer(flat)]
    return [
        hit
        for hit in pattern.finditer(flat)
        if not any(start <= hit.start() < end for start, end in non_code)
    ]


def _synthesize_errctx(message: str, query: str | None, width: int = 40) -> str:
    """Return a short fragment of the query around the failing token, or '' if unavailable.

    When the token occurs several times in code position the LAST one wins: AGE reports the
    first construct its parser could not accept, and for the alias family that is the trailing
    `ORDER BY <alias> DESC`, not an earlier innocent mention.
    """
    if not query:
        return ""
    flat = " ".join(query.split())
    m = _AT_OR_NEAR_RE.search(message or "")
    if m:
        hits = _token_occurrences(flat, m.group(1))
        if hits:
            hit = hits[-1]
            start = max(0, hit.start() - width)
            end = min(len(flat), hit.end() + width)
            prefix = "..." if start > 0 else ""
            suffix = "..." if end < len(flat) else ""
            return f"{prefix}{flat[start:end]}{suffix}"
    return flat[:120] + ("..." if len(flat) > 120 else "")


# openCypher `$name` parameters. AGE's cypher() SQL function takes no openCypher parameters,
# and run_cypher binds none, so every $name reaches the parser raw. Matched on code spans only
# so `'costs $50'` inside a string literal is not read as a parameter.
_PARAM_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")

# A `$` that is NOT the start of a `$name` parameter — `$$`, `$1`, a bare trailing `$`.
# The AGE driver wraps the query as `cypher('g', $$ <query> $$)`, so a `$$` inside the query
# terminates the dollar-quoted literal and the remainder is parsed as raw SQL. Rejecting it
# here is the UX layer; the structural driver guard is a separate change.
_DOLLAR_QUOTE_RE = re.compile(r"\$(?![A-Za-z_])")


def _find_parameters(query: str) -> list[str]:
    """Ordered, de-duplicated `$name` parameters used in code position."""
    seen: list[str] = []
    for m in _PARAM_RE.finditer(_strip_non_code_spans(query or "")):
        name = m.group(0)
        if name not in seen:
            seen.append(name)
    return seen


def _suggest_inline_parameters(_message: str, query: str) -> str:
    params = _find_parameters(query)
    named = ", ".join(params) if params else "the `$name` tokens"
    return (
        f"This query uses openCypher parameters ({named}) but Apache AGE's cypher() function "
        "cannot bind them and this tool never supplies a parameter map — so they reach the "
        "parser unbound. Inline the literal values directly into the query "
        "(e.g. `WHERE n.name = 'OMAR MOHAMED'` instead of `WHERE n.name = $name`), quoting "
        "strings with single quotes and leaving numbers bare."
    )


# A label test written where AGE's grammar does not accept one: an identifier followed by `:`
# that is NOT in pattern position. `(n:Persona)` / `[r:KNOWS]` (preceded by `(` or `[`) and map
# keys `{id: 5}` / `, k: 1` (preceded by `{` or `,`) are excluded, so what remains is the
# `WHERE n:Persona` / `NOT n:Persona` / `CASE WHEN n:Persona` / `MATCH (n:A OR n:B)` family.
_LABEL_TEST_RE = re.compile(r"(?<![(\[{,\w.`])\b[A-Za-z_]\w*\s*:\s*`?[A-Za-z_]\w*")


# Words AGE's grammar refuses as a bare identifier. Swept live on the :5433 bed with
# `RETURN label(n) AS t, count(n) AS <w> ORDER BY <w> DESC`: every word below errored,
# while filter/extract/type/label/keys/size/head/last/min/sum were accepted.
_AGE_RESERVED_ALIASES: tuple[str, ...] = (
    "all", "any", "coalesce", "contains", "count", "distinct", "end", "ends",
    "exists", "false", "none", "null", "single", "starts", "true",
)

_RESERVED_ALIAS_RE = re.compile(
    r"\bAS\s+(" + "|".join(_AGE_RESERVED_ALIASES) + r")\b", re.IGNORECASE
)


def _find_reserved_aliases(query: str) -> list[str]:
    """Ordered, de-duplicated reserved words used as `AS <alias>` in code position."""
    seen: list[str] = []
    for m in _RESERVED_ALIAS_RE.finditer(_strip_non_code_spans(query or "")):
        alias = m.group(1)
        if alias not in seen:
            seen.append(alias)
    return seen


def _suggest_rename_reserved_alias(_message: str, query: str) -> str:
    aliases = _find_reserved_aliases(query)
    named = ", ".join(f"`{a}`" for a in aliases) if aliases else "the alias"
    return (
        f"Apache AGE's grammar reserves {named}, so a bare reference to it (in ORDER BY, in a "
        "later RETURN, or in a WITH chain) is a syntax error even though the `AS` itself "
        "parses. Rename the alias to a non-reserved word and update every reference — e.g. "
        "`count(n) AS cnt ... ORDER BY cnt DESC`. Reserved here: "
        + ", ".join(_AGE_RESERVED_ALIASES)
        + "."
    )


# An `AS <alias>` whose name carries an uppercase letter. The AGE driver builds an UNQUOTED
# SQL column-definition list from these aliases; PostgreSQL folds unquoted identifiers to
# lower case, so `AS TipoDelito` yields a result column named `tipodelito` while the driver
# still looks up `TipoDelito` -> KeyError. Verified live on the :5433 bed.
_MIXED_CASE_ALIAS_RE = re.compile(r"\bAS\s+([A-Za-z_]*[A-Z]\w*)\b")


def _find_mixed_case_aliases(query: str) -> list[str]:
    seen: list[str] = []
    for m in _MIXED_CASE_ALIAS_RE.finditer(_strip_non_code_spans(query or "")):
        alias = m.group(1)
        if alias not in seen:
            seen.append(alias)
    return seen


# A UNION between query branches. Used to tell the two duplicate-column cases apart: a
# UNION's branches MUST project the same aliases (that is what makes the UNION well-formed),
# so "rename them" is an impossible repair there — it is only the right answer when one
# branch repeats an alias on its own. Tested against the code-only view of the query, so a
# `union` inside a string literal never fires it.
_UNION_RE = re.compile(r"\bunion\b", re.IGNORECASE)


def _code_view_no_idents(query: str) -> str:
    """``_strip_non_code_spans`` plus backtick-quoted identifiers.

    The base helper masks string literals and comments but leaves backticked identifiers in
    place, so ``n.`credit union``` still reads as a UNION to a keyword gate. Same placeholder
    convention as ``_extract_keyword_tokens``.
    """
    return _BACKTICK_IDENT_RE.sub(" _BT_ ", _strip_non_code_spans(query))


def _suggest_lowercase_alias(_message: str, query: str) -> str:
    aliases = _find_mixed_case_aliases(query)
    named = ", ".join(f"`{a}`" for a in aliases) if aliases else "the aliases"
    return (
        f"Apache AGE returns results through an unquoted SQL column list, and PostgreSQL folds "
        f"unquoted identifiers to lower case — so a mixed-case alias ({named}) is requested "
        "under one name and delivered under another, and the lookup fails. Use all-lowercase "
        "snake_case aliases: `AS tipo_delito`, `AS nombre_completo`."
    )


# Order matters: specific message fingerprints first, then the patterns that lean on
# query_check to disambiguate a generic `syntax error at or near "..."`.
_AGE_EXECUTION_ERROR_PATTERNS: list[ExecutionErrorPattern] = [
    ExecutionErrorPattern(
        name="unbound_parameter",
        matcher=re.compile(
            r"parameters argument is missing from cypher\(\) function call", re.IGNORECASE
        ),
        suggestion="This query uses openCypher `$name` parameters. Apache AGE cannot bind "
        "them — inline the literal values directly into the query instead.",
        doc_hint="Apache AGE: no `$param` binding; inline literals into the Cypher text.",
        example_fix="MATCH (n) WHERE n.name = 'OMAR MOHAMED' RETURN n.name LIMIT 25",
        suggestion_fn=_suggest_inline_parameters,
    ),
    ExecutionErrorPattern(
        name="reserved_id_variable",
        # Anchored on the FULL message, not the bare word `graphid`: several unrelated AGE
        # errors mention graphid (notably the agtype cast below), and answering those with
        # "rename your variable" names a variable the query never bound.
        matcher=re.compile(r"column notation \.id applied to type graphid", re.IGNORECASE),
        suggestion="A variable named `id` collides with AGE's built-in id()/graphid. "
        "Rename the variable (e.g. `ident`, `x`) and retry.",
        doc_hint="AGE reserves id()/graphid; never name a variable `id`.",
        example_fix="MATCH (ident) RETURN ident.name LIMIT 25",
    ),
    ExecutionErrorPattern(
        name="agtype_cast_error",
        # Live: `MATCH (n) WHERE id(n) = 'a' RETURN n` -> cannot cast agtype string to graphid.
        matcher=re.compile(r"cannot cast agtype (\w+) to type (\w+)", re.IGNORECASE),
        suggestion="A value in this query has the wrong agtype for the operation, and Apache "
        "AGE will not cast between agtype scalars implicitly. Compare like with like: `id(n)` "
        "yields a graphid, so compare it against an integer id (or another `id(...)` call), "
        "never against a quoted string. Quote strings only where a string is expected.",
        doc_hint="Apache AGE: agtype scalars are not auto-coerced; id() returns a graphid, "
        "not a string.",
        example_fix="MATCH (n) WHERE n.uuid = 'a' RETURN n.name AS name LIMIT 25",
    ),
    # Must precede duplicate_return_column: same message, opposite repair.
    ExecutionErrorPattern(
        name="union_column_collision",
        matcher=re.compile(r'column name "([^"]+)" specified more than once', re.IGNORECASE),
        suggestion="This query is a UNION, and its branches project the same aliases — as they "
        "must. Apache AGE builds ONE SQL column-definition list for the whole statement, so the "
        "branch columns land in it more than once and PostgreSQL rejects the repeated name. "
        "Renaming the aliases is NOT the fix: a UNION whose branches disagree on their columns "
        "is invalid Cypher. Drop the UNION instead — run each branch as a separate query, or fold "
        "the branches into a single MATCH over a generic relationship and filter the type: "
        "`MATCH (a)-[r]->(b)-[:LINKED_TO]->(c) WHERE type(r) IN ['REL_A','REL_B'] "
        "RETURN b.name AS item, c.name AS place`. (If instead ONE branch repeats an alias on "
        "its own, rename that one.)",
        doc_hint="Apache AGE: the branches of a UNION share a single SQL column-definition "
        "list — prefer separate queries, or one MATCH with a `type(r) IN [...]` filter.",
        example_fix="MATCH (a)-[r]->(b)-[:LINKED_TO]->(c) WHERE type(r) IN ['REL_A','REL_B'] "
        "RETURN b.name AS item, c.name AS place LIMIT 25",
        query_check=_UNION_RE,
        # `union` is also a plausible identifier, so the gate must not see backticked spans.
        query_view=_code_view_no_idents,
    ),
    ExecutionErrorPattern(
        name="duplicate_return_column",
        matcher=re.compile(r'column name "([^"]+)" specified more than once', re.IGNORECASE),
        suggestion="Two RETURN columns share the same alias. Apache AGE turns the RETURN "
        "aliases into SQL column names, which must be unique. Give every projected column a "
        "distinct alias (e.g. `p.uuid AS parte_uuid, r.uuid AS rol_uuid`).",
        doc_hint="Apache AGE: RETURN aliases become SQL column names and must be unique.",
        example_fix="MATCH (p)-[]->(r) RETURN p.uuid AS parte_uuid, r.uuid AS rol_uuid LIMIT 25",
    ),
    ExecutionErrorPattern(
        name="projection_column_mismatch",
        matcher=re.compile(
            r"return row and column definition list do not match", re.IGNORECASE
        ),
        suggestion="Apache AGE derives the result columns from the RETURN clause, and this "
        "projection could not be reconciled. Give EVERY returned expression an explicit, "
        "unique, lowercase alias — `RETURN n.name AS name, count(n) AS cnt` — and avoid "
        "returning whole nodes alongside expressions in the same clause.",
        doc_hint="Apache AGE: alias every RETURN item explicitly; unaliased projections lose "
        "their name (they come back as `col0`).",
        example_fix="MATCH (n) RETURN n.name AS name, n.uuid AS uuid LIMIT 25",
    ),
    ExecutionErrorPattern(
        name="mixed_case_alias",
        # str(KeyError('X')) is "'X'" — the entire message is a quoted identifier.
        matcher=re.compile(r"^'[A-Za-z_]*[A-Z]\w*'$"),
        suggestion="A mixed-case RETURN alias was lost to PostgreSQL identifier folding. "
        "Use all-lowercase snake_case aliases.",
        doc_hint="Apache AGE: RETURN aliases are folded to lower case — use snake_case.",
        example_fix="MATCH (n) RETURN label(n) AS tipo_delito, count(n) AS cnt LIMIT 25",
        query_check=_MIXED_CASE_ALIAS_RE,
        suggestion_fn=_suggest_lowercase_alias,
    ),
    ExecutionErrorPattern(
        name="internal_resource_owner",
        # PostgreSQL resource-owner bookkeeping error surfaced through AGE. Seen once in the
        # v0.31.0 bench run; not reproducible on demand, so the matcher keys only on the two
        # stable fragments of the standard PG message.
        matcher=re.compile(r"tupdesc reference.*is not owned by resource owner", re.IGNORECASE | re.DOTALL),
        suggestion="This is an internal PostgreSQL/AGE bookkeeping error, not a problem with "
        "your Cypher. Retry the same query once. If it recurs, simplify the query (fewer "
        "returned columns, a smaller LIMIT) or split it into two calls.",
        doc_hint="Apache AGE: transient internal error — retry, then simplify.",
    ),
    ExecutionErrorPattern(
        name="not_equals_operator",
        matcher=re.compile(r"operator does not exist: agtype != agtype", re.IGNORECASE),
        suggestion="Apache AGE has no `!=` operator. Use the openCypher canonical form `<>`. "
        "The stage-2b auto-fix normally rewrites this before execution; seeing it here means "
        "the `!=` sat somewhere the auto-fixer skips (inside a string-adjacent construct).",
        doc_hint="Apache AGE openCypher: `<>` is the not-equals operator; `!=` does not exist.",
        example_fix="MATCH (n) WHERE n.name <> 'zzz' RETURN n.name LIMIT 25",
    ),
    ExecutionErrorPattern(
        name="reltype_disjunction_unsupported",
        matcher=re.compile(r'at or near "\|"'),
        suggestion="AGE does not support relationship-type disjunction like [:A|B|C]. "
        "Match a generic edge and filter: MATCH (a)-[r]->(b) WHERE type(r) IN ['A','B','C'].",
        doc_hint="AGE has no [:A|B|C]; use WHERE type(r) IN [...].",
        example_fix="MATCH (a)-[r]->(b) WHERE type(r) IN ['DETIENE','INVESTIGA'] RETURN b LIMIT 25",
    ),
    ExecutionErrorPattern(
        name="boolean_label_test",
        matcher=re.compile(r'at or near "(?::|OR)"', re.IGNORECASE),
        suggestion="Apache AGE's parser does not accept a bare label test (`n:Label`) in an "
        "expression, and does not accept a disjunction inside a pattern (`MATCH (n:A OR n:B)`). "
        "Parenthesise each test — `WHERE (n:Persona) OR (n:Ubicacion)` — or compare the leaf "
        "label directly: `WHERE label(n) IN ['Persona', 'Ubicacion']`. To test the full "
        "ontology hierarchy use the stored list instead: `WHERE 'Actor' IN n.labels`.",
        doc_hint="Apache AGE: a label test outside a MATCH pattern must be parenthesised "
        "`(n:Label)`, or written as `label(n) = 'Label'` / `'Label' IN n.labels`.",
        example_fix="MATCH (n) WHERE (n:Persona) OR (n:Ubicacion) RETURN n.name LIMIT 25",
        query_check=_LABEL_TEST_RE,
    ),
    ExecutionErrorPattern(
        name="reserved_alias",
        matcher=re.compile(r"syntax error at or near", re.IGNORECASE),
        suggestion="An alias in this query is a word Apache AGE reserves. Rename it "
        "(e.g. `count(n) AS cnt ... ORDER BY cnt DESC`) and update every reference.",
        doc_hint="Apache AGE reserves count/exists/all/any/none/single/distinct/end/contains/"
        "starts/ends/null/true/false/coalesce as bare identifiers — never use them as aliases.",
        example_fix="MATCH (n) RETURN label(n) AS type, count(n) AS cnt ORDER BY cnt DESC LIMIT 25",
        query_check=_RESERVED_ALIAS_RE,
        suggestion_fn=_suggest_rename_reserved_alias,
    ),
]


def classify_age_execution_error(msg: str, query: str | None = None) -> CypherError:
    """Map an AGE/Postgres execution-error message to an actionable CypherError."""
    for pattern in _AGE_EXECUTION_ERROR_PATTERNS:
        if not pattern.matcher.search(msg or ""):
            continue
        if pattern.query_check is not None and (
            not query or not pattern.query_check.search(pattern.query_view(query))
        ):
            continue
        explanation = f"Apache AGE returned an error: {msg}"
        errctx = _synthesize_errctx(msg or "", query)
        if errctx:
            explanation += f"\nerrCtx: {errctx}"
        if pattern.example_fix:
            explanation += f"\nCorrected example: {pattern.example_fix}"
        suggestion = pattern.suggestion
        if pattern.suggestion_fn is not None and query:
            suggestion = pattern.suggestion_fn(msg or "", query)
        return CypherError(
            stage="execution",
            reason=pattern.name,
            found=msg,
            explanation=explanation,
            suggestion=suggestion,
            doc_hint=pattern.doc_hint,
        )

    explanation = f"Apache AGE returned an error: {msg}"
    errctx = _synthesize_errctx(msg or "", query)
    if errctx:
        explanation += f"\nerrCtx: {errctx}"
    return CypherError(
        stage="execution",
        reason="query_failed",
        found=msg,
        explanation=explanation,
        suggestion="Check your Cypher against this graph's dialect. Use get_schema to verify "
        "label and property names and to read `dialect_reference`.",
        doc_hint="Apache AGE openCypher — see get_schema `dialect_reference`.",
    )


# ---------------------------------------------------------------------------
# Stage 2b: AGE auto-fixes. Bench-observed patterns only — each verified against the live
# :5433 bed before inclusion. Every fix appends a human-readable note so the agent sees the
# transform in the envelope's `auto_fixes`.
# ---------------------------------------------------------------------------

# AGE has no `!=` (live: `operator does not exist: agtype != agtype`); `<>` works.
_NEQ_RE = re.compile(r"!=")
# `PROFILE` is a syntax error; `EXPLAIN` is worse — it silently escalates to a SQL-level
# EXPLAIN and returns a QUERY PLAN column the driver cannot read.
_PROFILE_EXPLAIN_RE = re.compile(r"^\s*(PROFILE|EXPLAIN)\s+", re.IGNORECASE)
# `AS count` parses, but a bare REFERENCE to that alias is a syntax error. Rename the alias
# and its references. Excluded by the lookbehind/lookahead: the aggregate call `count(`, the
# property `n.count`, a map key `{count:`, and — critically — a label or relationship type
# written `:count`, where renaming would silently change which pattern is matched.
_AS_COUNT_RE = re.compile(r"\bAS\s+count\b", re.IGNORECASE)
_BARE_COUNT_RE = re.compile(r"(?<![.\w`:])count\b(?!\s*[:(])", re.IGNORECASE)


def _has_bare_count_reference(code_only: str) -> bool:
    """True when `count` is referenced somewhere OTHER than its own `AS count` alias.

    `RETURN count(n) AS count` is valid AGE — the alias is never dereferenced, so nothing
    fails and rewriting it is churn. What breaks is a later bare mention (`ORDER BY count`,
    `WHERE count > 5`, `RETURN count`), so the alias occurrences are blanked out first and the
    rename only fires if a reference survives.
    """
    return bool(_BARE_COUNT_RE.search(_AS_COUNT_RE.sub(" ", code_only)))


def fix_age_dialect(query: str) -> tuple[str, list[str]]:
    """Auto-fix track — lossless rewrites for AGE compatibility.

    Returns the fixed query and human-readable descriptions of each fix applied. Every step
    runs through :func:`_apply_to_code_spans`, so string literals and comments are never
    rewritten and the ``fixes`` list never reports a change that did not happen.
    """
    fixes: list[str] = []

    # 1. `!=` -> `<>`
    new_query = _apply_to_code_spans(query, lambda s: _NEQ_RE.sub("<>", s))
    if new_query != query:
        query = new_query
        fixes.append("Replaced != with <> (Apache AGE has no != operator)")

    # 2. Strip a leading PROFILE/EXPLAIN. Applied to the WHOLE query, not per code span:
    # _apply_to_code_spans restarts each segment, so a `^`-anchored pattern would re-anchor
    # after every string literal and delete a mid-query PROFILE the agent needs to see.
    # A leading keyword cannot sit inside a literal, so nothing is lost by skipping the mask.
    new_query = _PROFILE_EXPLAIN_RE.sub("", query, count=1)
    if new_query != query:
        query = new_query
        fixes.append(
            "Stripped PROFILE/EXPLAIN prefix (PROFILE is a syntax error on Apache AGE and "
            "EXPLAIN returns a query plan instead of rows)"
        )

    # 3. Rename the reserved alias `count`. Gated on BOTH an actual `AS count` (so a bare
    # `count` in an already-broken query is not turned into an unbound variable) AND a
    # surviving reference to it (so a working `RETURN count(n) AS count` is left alone).
    code_only = _strip_non_code_spans(query)
    if _AS_COUNT_RE.search(code_only) and _has_bare_count_reference(code_only):
        new_query = _apply_to_code_spans(query, lambda s: _BARE_COUNT_RE.sub("count_", s))
        if new_query != query:
            query = new_query
            fixes.append(
                "Renamed the reserved alias `count` to `count_` (Apache AGE reserves `count`; "
                "a bare `count` reference is a syntax error)"
            )

    return query, fixes


# AGE variants of the startup probes. Differences from the base text, each verified live:
#  * no `:Entity` scoping — only 6 of 2228 vertices carry it on a real AGE graph
#  * `n.labels` (the stored ordered list) instead of `labels(n)`, which returns only the leaf
#  * `n.labels IS NOT NULL` — Episodic vertices carry no labels list
#  * `$label IN n.labels` for the sample-name lookup
# The endpoint `labels IS NOT NULL` tests on the edge probes are the AGE equivalent of the
# base query's `:Entity` endpoint scoping: they exclude Graphiti's bookkeeping edges, which
# hang off label-less Episodic vertices. Without them the probe advertised `MENTIONS: 11` on
# the live bed as though it were a domain relationship.
_AGE_PROFILE_QUERIES: dict[str, str] = {
    "entity_types": (
        "MATCH (n) "
        "WHERE n.group_id = $group_id AND n.labels IS NOT NULL "
        "RETURN n.labels AS entity_type, count(n) AS cnt "
        "ORDER BY cnt DESC"
    ),
    "edge_types": (
        "MATCH (s)-[r]->(t) "
        "WHERE s.group_id = $group_id AND t.group_id = $group_id "
        "AND s.labels IS NOT NULL AND t.labels IS NOT NULL "
        "RETURN type(r) AS relationship_type, count(r) AS cnt "
        "ORDER BY cnt DESC"
    ),
    "sample_names": (
        "MATCH (n) "
        "WHERE $label IN n.labels AND n.group_id = $group_id "
        "RETURN n.name AS name "
        "LIMIT $limit"
    ),
    "time_range": (
        "MATCH (s)-[r]->(t) "
        "WHERE s.group_id = $group_id AND r.created_at IS NOT NULL "
        "AND s.labels IS NOT NULL AND t.labels IS NOT NULL "
        "RETURN min(r.created_at) AS earliest, max(r.created_at) AS latest"
    ),
}


# The ONLY properties an AGE vertex stores at the top level — Graphiti's own
# bookkeeping columns. Everything else a domain writes lands inside the nested
# `attributes` agtype map, which is what flatten_node_props / property_accessor
# reconcile. `attributes` itself is listed so it is never treated as a domain field.
_AGE_TOP_LEVEL_KEYS: frozenset[str] = RESERVED_KEYS | {"attributes"}


class AgeFlavour(BaseFlavour):
    """AGE flavour — reject-with-hint for AGE-unsupported constructs, nested-map attribute_keys.

    check_dialect rejects (with an actionable hint) a variable named ``id`` and relationship-type
    disjunction ``[:A|B|C]`` — both AGE-unsupported, and both safer to reject than to auto-rewrite.
    auto_fix applies three bench-observed rewrites (``!=`` → ``<>``, strip PROFILE/EXPLAIN,
    rename the reserved ``count`` alias); everything AGE cannot express is reject-with-hint.
    classify_execution_error is a post-hoc net for anything check_dialect misses.
    """

    name = "age"
    dialect_id = "age-opencypher"
    dialect_reference = _AGE_DIALECT
    dialect_summary = (
        "Apache AGE openCypher (NOT FalkorDB/Neo4j): no APOC; no $param bindings — inline "
        "literals; attributes are a queryable map (n.attributes.<field>); one stored label per "
        "vertex, `label(n)` for the leaf and `'X' IN n.labels` for the hierarchy; label tests "
        "must be parenthesised `(n:Label)`; no [:A|B|C] disjunction (use type(r) IN [...]); "
        "`<>` not `!=`; alias everything in lowercase snake_case and never alias to `count`; "
        "never name a variable `id`."
    )

    def check_dialect(self, query: str) -> CypherError | None:
        return check_age_dialect(query)

    def auto_fix(self, query: str) -> tuple[str, list[str]]:
        return fix_age_dialect(query)

    def classify_execution_error(self, message: str, query: str | None = None) -> CypherError:
        return classify_age_execution_error(message, query)

    def profile_queries(self) -> dict[str, str]:
        return dict(_AGE_PROFILE_QUERIES)

    def census_queries(self) -> dict[str, str]:
        # The stored n.labels list is the hierarchy; labels(n) would hide every
        # abstract supertype from the census. IS NOT NULL excludes Episodic.
        return {
            "label_counts": (
                "MATCH (n) WHERE n.labels IS NOT NULL "
                "RETURN n.labels AS lbls, count(n) AS cnt"
            ),
            # `source_labels`/`target_labels` are the whole HIERARCHY here, so a
            # consumer picking positionally from them lands on an abstract
            # supertype: not `(n:X)`-matchable, and it merges genuinely distinct
            # leaf patterns (Actor->X swallows both Persona->X and Empresa->X).
            # `label(n)` returns exactly the stored leaf (dialect_reference,
            # "Labels"), so announce it as the optional extra columns the shared
            # parsing prefers. Aliases are lowercase snake_case and unique, as AGE
            # requires; the leaf is functionally determined by the labels list, so
            # DISTINCT returns no more rows than before.
            "rel_patterns": (
                "MATCH (s)-[r:`{rel_type}`]->(t) "
                "WHERE s.labels IS NOT NULL AND t.labels IS NOT NULL "
                "RETURN DISTINCT s.labels AS source_labels, t.labels AS target_labels, "
                "label(s) AS source_leaf, label(t) AS target_leaf LIMIT 20"
            ),
        }

    def census_notes(self) -> list[str]:
        # The census counts the hierarchy, so node_labels now advertises abstract
        # supertypes that NO vertex carries as its stored label. `MATCH (n:Actor)`
        # returns zero rows for those — silently, which reads as "no Actors exist".
        # Announce the working form instead of leaving the agent to infer it.
        return [
            "IMPORTANT: `node_labels` counts the FULL ontology hierarchy on this backend, "
            "so it includes abstract supertypes (e.g. Actor) that no vertex carries as its "
            "stored label. `MATCH (n:Actor)` matches ZERO rows for those — it silently looks "
            "like the type is empty. Match a hierarchy label with "
            "`MATCH (n) WHERE 'Actor' IN n.labels`; `(n:X)` and `label(n)` only ever address "
            "the LEAF label.",
        ]

    def ontology_queries(self) -> dict[str, str]:
        # Two AGE storage facts, both proven on the live bench ontology graph:
        #
        # (a) NESTED ATTRIBUTES. Only uuid/name/summary/group_id/labels/created_at
        #     are top-level node properties; every descriptive ontology field lives
        #     in the queryable `attributes` agtype map (dialect_reference,
        #     "Attributes & agtype"). The base text read them top-level, so
        #     `n.ontology_type` was NULL on 30/30 classes while
        #     `n.attributes.ontology_type` was populated on 30/30 — the whole
        #     structured payload came back empty and NOTHING errored.
        #
        # (b) TYPED EDGES. AGE materializes the relationship NAME as the edge
        #     LABEL (graphiti_core age_graph_operations._edge_label), so this graph
        #     has ZERO `RELATES_TO` edges: `[r:RELATES_TO]` between OntologyClass
        #     nodes matched 0 rows while a generic match found 918 typed edges.
        #     So the edge pattern drops its TYPE CONSTRAINT — and only that. The
        #     relation NAME still comes from `r.name`, exactly as on the base
        #     flavour: both AGE edge write paths persist `name` top-level (the same
        #     props dict the `fact` read below relies on), and `r.name` is the ONE
        #     source that is never lossy. `type(r)` is NOT a general substitute for
        #     it here: `_edge_label`'s identifier test is ASCII-only, so a relation
        #     whose name carries an accent (INVOLUCRA_MUNICIÓN — the ontology loader
        #     does no accent folding) is stored under the `RELATES_TO` FALLBACK
        #     label. Reading `type(r)` would rename that relation to "RELATES_TO"
        #     and, because _edge_relationship_entry rebuilds the fact prefix from
        #     the name, leave its summary unstripped as well.
        #
        # Aliases are IDENTICAL to the base variant's (the parsing is shared), all
        # lowercase snake_case and unique, as AGE requires.
        nested = (
            "n.attributes.ontology_type AS ontology_type, "
            "n.attributes.inherits_from AS inherits_from, "
            "n.summary AS summary, "
            "n.attributes.alt_labels AS alt_labels, "
            "n.attributes.source_entity AS source_entity, "
            "n.attributes.target_entity AS target_entity, "
            "n.attributes.examples AS examples"
        )
        return {
            "class_context": (
                "MATCH (n:OntologyClass) "
                "RETURN n.uuid AS uuid, "
                "n.name AS name, "
                f"{nested}, "
                "n.attributes.properties AS properties, "
                "n.attributes.identity AS identity"
            ),
            "structure": (
                "MATCH (n:OntologyClass) "
                "RETURN n.name AS name, "
                f"{nested}"
            ),
            # A token-for-token mirror of the base text; the ONLY difference is
            # the dropped `:RELATES_TO` type constraint.
            # No SUBCLASS_OF filter: hierarchy edges are in the base row set too,
            # and _combine_relationship_entries drops them for both arms — that
            # shared filter keys on this same `name` value, which is another reason
            # it must be the stored name and not the label.
            # `name` and `fact` are top-level EDGE properties on AGE (edge_save
            # writes them into the edge props; only custom `attributes` nest).
            "relates": (
                "MATCH (a:OntologyClass)-[r]->(b:OntologyClass) "
                "RETURN a.name AS source, r.name AS name, r.fact AS fact, b.name AS target"
            ),
        }

    def subgraph_node_query(self) -> str:
        # No :Entity scope (only a handful of AGE vertices carry it); the stored
        # n.labels list is the hierarchy; `IS NOT NULL` excludes Episodic.
        #
        # ORDER BY n.created_at is a DIVERSIFIER, not a sort the caller asked for.
        # AGE keeps ONE TABLE PER LABEL, so an unordered `MATCH (n)` scan is
        # label-sequential: live on policia_partes_bench_v1 the first hundreds of
        # vertices were all ParteDeIntervencion, giving 1 label set and 0 edges at
        # limit 25 where falkor gave 10 sets and 26 edges.
        #
        # created_at is ingestion time, and a parte is written together with its
        # roles and actors — so ordering by it breaks the per-label grouping while
        # PRESERVING that cluster locality, which is what puts both endpoints of an
        # edge in the same sample. Measured live against the alternatives:
        #
        #     limit 25   label_sets   edges
        #     none            1         0
        #     n.uuid          9         0     <- max diversity, locality destroyed
        #     n.name          1         0
        #     n.created_at    6        23     <- falkor for comparison: 10 / 26
        #
        # A uuid4 sort is the better pseudo-random draw and the worse SAMPLE: 25
        # nodes scattered over ~1400 contain ~0.6 edges by expectation, so the view
        # renders unconnected dots. Base and FalkorDB keep storage order (already
        # interleaved) and stay byte-identical.
        return (
            "MATCH (n) WHERE n.labels IS NOT NULL "
            "RETURN n.uuid AS uuid, n.name AS name, n.labels AS labels, "
            "n.created_at AS created_at, n.summary AS summary, n.group_id AS group_id "
            "ORDER BY n.created_at "
            "LIMIT $limit"
        )

    def subgraph_edge_query(self, uuids: list[str]) -> str:
        # Unlabelled endpoints; the uuid IN-list already restricts both ends to
        # sampled entity nodes, which is what excludes MENTIONS/Episodic edges.
        lit = _uuid_list_literal(uuids)
        return (
            f"MATCH (s)-[r]->(t) "
            f"WHERE s.uuid IN {lit} AND t.uuid IN {lit} "
            "RETURN r.uuid AS uuid, type(r) AS name, r.fact AS fact, "
            "s.uuid AS source_node_uuid, t.uuid AS target_node_uuid, "
            "r.created_at AS created_at LIMIT $limit"
        )

    def node_sample_query(self) -> str:
        # `RETURN n` is an UNALIASED variable projection and AGE names those
        # `col0` (live: the driver returned header ['col0']), so the profiler's
        # `rec.get('n')` found nothing while rows still existed — which is exactly
        # how 19 AGE leaves came back `sampled: true` with `properties: {}`.
        # Same fix as get_schema's `RETURN DISTINCT key AS key`.
        return 'MATCH (n:`{label}`) RETURN n AS n LIMIT {limit}'

    def flatten_node_props(self, props: dict[str, Any]) -> dict[str, Any]:
        """Unwrap AGE's vertex envelope, then merge the nested `attributes` map up.

        Two layers of transport sit between the query and the domain fields, both
        confirmed against the live driver:

        1. The value is the whole VERTEX — `{'id', 'label', 'properties': {...}}` —
           not a flat property dict.
        2. Inside it, only Graphiti's own columns (`_AGE_TOP_LEVEL_KEYS`) are real
           properties; every DOMAIN field lives in the queryable `attributes` map.

        A reader that walks neither layer finds no domain fields at all — live,
        `properties: {}` on all 23 entity types, including the 19 the profiler
        flags `sampled: true`.

        Collision rule: the TOP LEVEL WINS. `name`/`summary` are Graphiti's own
        columns and stay authoritative, so a same-named field inside the map can
        never overwrite them. The `attributes` container itself is dropped — it is
        the transport, not a domain property.
        """
        # Unwrap only a real vertex envelope: a `properties` MAP alongside the
        # vertex identity. A domain field named `properties` cannot be mistaken
        # for it — domain fields live under `attributes`, one level further down.
        inner = props.get("properties")
        if isinstance(inner, dict) and ("id" in props or "label" in props):
            props = inner
        nested = props.get("attributes")
        if not isinstance(nested, dict):
            return props
        flat = dict(nested)
        flat.update({k: v for k, v in props.items() if k != "attributes"})
        return flat

    def property_accessor(self, prop: str) -> str:
        """Read `prop` through the nested map unless it is one of Graphiti's own
        top-level columns — the mirror of :meth:`flatten_node_props`.

        A full scan on the TOP-level path for a domain field is valid Cypher that
        matches nothing: it returns 0/0 and silently overwrites a real sample-based
        coverage with 0.0, which is worse than failing.
        """
        if prop in _AGE_TOP_LEVEL_KEYS:
            return f"n.`{prop}`"
        return f"n.attributes.`{prop}`"

    async def attribute_keys(self, driver: Any, label: str, sample: int = 50) -> list[str]:
        """Keys of the nested `attributes` agtype map, UNIONED across a small sample."""
        try:
            records, _, _ = await driver.execute_query(
                f"MATCH (n:`{label}`) WHERE n.attributes IS NOT NULL "
                f"RETURN keys(n.attributes) AS ks LIMIT {int(sample)}"
            )
        except Exception:  # noqa: BLE001 — a non-map label must not break schema
            return []
        keys: set[str] = set()
        for r in records:
            ks = r.get("ks")
            if ks:
                keys.update(ks)
        return sorted(keys)
