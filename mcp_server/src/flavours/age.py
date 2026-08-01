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

from flavours.base import BaseFlavour, RESERVED_KEYS  # noqa: F401
from utils.cypher import CypherError, _apply_to_code_spans, _strip_non_code_spans

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
    "- A `$$` anywhere in the query is rejected before execution.\n\n"
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
    """

    name: str
    matcher: re.Pattern
    suggestion: str
    doc_hint: str
    example_fix: str | None = None
    query_check: re.Pattern | None = None
    suggestion_fn: Callable[[str, str], str] | None = None


# Postgres reports the offending token as `at or near "X"` but never echoes the query, so the
# AGE classifier reconstructs FalkorDB's `errCtx` itself: find X in the submitted query and
# quote the surrounding fragment. That echo is what let the FalkorDB arm of the v0.31.0 bench
# self-correct; without it the agent has nothing to diff against.
_AT_OR_NEAR_RE = re.compile(r'at or near "([^"]+)"')


def _synthesize_errctx(message: str, query: str | None, width: int = 40) -> str:
    """Return a short fragment of ``query`` around the failing token, or '' if unavailable."""
    if not query:
        return ""
    flat = " ".join(query.split())
    m = _AT_OR_NEAR_RE.search(message or "")
    if m:
        token = m.group(1)
        idx = flat.lower().find(token.lower())
        if idx >= 0:
            start = max(0, idx - width)
            end = min(len(flat), idx + len(token) + width)
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
        matcher=re.compile(r"graphid", re.IGNORECASE),
        suggestion="A variable named `id` collides with AGE's built-in id()/graphid. "
        "Rename the variable (e.g. `ident`, `x`) and retry.",
        doc_hint="AGE reserves id()/graphid; never name a variable `id`.",
        example_fix="MATCH (ident) RETURN ident.name LIMIT 25",
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
        if pattern.query_check is not None:
            if not query or not pattern.query_check.search(_strip_non_code_spans(query)):
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
# `AS count` parses, but every bare reference to it is a syntax error. Rename the alias AND
# its references; `count(` (the aggregate) and `n.count` / `{count:` are excluded.
_AS_COUNT_RE = re.compile(r"\bAS\s+count\b", re.IGNORECASE)
_BARE_COUNT_RE = re.compile(r"(?<![.\w`])count\b(?!\s*[:(])", re.IGNORECASE)


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

    # 2. Strip a leading PROFILE/EXPLAIN.
    new_query = _apply_to_code_spans(query, lambda s: _PROFILE_EXPLAIN_RE.sub("", s))
    if new_query != query:
        query = new_query
        fixes.append(
            "Stripped PROFILE/EXPLAIN prefix (PROFILE is a syntax error on Apache AGE and "
            "EXPLAIN returns a query plan instead of rows)"
        )

    # 3. Rename the reserved alias `count`. Gated on an actual `AS count` so a bare `count`
    # reference in an already-broken query is not turned into an unbound variable.
    if _AS_COUNT_RE.search(_strip_non_code_spans(query)):
        new_query = _apply_to_code_spans(query, lambda s: _BARE_COUNT_RE.sub("count_", s))
        if new_query != query:
            query = new_query
            fixes.append(
                "Renamed the reserved alias `count` to `count_` (Apache AGE reserves `count`; "
                "a bare `count` reference is a syntax error)"
            )

    return query, fixes


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
