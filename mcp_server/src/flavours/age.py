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
from utils.cypher import CypherError, _strip_non_code_spans

_AGE_DIALECT = (
    "Backend: Apache AGE (openCypher over PostgreSQL) — this is NOT FalkorDB or Neo4j. "
    "APOC is unavailable and string helpers such as indexOf/split/replace are absent or "
    "behave differently; do NOT use them. "
    "Match nodes by a property, e.g. `WHERE n.name = '...'` (see get_schema for the label "
    "and relationship-type inventory of THIS graph). "
    "ATTRIBUTES: each node's descriptive fields live in a queryable MAP property `attributes` "
    "— address members directly, e.g. `n.attributes.<field>`; NEVER string-parse it. See "
    "`attribute_keys` in get_schema for the fields per label, or `RETURN keys(n.attributes)`. "
    "Dates are ISO 'YYYY-MM-DD' strings: get the year with "
    "`toInteger(substring(n.attributes.<field>, 0, 4))` and derive ranges with a CASE. "
    "PITFALLS (AGE-specific): (1) NEVER name a variable `id` — it collides with AGE's built-in "
    "id()/graphid and fails ('column notation .id applied to type graphid'); use `ident`/`x`. "
    "(2) relationship-type disjunction `[:A|B|C]` is NOT supported — match a generic edge and "
    "filter: `MATCH (a)-[r]->(b) WHERE type(r) IN ['A','B','C']`. Always include a LIMIT."
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

    # (1) `id` used as a VARIABLE — a pattern variable ((id / [id) or an alias (AS id).
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

    # (2) relationship-type disjunction [:A|B|C] — unsupported by AGE.
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
        name="reltype_disjunction_unsupported",
        matcher=re.compile(r'at or near "\|"'),
        suggestion="AGE does not support relationship-type disjunction like [:A|B|C]. "
        "Match a generic edge and filter: MATCH (a)-[r]->(b) WHERE type(r) IN ['A','B','C'].",
        doc_hint="AGE has no [:A|B|C]; use WHERE type(r) IN [...].",
        example_fix="MATCH (a)-[r]->(b) WHERE type(r) IN ['DETIENE','INVESTIGA'] RETURN b LIMIT 25",
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


class AgeFlavour(BaseFlavour):
    """AGE flavour — reject-with-hint for AGE-unsupported constructs, nested-map attribute_keys.

    check_dialect rejects (with an actionable hint) a variable named ``id`` and relationship-type
    disjunction ``[:A|B|C]`` — both AGE-unsupported, and both safer to reject than to auto-rewrite.
    The only AGE auto-fix is the shared pipeline's LIMIT injection (BaseFlavour.auto_fix is a
    no-op here). classify_execution_error is a post-hoc net for anything check_dialect misses.
    """

    name = "age"
    dialect_id = "age-opencypher"
    dialect_reference = _AGE_DIALECT
    dialect_summary = (
        "Apache AGE openCypher (NOT FalkorDB/Neo4j): no APOC/indexOf/split; attributes are a "
        "queryable map (n.attributes.<field>); never name a variable `id`; no [:A|B|C] "
        "disjunction (use MATCH (a)-[r]->(b) WHERE type(r) IN [...])."
    )

    def check_dialect(self, query: str) -> CypherError | None:
        return check_age_dialect(query)

    # auto_fix: inherited from BaseFlavour (no-op). AGE's only auto-fix is the shared pipeline's
    # LIMIT injection; AGE-unsupported constructs are reject-with-hint via check_dialect.

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
