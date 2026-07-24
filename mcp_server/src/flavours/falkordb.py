"""FalkorDB flavour — the relocated FalkorDB openCypher dialect machinery.

This module owns everything FalkorDB-specific about validating and running a Cypher query:
the reject-track checks, the auto-fix transforms, execution-error classification, and the
FalkorDB Cypher dialect reference text. The generic span-masking primitives it builds on
(`_apply_to_code_spans`, `_apply_outside_comments`, `_strip_non_code_spans`, `_STRING_LITERAL_RE`)
stay backend-agnostic in `utils/cypher.py` and are imported here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from flavours.base import BaseFlavour
from utils.cypher import (
    CypherError,
    _apply_outside_comments,
    _apply_to_code_spans,
    _strip_non_code_spans,
    _STRING_LITERAL_RE,
)

# ---------------------------------------------------------------------------
# Stage 2: FalkorDB dialect checks
# ---------------------------------------------------------------------------

# Reject-track patterns
_APOC_RE = re.compile(r'apoc\.\w+[\.\w]*\(', re.IGNORECASE)
_EXISTS_SUBQUERY_RE = re.compile(r'\bEXISTS\s*\{', re.IGNORECASE)

# UNWIND ... AS x [then] WHERE   (no WITH or MATCH between UNWIND and WHERE)
# Multiline-aware; case-insensitive.  Matches when the token after the
# alias-and-any-whitespace is WHERE (not MATCH, WITH, RETURN, etc.).
# Uses a tempered pattern to stop at intervening WITH/MATCH/OPTIONAL/RETURN
# keywords so legitimate UNWIND...WITH...WHERE forms are not matched.
#
# Known limitations of this regex-only approach:
#  - `CALL` is intentionally NOT in the stop-set: practical CALL{} subquery
#    bodies open with WITH or contain RETURN, both of which already terminate
#    the span.  A pathological `UNWIND a AS n CALL { CREATE ... } WHERE ...`
#    would over-match, but write operations are blocked by the security stage.
#  - String literals containing the stop-words (e.g.
#    `UNWIND ['WITH','MATCH'] AS k WHERE k IS NOT NULL`) are false negatives
#    because the regex word-boundaries match inside quotes.  Such queries
#    fall through to FalkorDB and surface via classify_falkordb_execution_error().
# A proper fix for these cases requires AST parsing.
_UNWIND_WHERE_NO_WITH_RE = re.compile(
    r'\bUNWIND\b(?:(?!\b(?:WITH|MATCH|OPTIONAL|RETURN)\b).)+?\bAS\s+\w+\s+WHERE\b',
    re.IGNORECASE | re.DOTALL,
)

# Auto-fix patterns
_DATE_WRAPPER_RE = re.compile(
    r'\b(?:date|datetime|localDateTime)\s*\(\s*([\'"][^\'"]+[\'"])\s*\)',
    re.IGNORECASE,
)
_LOWER_RE = re.compile(r'\blower\s*\(', re.IGNORECASE)
_UPPER_RE = re.compile(r'\bupper\s*\(', re.IGNORECASE)
_PROFILE_EXPLAIN_RE = re.compile(r'^\s*(PROFILE|EXPLAIN)\s+', re.IGNORECASE)
# `!=` is a Neo4j/SQL convenience that FalkorDB does not support — only the
# canonical openCypher `<>` works. Match a bare `!=` (the `!` is otherwise
# unused in Cypher syntax). Run via `_apply_to_code_spans` so occurrences
# inside string literals/comments are preserved.
_NEQ_RE = re.compile(r'!=')

# `<expr> NOT IN [list]` is rejected by FalkorDB's parser; the equivalent
# `NOT (<expr> IN [list])` parses correctly. Match a simple LHS (a bare
# identifier or single-dot property access) followed by NOT IN [<flat list>].
# Limiting to flat lists (no nested brackets) keeps the regex safe; balanced
# function-call expressions (e.g. `coalesce(x, y) NOT IN [...]`) fall through
# to the Layer-3 classifier with its actionable suggestion.
# Group 1: the LHS expression (e.g. `n.kind` or `x`).
# Group 2: the bracketed list literal verbatim (`[...]`).
_NOT_IN_RE = re.compile(
    r'\b((?:\w+\.)?\w+)\s+NOT\s+IN\s+(\[[^\[\]]+\])',
    re.IGNORECASE,
)

# Non-ASCII identifier transliteration.  FalkorDB only accepts ASCII in
# variable names and aliases.  We transliterate common accented characters
# in Cypher identifiers (outside of string literals and comments) to their
# ASCII equivalents.  This handles the most common Spanish/French/German cases.
_NON_ASCII_TRANS = str.maketrans(
    'áàâäãåéèêëíìîïóòôöõúùûüñçÁÀÂÄÃÅÉÈÊËÍÌÎÏÓÒÔÖÕÚÙÛÜÑÇ',
    'aaaaaaeeeeiiiiooooouuuuncAAAAAAEEEEIIIIOOOOOUUUUNC',
)


def _transliterate_non_ascii_identifiers(query: str) -> str:
    """Replace non-ASCII characters in identifiers with ASCII equivalents.

    Preserves string literals and comments untouched — only identifiers
    (variable names, aliases) are transliterated.  Implemented on top of
    :func:`_apply_to_code_spans` so the non-code-span policy is consistent
    across every Stage 2b fix.
    """
    return _apply_to_code_spans(query, lambda s: s.translate(_NON_ASCII_TRANS))

# Bare variable in pattern:  MATCH parte-[:R]->(x)  ->  MATCH (parte)-[:R]->(x)
# Matches a word-variable immediately after MATCH/OPTIONAL MATCH that is
# followed by `-[` without intervening parens.
# Group 1: the MATCH keyword (preserved in substitution).
# Group 2: the bare variable name (wrapped in parens).
#
# Known limitations of this regex-only approach (AST parsing would cover them):
#  - Comma-separated patterns in one MATCH don't get second-position vars
#    wrapped (e.g. `MATCH (a)-[:R]->(b), c-[:R]->(d)` leaves `c` alone —
#    the classify_falkordb_execution_error layer still returns a useful error).
#  - Whitespace (tabs, multiple spaces) between MATCH and the variable is
#    normalized to a single space.
#  - Text inside string literals matching the pattern will also be rewritten;
#    probability of this occurring in LLM-generated Cypher is very low.
_BARE_VAR_IN_PATTERN_RE = re.compile(
    r'(\bOPTIONAL\s+MATCH\b|\bMATCH\b)\s+([A-Za-z_]\w*)(?=\s*-\s*\[)',
    re.IGNORECASE,
)


def check_falkordb_dialect(query: str) -> CypherError | None:
    """Reject track — return a CypherError for FalkorDB-incompatible patterns.

    Returns None if the query is clean.  Checks are ordered by severity; the
    first match wins.  String literals and comments are masked via
    :func:`_strip_non_code_spans` before each reject pattern is evaluated so
    mentions inside quotes or comments cannot falsely trigger rejection.
    """
    # Code-only view: strings replaced by ` _STR_ `, comments by ` _CMT_ `.
    # Reject patterns below look for real code usage, not mere mentions.
    code_only = _strip_non_code_spans(query)

    # 1. APOC procedures
    m = _APOC_RE.search(code_only)
    if m:
        return CypherError(
            stage='falkordb_dialect',
            reason='apoc_unsupported',
            found=m.group(0),
            explanation='APOC procedures are not available in FalkorDB.',
            suggestion='Use variable-length path patterns instead: MATCH path = (n)-[*1..3]->(end) RETURN path',
            doc_hint='FalkorDB supports openCypher variable-length paths with [*min..max] syntax',
        )

    # 2. EXISTS {} subqueries (NOT EXISTS((n)--()) which is valid)
    m = _EXISTS_SUBQUERY_RE.search(code_only)
    if m:
        return CypherError(
            stage='falkordb_dialect',
            reason='exists_subquery_unsupported',
            found=m.group(0),
            explanation='EXISTS {} subqueries are not supported in FalkorDB.',
            suggestion='Use WHERE EXISTS((n)-[:REL]->()) pattern syntax instead',
            doc_hint='FalkorDB supports EXISTS with inline path patterns, not subquery blocks',
        )

    # 3. UNWIND ... WHERE (must be UNWIND ... WITH ... WHERE)
    m = _UNWIND_WHERE_NO_WITH_RE.search(code_only)
    if m:
        return CypherError(
            stage='falkordb_dialect',
            reason='unwind_where_missing_with',
            found=m.group(0)[:80],
            explanation='WHERE cannot attach directly to UNWIND. Insert a WITH clause between UNWIND and WHERE.',
            suggestion=(
                'Rewrite `UNWIND list AS x WHERE ...` as '
                '`UNWIND list AS x WITH x, <other_bound_vars> WHERE ...` '
                '(include any prior-bound variables you need to keep in scope).'
            ),
            doc_hint='FalkorDB openCypher: WHERE attaches to MATCH / OPTIONAL MATCH / WITH only.',
        )

    return None


def fix_falkordb_dialect(query: str) -> tuple[str, list[str]]:
    """Auto-fix track — apply lossless transformations for FalkorDB compatibility.

    Returns the fixed query and a list of human-readable descriptions of
    each fix applied.  Every step operates only on code spans via
    :func:`_apply_to_code_spans`, so text inside string literals and
    comments is never silently rewritten and the ``fixes`` list never
    reports a false positive for content that didn't need changing.
    """
    fixes: list[str] = []

    # 1. Strip date/datetime/localDateTime wrappers.
    # The wrapper regex legitimately consumes a string literal as part of
    # its match (`date('2024-01-01')`), so we can only mask comments here —
    # not string literals.
    new_query = _apply_outside_comments(query, lambda s: _DATE_WRAPPER_RE.sub(r'\1', s))
    if new_query != query:
        query = new_query
        fixes.append('Stripped date/datetime wrapper functions (FalkorDB uses string dates)')

    # 2. Fix lower() -> toLower() (code only)
    new_query = _apply_to_code_spans(query, lambda s: _LOWER_RE.sub('toLower(', s))
    if new_query != query:
        query = new_query
        fixes.append('Replaced lower() with toLower()')

    # 3. Fix upper() -> toUpper() (code only)
    new_query = _apply_to_code_spans(query, lambda s: _UPPER_RE.sub('toUpper(', s))
    if new_query != query:
        query = new_query
        fixes.append('Replaced upper() with toUpper()')

    # 3b. Fix `!=` -> `<>` (code only — string literals and comments preserved).
    new_query = _apply_to_code_spans(query, lambda s: _NEQ_RE.sub('<>', s))
    if new_query != query:
        query = new_query
        fixes.append('Replaced != with <> (FalkorDB only accepts the openCypher canonical form)')

    # 3c. Rewrite `<expr> NOT IN [list]` -> `NOT (<expr> IN [list])`.
    # FalkorDB rejects the prefix-style form; the parenthesized form is the
    # canonical openCypher equivalent. Limited to simple LHS expressions
    # (identifier or single-dot property access) and flat list literals.
    #
    # Cannot use `_apply_to_code_spans` because list items are themselves
    # string literals (`['a', 'b']`), so the per-chunk approach would split
    # the match. Cannot blindly use `_apply_outside_comments` either, because
    # a literal `... NOT IN [...]` substring inside a string value would get
    # rewritten. Walk all candidate matches on the full query and skip those
    # whose LHS starts inside a string literal.
    string_spans = [(m.start(), m.end()) for m in _STRING_LITERAL_RE.finditer(query)]
    def _lhs_is_in_string(pos: int) -> bool:
        return any(s <= pos < e for s, e in string_spans)
    out_parts: list[str] = []
    last = 0
    rewrote = False
    for m in _NOT_IN_RE.finditer(query):
        if _lhs_is_in_string(m.start()):
            continue
        out_parts.append(query[last:m.start()])
        out_parts.append(f'NOT ({m.group(1)} IN {m.group(2)})')
        last = m.end()
        rewrote = True
    if rewrote:
        out_parts.append(query[last:])
        query = ''.join(out_parts)
        fixes.append('Rewrote `x NOT IN [...]` to `NOT (x IN [...])` (FalkorDB only accepts the parenthesized form)')

    # 4. Strip PROFILE/EXPLAIN prefix (code only; anchored to start-of-span)
    new_query = _apply_to_code_spans(query, lambda s: _PROFILE_EXPLAIN_RE.sub('', s))
    if new_query != query:
        query = new_query
        fixes.append('Stripped PROFILE/EXPLAIN prefix (not supported in FalkorDB)')

    # 5. Wrap bare variables in MATCH/OPTIONAL MATCH patterns (code only).
    new_query = _apply_to_code_spans(query, lambda s: _BARE_VAR_IN_PATTERN_RE.sub(r'\1 (\2)', s))
    if new_query != query:
        query = new_query
        fixes.append('Wrapped bare variable in parens (FalkorDB pattern syntax requires parenthesized nodes)')

    # 6. Transliterate non-ASCII characters in identifiers/aliases.
    # FalkorDB's parser only accepts ASCII in variable names and aliases.
    # Common with Spanish-language LLM output (año → ano, señal → senal).
    # Uses the shared helper so content inside string literals and
    # comments is preserved.
    new_query = _transliterate_non_ascii_identifiers(query)
    if new_query != query:
        query = new_query
        fixes.append('Transliterated non-ASCII characters in identifiers (FalkorDB requires ASCII-only names)')

    return query, fixes


# ---------------------------------------------------------------------------
# Execution-error classification (Layer 3)
# ---------------------------------------------------------------------------

@dataclass
class ExecutionErrorPattern:
    """Maps a FalkorDB parser-error string to an actionable CypherError.

    Walked in order by :func:`classify_falkordb_execution_error`; first match wins.
    """

    name: str
    matcher: re.Pattern
    suggestion: str
    doc_hint: str
    example_fix: str | None = None


# Order matters: more specific patterns before general ones.  The first
# matching pattern wins.  Append new patterns at the end of the list and
# tighten matchers if a new pattern overlaps an existing one.
_EXECUTION_ERROR_PATTERNS: list[ExecutionErrorPattern] = [
    ExecutionErrorPattern(
        name='bare_variable_in_pattern',
        matcher=re.compile(r"Invalid input '-': expected '='"),
        suggestion=(
            'A bound variable inside a MATCH/OPTIONAL MATCH pattern must be '
            'wrapped in parentheses. Use `(var)-[:REL]->(other)` instead of '
            '`var-[:REL]->(other)`.'
        ),
        doc_hint='FalkorDB openCypher: pattern nodes must be parenthesized.',
        example_fix='OPTIONAL MATCH (n)-[:REL]->(m)',
    ),
    ExecutionErrorPattern(
        name='where_needs_with',
        matcher=re.compile(r"expected WITH.+errCtx:\s*WHERE\b", re.IGNORECASE | re.DOTALL),
        suggestion=(
            'WHERE cannot attach to RETURN or UNWIND. Compute projections or '
            'unwound aliases in a WITH clause first, then filter, then RETURN. '
            'Example: `WITH expr AS alias WHERE alias IS NOT NULL RETURN alias`. '
            'For UNWIND: `UNWIND list AS x WITH x WHERE x.prop IS NOT NULL RETURN x`.'
        ),
        doc_hint='FalkorDB openCypher: WHERE attaches only to MATCH/OPTIONAL MATCH/WITH.',
        example_fix='WITH expr AS alias WHERE alias IS NOT NULL RETURN alias',
    ),
    ExecutionErrorPattern(
        name='non_ascii_identifier',
        matcher=re.compile(r"Invalid input '.*?[^\x00-\x7F]|Invalid input '\ufffd'"),
        suggestion=(
            'FalkorDB identifiers (variable names, aliases) must be ASCII-only. '
            'Replace accented characters: año → anno, señal → sennal, etc.'
        ),
        doc_hint='FalkorDB openCypher: identifiers are ASCII [A-Za-z0-9_] only.',
        example_fix='WITH toInteger(val) AS anno_nacimiento',
    ),
    ExecutionErrorPattern(
        name='function_arity_mismatch',
        matcher=re.compile(
            r"Received (\d+) arguments? to function '(\w+)', expected (at most \d+|at least \d+|exactly \d+|\d+)",
            re.IGNORECASE,
        ),
        suggestion=(
            "FalkorDB's function signature differs from Neo4j's — some functions take "
            "fewer arguments in FalkorDB.  Most common: `round(x, precision)` "
            "(Neo4j) is unsupported; FalkorDB's `round(x)` takes only one argument.  "
            "For decimal precision use arithmetic: "
            "`round(x * 100.0) / 100.0` for 2 decimals.  For other functions, check "
            "FalkorDB docs for the correct signature."
        ),
        doc_hint='FalkorDB openCypher: function arities can differ from Neo4j.',
        example_fix='round(avg(val) * 100.0) / 100.0  // 2-decimal precision',
    ),
    ExecutionErrorPattern(
        name='not_equals_operator',
        # FalkorDB rejects `!=` with `Invalid input '!'`. The expected-token
        # list always includes `<>` (the canonical openCypher form). Match on
        # the combination — `Invalid input '!'` plus a `<>` mention in the
        # expected list — to avoid catching unrelated `!`-bearing errors.
        matcher=re.compile(
            r"Invalid input '!'.*'<>'",
            re.IGNORECASE | re.DOTALL,
        ),
        suggestion=(
            'FalkorDB does not support the `!=` operator. Use the openCypher '
            'canonical form `<>` instead. Layer-2 auto-fix normally rewrites '
            'this before execution; if you see this error, double-check that '
            '`!=` was not embedded in something the auto-fixer skipped (e.g. a '
            'parameterized expression).'
        ),
        doc_hint='FalkorDB openCypher: only `<>` is the not-equals operator; `!=` is not accepted.',
        example_fix='WHERE n.status <> "None"',
    ),
    ExecutionErrorPattern(
        name='sql_window_function',
        # SQL window functions like `<agg>(...) OVER (PARTITION BY ...)` are
        # not openCypher. FalkorDB rejects with `Invalid input 'V'` (from the
        # V in OVER) and the errCtx contains `OVER (`. Match the combination
        # so unrelated `Invalid input 'V'` errors don't false-positive.
        matcher=re.compile(
            r"Invalid input 'V'.*errCtx:[^\n]*\bOVER\s*\(",
            re.IGNORECASE | re.DOTALL,
        ),
        suggestion=(
            'openCypher does not have SQL window functions. '
            '`<agg>(x) OVER (PARTITION BY key)` is not supported. '
            'Use a chained WITH that aggregates by the partition key, then '
            'a second WITH (or RETURN) to combine: '
            'first `WITH key, sum(x) AS group_total`, then carry that into '
            'the next clause alongside the per-row data you need.'
        ),
        doc_hint='openCypher: no window functions; group with chained WITH clauses.',
        example_fix=(
            'WITH op, sum(weight) AS group_total '
            'WITH op, group_total RETURN op, group_total ORDER BY group_total DESC'
        ),
    ),
    ExecutionErrorPattern(
        name='not_in_list_form',
        # FalkorDB rejects `x NOT IN [a, b, c]` — must be `NOT (x IN [a, b, c])`.
        # Error fingerprint: `Invalid input ','` plus `expected ... ']'` plus
        # the errCtx containing `NOT IN [`. Scope by all three so we don't
        # over-match generic comma errors.
        matcher=re.compile(
            r"Invalid input ','.*\]'.*errCtx:[^\n]*\bNOT IN \[",
            re.IGNORECASE | re.DOTALL,
        ),
        suggestion=(
            'FalkorDB does not accept `x NOT IN [list]`. Wrap the IN test in '
            'parens and prefix NOT: `NOT (x IN [list])`. This is semantically '
            'equivalent and parses correctly.'
        ),
        doc_hint='FalkorDB openCypher: NOT IN with a list literal must be written as NOT (x IN [...]).',
        example_fix='WHERE NOT (n.kind IN ["a", "b", "c"])',
    ),
    ExecutionErrorPattern(
        name='variable_not_in_scope',
        # FalkorDB emits messages like `'pi' not defined` when a variable is
        # referenced where it isn't bound. The most common cause we see is
        # cross-side reference in UNION / UNION ALL (each side has its own
        # scope), followed by missing carry-through in WITH and plain typos.
        matcher=re.compile(r"'(\w+)' not defined", re.IGNORECASE),
        suggestion=(
            'A referenced variable is not in scope at the point of use. '
            'Common causes: (1) you referenced it on the other side of '
            'UNION / UNION ALL — each side has its own scope, so variables '
            'from the left half are not visible on the right; (2) it was '
            'not carried through a preceding WITH; (3) typo in the name. '
            'Fix: either repeat the MATCH chain on each side of UNION, or '
            'collapse to a single MATCH chain and use CASE / coalesce to '
            'pick between alternatives.'
        ),
        doc_hint='FalkorDB openCypher: UNION sides have independent scope; WITH must carry every variable used downstream.',
        example_fix=(
            'MATCH (n) OPTIONAL MATCH (n)-[:R1]->(a) '
            'OPTIONAL MATCH (n)-[:R2]->(b) '
            'RETURN n, CASE WHEN a IS NOT NULL THEN a.x ELSE b.x END AS x'
        ),
    ),
    # More patterns added as they surface from Langfuse observations.
]


def classify_falkordb_execution_error(msg: str) -> CypherError:
    """Map a FalkorDB execution-error message to an actionable CypherError.

    Walks :data:`_EXECUTION_ERROR_PATTERNS` in order.  First match wins.
    On no match, returns the generic envelope to preserve current
    behavior for unknown errors.
    """
    for pattern in _EXECUTION_ERROR_PATTERNS:
        if pattern.matcher.search(msg):
            explanation = f'FalkorDB returned an error: {msg}'
            if pattern.example_fix:
                explanation += f'\nCorrected example: {pattern.example_fix}'
            return CypherError(
                stage='execution',
                reason=pattern.name,
                found=msg,
                explanation=explanation,
                suggestion=pattern.suggestion,
                doc_hint=pattern.doc_hint,
            )

    return CypherError(
        stage='execution',
        reason='query_failed',
        found=msg,
        explanation=f'FalkorDB returned an error: {msg}',
        suggestion='Check your Cypher syntax. Use get_schema to verify label and property names.',
    )


# ---------------------------------------------------------------------------
# FalkorDB Cypher dialect reference. Curated from https://github.com/FalkorDB/skills.
# Task 7 wires get_schema to surface this via flavour.dialect_reference; until that lands,
# get_schema() also holds an identical copy of this text.
# ---------------------------------------------------------------------------

_FALKORDB_DIALECT = (
    "## Cypher Quick Reference (FalkorDB)\n\n"
    "### Property & Label Escaping\n"
    "- Multi-word labels: MATCH (n:`My Label`) RETURN n\n"
    "- Multi-word properties: WHERE n.`my property` = 'value'\n"
    "- Always use backticks for identifiers with spaces or special chars\n\n"
    "### Variable-Length Paths (no APOC)\n"
    "- MATCH (a)-[*1..3]->(b) RETURN a, b\n"
    "- MATCH path = (a)-[*..5]->(b) RETURN nodes(path), relationships(path)\n\n"
    "### Aggregation Patterns\n"
    "- GROUP BY is implicit: MATCH (n) RETURN n.type, count(n)\n"
    "- Mid-query: MATCH (n)-[:REL]->(m) WITH m, count(n) AS cnt "
    "WHERE cnt > 1 RETURN m.name, cnt\n\n"
    "### Date Handling\n"
    "- No date() function — compare strings: WHERE n.date > '2024-01-01'\n\n"
    "### String Functions\n"
    "- toLower() / toUpper() (NOT lower() / upper())\n"
    "- starts with / ends with / contains\n\n"
    "### Index-Aware Filtering\n"
    "- Accelerated: =, <, >, <=, >=, IN, starts with\n"
    "- NOT accelerated: <> (not-equal), contains, ends with\n"
    "- Full-text: CALL db.idx.fulltext.queryNodes('idx', 'term')\n\n"
    "### Known Limitations\n"
    "- No APOC — use variable-length paths\n"
    "- No pattern comprehensions — use OPTIONAL MATCH + collect()\n"
    "- No EXISTS {} subqueries — use EXISTS(pattern) syntax\n"
    "- No CALL {} subqueries — use WITH + OPTIONAL MATCH\n"
    "- No map projections — return properties individually\n"
    "- LIMIT auto-injected (200) if not specified"
)


class FalkorDbFlavour(BaseFlavour):
    """FalkorDB flavour — full dialect machinery + DB-enforced read-only via ro_query."""

    name = "falkordb"
    dialect_id = "falkordb-cypher"
    dialect_reference = _FALKORDB_DIALECT

    def check_dialect(self, query: str) -> CypherError | None:
        return check_falkordb_dialect(query)

    def auto_fix(self, query: str) -> tuple[str, list[str]]:
        return fix_falkordb_dialect(query)

    def classify_execution_error(self, message: str) -> CypherError:
        return classify_falkordb_execution_error(message)

    # attribute_keys: inherited from BaseFlavour (top-level keys minus reserved) — correct for
    # FalkorDB, whose properties are top-level.

    async def execute_graph_query(self, driver, query: str) -> tuple[list[dict], list[str]]:
        """DB-enforced read-only via ro_query on the internal graph handle.

        The public execute_query() uses graph.query() (read-write), so we access the internal
        graph handle for ro_query — same pattern as graphiti_core.
        """
        graph = driver._get_graph(driver._database)
        query_result = await graph.ro_query(query)
        header = [h[1] for h in query_result.header] if query_result.header else []
        records: list[dict] = []
        for row in (query_result.result_set or []):
            records.append(
                {name: (row[i] if i < len(row) else None) for i, name in enumerate(header)}
            )
        return records, header
