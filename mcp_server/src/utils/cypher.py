"""Data models for the Cypher validation pipeline and result formatting."""
from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from utils.cypher_quality import assess_quality, compute_result_signals, refine_verdict


@dataclass
class SanitizedQuery:
    """Result of successful validation: the sanitized query and applied fixes."""

    query: str
    auto_fixes: list[str] = field(default_factory=list)
    effective_limit: int = 200


@dataclass
class CypherError:
    """Structured rejection from the validation pipeline."""

    stage: str
    reason: str
    found: str
    explanation: str
    suggestion: str
    doc_hint: str = ''


# ---------------------------------------------------------------------------
# Stage 1 helpers
# ---------------------------------------------------------------------------

# Smart-quote mapping: all unicode curly/fancy quotes to their ASCII equivalents
_SMART_QUOTE_MAP: dict[str, str] = {
    '\u201c': '"',  # left double quotation mark
    '\u201d': '"',  # right double quotation mark
    '\u201e': '"',  # double low-9 quotation mark
    '\u201f': '"',  # double high-reversed-9 quotation mark
    '\u2018': "'",  # left single quotation mark
    '\u2019': "'",  # right single quotation mark
    '\u201a': "'",  # single low-9 quotation mark
    '\u201b': "'",  # single high-reversed-9 quotation mark
}

# Regex for code-block wrapping
_CODE_BLOCK_RE = re.compile(r'```(?:cypher)?\s*\n?(.*?)\n?\s*```', re.DOTALL)

# Multi-word identifier quoting regexes (from neo4j-graphrag-python)
# Node labels:  :Data Science  ->  :`Data Science`
_LABEL_RE = re.compile(r'(?<=:)(?!`)([A-Za-z]\w* \w[\w ]*?)(?=[\s)}\]{,|])')
# Property keys:  {first name: ...}  ->  {`first name`: ...}
_PROP_KEY_RE = re.compile(r'(?<=[{,])\s*(?!`)([A-Za-z]\w* \w[\w ]*?)(?=\s*:)')
# Relationship types:  [:WORKS WITH]  ->  [:`WORKS WITH`]
_REL_TYPE_RE = re.compile(r'(?<=\[)(:)(?!`)([A-Za-z]\w* \w[\w ]*?)(?=[\]|])')

# Cypher keywords that must NOT be swallowed into multi-word identifiers.
# When the backtick fixer sees `:Ubicacion OR loc:Municipio`, it should NOT
# treat "Ubicacion OR" as a multi-word label — OR is a keyword.
_CYPHER_KEYWORDS = frozenset({
    'AND', 'OR', 'NOT', 'XOR', 'IN', 'IS', 'NULL',
    'WHERE', 'WITH', 'RETURN', 'MATCH', 'OPTIONAL', 'CREATE', 'DELETE',
    'SET', 'REMOVE', 'MERGE', 'DETACH', 'ORDER', 'BY', 'SKIP', 'LIMIT',
    'UNION', 'ALL', 'AS', 'DISTINCT', 'ON', 'CASE', 'WHEN', 'THEN',
    'ELSE', 'END', 'EXISTS', 'CONTAINS', 'STARTS', 'ENDS', 'TRUE',
    'FALSE', 'UNWIND', 'FOREACH', 'CALL', 'YIELD', 'DESC', 'ASC',
})


def _contains_keyword(identifier: str) -> bool:
    """Check if a multi-word identifier contains a Cypher keyword."""
    return any(word.upper() in _CYPHER_KEYWORDS for word in identifier.split())


def _extract_alias(expr: str) -> str | None:
    """Extract the alias from an expression like 'count(o) AS cnt' or a simple variable name.

    Returns the alias name, or None if the expression is empty/unparseable.
    """
    expr = expr.strip()
    if not expr:
        return None

    # Check for AS alias  (case insensitive)
    as_match = re.search(r'\bAS\s+(\w+)\s*$', expr, re.IGNORECASE)
    if as_match:
        return as_match.group(1)

    # Simple variable name (single word, no dots, no parens)
    if re.fullmatch(r'[A-Za-z_]\w*', expr):
        return expr

    return None


def _parse_variable_list(clause: str) -> list[str]:
    """Split a comma-separated variable list respecting parentheses.

    For example: 'op, count(o) AS cnt' -> ['op', 'count(o) AS cnt']
    """
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in clause:
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
            parts.append(''.join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append(''.join(current))
    return parts


def _extract_return_variables(query: str) -> list[str]:
    """Extract variables to use in a synthesized RETURN clause.

    If the query has WITH clauses, uses the variables from the **last** WITH
    (since WITH redefines scope). Otherwise extracts variables from MATCH
    patterns.
    """
    # Check for WITH clauses — use the last one
    with_matches = list(re.finditer(
        r'\bWITH\b\s+(.*?)(?=\b(?:MATCH|WHERE|WITH|ORDER|LIMIT|RETURN|CREATE|MERGE|SET|DELETE|REMOVE|UNWIND|CALL)\b|$)',
        query,
        re.IGNORECASE,
    ))
    if with_matches:
        last_with = with_matches[-1].group(1).strip()
        parts = _parse_variable_list(last_with)
        variables = []
        for part in parts:
            alias = _extract_alias(part)
            if alias:
                variables.append(alias)
        if variables:
            return variables

    # Fall back to MATCH pattern variables: find all (varName...) groups
    match_vars: list[str] = []
    for m in re.finditer(r'\((\w+)', query):
        var = m.group(1)
        if var not in match_vars:
            match_vars.append(var)
    return match_vars


# ---------------------------------------------------------------------------
# Stage 1: LLM syntax fixups
# ---------------------------------------------------------------------------

def _fix_llm_syntax(query: str) -> tuple[str, list[str]]:
    """Fix common LLM Cypher generation mistakes.

    Returns the fixed query and a list of human-readable descriptions of
    each fix applied.  This is a pure function — no I/O, no side-effects.
    """
    fixes: list[str] = []

    # 1. Code block extraction
    m = _CODE_BLOCK_RE.search(query)
    if m:
        query = m.group(1)
        fixes.append('Extracted query from code block wrapper')

    # 2. Smart quote replacement
    replaced_smart = False
    for smart, straight in _SMART_QUOTE_MAP.items():
        if smart in query:
            query = query.replace(smart, straight)
            replaced_smart = True
    if replaced_smart:
        fixes.append('Replaced smart quotes with straight quotes')

    # 3. HTML entity decoding
    decoded = html.unescape(query)
    if decoded != query:
        query = decoded
        fixes.append('Decoded HTML entities')

    # 4. Multi-word identifier quoting (skip if match contains a Cypher keyword)
    # Node labels
    new_query = _LABEL_RE.sub(
        lambda m: f'`{m.group(1)}`' if not _contains_keyword(m.group(1)) else m.group(1),
        query,
    )
    # Property keys
    new_query = _PROP_KEY_RE.sub(
        lambda m: f'`{m.group(1)}`' if not _contains_keyword(m.group(1)) else m.group(1),
        new_query,
    )
    # Relationship types
    new_query = _REL_TYPE_RE.sub(
        lambda m: f'{m.group(1)}`{m.group(2)}`' if not _contains_keyword(m.group(2)) else m.group(0),
        new_query,
    )
    if new_query != query:
        query = new_query
        fixes.append('Backtick-quoted multi-word identifiers')

    # 5. RETURN injection
    query_upper = query.upper()
    has_return = 'RETURN' in query_upper
    has_call = 'CALL' in query_upper
    if not has_return and not has_call:
        variables = _extract_return_variables(query)
        if variables:
            var_list = ', '.join(variables)
            query = f'{query} RETURN {var_list}'
            fixes.append(f'Injected RETURN {var_list}')

    return query, fixes


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
#    fall through to FalkorDB and surface via classify_execution_error().
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

# Bare variable in pattern:  MATCH parte-[:R]->(x)  ->  MATCH (parte)-[:R]->(x)
# Matches a word-variable immediately after MATCH/OPTIONAL MATCH that is
# followed by `-[` without intervening parens.
# Group 1: the MATCH keyword (preserved in substitution).
# Group 2: the bare variable name (wrapped in parens).
#
# Known limitations of this regex-only approach (AST parsing would cover them):
#  - Comma-separated patterns in one MATCH don't get second-position vars
#    wrapped (e.g. `MATCH (a)-[:R]->(b), c-[:R]->(d)` leaves `c` alone —
#    the classify_execution_error layer still returns a useful error).
#  - Whitespace (tabs, multiple spaces) between MATCH and the variable is
#    normalized to a single space.
#  - Text inside string literals matching the pattern will also be rewritten;
#    probability of this occurring in LLM-generated Cypher is very low.
_BARE_VAR_IN_PATTERN_RE = re.compile(
    r'(\bOPTIONAL\s+MATCH\b|\bMATCH\b)\s+([A-Za-z_]\w*)(?=\s*-\s*\[)',
    re.IGNORECASE,
)


def _check_falkordb_dialect(query: str) -> CypherError | None:
    """Reject track — return a CypherError for FalkorDB-incompatible patterns.

    Returns None if the query is clean.  Checks are ordered by severity; the
    first match wins.
    """
    # 1. APOC procedures
    m = _APOC_RE.search(query)
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
    m = _EXISTS_SUBQUERY_RE.search(query)
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
    m = _UNWIND_WHERE_NO_WITH_RE.search(query)
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


def _fix_falkordb_dialect(query: str) -> tuple[str, list[str]]:
    """Auto-fix track — apply lossless transformations for FalkorDB compatibility.

    Returns the fixed query and a list of human-readable descriptions of
    each fix applied.
    """
    fixes: list[str] = []

    # 1. Strip date/datetime/localDateTime wrappers
    new_query = _DATE_WRAPPER_RE.sub(r'\1', query)
    if new_query != query:
        query = new_query
        fixes.append('Stripped date/datetime wrapper functions (FalkorDB uses string dates)')

    # 2. Fix lower() -> toLower()
    new_query = _LOWER_RE.sub('toLower(', query)
    if new_query != query:
        query = new_query
        fixes.append('Replaced lower() with toLower()')

    # 3. Fix upper() -> toUpper()
    new_query = _UPPER_RE.sub('toUpper(', query)
    if new_query != query:
        query = new_query
        fixes.append('Replaced upper() with toUpper()')

    # 4. Strip PROFILE/EXPLAIN prefix
    new_query = _PROFILE_EXPLAIN_RE.sub('', query)
    if new_query != query:
        query = new_query
        fixes.append('Stripped PROFILE/EXPLAIN prefix (not supported in FalkorDB)')

    # 5. Wrap bare variables in MATCH/OPTIONAL MATCH patterns.
    new_query = _BARE_VAR_IN_PATTERN_RE.sub(r'\1 (\2)', query)
    if new_query != query:
        query = new_query
        fixes.append('Wrapped bare variable in parens (FalkorDB pattern syntax requires parenthesized nodes)')

    return query, fixes


# ---------------------------------------------------------------------------
# Execution-error classification (Layer 3)
# ---------------------------------------------------------------------------

@dataclass
class ExecutionErrorPattern:
    """Maps a FalkorDB parser-error string to an actionable CypherError.

    Walked in order by :func:`classify_execution_error`; first match wins.
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
        name='unwind_where_missing_with',
        matcher=re.compile(r"expected WITH.+errCtx:\s*WHERE\b", re.IGNORECASE | re.DOTALL),
        suggestion=(
            'WHERE cannot attach to UNWIND directly. Insert WITH between them: '
            '`UNWIND list AS x WITH x WHERE x.prop IS NOT NULL`.'
        ),
        doc_hint='FalkorDB openCypher: WHERE attaches only to MATCH/OPTIONAL MATCH/WITH.',
        example_fix='UNWIND list AS x WITH x WHERE x.prop IS NOT NULL RETURN x',
    ),
    # More patterns added as they surface from Langfuse observations.
]


def classify_execution_error(msg: str) -> CypherError:
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
# Stage 3: Security whitelist — fail-safe keyword gate
# ---------------------------------------------------------------------------

_BLOCKED_KEYWORDS: set[str] = {
    'CREATE', 'DELETE', 'DETACH', 'SET', 'MERGE', 'REMOVE',
    'DROP', 'FOREACH', 'LOAD', 'CSV',
}

# Patterns for stripping non-keyword tokens before scanning
_STRING_LITERAL_RE = re.compile(r"""(?:"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')""")
_BACKTICK_IDENT_RE = re.compile(r'`[^`]*`')
_PROPERTY_ACCESS_RE = re.compile(r'\.(\w+)')
_CALL_PROCEDURE_RE = re.compile(r'\bCALL\s+([\w.]+)', re.IGNORECASE)


def _extract_keyword_tokens(query: str) -> list[str]:
    """Extract keyword-level tokens from a Cypher query for security scanning.

    Strips string literals, backtick-quoted identifiers, and property-access
    names so that blocked keywords inside those contexts are not matched.
    Returns uppercase tokens.
    """
    # 1. Remove string literals (replace with placeholder to preserve spacing)
    cleaned = _STRING_LITERAL_RE.sub(' _STR_ ', query)
    # 2. Remove backtick-quoted identifiers
    cleaned = _BACKTICK_IDENT_RE.sub(' _BT_ ', cleaned)
    # 3. Remove property access (.word) so e.g. n.description won't match
    cleaned = _PROPERTY_ACCESS_RE.sub(' ', cleaned)
    # 4. Split on non-word characters and return uppercase tokens
    tokens = re.findall(r'\b[A-Za-z_]\w*\b', cleaned)
    return [t.upper() for t in tokens]


def _check_whitelist(query: str) -> CypherError | None:
    """Stage 3: reject queries containing write/admin Cypher keywords.

    Returns a CypherError with stage='security' and reason='write_operation'
    if a blocked keyword is found.  Returns None if the query is clean.
    """
    # Special check: CALL is allowed only for db.* procedures
    for m in _CALL_PROCEDURE_RE.finditer(query):
        proc_name = m.group(1).lower()
        if not proc_name.startswith('db.'):
            return CypherError(
                stage='security',
                reason='write_operation',
                found=m.group(0),
                explanation=(
                    f'Only CALL db.* procedures are allowed for read-only access. '
                    f'Found: {m.group(1)}'
                ),
                suggestion='Use CALL db.labels() or CALL db.relationshipTypes() for schema introspection.',
                doc_hint='Read-only procedure namespaces: db.*',
            )

    tokens = _extract_keyword_tokens(query)
    for token in tokens:
        if token in _BLOCKED_KEYWORDS:
            return CypherError(
                stage='security',
                reason='write_operation',
                found=token,
                explanation=(
                    f'Write operation "{token}" is not allowed. '
                    f'This endpoint only accepts read-only Cypher queries.'
                ),
                suggestion='Use MATCH ... RETURN for read-only queries.',
                doc_hint='Allowed operations: MATCH, OPTIONAL MATCH, WHERE, WITH, RETURN, ORDER BY, SKIP, LIMIT, UNION, UNWIND, CALL db.*',
            )

    return None


# ---------------------------------------------------------------------------
# Stage 4: Safety injection
# ---------------------------------------------------------------------------

DEFAULT_LIMIT = 200


def _inject_safety(query: str, limit: int = DEFAULT_LIMIT) -> tuple[str, list[str], int]:
    """Stage 4: Inject safety measures.

    - Appends LIMIT (N+1) if no LIMIT clause present (for truncation detection)
    - CALL queries are exempt from LIMIT injection

    Returns (query, fixes, effective_limit).
    """
    fixes: list[str] = []
    query_upper = query.upper().strip()

    if query_upper.startswith('CALL '):
        return query, fixes, limit

    all_limits = re.findall(r'\bLIMIT\s+(\d+)', query, re.IGNORECASE)
    if all_limits:
        return query, fixes, int(all_limits[-1])

    query = f'{query.rstrip().rstrip(";")} LIMIT {limit + 1}'
    fixes.append(f'Injected LIMIT {limit}')

    return query, fixes, limit


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


def validate_and_sanitize(query: str, limit: int = DEFAULT_LIMIT) -> SanitizedQuery | CypherError:
    """Validate and sanitize a Cypher query through the four-stage pipeline.

    Stages:
        1. LLM syntax fixups (smart quotes, code blocks, RETURN injection)
        2. FalkorDB dialect (reject unsupported features, then auto-fix)
        3. Security whitelist (block write operations)
        4. Safety injection (LIMIT)

    Returns SanitizedQuery on success, CypherError on rejection.
    """
    # Stage 1: LLM fixups — always runs
    query, fixes_1 = _fix_llm_syntax(query)

    # Stage 2a: FalkorDB reject — fail fast
    if err := _check_falkordb_dialect(query):
        return err

    # Stage 2b: FalkorDB auto-fix
    query, fixes_2 = _fix_falkordb_dialect(query)

    # Stage 3: Security whitelist
    if err := _check_whitelist(query):
        return err

    # Stage 4: Safety injection
    query, fixes_4, effective_limit = _inject_safety(query, limit=limit)

    return SanitizedQuery(
        query=query,
        auto_fixes=fixes_1 + fixes_2 + fixes_4,
        effective_limit=effective_limit,
    )


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

ResultType = Literal['scalar', 'tabular', 'graph', 'path', 'error']


def _classify_result(
    records: list[dict[str, Any]], header: list[str]
) -> ResultType:
    """Classify the result type based on record shape and value types.

    Classification rules (checked in order):
    - scalar: 0 records, OR 1 record with exactly 1 column
    - path: any value has both ``nodes`` and ``edges`` attributes
    - graph: any value has a ``labels`` attribute (Node) or ``relation`` attribute (Edge)
    - tabular: everything else (multiple records with primitive values)
    """
    if len(records) == 0:
        return 'scalar'
    if len(records) == 1 and len(header) <= 1:
        return 'scalar'

    # Inspect all values for graph/path objects
    for record in records:
        for value in record.values():
            # Path: has both nodes and edges
            if hasattr(value, 'nodes') and hasattr(value, 'edges'):
                return 'path'
            # Graph: Node (has labels) or Edge (has relation)
            if hasattr(value, 'labels') or hasattr(value, 'relation'):
                return 'graph'

    return 'tabular'


def _format_scalar(records: list[dict[str, Any]], header: list[str]) -> dict[str, Any]:
    """Format a scalar result (0 or 1 record with 1 column)."""
    if len(records) == 0:
        return {'result': None}
    # Single record — return the sole value
    record = records[0]
    if header:
        return {'result': record[header[0]]}
    # Fallback: get first value from dict
    return {'result': next(iter(record.values()), None)}


def _format_tabular(
    records: list[dict[str, Any]], header: list[str]
) -> dict[str, Any]:
    """Format tabular results as columnar array-of-arrays (token-efficient)."""
    rows = [[record[col] for col in header] for record in records]
    return {'columns': header, 'rows': rows}


def _format_node(node: Any) -> dict[str, Any]:
    """Format a FalkorDB Node into a serialisable dict, filtering :Entity label."""
    labels = [lbl for lbl in (node.labels if hasattr(node, 'labels') else []) if lbl != 'Entity']
    props = node.properties if hasattr(node, 'properties') else {}
    return {
        'id': node.id if hasattr(node, 'id') else None,
        'labels': labels,
        'properties': dict(props),
    }


def _format_edge(edge: Any) -> dict[str, Any]:
    """Format a FalkorDB Edge into a serialisable dict."""
    return {
        'id': edge.id if hasattr(edge, 'id') else None,
        'type': edge.relation if hasattr(edge, 'relation') else None,
        'src': edge.src_node if hasattr(edge, 'src_node') else None,
        'dest': edge.dest_node if hasattr(edge, 'dest_node') else None,
    }


def _format_graph(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Format graph results — collect unique nodes and edges."""
    nodes_seen: set[int] = set()
    edges_seen: set[int] = set()
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    for record in records:
        for value in record.values():
            if hasattr(value, 'labels'):
                nid = value.id if hasattr(value, 'id') else id(value)
                if nid not in nodes_seen:
                    nodes_seen.add(nid)
                    nodes.append(_format_node(value))
            elif hasattr(value, 'relation'):
                eid = value.id if hasattr(value, 'id') else id(value)
                if eid not in edges_seen:
                    edges_seen.add(eid)
                    edges.append(_format_edge(value))

    return {'nodes': nodes, 'edges': edges}


def _format_path(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Format path results — alternating node/edge sequence per path."""
    paths: list[list[dict[str, Any]]] = []

    for record in records:
        for value in record.values():
            if not (hasattr(value, 'nodes') and hasattr(value, 'edges')):
                continue
            # nodes/edges may be properties or callables
            path_nodes = value.nodes() if callable(value.nodes) else value.nodes
            path_edges = value.edges() if callable(value.edges) else value.edges

            steps: list[dict[str, Any]] = []
            for i, node in enumerate(path_nodes):
                steps.append(_format_node(node))
                if i < len(path_edges):
                    steps.append(_format_edge(path_edges[i]))
            paths.append(steps)

    return {'steps': paths}


def format_result(
    records: list[dict[str, Any]],
    header: list[str],
    query: str,
    auto_fixes: list[str],
    execution_ms: float,
    limit: int,
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify results and build a typed JSON envelope with metadata.

    Uses the N+1 trick for truncation detection: if ``len(records) > limit``,
    the result set is truncated to ``limit`` rows and ``truncated`` is set to
    ``True``.
    """
    # Truncation detection (N+1 trick)
    truncated = len(records) > limit
    if truncated:
        records = records[:limit]

    result_type = _classify_result(records, header)

    # Build type-specific payload
    if result_type == 'scalar':
        payload = _format_scalar(records, header)
    elif result_type == 'graph':
        payload = _format_graph(records)
    elif result_type == 'path':
        payload = _format_path(records)
    else:
        payload = _format_tabular(records, header)

    # Metadata envelope (always present)
    envelope: dict[str, Any] = {
        'query': query,
        'auto_fixes': auto_fixes,
        'type': result_type,
        'row_count': len(records),
        'truncated': truncated,
        'limit_applied': limit,
        'execution_ms': execution_ms,
    }

    # Merge type-specific payload into envelope
    envelope.update(payload)

    # Quality assessment
    quality = assess_quality(query, schema=schema)
    quality.result_signals = compute_result_signals(records, header, truncated=truncated)
    quality = refine_verdict(quality)
    envelope['cypher_quality'] = quality.to_dict()

    return envelope


def format_error(query: str, error: CypherError) -> dict[str, Any]:
    """Format a CypherError into the standard envelope structure."""
    outcome = 'error' if error.stage == 'execution' else 'rejected'
    return {
        'query': query,
        'type': 'error',
        'error': {
            'stage': error.stage,
            'reason': error.reason,
            'found': error.found,
            'explanation': error.explanation,
            'suggestion': error.suggestion,
            'doc_hint': error.doc_hint,
        },
        'execution_ms': 0,
        'cypher_quality': {
            'outcome': outcome,
            'verdict': outcome,
        },
    }
