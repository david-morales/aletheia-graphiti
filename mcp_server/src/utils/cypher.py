"""Data models for the Cypher validation pipeline and result formatting."""
from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class SanitizedQuery:
    """Result of successful validation: the sanitized query and applied fixes."""

    query: str
    auto_fixes: list[str] = field(default_factory=list)


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

    # 4. Multi-word identifier quoting
    # Node labels
    new_query = _LABEL_RE.sub(r'`\1`', query)
    # Property keys
    new_query = _PROP_KEY_RE.sub(lambda m: f'`{m.group(1)}`', new_query)
    # Relationship types
    new_query = _REL_TYPE_RE.sub(lambda m: f'{m.group(1)}`{m.group(2)}`', new_query)
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
_PATTERN_COMPREHENSION_RE = re.compile(r'\[\s*\(.*?\|', re.DOTALL)
_EXISTS_SUBQUERY_RE = re.compile(r'\bEXISTS\s*\{', re.IGNORECASE)
_CALL_SUBQUERY_RE = re.compile(r'\bCALL\s*\{', re.IGNORECASE)
_MAP_PROJECTION_RE = re.compile(r'\w+\s*\{\s*\.\w+')

# Auto-fix patterns
_DATE_WRAPPER_RE = re.compile(
    r'\b(?:date|datetime|localDateTime)\s*\(\s*([\'"][^\'"]+[\'"])\s*\)',
    re.IGNORECASE,
)
_LOWER_RE = re.compile(r'\blower\s*\(', re.IGNORECASE)
_UPPER_RE = re.compile(r'\bupper\s*\(', re.IGNORECASE)
_PROFILE_EXPLAIN_RE = re.compile(r'^\s*(PROFILE|EXPLAIN)\s+', re.IGNORECASE)


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

    # 2. Pattern comprehensions  [(n)-[:REL]->(m) | m.prop]
    m = _PATTERN_COMPREHENSION_RE.search(query)
    if m:
        return CypherError(
            stage='falkordb_dialect',
            reason='pattern_comprehension_unsupported',
            found=m.group(0),
            explanation='Pattern comprehensions are not supported in FalkorDB.',
            suggestion='Use WITH + MATCH + collect() to achieve the same result',
            doc_hint='Rewrite as: MATCH (n)-[:REL]->(m) WITH n, collect(m.prop) AS props',
        )

    # 3. EXISTS {} subqueries (NOT EXISTS((n)--()) which is valid)
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

    # 4. CALL {} subqueries
    m = _CALL_SUBQUERY_RE.search(query)
    if m:
        return CypherError(
            stage='falkordb_dialect',
            reason='call_subquery_unsupported',
            found=m.group(0),
            explanation='CALL {} subqueries are not supported in FalkorDB.',
            suggestion='Use WITH + OPTIONAL MATCH to achieve similar results',
            doc_hint='Rewrite CALL {} blocks as sequential WITH + MATCH clauses',
        )

    # 5. Map projections  n {.name, .date}
    m = _MAP_PROJECTION_RE.search(query)
    if m:
        return CypherError(
            stage='falkordb_dialect',
            reason='map_projection_unsupported',
            found=m.group(0),
            explanation='Map projections are not supported in FalkorDB.',
            suggestion='Return properties individually: RETURN n.name, n.date',
            doc_hint='Use explicit property access instead of map projection syntax',
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

    return query, fixes


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


def _inject_safety(query: str, limit: int = DEFAULT_LIMIT) -> tuple[str, list[str]]:
    """Stage 4: Inject safety measures.

    - Appends LIMIT (N+1) if no LIMIT clause present (for truncation detection)
    - CALL queries are exempt from LIMIT injection
    """
    fixes: list[str] = []
    query_upper = query.upper().strip()

    if query_upper.startswith('CALL '):
        return query, fixes

    if not re.search(r'\bLIMIT\b', query, re.IGNORECASE):
        query = f'{query.rstrip().rstrip(";")} LIMIT {limit + 1}'
        fixes.append(f'Injected LIMIT {limit}')

    return query, fixes


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
    query, fixes_4 = _inject_safety(query, limit=limit)

    return SanitizedQuery(
        query=query,
        auto_fixes=fixes_1 + fixes_2 + fixes_4,
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

    return envelope


def format_error(query: str, error: CypherError) -> dict[str, Any]:
    """Format a CypherError into the standard envelope structure."""
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
    }
