"""Data models for the Cypher validation pipeline and result formatting."""
from __future__ import annotations

import html
import re
from dataclasses import dataclass, field


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
