"""AGE flavour — Apache AGE (openCypher over PostgreSQL).

Dialect data (reference text + error patterns) is carried over from the aletheia GraphRAG
scaffold's age.py; the method shapes match the fork's CypherError protocol. AGE has no
ro_query, so execute_graph_query (inherited from BaseFlavour) relies on the pipeline's
whitelist as the read-only guard.
"""
from __future__ import annotations

import re
from typing import Any

from flavours.base import BaseFlavour, RESERVED_KEYS  # noqa: F401
from utils.cypher import CypherError

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

# String literals + comments, matched for LENGTH-PRESERVING masking so structural regexes
# (clause keywords, bracket patterns) never trip on text inside quotes/comments while byte
# positions stay aligned with the raw query for splicing.
_LITERAL_OR_COMMENT_RE = re.compile(
    r'"(?:[^"\\]|\\.)*"'      # double-quoted string
    r"|'(?:[^'\\]|\\.)*'"     # single-quoted string
    r"|//[^\n]*"              # line comment
    r"|/\*[\s\S]*?\*/"        # block comment
)


def _mask_literals(query: str) -> str:
    """Same-length copy of query with string-literal / comment chars replaced by 'x'.

    Structural analysis (finding disjunctions, clause keywords, WHERE-condition ends) runs on
    the masked view so a keyword or bracket INSIDE a string literal can never misfire; byte
    positions are preserved, so edits are spliced from the RAW query at masked-derived offsets.
    """
    return _LITERAL_OR_COMMENT_RE.sub(lambda m: "x" * (m.end() - m.start()), query)


# `id` as a node/relationship PATTERN variable — `(id` / `[id` not followed by `.`(property)
# or `(`(function). Brace maps `{id:` are intentionally NOT matched (map key, not a variable).
_ID_PATTERN_VAR_RE = re.compile(r"[\(\[]\s*id\b(?!\s*[.(])", re.IGNORECASE)
# `id` as an alias — `AS id` (an aliased column is a variable named id).
_ID_ALIAS_RE = re.compile(r"\bAS\s+id\b", re.IGNORECASE)

# Relationship-type disjunction inside a [...] pattern: [:A|B|C] or [r:A|B|C].
_REL_DISJUNCTION_RE = re.compile(
    r"\[\s*(?P<var>[A-Za-z_]\w*)?\s*:\s*"
    r"(?P<types>`?\w[\w`]*`?(?:\s*\|\s*`?\w[\w`]*`?)+)\s*\]"
)

# Clause keywords that bound a MATCH pattern (used to find where to attach the type filter).
_CLAUSE_KW_RE = re.compile(
    r"\b(OPTIONAL\s+MATCH|MATCH|WHERE|WITH|RETURN|ORDER\s+BY|SKIP|LIMIT|UNION"
    r"|CREATE|MERGE|SET|DELETE|REMOVE|CALL|UNWIND|FOREACH)\b",
    re.IGNORECASE,
)


def _unique_rel_var(masked: str) -> str:
    """A relationship variable name guaranteed not to collide with one already in the query.

    Scans the MASKED view so a token that appears only inside a string literal is not treated
    as a collision.
    """
    for name in ("r", "rt", "rel", "_r0", "_r1", "_r2"):
        if not re.search(rf"\b{name}\b", masked, re.IGNORECASE):
            return name
    i = 0
    while re.search(rf"\b_r{i}\b", masked, re.IGNORECASE):
        i += 1
    return f"_r{i}"


def _rewrite_rel_disjunction(query: str) -> tuple[str, bool]:
    """Conservatively rewrite a single [:A|B|C] disjunction to a generic edge + type filter.

    Fires ONLY on the unambiguous single-disjunction case (bails when there are 0 or >1,
    leaving the query for the execution-error hint). Uses a collision-free relationship
    variable, and attaches ``WHERE type(<var>) IN [...]`` to the clause that CONTAINS the
    disjunction — the next clause keyword is found by scanning forward FROM the disjunction
    (so a preceding WHERE/WITH cannot capture the filter). An existing WHERE for that clause is
    AND-ed with its condition parenthesized (so a top-level OR is not silently defeated). All
    position-finding runs on a length-preserving masked view; the result is spliced from the
    RAW query at those offsets, so string-literal contents never affect the structural edit.
    """
    masked = _mask_literals(query)
    matches = list(_REL_DISJUNCTION_RE.finditer(masked))
    if len(matches) != 1:
        return query, False
    m = matches[0]
    disj_start, disj_end = m.start(), m.end()
    var = m.group("var") if m.group("var") else _unique_rel_var(masked)
    types = [t.strip().strip("`") for t in m.group("types").split("|")]
    type_list = ", ".join(f"'{t}'" for t in types)
    filter_clause = f"type({var}) IN [{type_list}]"
    edge = f"[{var}]"

    nk = _CLAUSE_KW_RE.search(masked, disj_end)
    if nk and nk.group(1).upper() == "WHERE":
        # This clause has a WHERE — AND our filter in, parenthesizing the existing condition
        # (which ends at the next clause keyword) so precedence with a top-level OR is safe.
        where_end = nk.end()
        after = _CLAUSE_KW_RE.search(masked, where_end)
        cond_end = after.start() if after else len(query)
        existing = query[where_end:cond_end].strip()
        new_query = (
            query[:disj_start] + edge + query[disj_end:where_end]
            + f" {filter_clause} AND ({existing}) " + query[cond_end:]
        )
    elif nk:
        # No WHERE for this clause; insert one before the next clause keyword.
        i = nk.start()
        new_query = query[:disj_start] + edge + query[disj_end:i] + f"WHERE {filter_clause} " + query[i:]
    else:
        # No following clause keyword — append.
        new_query = query[:disj_start] + edge + query[disj_end:] + f" WHERE {filter_clause}"
    return new_query, True


class AgeFlavour(BaseFlavour):
    """AGE flavour — safe auto-fix subset, id-reject-with-hint, nested-map attribute_keys."""

    name = "age"
    dialect_id = "age-opencypher"
    dialect_reference = _AGE_DIALECT

    def check_dialect(self, query: str) -> CypherError | None:
        # Reject `id` used as a VARIABLE — a pattern variable ((id / [id) or an alias (AS id).
        # Renaming a variable is semantically risky, so we reject-with-hint rather than
        # auto-rewrite. We stay PRECISE (only high-confidence variable positions): property
        # access (n.id), the id() function, and map keys ({id: ...}) are allowed, and ambiguous
        # bare references fall through to the execution-error hint net. Matching runs on the
        # masked view so `id` inside a string literal / comment never trips this.
        masked = _mask_literals(query)
        if _ID_PATTERN_VAR_RE.search(masked) or _ID_ALIAS_RE.search(masked):
            return CypherError(
                stage="age_dialect",
                reason="reserved_id_variable",
                found="id",
                explanation="A variable named `id` collides with AGE's built-in id()/graphid "
                "and fails at execution ('column notation .id applied to type graphid').",
                suggestion="Rename the variable (e.g. `ident`, `x`) and retry.",
                doc_hint="AGE reserves id()/graphid; never name a variable `id`.",
            )
        return None

    def auto_fix(self, query: str) -> tuple[str, list[str]]:
        fixes: list[str] = []
        # Conservatively rewrite a single [:A|B|C] disjunction (no-op on 0 or >1 — those fall
        # through to the execution-error hint). LIMIT injection is the shared pipeline's job.
        new_query, rewrote = _rewrite_rel_disjunction(query)
        if rewrote:
            query = new_query
            fixes.append(
                "Rewrote relationship-type disjunction [:A|B|C] to a generic edge with "
                "WHERE type(r) IN [...] (AGE does not support disjunction)."
            )
        return query, fixes

    def classify_execution_error(self, message: str) -> CypherError:
        m = (message or "").lower()
        if "graphid" in m:
            return CypherError(
                stage="execution",
                reason="reserved_id_variable",
                found="id",
                explanation="A variable named `id` collides with AGE's built-in id()/graphid.",
                suggestion="Rename the variable (e.g. `ident`, `x`) and retry.",
                doc_hint="AGE reserves id()/graphid.",
            )
        if 'syntax error at or near "|"' in m or 'at or near "|"' in m:
            return CypherError(
                stage="execution",
                reason="reltype_disjunction_unsupported",
                found="|",
                explanation="AGE does not support relationship-type disjunction like [:A|B|C].",
                suggestion="Match a generic edge and filter: MATCH (a)-[r]->(b) "
                "WHERE type(r) IN ['A','B','C'].",
                doc_hint="AGE has no [:A|B|C]; use WHERE type(r) IN [...].",
            )
        return super().classify_execution_error(message)

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
