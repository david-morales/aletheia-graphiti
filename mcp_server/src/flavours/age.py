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

# A standalone `id` token used as a variable (NOT n.id property access, NOT id() function).
# Applied to the string/comment-masked view so mentions inside literals never trip it.
_ID_VAR_RE = re.compile(r"(?<![\w.])id(?![\w.(])", re.IGNORECASE)

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


def _unique_rel_var(query: str) -> str:
    """A relationship variable name guaranteed not to collide with one already in the query."""
    for name in ("r", "rt", "rel", "_r0", "_r1", "_r2"):
        if not re.search(rf"\b{name}\b", query, re.IGNORECASE):
            return name
    i = 0
    while re.search(rf"\b_r{i}\b", query, re.IGNORECASE):
        i += 1
    return f"_r{i}"


def _rewrite_rel_disjunction(query: str) -> tuple[str, bool]:
    """Conservatively rewrite a single [:A|B|C] disjunction to a generic edge + type filter.

    Fires ONLY on the unambiguous single-disjunction case (bails when there are 0 or >1,
    leaving the query for the execution-error hint). Uses a collision-free relationship
    variable, and attaches ``WHERE type(<var>) IN [...]`` to the clause that CONTAINS the
    disjunction — found by scanning forward from the disjunction to the next clause keyword,
    not from the start of the query (so a preceding WHERE/WITH cannot capture the filter).
    An existing WHERE for that clause is AND-ed with its condition parenthesized (so a
    top-level OR is not silently defeated).
    """
    matches = list(_REL_DISJUNCTION_RE.finditer(query))
    if len(matches) != 1:
        return query, False
    m = matches[0]
    var = m.group("var") if m.group("var") else _unique_rel_var(query)
    types = [t.strip().strip("`") for t in m.group("types").split("|")]
    type_list = ", ".join(f"'{t}'" for t in types)
    filter_clause = f"type({var}) IN [{type_list}]"

    # Replace the pattern with a generic edge carrying the variable.
    replacement = f"[{var}]"
    rewritten = query[: m.start()] + replacement + query[m.end():]
    scan_from = m.start() + len(replacement)

    nk = _CLAUSE_KW_RE.search(rewritten, scan_from)
    if nk and nk.group(1).upper() == "WHERE":
        # This clause has a WHERE — AND our filter in, parenthesizing the existing condition
        # (which ends at the next clause keyword) so precedence with a top-level OR is safe.
        where_end = nk.end()
        after = _CLAUSE_KW_RE.search(rewritten, where_end)
        cond_end = after.start() if after else len(rewritten)
        existing = rewritten[where_end:cond_end].strip()
        new_query = (
            rewritten[:where_end]
            + f" {filter_clause} AND ({existing}) "
            + rewritten[cond_end:]
        )
    elif nk:
        # No WHERE for this clause; insert one before the next clause keyword.
        i = nk.start()
        new_query = rewritten[:i] + f"WHERE {filter_clause} " + rewritten[i:]
    else:
        # No following clause keyword — append.
        new_query = f"{rewritten} WHERE {filter_clause}"
    return new_query, True


class AgeFlavour(BaseFlavour):
    """AGE flavour — safe auto-fix subset, id-reject-with-hint, nested-map attribute_keys."""

    name = "age"
    dialect_id = "age-opencypher"
    dialect_reference = _AGE_DIALECT

    def check_dialect(self, query: str) -> CypherError | None:
        # Only a variable named `id` is rejected (renaming a variable is semantically risky, so
        # we reject-with-hint rather than auto-rewrite). [:A|B|C] is handled by auto_fix.
        # Match on the string/comment-masked view so `id` inside a literal or comment never
        # trips this; _ID_VAR_RE excludes property access (n.id) and the id() function.
        code_only = _strip_non_code_spans(query)
        if _ID_VAR_RE.search(code_only):
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
