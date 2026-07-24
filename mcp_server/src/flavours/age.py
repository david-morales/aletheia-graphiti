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

# Relationship-type disjunction inside a [...] pattern: [:A|B|C] or [r:A|B|C].
_REL_DISJUNCTION_RE = re.compile(
    r"\[\s*(?P<var>[A-Za-z_]\w*)?\s*:\s*"
    r"(?P<types>`?\w[\w`]*`?(?:\s*\|\s*`?\w[\w`]*`?)+)\s*\]"
)


def _rewrite_rel_disjunction(query: str) -> tuple[str, bool]:
    """Rewrite the FIRST [:A|B|C] disjunction to a generic edge + WHERE type(r) IN [...].

    Introduces a relationship variable `r` when absent, and appends the type filter to the
    query's WHERE (or inserts a new WHERE before RETURN/WITH). Only the first disjunction is
    handled per call; the caller loops until stable to cover multiple disjunctions.
    """
    m = _REL_DISJUNCTION_RE.search(query)
    if not m:
        return query, False
    var = m.group("var") or "r"
    types = [t.strip().strip("`") for t in m.group("types").split("|")]
    type_list = ", ".join(f"'{t}'" for t in types)
    # Replace the pattern with a generic edge carrying the variable.
    replacement = f"[{var}]"
    new_query = query[: m.start()] + replacement + query[m.end():]
    filter_clause = f"type({var}) IN [{type_list}]"
    # Attach the filter: prefer an existing WHERE; else insert before RETURN (or WITH).
    if re.search(r"\bWHERE\b", new_query, re.IGNORECASE):
        new_query = re.sub(
            r"\bWHERE\b", f"WHERE {filter_clause} AND", new_query, count=1, flags=re.IGNORECASE
        )
    else:
        insert = re.search(r"\b(RETURN|WITH)\b", new_query, re.IGNORECASE)
        if insert:
            i = insert.start()
            new_query = new_query[:i] + f"WHERE {filter_clause} " + new_query[i:]
        else:
            new_query = f"{new_query} WHERE {filter_clause}"
    return new_query, True


class AgeFlavour(BaseFlavour):
    """AGE flavour — safe auto-fix subset, id-reject-with-hint, nested-map attribute_keys."""

    name = "age"
    dialect_id = "age-opencypher"
    dialect_reference = _AGE_DIALECT

    def check_dialect(self, query: str) -> CypherError | None:
        # Only the `id`-variable is rejected (renaming a variable is semantically risky, so we
        # reject-with-hint rather than auto-rewrite). [:A|B|C] is handled by auto_fix.
        # Match a bare `id` used as a pattern variable (`(id`/`[id`) or a bare RETURN item
        # (RETURN ... id, not followed by a `.` — i.e. not the property access n.id).
        if re.search(r"[\(\[]\s*id\b", query, re.IGNORECASE) or re.search(
            r"\breturn\b[^,]*\bid\b(?!\s*\.)", query, re.IGNORECASE
        ):
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
        # Rewrite every [:A|B|C] disjunction (loop until stable — one per pass).
        rewrote_any = False
        for _ in range(8):  # bound the loop
            query, rewrote = _rewrite_rel_disjunction(query)
            if not rewrote:
                break
            rewrote_any = True
        if rewrote_any:
            fixes.append(
                "Rewrote relationship-type disjunction [:A|B|C] to a generic edge with "
                "WHERE type(r) IN [...] (AGE does not support disjunction)."
            )
        # LIMIT injection is handled by the shared _inject_safety stage in validate_and_sanitize.
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
