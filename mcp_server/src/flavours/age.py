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
        return super().classify_execution_error(message, query)

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
