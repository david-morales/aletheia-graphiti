"""Backend flavour protocol + the generic (openCypher / Neo4j) flavour.

A Flavour owns everything backend-specific about running a read-only Cypher query:
the dialect reference text (announced via get_schema / instructions), the pre-flight
dialect reject, the safe auto-fix, execution-error classification, per-label attribute-key
extraction, and query execution normalized to (records, header). The generic BaseFlavour
does the backend-agnostic minimum and is used as-is for Neo4j and as the parent of the
FalkorDB/AGE flavours.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from utils.cypher import CypherError

# Property keys never surfaced as domain attribute_keys (internal Graphiti bookkeeping).
RESERVED_KEYS: frozenset[str] = frozenset(
    {"uuid", "name", "group_id", "summary", "created_at", "name_embedding", "labels"}
)


@runtime_checkable
class Flavour(Protocol):
    name: str
    dialect_id: str
    dialect_reference: str   # full dialect teaching text (ADR-019 R5, surfaced via get_schema)
    dialect_summary: str     # short form for server instructions / tool descriptions (ADR-019 R1/R6)

    def check_dialect(self, query: str) -> CypherError | None: ...
    def auto_fix(self, query: str) -> tuple[str, list[str]]: ...
    def classify_execution_error(self, message: str) -> CypherError: ...
    async def attribute_keys(self, driver: Any, label: str, sample: int = 50) -> list[str]: ...
    async def execute_graph_query(
        self, driver: Any, query: str
    ) -> tuple[list[dict], list[str]]: ...


class BaseFlavour:
    """Generic openCypher flavour — the Neo4j path and the shared parent."""

    name = "opencypher"
    dialect_id = "opencypher"
    dialect_reference = ""
    dialect_summary = ""

    def check_dialect(self, query: str) -> CypherError | None:
        return None

    def auto_fix(self, query: str) -> tuple[str, list[str]]:
        return query, []

    def classify_execution_error(self, message: str) -> CypherError:
        return CypherError(
            stage="execution",
            reason="execution_error",
            found="",
            explanation=message,
            suggestion="Check the query against the schema (get_schema) and retry.",
        )

    async def attribute_keys(self, driver: Any, label: str, sample: int = 50) -> list[str]:
        """Top-level property keys for a label, minus reserved bookkeeping keys."""
        records, _, _ = await driver.execute_query(
            f"MATCH (n:`{label}`) WITH keys(n) AS k LIMIT {int(sample)} "
            f"UNWIND k AS key RETURN DISTINCT key"
        )
        keys = [
            r["key"]
            for r in records
            if r.get("key")
            and r["key"] not in RESERVED_KEYS
            and "embedding" not in r["key"].lower()
        ]
        return sorted(keys)

    async def execute_graph_query(
        self, driver: Any, query: str
    ) -> tuple[list[dict], list[str]]:
        """Execute via the public driver API; returns (records: list[dict], header: list[str]).

        The whitelist guard in validate_and_sanitize is the read-only enforcement for backends
        without a DB-enforced read-only mode. Normalizes across driver return shapes:
        FalkorDB/AGE return ``(records, header, summary)`` with dict rows; Neo4j returns a neo4j
        ``EagerResult`` (``records``/``keys``/``summary`` attributes) with Record rows.
        """
        result = await driver.execute_query(query)
        if hasattr(result, "records") and hasattr(result, "keys"):
            # neo4j EagerResult: Record rows + explicit keys.
            records = [dict(r) for r in result.records]
            header = list(result.keys)
            return records, header
        records, header, _ = result
        header = list(header) if header else (list(records[0].keys()) if records else [])
        return list(records), header
