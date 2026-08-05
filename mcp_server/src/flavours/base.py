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


# Startup domain-profile probes. The FLAVOUR owns the query text (dialects disagree on label
# semantics and reserved words); domain_profile owns the parsing. Every variant MUST return the
# same column names, or domain_profile cannot read it:
#   entity_types -> (entity_type: list[str], cnt: int)
#   edge_types   -> (relationship_type: str, cnt: int)
#   sample_names -> (name: str)
#   time_range   -> (earliest, latest)
# The count column is aliased `cnt`, not `count`: `count` is a reserved word on Apache AGE.
_BASE_PROFILE_QUERIES: dict[str, str] = {
    "entity_types": (
        "MATCH (n:Entity) "
        "WHERE n.group_id = $group_id "
        "RETURN labels(n) AS entity_type, count(n) AS cnt "
        "ORDER BY cnt DESC"
    ),
    "edge_types": (
        "MATCH (s:Entity)-[r]->(t:Entity) "
        "WHERE s.group_id = $group_id AND t.group_id = $group_id "
        "RETURN type(r) AS relationship_type, count(r) AS cnt "
        "ORDER BY cnt DESC"
    ),
    "sample_names": (
        "MATCH (n:Entity) "
        "WHERE $label IN labels(n) AND n.group_id = $group_id "
        "RETURN n.name AS name "
        "LIMIT $limit"
    ),
    "time_range": (
        "MATCH (s:Entity)-[r]->(t:Entity) "
        "WHERE s.group_id = $group_id AND r.created_at IS NOT NULL "
        "RETURN min(r.created_at) AS earliest, max(r.created_at) AS latest"
    ),
}


_UUID_SAFE = frozenset("0123456789abcdefABCDEF-")


def _uuid_list_literal(uuids: list[str]) -> str:
    """Inline a sanitized, quoted uuid list. Inlined (not a $param) because AGE's
    param path is proven for scalars only (_AGE_PROFILE_QUERIES); values come from
    our own node query, sanitization is defense in depth."""
    safe = [u for u in uuids if u and set(u) <= _UUID_SAFE]
    return "[" + ", ".join(f'"{u}"' for u in safe) + "]"


@runtime_checkable
class Flavour(Protocol):
    name: str
    dialect_id: str
    dialect_reference: str   # full dialect teaching text (ADR-019 R5, surfaced via get_schema)
    dialect_summary: str     # short form for server instructions / tool descriptions (ADR-019 R1/R6)

    def check_dialect(self, query: str) -> CypherError | None: ...
    def auto_fix(self, query: str) -> tuple[str, list[str]]: ...
    def classify_execution_error(self, message: str, query: str | None = None) -> CypherError: ...
    def profile_queries(self) -> dict[str, str]: ...
    def census_queries(self) -> dict[str, str]: ...
    def census_notes(self) -> list[str]: ...
    def ontology_queries(self) -> dict[str, str]: ...
    def subgraph_node_query(self) -> str: ...
    def subgraph_edge_query(self, uuids: list[str]) -> str: ...
    def node_sample_query(self) -> str: ...
    def flatten_node_props(self, props: dict[str, Any]) -> dict[str, Any]: ...
    def property_accessor(self, prop: str) -> str: ...
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

    def classify_execution_error(self, message: str, query: str | None = None) -> CypherError:
        """Generic envelope. ``query`` is accepted for protocol parity and ignored here."""
        return CypherError(
            stage="execution",
            reason="execution_error",
            found="",
            explanation=message,
            suggestion="Check the query against the schema (get_schema) and retry.",
        )

    def profile_queries(self) -> dict[str, str]:
        """Cypher for the four startup domain-profile probes, keyed by probe name."""
        return dict(_BASE_PROFILE_QUERIES)

    def census_queries(self) -> dict[str, str]:
        """get_schema's structural censuses. `rel_patterns` carries a {rel_type}
        placeholder (str.format) — rel_type values come from the graph itself.

        Both variants MUST project the same aliases (`lbls`/`cnt`,
        `source_labels`/`target_labels`): get_schema parses either flavour's rows
        with one shared loop.

        A flavour whose `source_labels` is a HIERARCHY (AGE) additionally projects
        `source_leaf`/`target_leaf` — the single matchable endpoint label. Those
        columns are OPTIONAL: consumers prefer them when present and fall back to
        picking positionally from the label list when they are absent, which is
        what keeps this base variant's behaviour unchanged."""
        return {
            "label_counts": "MATCH (n) RETURN labels(n) AS lbls, count(n) AS cnt",
            "rel_patterns": (
                "MATCH (s)-[r:`{rel_type}`]->(t) "
                "RETURN DISTINCT labels(s) AS source_labels, labels(t) AS target_labels LIMIT 20"
            ),
        }

    def census_notes(self) -> list[str]:
        """Caveats about how to READ this backend's census, announced as data
        (ADR-019 R6) — get_schema appends them to `analysis_notes`.

        Empty here: on openCypher/FalkorDB every censused label is matchable with
        `(n:Label)`, so the schema needs no reading instructions."""
        return []

    def ontology_queries(self) -> dict[str, str]:
        """Read path for the three ontology tiers, keyed by tier.

        `class_context` serves get_ontology_documentation and explore_ontology
        (the full projection); `structure` serves get_ontology_structure (the
        lightweight map tier, whose shape is frozen — no uuid, no
        properties/identity); `relates` serves the edge-derived relationships
        both full tiers union in.

        Both variants MUST project the same aliases — `_ontology_full_entry`,
        `_combine_relationship_entries` and the structure loop are shared across
        flavours, and they read rows by key:
            class_context -> uuid, name, ontology_type, inherits_from, summary,
                             alt_labels, source_entity, target_entity, examples,
                             properties, identity
            structure     -> the same, minus uuid / properties / identity
            relates       -> source, name, fact, target
            summaries     -> name, summary  (domain_profile's description probe)

        These texts are FalkorDB-shaped: every descriptive ontology field is a
        TOP-LEVEL node property, and relationships are RELATES_TO edges whose
        `name` property carries the real relation name. Apache AGE stores both
        differently (nested `attributes` map; the relation name becomes the edge
        LABEL), which is why the text lives behind the flavour at all.

        `relates` deliberately does NOT filter SUBCLASS_OF. Hierarchy edges are
        part of this row set on every flavour, and `_combine_relationship_entries`
        drops them downstream — ONE filter shared by both arms. Moving that
        decision into a flavour would make the two arms disagree about what the
        query returns, which the shared parsing cannot survive.
        """
        return {
            # `properties` (a JSON string) and `identity` (a bool) may be
            # null/absent on graphs built before v1.0.3 — the parsing defaults.
            "class_context": (
                "MATCH (n:OntologyClass) "
                "RETURN n.uuid AS uuid, "
                "n.name AS name, "
                "n.ontology_type AS ontology_type, "
                "n.inherits_from AS inherits_from, "
                "n.summary AS summary, "
                "n.alt_labels AS alt_labels, "
                "n.source_entity AS source_entity, "
                "n.target_entity AS target_entity, "
                "n.examples AS examples, "
                "n.properties AS properties, "
                "n.identity AS identity"
            ),
            "structure": (
                "MATCH (n:OntologyClass) "
                "RETURN n.name AS name, "
                "n.ontology_type AS ontology_type, "
                "n.inherits_from AS inherits_from, "
                "n.summary AS summary, "
                "n.alt_labels AS alt_labels, "
                "n.source_entity AS source_entity, "
                "n.target_entity AS target_entity, "
                "n.examples AS examples"
            ),
            # Ontology families that model relationships as owl:ObjectProperty
            # store them as edges between OntologyClass nodes (no
            # relationship_class nodes at all). The edges carry `name`
            # (e.g. ES_DETENIDO) and `fact`
            # (e.g. "Persona ES_DETENIDO Detencion: <prose>").
            "relates": (
                "MATCH (a:OntologyClass)-[r:RELATES_TO]->(b:OntologyClass) "
                "RETURN a.name AS source, r.name AS name, r.fact AS fact, b.name AS target"
            ),
            # domain_profile's description probe. Scoped by `:Entity` here because
            # a FalkorDB ontology vertex carries BOTH labels and this is the text
            # that arm has always issued — kept byte-identical. AGE's ontology
            # vertices carry only the leaf `OntologyClass`, so it overrides.
            "summaries": (
                "MATCH (n:Entity) "
                "WHERE n.summary IS NOT NULL "
                "RETURN n.name AS name, n.summary AS summary"
            ),
        }

    def subgraph_node_query(self) -> str:
        """Node sample for the UI Knowledge view. Columns are the wire contract:
        uuid, name, labels (the hierarchy), created_at, summary, group_id."""
        return (
            "MATCH (n:Entity) "
            "RETURN n.uuid AS uuid, n.name AS name, labels(n) AS labels, "
            "n.created_at AS created_at, n.summary AS summary, n.group_id AS group_id "
            "LIMIT $limit"
        )

    def subgraph_edge_query(self, uuids: list[str]) -> str:
        """Edges among the sampled nodes. Columns: uuid, name, fact,
        source_node_uuid, target_node_uuid, created_at."""
        lit = _uuid_list_literal(uuids)
        return (
            f"MATCH (s:Entity)-[r]->(t:Entity) "
            f"WHERE s.uuid IN {lit} AND t.uuid IN {lit} "
            "RETURN r.uuid AS uuid, type(r) AS name, r.fact AS fact, "
            "s.uuid AS source_node_uuid, t.uuid AS target_node_uuid, "
            "r.created_at AS created_at LIMIT $limit"
        )

    def node_sample_query(self) -> str:
        """Per-label node sample for the profiler. `{label}` / `{limit}` (str.format).

        The projection MUST land in a column the caller can read back as `n`: AGE
        names an unaliased variable projection `col0`, so a flavour that needs an
        explicit alias says so here."""
        return 'MATCH (n:`{label}`) RETURN n LIMIT {limit}'

    def flatten_node_props(self, props: dict[str, Any]) -> dict[str, Any]:
        """Normalize one sampled node's properties to a flat domain-field mapping.

        Identity here: on openCypher/FalkorDB every domain field is already a
        top-level property. A flavour that nests them (AGE's `attributes` agtype
        map) merges them up — see AgeFlavour for the collision rule."""
        return props

    def property_accessor(self, prop: str) -> str:
        """Cypher expression reading `prop` off the node bound to `n`.

        Pairs with :meth:`flatten_node_props`: whatever that surfaces as a domain
        field, this must be able to probe on a full scan, or the scan silently
        reports 0 rows and overwrites the sample-based coverage with 0.0."""
        return f'n.`{prop}`'

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
