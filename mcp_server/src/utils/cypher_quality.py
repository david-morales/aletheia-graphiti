"""Schema-aware Cypher quality assessment.

Diffs AST-extracted Cypher elements against a cached graph schema to
detect hallucinated labels, wrong-direction relationships, and
misplaced properties.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any

from utils.cypher_extractor import PropertyAccess, RelPattern, extract_elements

# Internal Graphiti properties that are always valid on any node.
_INTERNAL_PROPERTIES: frozenset[str] = frozenset({
    'name',
    'name_embedding',
    'summary',
    'summary_embedding',
    'group_id',
    'created_at',
    'updated_at',
    'uuid',
    'expired_at',
})


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class UnknownProp:
    property_name: str
    on_label: str


@dataclass
class WrongLabelProp:
    property_name: str
    on_label: str
    exists_on: list[str]


@dataclass
class LabelMatch:
    found: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    suggestions: dict[str, str] = field(default_factory=dict)


@dataclass
class RelMatch:
    found: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    wrong_direction: list[str] = field(default_factory=list)


@dataclass
class PropMatch:
    found: list[str] = field(default_factory=list)
    unknown: list[UnknownProp] = field(default_factory=list)
    wrong_label: list[WrongLabelProp] = field(default_factory=list)


@dataclass
class SchemaMatch:
    labels: LabelMatch = field(default_factory=LabelMatch)
    relationships: RelMatch = field(default_factory=RelMatch)
    properties: PropMatch = field(default_factory=PropMatch)


@dataclass
class ResultSignals:
    row_count: int = 0
    null_ratio: float = 0.0
    truncated: bool = False


@dataclass
class CypherQuality:
    outcome: str  # "rejected" | "error" | "ok" | "suspect"
    verdict: str  # "success" | "schema_mismatch" | "empty_legit" | "degraded" | "parse_failed"
    schema_match: SchemaMatch | None = None
    result_signals: ResultSignals | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON inclusion in result envelope."""
        d: dict[str, Any] = {
            'outcome': self.outcome,
            'verdict': self.verdict,
        }

        if self.schema_match is not None:
            sm = self.schema_match
            d['schema_match'] = {
                'labels': {
                    'found': sm.labels.found,
                    'unknown': sm.labels.unknown,
                    'suggestions': sm.labels.suggestions,
                },
                'relationships': {
                    'found': sm.relationships.found,
                    'unknown': sm.relationships.unknown,
                    'wrong_direction': sm.relationships.wrong_direction,
                },
                'properties': {
                    'found': sm.properties.found,
                    'unknown': [
                        {'property': u.property_name, 'on_label': u.on_label}
                        for u in sm.properties.unknown
                    ],
                    'wrong_label': [
                        {
                            'property': w.property_name,
                            'on_label': w.on_label,
                            'exists_on': w.exists_on,
                        }
                        for w in sm.properties.wrong_label
                    ],
                },
            }
        else:
            d['schema_match'] = None

        if self.result_signals is not None:
            d['result_signals'] = {
                'row_count': self.result_signals.row_count,
                'null_ratio': self.result_signals.null_ratio,
                'truncated': self.result_signals.truncated,
            }
        else:
            d['result_signals'] = None

        return d


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_labels(
    labels: list[str],
    known_labels: list[str],
) -> LabelMatch:
    """Check extracted labels against schema node labels."""
    match = LabelMatch()
    for label in labels:
        if label in known_labels:
            match.found.append(label)
        else:
            match.unknown.append(label)
            close = difflib.get_close_matches(label, known_labels, n=1, cutoff=0.6)
            if close:
                match.suggestions[label] = close[0]
    return match


def _validate_relationships(
    rel_types: list[str],
    rel_patterns: list[RelPattern],
    var_labels: dict[str, str],
    schema: dict[str, Any],
) -> RelMatch:
    """Check relationship types and directions against schema."""
    known_rels = list(schema.get('relationship_types', {}).keys())
    match = RelMatch()

    # Validate rel types
    seen: set[str] = set()
    for rt in rel_types:
        if rt in seen:
            continue
        seen.add(rt)
        if rt in known_rels:
            match.found.append(rt)
        else:
            match.unknown.append(rt)

    # Validate directions
    seen_patterns: set[str] = set()
    for rp in rel_patterns:
        if rp.rel_type is None or rp.rel_type not in known_rels:
            continue
        if rp.direction == 'undirected':
            continue

        # Resolve variable names to labels
        src_label = var_labels.get(rp.source_var) if rp.source_var else None
        tgt_label = var_labels.get(rp.target_var) if rp.target_var else None

        if not src_label or not tgt_label:
            continue

        # Determine effective direction
        if rp.direction == 'right':
            effective_src, effective_tgt = src_label, tgt_label
        else:  # left: arrow goes target->source, effective is reversed
            effective_src, effective_tgt = tgt_label, src_label

        schema_patterns = schema['relationship_types'][rp.rel_type].get('patterns', [])
        forward_match = any(
            p[0] == effective_src and p[1] == effective_tgt for p in schema_patterns
        )
        reverse_match = any(
            p[0] == effective_tgt and p[1] == effective_src for p in schema_patterns
        )

        pattern_key = f'{rp.rel_type}:{effective_src}->{effective_tgt}'
        if pattern_key in seen_patterns:
            continue
        seen_patterns.add(pattern_key)

        if not forward_match and reverse_match:
            if rp.rel_type not in match.wrong_direction:
                match.wrong_direction.append(rp.rel_type)

    return match


def _addressable_properties(info: dict[str, Any]) -> list[str]:
    """Every property name a query may legitimately name on one label.

    The union of the two key sets get_schema announces per label:

    * ``properties`` — the TOP-LEVEL keys, i.e. what ``keys(n)`` returns. On a
      flat backend (FalkorDB / openCypher) that is already the whole domain.
    * ``attribute_keys`` — the canonical ADR-019 R5 domain-queryable keys. On a
      backend that NESTS its domain fields (Apache AGE) these live one level
      down, inside the announced ``attribute_container`` map, and appear in
      ``properties`` nowhere at all.

    Validating against ``properties`` alone therefore called every correct
    nested-path query a schema mismatch on the nesting backend (BLK-1). Both
    key sets are produced by the FLAVOUR; this function only unions what the
    schema already announces, so no backend is named here.

    Used for the CROSS-LABEL index only. Whether a key is reachable on the label
    the query actually named is a finer question — see
    :func:`_container_only_keys`: a union is container-blind, and accepting
    ``n.edad`` where the key exists only inside the container would swap a false
    alarm for a silently null column.
    """
    return list(info.get('properties') or []) + list(info.get('attribute_keys') or [])


def _container_only_keys(info: dict[str, Any]) -> set[str]:
    """Keys reachable ONLY through the container on this label.

    A key that is in ``attribute_keys`` and NOT in the top-level ``properties``
    has no top-level existence: ``n.<key>`` parses, runs, and returns null for
    every row. The schema announces both sets, so it already carries enough to
    say so — no backend knowledge is needed here.
    """
    top_level = set(info.get('properties') or [])
    return {k for k in (info.get('attribute_keys') or []) if k not in top_level}


def _label_was_probed(info: dict[str, Any]) -> bool:
    """Did get_schema actually learn this label's attributes?

    Two ways it did not, both reachable without the call failing: the top-level
    probe degraded (get_schema reports the entry ``sampled: False`` rather than
    taking the whole schema down), or the attribute probe swallowed its own
    exception and returned nothing (``AgeFlavour.attribute_keys``). Either way
    the connector has no evidence about this label, and no-evidence must not be
    spent as an accusation — a `schema_mismatch` is sticky (`refine_verdict`
    never downgrades it), so a wrong one outlives the query that caused it.

    A schema entry that omits ``sampled`` entirely is treated as probed:
    get_schema always sets it, so absence means a caller that is not modelling
    the degraded case rather than a degraded label.
    """
    return bool(info.get('sampled', True)) and bool(info.get('attribute_keys'))


def _container_prefixed_accesses(query: str, container: str) -> set[tuple[str, str]]:
    """``(variable, key)`` pairs the query writes as ``<var>.<container>.<key>``.

    The AST extractor flattens a nested path: ``n.attributes.edad`` arrives as
    two independent accesses, ``attributes`` and ``edad``, with no record that
    one was written through the other. That is exactly the fact needed to tell a
    correct nested read from a bare one, so it is recovered from the query text
    here rather than by widening ``PropertyAccess`` — whose identity
    (``__eq__``/``__hash__`` over variable + name) other call sites depend on.

    Backticks are optional around either identifier, matching how the dialects
    quote them.
    """
    ident = r'`?([A-Za-z_][A-Za-z0-9_]*)`?'
    pattern = rf'{ident}\s*\.\s*`?{re.escape(container)}`?\s*\.\s*{ident}'
    return {(m.group(1), m.group(2)) for m in re.finditer(pattern, query)}


def _validate_properties(
    properties: list[PropertyAccess],
    var_labels: dict[str, str],
    schema: dict[str, Any],
    container_paths: set[tuple[str, str]] | None = None,
) -> PropMatch:
    """Check property accesses against schema node properties.

    ``container_paths`` carries which accesses were written through the
    announced container (see :func:`_container_prefixed_accesses`); empty on a
    flat backend, where nothing is path-restricted.
    """
    container_paths = container_paths or set()
    node_labels = schema.get('node_labels', {})
    known_label_set = set(node_labels.keys())
    match = PropMatch()

    # The map a nesting backend keeps its domain fields in, announced as DATA by
    # the flavour (ADR-019 R6) rather than hardcoded here. `n.attributes.edad`
    # parses to TWO property accesses — the container and the field — so without
    # knowing the container's name the transport half reads as a hallucinated
    # property. None on flat backends, where the nested form really is wrong.
    container = schema.get('attribute_container')

    # Build a reverse index: property_name -> list of labels that have it
    prop_to_labels: dict[str, list[str]] = {}
    for label, info in node_labels.items():
        for prop in _addressable_properties(info):
            prop_to_labels.setdefault(prop, []).append(label)

    seen: set[tuple[str, str]] = set()
    for pa in properties:
        label = var_labels.get(pa.variable)

        # No label binding -> skip
        if label is None:
            continue

        # Unknown label -> skip (label validation catches this)
        if label not in known_label_set:
            continue

        key = (pa.property_name, label)
        if key in seen:
            continue
        seen.add(key)

        # Internal properties are always valid
        if pa.property_name in _INTERNAL_PROPERTIES:
            if pa.property_name not in match.found:
                match.found.append(pa.property_name)
            continue

        # The container is transport, not a domain field: on a nesting backend
        # EVERY entity vertex carries it by construction, so its validity is a
        # backend fact and must not depend on whether one label's key sample
        # happened to observe it (a degraded probe reports `properties: []`).
        if container and pa.property_name == container:
            if pa.property_name not in match.found:
                match.found.append(pa.property_name)
            continue

        info = node_labels[label]
        prefixed = bool(container) and (pa.variable, pa.property_name) in container_paths

        # No evidence about this label's attributes -> no field-level verdict on
        # a nested access. The container name above is still checked, which is
        # the part that holds without a sample.
        if prefixed and not _label_was_probed(info):
            continue

        # A key that lives ONLY inside the container is unreachable at the top
        # level: `n.edad` runs and returns null for every row, which is the
        # silent wrong answer this module exists to catch. Reported as unknown
        # because that is what it is where the query looked — at the top level.
        if not prefixed and pa.property_name in _container_only_keys(info):
            match.unknown.append(
                UnknownProp(property_name=pa.property_name, on_label=label)
            )
            continue

        label_props = _addressable_properties(info)
        if pa.property_name in label_props:
            if pa.property_name not in match.found:
                match.found.append(pa.property_name)
        elif pa.property_name in prop_to_labels:
            match.wrong_label.append(
                WrongLabelProp(
                    property_name=pa.property_name,
                    on_label=label,
                    exists_on=prop_to_labels[pa.property_name],
                )
            )
        else:
            match.unknown.append(
                UnknownProp(property_name=pa.property_name, on_label=label)
            )

    return match


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_NULL_RATIO_THRESHOLD = 0.5


def compute_result_signals(
    records: list[dict[str, Any]],
    header: list[str],
    *,
    truncated: bool,
) -> ResultSignals:
    """Compute post-execution quality signals from query results."""
    row_count = len(records)
    total_cells = row_count * len(header)
    if total_cells == 0:
        null_ratio = 0.0
    else:
        null_count = sum(1 for row in records for col in header if row.get(col) is None)
        null_ratio = round(null_count / total_cells, 4)
    return ResultSignals(row_count=row_count, null_ratio=null_ratio, truncated=truncated)


def refine_verdict(quality: CypherQuality) -> CypherQuality:
    """Refine verdict using post-execution result signals.

    Rules (in priority order):
    1. schema_mismatch -> keep as-is (don't downgrade)
    2. parse_failed + successful execution (rows > 0) -> downgrade to success
       (our parser doesn't cover all valid Cypher — e.g. consecutive WITH clauses)
    3. result_signals is None -> keep as-is
    4. row_count == 0 and verdict == "success" -> change to "empty_legit", outcome stays "ok"
    5. null_ratio > threshold and verdict == "success" -> change to "degraded", outcome="suspect"
    6. Otherwise -> keep as-is
    """
    if quality.verdict == 'schema_mismatch':
        return quality
    if quality.verdict == 'parse_failed' and quality.result_signals is not None:
        if quality.result_signals.row_count > 0:
            quality.verdict = 'success'
            quality.outcome = 'ok'
            return quality
    if quality.result_signals is None:
        return quality
    if quality.result_signals.row_count == 0 and quality.verdict == 'success':
        quality.verdict = 'empty_legit'
        return quality
    if quality.result_signals.null_ratio > _NULL_RATIO_THRESHOLD and quality.verdict == 'success':
        quality.verdict = 'degraded'
        quality.outcome = 'suspect'
        return quality
    return quality


def assess_quality(query: str, *, schema: dict[str, Any] | None) -> CypherQuality:
    """Assess quality of a Cypher query against a graph schema."""
    elements = extract_elements(query)

    # Parse errors -> suspect / parse_failed
    if elements.parse_errors > 0:
        return CypherQuality(
            outcome='suspect',
            verdict='parse_failed',
            schema_match=SchemaMatch() if schema is not None else None,
        )

    # No schema -> can't validate
    if schema is None:
        return CypherQuality(outcome='ok', verdict='success')

    known_labels = list(schema.get('node_labels', {}).keys())

    labels = _validate_labels(elements.labels, known_labels)
    relationships = _validate_relationships(
        elements.rel_types,
        elements.rel_patterns,
        elements.var_labels,
        schema,
    )
    # Which accesses were written through the announced nesting container. Read
    # from the query text because the AST extractor flattens the path away.
    container = schema.get('attribute_container')
    container_paths = (
        _container_prefixed_accesses(query, container) if container else set()
    )
    properties = _validate_properties(
        elements.properties, elements.var_labels, schema, container_paths
    )

    schema_match = SchemaMatch(
        labels=labels,
        relationships=relationships,
        properties=properties,
    )

    # Determine verdict
    has_issues = (
        labels.unknown
        or relationships.unknown
        or relationships.wrong_direction
        or properties.unknown
        or properties.wrong_label
    )

    if has_issues:
        return CypherQuality(
            outcome='suspect',
            verdict='schema_mismatch',
            schema_match=schema_match,
        )

    return CypherQuality(
        outcome='ok',
        verdict='success',
        schema_match=schema_match,
    )
