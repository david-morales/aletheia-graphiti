"""Schema-aware Cypher quality assessment.

Diffs AST-extracted Cypher elements against a cached graph schema to
detect hallucinated labels, wrong-direction relationships, and
misplaced properties.
"""

from __future__ import annotations

import difflib
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


def _validate_properties(
    properties: list[PropertyAccess],
    var_labels: dict[str, str],
    schema: dict[str, Any],
) -> PropMatch:
    """Check property accesses against schema node properties."""
    node_labels = schema.get('node_labels', {})
    known_label_set = set(node_labels.keys())
    match = PropMatch()

    # Build a reverse index: property_name -> list of labels that have it
    prop_to_labels: dict[str, list[str]] = {}
    for label, info in node_labels.items():
        for prop in info.get('properties', []):
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

        label_props = node_labels[label].get('properties', [])
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
    1. schema_mismatch or parse_failed -> keep as-is (don't downgrade)
    2. result_signals is None -> keep as-is
    3. row_count == 0 and verdict == "success" -> change to "empty_legit", outcome stays "ok"
    4. null_ratio > threshold and verdict == "success" -> change to "degraded", outcome="suspect"
    5. Otherwise -> keep as-is
    """
    if quality.verdict in ('schema_mismatch', 'parse_failed'):
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
    properties = _validate_properties(elements.properties, elements.var_labels, schema)

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
