"""Census-validated property references for the graph_query response.

A property name the graph does not carry is not an error on either backend. The
query parses, runs, and returns a null column for every row — so an agent that
guessed `fecha_inicio` where the graph stores `fecha_de_inicio` reads a clean
empty answer and concludes the fact is absent. Nothing in the pipeline says
otherwise, because nothing failed.

The exhortation channel for this already exists (`get_schema`'s description says
to call it before writing Cypher) and is measured insufficient. This module adds
the channel that arrives at the moment it can act: a WARNING in the result the
model reads next turn, while it still holds the query that produced the nulls.

Two rules govern every warning here:

* **It never blocks and never rewrites.** The query has already run. A warning is
  advice attached to the answer, not a verdict on it.
* **It fires only from MEASURED knowledge.** A census with an unsampled label or
  an empty property union is not evidence of absence, and no-evidence must not be
  spent as an accusation. Where the census cannot support a verdict, this module
  is silent.

Relationship to :mod:`utils.cypher_quality`: that module answers the finer,
per-label question (is this property reachable on the label this alias is bound
to?) and reports it as structured `schema_match` detail. It necessarily SKIPS any
reference whose alias carries no label binding — after a `WITH` rebinding, or in
an unlabelled `MATCH (n)` — which is where the measured misses landed. This
module is deliberately coarser and label-agnostic: one union over the whole
census, so a reference is judged wherever it appears, and the finding is stated
in prose the model reads rather than nested under a structured key.
"""

from __future__ import annotations

import difflib
from typing import Any

from utils.cypher_extractor import extract_elements
from utils.cypher_quality import _INTERNAL_PROPERTIES as _NODE_INTERNAL_PROPERTIES

# Properties Graphiti itself stores, which no domain census needs to announce.
#
# Seeded from cypher_quality's node-level set so the two cannot drift, then
# extended with the EDGE and EPISODIC internals: this module's check is
# label-agnostic and therefore sees references the per-label one filters out
# before it gets this far. Every name below is a field on a graphiti_core model
# (Node / EntityNode / EpisodicNode / Edge / EntityEdge), not a domain term.
_INTERNAL_PROPERTIES: frozenset[str] = _NODE_INTERNAL_PROPERTIES | frozenset({
    # Node
    'labels',
    # EpisodicNode
    'content',
    'source',
    'source_description',
    'valid_at',
    'entity_edges',
    'episode_metadata',
    # Edge / EntityEdge
    'source_node_uuid',
    'target_node_uuid',
    'fact',
    'episodes',
    'invalid_at',
    'reference_time',
})

# How many census names a did-you-mean may offer, and how close they must be.
_SUGGESTION_COUNT = 3
_SUGGESTION_CUTOFF = 0.6


def _is_internal(name: str) -> bool:
    """Is this a Graphiti-owned property rather than a domain one?"""
    return name in _INTERNAL_PROPERTIES or name.endswith('_embedding')


def _census_property_union(schema: dict[str, Any]) -> set[str]:
    """Every property name ANY censused label announces.

    The union of both key sets get_schema publishes per label: `properties` (the
    top-level keys) and `attribute_keys` (the canonical domain-queryable keys,
    which on a nesting backend appear in `properties` nowhere at all). Unioning
    what the schema already announces keeps this function backend-blind — the
    same reason `cypher_quality._addressable_properties` unions them.
    """
    union: set[str] = set()
    for info in (schema.get('node_labels') or {}).values():
        union.update(info.get('properties') or [])
        union.update(info.get('attribute_keys') or [])
    return union


def _census_is_positive(schema: dict[str, Any]) -> bool:
    """May an ABSENCE from this census be reported as a finding?

    Only when the census is complete evidence: at least one label, every one of
    them actually sampled, and a non-empty property union. A label whose probe
    degraded reports `sampled: False` with empty `properties`, and reading that
    as "this type has no properties" is exactly the misreading the profiler and
    get_schema both warn about. An entry that omits `sampled` is treated as
    sampled — get_schema always sets it, so absence means a caller that is not
    modelling the degraded case.
    """
    node_labels = schema.get('node_labels') or {}
    if not node_labels:
        return False
    if any(not bool(info.get('sampled', True)) for info in node_labels.values()):
        return False
    return bool(_census_property_union(schema))


def _did_you_mean(name: str, union: set[str]) -> list[str]:
    """The nearest census names to a name the census does not carry."""
    return difflib.get_close_matches(
        name, sorted(union), n=_SUGGESTION_COUNT, cutoff=_SUGGESTION_CUTOFF
    )


def _unknown_property_warning(name: str, union: set[str]) -> str:
    suggestions = _did_you_mean(name, union)
    text = (
        f'Property `{name}` is not carried by any node type in this graph. '
        f'A property that does not exist is not an error here: the column comes '
        f'back null for every row, so an empty result does NOT mean the fact is '
        f'absent.'
    )
    if suggestions:
        named = ', '.join(f'`{s}`' for s in suggestions)
        text += f' Closest names this graph does carry: {named}.'
    else:
        text += ' Call get_schema for this graph\'s exact property names.'
    return text


def _chain_warning(variable: str, path: tuple[str, ...], container: str | None) -> str:
    """Report a dotted path that does not address a stored property.

    The correct form is read off the census's own announcement
    (`attribute_container`, ADR-019 R6 — dialect as DATA), never off a backend
    name. Where the census announces no container the stored properties are
    flat; where it announces one, that map is the only legitimate middle segment.
    """
    written = '.'.join((variable, *path))
    leaf = path[-1]
    if container:
        correct = f'{variable}.{container}.{leaf}'
        shape = (
            f'this graph nests its domain fields in the `{container}` map, '
            f'so the only valid form is `{correct}`'
        )
    else:
        correct = f'{variable}.{leaf}'
        shape = (
            f'node properties in this graph are FLAT, so the valid form is '
            f'`{correct}`'
        )
    return (
        f'`{written}` reads through a map that stored nodes do not have — '
        f'{shape}. Nested field maps appear in the JSON that search and explore '
        f'return, but the stored node does not carry that shape, so this path '
        f'returns null for every row rather than failing.'
    )


def build_schema_warnings(query: str, schema: dict[str, Any] | None) -> list[str]:
    """Warnings about property references the census cannot account for.

    ``query`` is the SANITIZED query — the text that actually ran, so a warning
    can never describe something the pipeline already rewrote away.

    Returns an empty list wherever a verdict would not be grounded: no census, a
    query this parser could not read, or a census too degraded to prove an
    absence. The result is advisory only — the caller attaches it to a response
    that has already been produced.
    """
    if not schema:
        return []

    elements = extract_elements(query)
    # A partial parse yields partial references, and a warning derived from one
    # would be a guess about a guess. cypher_quality takes the same position
    # (`parse_failed` suppresses its schema verdict).
    if elements.parse_errors > 0:
        return []

    container = schema.get('attribute_container')
    union = _census_property_union(schema)
    census_is_positive = _census_is_positive(schema)

    chain_warnings: list[str] = []
    unknown_warnings: dict[str, str] = {}

    for chain in elements.property_chains:
        # The census describes NODE labels. An edge-bound reference is outside
        # what it can speak to, so it is left alone rather than guessed at.
        if chain.variable in elements.rel_vars:
            continue

        nested = len(chain.path) > 1
        through_container = (
            nested and container is not None and chain.path[0] == container
        )

        # A nested read is a defect unless it goes through the map this backend
        # announced. Schema-INDEPENDENT: it does not consult the property union,
        # so the positive-census gate does not bind it.
        if nested and not through_container:
            warning = _chain_warning(chain.variable, chain.path, container)
            if warning not in chain_warnings:
                chain_warnings.append(warning)

        # The leaf is the name the query is actually reaching for, whatever
        # shape it was written through — so a chain that was ALSO misspelled
        # earns both findings, which is the measured failure.
        leaf = chain.path[-1]
        if not census_is_positive:
            continue
        if _is_internal(leaf):
            continue
        # The container is transport, not a domain field: naming it is never a
        # missing property.
        if container is not None and leaf == container:
            continue
        if leaf in union:
            continue
        if leaf not in unknown_warnings:
            unknown_warnings[leaf] = _unknown_property_warning(leaf, union)

    return chain_warnings + [unknown_warnings[k] for k in sorted(unknown_warnings)]
