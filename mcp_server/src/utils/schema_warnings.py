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

Three rules govern every warning here:

* **It never blocks and never rewrites.** The query has already run. A warning is
  advice attached to the answer, not a verdict on it.
* **It fires only from MEASURED knowledge.** A census with an unsampled label or
  an empty property union is not evidence of absence, and no-evidence must not be
  spent as an accusation. Where the census cannot support a verdict, this module
  is silent.
* **It never claims more than the census measured.** The census SAMPLES a bounded
  number of nodes per label; `sampled: True` means "the probe returned rows", not
  "every key was observed". A live scan of a bench graph found a real domain
  property the sample never saw. So an unknown name is reported as ABSENT FROM
  THE CENSUS, never as absent from the graph — the difference decides whether a
  model fixes a broken query or breaks a working one.

**Silence is the safe failure.** Every uncertainty here resolves toward emitting
nothing. A missed warning costs the model one wrong query it was going to write
anyway; a wrong warning tells it to edit a query that works, on the connector's
authority.

Known limits — what this cannot see (all of them cost silence, never a false
warning):

* **UNION arms after the first.** The vendored grammar's entry rules stop at the
  first `singleQuery` and report NO parse error, so references in later arms are
  invisible. Measured prevalence in the reference artifact: 4/120 queries.
* **`exists()` and `CALL {}` subqueries.** These fail to parse, and a non-zero
  `parse_errors` suppresses every warning for the whole query. Measured
  prevalence: 0/120.
* **Consecutive `WITH` clauses.** Same parser gap, same suppression.
* **Cross-label property confusion.** The union is graph-wide and label-blind by
  design, so a real property read on the wrong label — `p.municipio.fecha` where
  both names are censused somewhere — passes unremarked. Judging it would need
  alias-to-label resolution, whose failure mode is the accusation this module
  must never make. `cypher_quality` answers the per-label question separately.

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

from utils.cypher_extractor import CypherElements, extract_elements
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

# A warning list is read by a model with a context budget, and a query can name
# an unbounded number of properties. Past this many findings of either kind the
# rest are COUNTED rather than spelled out — a truncation that says so beats
# both a silent drop and a wall of text.
_MAX_WARNINGS_PER_KIND = 5

# The consequence is identical for every unknown-name finding, so it is stated
# ONCE at the end rather than repeated per entry. Attached only where a NAME is
# in doubt: a chain finding disputes a SHAPE, and the note would be answering a
# question nobody asked. Phrased to the same evidence as the findings it
# summarises — the census sampled, so it can report a name missing from itself
# and nothing stronger.
_SHARED_NOTE = (
    'A name missing from the census is not an error here — the query runs and '
    'the column comes back null for every row, so an empty or null-filled '
    'result does NOT establish that the fact is absent. Call get_schema for '
    'this graph\'s property names before concluding anything from this result.'
)


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
    """Report a name the census did not observe — as exactly that, and no more.

    The census samples a bounded number of nodes per label, so it can say a name
    was NOT SEEN. It cannot say the name does not exist: a live scan of a bench
    graph turned up a real domain property the sample missed. Stated absolutely,
    that gap makes the connector tell a model to edit a query that works.
    """
    text = f'Property `{name}` does not appear in this graph\'s sampled property census.'
    suggestions = _did_you_mean(name, union)
    if suggestions:
        named = ', '.join(f'`{s}`' for s in suggestions)
        text += f' Closest names in the census: {named}.'
    else:
        text += (
            ' The census samples nodes rather than enumerating every key, so a '
            'rare property can be missing from it — check get_schema before '
            'assuming the name is wrong.'
        )
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
        f'`{written}` reads through a map the census does not show on stored '
        f'nodes — {shape}. Nested field maps appear in the JSON that search and '
        f'explore return; the stored node does not carry that shape.'
    )


def _capped(warnings: list[str], noun: str) -> list[str]:
    """At most ``_MAX_WARNINGS_PER_KIND`` findings, with the remainder counted."""
    if len(warnings) <= _MAX_WARNINGS_PER_KIND:
        return warnings
    hidden = len(warnings) - _MAX_WARNINGS_PER_KIND
    return [
        *warnings[:_MAX_WARNINGS_PER_KIND],
        f'... and {hidden} more {noun} the census cannot account for.',
    ]


def build_schema_warnings(
    query: str,
    schema: dict[str, Any] | None,
    *,
    elements: CypherElements | None = None,
) -> list[str]:
    """Warnings about property references the census cannot account for.

    ``query`` is the SANITIZED query — the text that actually ran, so a warning
    can never describe something the pipeline already rewrote away. ``elements``
    lets a caller that has already parsed the query pass the result in rather
    than paying for a second parse.

    Returns an empty list wherever a verdict would not be grounded: no census, a
    query this parser could not read, or a census too degraded to support an
    absence. The result is advisory only — the caller attaches it to a response
    that has already been produced.
    """
    if not schema:
        return []

    if elements is None:
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
        # ONLY node-bound variables are judged. The census describes node
        # labels, so an edge reference, a map projection, an UNWIND element or
        # a function result is outside what it can speak to — and every one of
        # those is a shape valid queries use routinely. An allow-list is the
        # only safe direction here: a deny-list would have to enumerate every
        # non-node shape, and each one it missed would be a wrong accusation
        # against working Cypher.
        if chain.variable not in elements.node_vars:
            continue

        head = chain.path[0]
        nested = len(chain.path) > 1

        # WHICH SEGMENT IS THE DOMAIN REFERENCE depends on the arm, and getting
        # it wrong on either one is silent.
        if nested and container is not None and head == container:
            # Through the ANNOUNCED container, the reference is path[1] — not
            # the head (that is transport) and not the last segment (that may
            # be a component of the value). Reading it as either killed this
            # check outright on the nesting backend: there `keys(n)` returns the
            # container, so the container is IN the property union and a
            # "head is a known property" test short-circuits every correct
            # reference form the backend has.
            target = chain.path[1]
        else:
            # `p.created_at.year` reads a COMPONENT of a real property, not a
            # path through a phantom map. Warned about naively it advised
            # `p.year` — a property that does not exist — so following the
            # advice made the query worse.
            if nested and (head in union or _is_internal(head)):
                continue
            # A nested read that reaches through neither the announced
            # container nor a known property is a defect. Schema-INDEPENDENT,
            # so the positive-census gate does not bind it.
            if nested:
                warning = _chain_warning(chain.variable, chain.path, container)
                if warning not in chain_warnings:
                    chain_warnings.append(warning)
            # The leaf is the name being reached for, whatever shape it was
            # written through — so a chain that was ALSO misspelled earns both
            # findings, which is the measured failure.
            target = chain.path[-1]

        if not census_is_positive:
            continue
        if _is_internal(target):
            continue
        # The container is transport, not a domain field: naming it is never a
        # missing property.
        if container is not None and target == container:
            continue
        if target in union:
            continue
        if target not in unknown_warnings:
            unknown_warnings[target] = _unknown_property_warning(target, union)

    # Unknown findings keep the order the query names them in, so the leaves of
    # the chains shown above sit next to them; sorting alphabetically split the
    # two halves of one defect across the truncation.
    unknowns = _capped(list(unknown_warnings.values()), 'property references')
    findings = _capped(chain_warnings, 'nested property paths') + unknowns
    # The note answers "what does an empty result mean here", which only arises
    # once a NAME is in doubt.
    return [*findings, _SHARED_NOTE] if unknowns else findings
