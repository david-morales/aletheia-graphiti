"""The `investigate` prompt — the connector's one MCP prompt (P2).

WHY THIS MODULE EXISTS. At the 2026-07-28 era the SDK announces the `prompts`
capability iff a `prompts/list` handler is registered, and `MCPServer` registers
one unconditionally (`mcp/server/mcpserver/server.py:213`). This connector has
therefore announced `prompts` since the modern-era migration while serving an
empty list: declared-and-empty, the ADR-019 R7 defect that the P3 comment block
in `graphiti_mcp_server.py` names and explicitly defers to P2.

WHAT IT SERVES. One prompt, `investigate`, split in two along the axis that
decides staleness:

- The LIST entry (name, title, description, arguments) is STATIC and
  domain-agnostic. It says what the prompt is for, never what this graph
  contains, so `prompts/list` cannot go stale when the graph moves.
- The MESSAGES are rendered HERE, at `prompts/get` time, from whatever profile
  the server holds at that moment. Nothing is cached, so nothing can be stale.

That split is why P2 needs none of P3's fingerprint-and-notify machinery: P3
exists because `tools/list` descriptions are rendered ONCE at startup and frozen
for the process lifetime. A prompt rendered per request has no such window.

DOMAIN AGNOSTICISM (Hard Rule 2 / ADR-003). Every string in this module is
static template text and must name no domain. All domain content — labels,
counts, relationship types, the partition name — enters through the
`DomainProfile` passed in at render time. `test_no_domain_leakage.py` scans this
file statically; `test_prompt_surface.py` scans what it renders.
"""

from __future__ import annotations

from domain_profile import DomainProfile

INVESTIGATE_PROMPT_NAME = 'investigate'

INVESTIGATE_PROMPT_TITLE = 'Investigate a topic in this graph'

INVESTIGATE_PROMPT_DESCRIPTION = (
    'Start a grounded investigation of a topic against this knowledge graph. '
    'Returns a step-by-step retrieval workflow together with a live census of '
    'the graph this connector serves, so the plan is written against the data '
    'that is actually here rather than against an assumed schema.'
)

TOPIC_ARGUMENT_DESCRIPTION = (
    'What to investigate, in natural language. A subject, an entity name, a '
    'question, or a claim to corroborate.'
)

NO_CENSUS_MARKER = '!! census unavailable: this graph has not been profiled yet !!'
"""Leads the census block when no profile exists.

CONSUMER-PINNED: aletheia's `tests/mcp_e2e/test_prompt_surface.py` asserts this
literal — changing the wording breaks that suite (P2 ledger, M13).

The same honesty rule `DEGRADED_INSTRUCTIONS_MARKER` follows for `instructions`
(BUG-50 / A-D2): a consumer reading a workflow with no census must be able to
SEE that the census is missing, not quietly infer an empty graph. The two states
are different — "not censused" is a connector that has not looked, "censused and
empty" is a graph that has nothing — and conflating them tells an agent the
graph is empty when it may be full.
"""

# The number of entity and relationship types the census lists before it stops
# and says how many more there are. A prompt is a message an agent reads in
# full, not a reference it indexes; a graph with hundreds of labels would
# otherwise bury the workflow under its own schema.
_CENSUS_TYPE_LIMIT = 25


def _census_lines(profile: DomainProfile | None) -> list[str]:
    """The live census, rendered from the profile object at call time.

    Reads the profile's own fields rather than reusing
    `render_domain_summary()`: that renderer is the body of the
    `graphiti://domain-summary` RESOURCE and is free to grow for that audience,
    while this block has to stay short enough to sit inside a prompt message.
    Coupling them would let a resource-side change silently inflate every
    prompt.
    """
    if profile is None:
        return [NO_CENSUS_MARKER, '', 'Call get_schema to census the graph before planning.']

    # CONSUMER-PINNED: aletheia's `tests/mcp_e2e/test_prompt_surface.py` asserts
    # this "Graph partition (group_id):" prefix verbatim (P2 ledger, M13).
    lines = [f'Graph partition (group_id): {profile.group_id}', '']

    # No early return on empty entities, and no affirmative "this graph holds
    # no entities" claim. The three census probes are INDEPENDENT and each
    # swallows its own exception (`domain_profile._query_entity_types` and
    # friends return {} on failure), so empty-entities alongside populated
    # edge_types is a REACHABLE state — and it means the entity probe failed,
    # not that the graph is empty. Returning early there also dropped the edges
    # and the time range that DID come back, which is the same not-looked /
    # looked-and-empty conflation `NO_CENSUS_MARKER` exists to prevent, one
    # level down. "none recorded" says what is true either way, and mirrors the
    # edge wording below.
    entities = sorted(profile.entity_types.values(), key=lambda i: -i.count)
    if entities:
        lines.append(f'Entity types ({len(entities)}), most populated first:')
        for info in entities[:_CENSUS_TYPE_LIMIT]:
            # `hierarchy` labels are censusable and searchable but reach no
            # stored vertex, so `MATCH (n:Label)` returns nothing for them. An
            # agent that escalates to graph_query against one gets an empty
            # result and no explanation — mark them here rather than let it
            # conclude the graph is empty.
            note = ' [hierarchy only — not matchable as (n:Label)]' if info.hierarchy else ''
            desc = f' — {info.description}' if info.description else ''
            lines.append(f'- {info.label} ({info.count}){note}{desc}')
        if len(entities) > _CENSUS_TYPE_LIMIT:
            lines.append(
                f'- ... and {len(entities) - _CENSUS_TYPE_LIMIT} more (see get_schema)'
            )
    else:
        lines.append('Entity types: none recorded.')

    lines.append('')
    edges = sorted(profile.edge_types.values(), key=lambda i: -i.count)
    if edges:
        lines.append(f'Relationship types ({len(edges)}), most populated first:')
        for info in edges[:_CENSUS_TYPE_LIMIT]:
            pattern = f' ({info.source_target_pattern})' if info.source_target_pattern else ''
            lines.append(f'- {info.name}{pattern} [{info.count}]')
        if len(edges) > _CENSUS_TYPE_LIMIT:
            lines.append(f'- ... and {len(edges) - _CENSUS_TYPE_LIMIT} more (see get_schema)')
    else:
        lines.append('Relationship types: none recorded.')

    if profile.time_range:
        lines.append('')
        lines.append(
            f'Fact time range: {profile.time_range[0]} to {profile.time_range[1]}'
        )

    return lines


def _workflow_lines() -> list[str]:
    """The retrieval workflow. Static, backend-agnostic, canonical names only.

    Ordered cheapest-and-most-grounded first. The escalation to `graph_query` is
    deliberately LAST: Cypher against a schema the agent has not read is the
    single most common way to get a confidently wrong empty result, and the
    dialect it would need is not written here — it is DATA the flavour owns and
    `get_schema` serves as `dialect_reference` (ADR-019 R6). Inlining dialect
    text into this prompt would fork it per-backend in a place no flavour test
    looks.
    """
    return [
        '1. Ground yourself in the schema. Call get_schema first, even if the',
        '   census above looks sufficient: it carries property keys, the',
        '   source->target patterns behind each relationship type, and this',
        "   backend's `dialect_reference`. Follow `dialect_reference` for any",
        '   query dialect question — do not assume a dialect from the census.',
        '',
        '2. Find the entry points. Call search with the topic to locate the',
        '   entities, facts and communities that mention it. Search is semantic:',
        '   prefer the full natural-language topic over keywords, and re-run it',
        '   with rephrasings before concluding the graph has nothing.',
        '',
        '3. Expand what you found. For each promising hit, call explore_entity',
        '   on its name to walk the neighbourhood. This is where relationships',
        '   turn a list of matches into an account of how they connect. Prefer',
        '   it over more searching once you know which entity matters.',
        '',
        '4. Escalate only if the question needs it. When you need counts,',
        '   aggregations, comparisons, gap detection or path queries that steps',
        '   2-3 cannot express, call graph_query with read-only Cypher built',
        "   from get_schema's labels, patterns and `dialect_reference`. Bridge",
        '   from what you already found (WHERE ... IN [...]) rather than',
        '   scanning the whole graph.',
        '',
        '5. Judge before you trust. If a count or a coverage claim carries the',
        '   answer, call profile_data to check property coverage, sample values',
        '   and relationship cardinality first. A count over a sparsely',
        '   populated property is a number, not a finding.',
    ]


def _reporting_lines() -> list[str]:
    """How to report. Static; keeps the prompt from ending at retrieval.

    The failure this addresses is not retrieval, it is the confident summary
    written over three weak hits. Naming the absence of evidence as a valid
    outcome is what makes that reportable rather than embarrassing.
    """
    return [
        '- Cite what you retrieved. Name the entities and relationships each',
        '  claim rests on, so a reader can re-run the same calls.',
        '- Distinguish what the graph says from what you inferred across hops.',
        '- If the graph does not cover the topic, say so and say which types you',
        '  searched. An honest gap is a result; an invented one is not.',
    ]


_HEADING_TOPIC_MAX_CHARS = 200
"""How much of the topic the `# Investigate:` heading carries.

The heading is a label, not the payload — the full topic reaches the agent
through the `topic` argument the client already sent. Capping it bounds the one
piece of caller-controlled text in the document.
"""


def _heading_safe(topic: str) -> str:
    """Flatten `topic` so it cannot introduce markdown structure.

    The heading interpolates caller-supplied text into a document whose sections
    an agent navigates by `##`. A topic containing newlines and a `## How to
    investigate` line therefore RESTRUCTURED the document — measured at two
    headings of that name — letting the caller forge or displace instructions
    the connector is supposed to own.

    Collapsing every run of whitespace (newlines included) to a single space
    makes structure unreachable: markdown block constructs need a line start,
    and after this there is exactly one line. The cap then bounds it. Escaping
    `#` was the alternative and is strictly worse — it leaves the newline, which
    is the half that actually matters.
    """
    flattened = ' '.join(topic.split())
    if len(flattened) > _HEADING_TOPIC_MAX_CHARS:
        return flattened[:_HEADING_TOPIC_MAX_CHARS].rstrip() + '...'
    return flattened


def build_investigate_prompt(topic: str, profile: DomainProfile | None) -> str:
    """Render the `investigate` message body for `topic` against `profile`.

    Called at `prompts/get` time with the CURRENT profile — see the module
    docstring. `profile` is `None` on every path where the connector has not
    censused its graph: before startup reaches the census, on an episode-only
    deployment, and after introspection fails and the surface degrades. All
    three render the marked no-census block rather than raising, because a
    `prompts/get` that errors is a worse answer than one that admits what it
    does not know.
    """
    sections = [
        f'# Investigate: {_heading_safe(topic)}',
        '',
        'You are investigating the topic above against a knowledge graph exposed',
        'through this connector. Work from what the graph actually contains — the',
        'census below is live, read at the moment this prompt was requested.',
        '',
        '## This graph',
        '',
        *_census_lines(profile),
        '',
        '## How to investigate',
        '',
        *_workflow_lines(),
        '',
        '## How to report',
        '',
        *_reporting_lines(),
    ]
    return '\n'.join(sections)
