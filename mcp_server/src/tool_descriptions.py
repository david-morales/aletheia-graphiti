"""Dynamic tool descriptions and MCP instructions built from DomainProfile."""

from __future__ import annotations

from typing import TYPE_CHECKING

from domain_profile import DomainProfile
from utils.formatting import COMBINED_EPISODE_LIMIT, EPISODE_CONTENT_CAP
from version import CONNECTOR_BUILD

if TYPE_CHECKING:
    from flavours.base import Flavour


def _episode_leg_is_live(flavour: Flavour | None) -> bool:
    """Whether THIS backend can actually answer an episode search.

    Announcement is a promise, so it is gated on the capability rather than on
    the recipe existing. The recipes exist everywhere — core asks every backend
    for an episode leg — but a driver with no episode full-text index answers
    with an empty list forever. Telling an agent to fall back to `episodes` when
    nodes and edges look thin is worse than silence where `episodes` cannot ever
    be non-empty: it spends a call and manufactures a false negative.

    An unknown flavour (None) is treated as cannot — under-promising costs a
    capability, over-promising costs a wrong answer.
    """
    return bool(flavour is not None and flavour.searches_episode_content())


def _search_catalog_entry(flavour: Flavour | None) -> list[str]:
    """The catalog's `search` entry, with the episode half only where it is real."""
    if not _episode_leg_is_live(flavour):
        return [
            '1. search -- Find entities, facts, or communities by natural language query.',
            '   Use when: the user asks a question or wants to find something.',
            '   Use explore_entity instead when: you already know which entity to examine.',
            '   RANKED TOP-K SAMPLE, not an enumeration: it returns the best matches up',
            '   to `limit`, and there is no offset -- calling it again with the same',
            '   query returns the same sample. It cannot produce a count, a ranking or',
            '   a superlative, however many times you call it.',
            '   Use graph_query instead for counts, rankings, superlatives and anything',
            '   computed exhaustively over the whole graph.',
            '',
        ]
    return [
        '1. search -- Find entities, facts, source narratives or communities by',
        '   natural language query.',
        '   Use when: the user asks a question or wants to find something.',
        '   Use explore_entity instead when: you already know which entity to examine.',
        '   RANKED TOP-K SAMPLE, not an enumeration: it returns the best matches up',
        '   to `limit`, and there is no offset -- calling it again with the same',
        '   query returns the same sample. It cannot produce a count, a ranking or',
        '   a superlative, however many times you call it.',
        '   Use graph_query instead for counts, rankings, superlatives and anything',
        '   computed exhaustively over the whole graph.',
        '   ALSO SEARCHES THE SOURCE TEXT. Alongside nodes and edges the result',
        '   carries `episodes` -- the ingested documents themselves, matched on',
        '   their full text. Extraction lifts only part of a document into',
        '   entities and relationships, so a detail absent from every node and',
        '   edge can still be present in an episode narrative: when nodes and',
        '   edges come back thin, READ `episodes` BEFORE CONCLUDING THE GRAPH',
        '   DOES NOT HOLD THE ANSWER.',
        f'   Other modes return at most {COMBINED_EPISODE_LIMIT} episodes; '
        'intent="narrative"',
        '   (or search_mode="episodes") searches ONLY that text and returns the',
        '   full limit, for when the question is about what a document says.',
        f'   Content is cut at {EPISODE_CONTENT_CAP} characters with',
        '   `content_truncated: true`; get_episode_context(episode_uuids=[uuid])',
        '   returns that episode\'s content in full.',
        '',
    ]


def _key_tools_lines(flavour: Flavour | None = None) -> list[str]:
    """The capability catalog (ADR-019 R1) — ALL 18 served tools.

    Profile-independent on purpose: the tools a connector serves do not depend on
    what its graph happens to contain, so the healthy and the DEGRADED
    announcements serve the same catalog and cannot drift apart.

    Completeness is the contract (A-D9). The catalog used to name 7 of 18, omitting
    `add_memory` — the connector's only ingestion path — along with `profile_data`
    and both ontology-bulk tools, so an agent reading the announcement could not
    learn that half the surface exists. The three destructive tools are named AND
    marked: announcing one without saying what it does is worse than omitting it.
    """
    return [
        '',
        'Key tools:',
        '',
        *_search_catalog_entry(flavour),
        "2. explore_entity -- Expand a known entity's neighborhood.",
        '   Use when: you have a specific entity name and want its connections.',
        "   Use search instead when: you don't know which entity to start from.",
        '   A neighborhood expansion around ONE entity, ranked by proximity and cut',
        '   at `limit`: not exhaustive, and it aggregates nothing.',
        '   Use graph_query instead for counts, rankings, superlatives and anything',
        '   computed exhaustively over the whole graph.',
        '',
        '3. search_ontology -- Look up schema definitions in the companion ontology.',
        '   Use when: you need to understand what types or properties are defined.',
        '   Use search instead when: you want actual data, not schema definitions.',
        '',
        '4. explore_ontology -- Expand a specific ontology class.',
        '   Use when: you want properties and parent classes for a specific type.',
        '',
        '5. sample_subgraph -- Sample nodes plus the edges among them, already',
        '   normalized across backends (labels are the full hierarchy, unordered).',
        '   Use when: a client needs a renderable slice of the graph (graph view).',
        '   Use search instead when: you are answering a question -- this samples,',
        '   it does not rank or filter by meaning.',
        '',
        'Schema and structure:',
        '',
        '6. get_schema -- This graph\'s labels, relationship types, counts, property',
        '   keys and the backend `dialect_reference`. Call it before writing Cypher.',
        '',
        '7. graph_query -- Read-only Cypher for counts, aggregations and path queries.',
        '   THE surface for counts, rankings and superlatives: aggregation runs over',
        '   EVERY matching row in the whole graph, not over a retrieved sample --',
        '   unless your own query limits its input first, since a LIMIT before the',
        '   aggregation truncates what it sees. Written without one, a count is exact',
        '   and an ORDER BY ... LIMIT n is the real top n. search and explore_entity',
        '   return ranked samples and can answer none of these.',
        '   Writes are rejected; 200 rows are auto-limited (the cap bounds the rows',
        '   RETURNED, not the rows aggregated over).',
        '',
        '8. profile_data -- Property coverage, sample values, detected languages and',
        '   relationship cardinality. Use when you need to judge data QUALITY before',
        '   trusting a count.',
        '',
        '9. get_ontology_structure -- Every ontology class in one compact call: the',
        '   surface map. Cheaper than search_ontology when you want the whole list.',
        '',
        '10. get_ontology_documentation -- The FULL ontology reference: complete prose',
        '    and per-class property definitions. LARGE -- prefer the three tools above',
        '    for agent use; this one is for UIs, exports and batch consumers.',
        '',
        'Episodes (the ingested source documents):',
        '',
        '11. get_episodes -- List recent episodes for a graph partition.',
        '',
        '12. get_episode_context -- The nodes and edges a given episode produced.',
        '    Use when: you need to trace a fact back to its source document.',
        '',
        'Writing to the graph:',
        '',
        '13. add_memory -- Ingest an episode (text, JSON or message). Single episodes',
        '    are queued and processed asynchronously; pass `episodes` for a bulk load',
        '    that returns when done. This is the ONLY ingestion path.',
        '',
        'Health:',
        '',
        '14. get_status -- Server and database reachability, and this connector build.',
        '',
        'Destructive -- these REMOVE data and cannot be undone:',
        '',
        '15. build_communities -- DESTRUCTIVE despite the name. Within the group_ids',
        '    you pass, it DELETES the existing communities before re-clustering, so',
        '    any Community uuid you already hold for those partitions stops resolving.',
        '    Partitions you did not name are left alone. Run it after a significant',
        '    ingestion, on a connector whose graph you own, then search with',
        '    search_mode="communities".',
        '',
        '16. delete_entity_edge -- DESTRUCTIVE. Deletes one relationship by uuid.',
        '',
        '17. delete_episode -- DESTRUCTIVE. Deletes an episode AND everything extracted',
        '    from it.',
        '',
        '18. clear_graph -- DESTRUCTIVE and IRREVERSIBLE. Deletes ALL data in the named',
        '    graph partitions. There is no undo and no backup on this side.',
    ]


def _analytical_queries_lines() -> list[str]:
    """The two-family access-pattern guidance (ADR-019 R1)."""
    return [
        '',
        '## Analytical Queries',
        '',
        'Two complementary tool families for this graph:',
        '- **Semantic discovery** (search, explore_entity): find entities, explore connections, community context',
        '- **Analytical queries** (get_schema, graph_query): counts, aggregations, path queries, comparisons, gap detection',
        '',
        '**When to use which:**',
        '- Use search/explore_entity when you need semantic similarity or entity discovery',
        '- Use get_schema + graph_query when you need counts, aggregations, comparisons, or gap detection',
        '- Use search -> then graph_query for chained workflows: discover entities semantically,',
        '  then compute metrics with Cypher using WHERE ... IN [...] to bridge results',
    ]


def _census_caveat_lines(flavour: Flavour | None) -> list[str]:
    """The flavour's census caveats, announced as data (ADR-019 R6).

    These govern how to read EVERY count and label above, so they belong in the
    instructions and not only in `get_schema.analysis_notes` — which was the one
    surface they reached (A-D4).
    """
    notes = list(flavour.census_notes()) if flavour is not None else []
    if not notes:
        return []
    return ['', 'How to read this graph\'s labels and counts:', *[f'- {n}' for n in notes]]


def _dialect_lines(flavour: Flavour | None) -> list[str]:
    """Backend Cypher dialect, short form (ADR-019 R1/R6).

    The full form is get_schema's `dialect_reference`. Sourced from the flavour,
    never hardcoded per-backend — and available even when graph introspection
    failed, which is why the degraded announcement can still carry it.
    """
    if flavour is None or not flavour.dialect_summary:
        return []
    return [
        '',
        f'**Cypher dialect:** {flavour.dialect_summary}',
        'See get_schema `dialect_reference` for the full dialect notes.',
    ]


def _other_primitives_lines() -> list[str]:
    """Where else to look: the served primitives this catalog is not about (R1).

    `instructions` is a TOOL catalog by construction, and it arrives with the
    handshake — which makes it the one document many consumers read and the
    only place they would learn that this server also serves resources, a
    prompt and a change-notification stream. Saying nothing left those three
    discoverable only by a client that thought to ask.

    A POINTER, not a copy. Enumerating the members here would create a second
    catalogue of `resources/list` and `prompts/list`, maintained by hand and
    free to drift from the lists it describes — the rendered-but-not-true
    defect class re-entering through prose. This names kinds and the call that
    answers for each, so there is nothing in it that can go stale: it stays
    true whether the server serves one resource or forty, which is exactly what
    the degraded arm needs (it serves fewer, and says so where it says why).

    KINDS, AND NOT BEHAVIOUR, for the same reason. An earlier wording promised
    the listen stream "pushes list changes and per-resource content updates as
    the graph moves" — false under a supported configuration, since
    `GRAPHITI_SURFACE_REFRESH_DEBOUNCE_SECONDS=0` disables the re-census
    scheduler outright and the stream then carries nothing. Announcing a
    behaviour an operator can switch off is the same class of defect as
    announcing a capability with no publisher; naming the endpoint is true in
    every configuration.
    """
    return [
        '',
        'Beyond this catalog: the same server serves resources (`resources/list`),',
        'prompts (`prompts/list`), and a `subscriptions/listen` stream carrying the',
        'change notifications for both. Read those lists for what they hold -- they',
        'are deliberately not repeated here, because a copy in prose is a copy that',
        'drifts from the list it describes.',
    ]


def build_degraded_instructions(
    *,
    group_id: str,
    flavour: Flavour | None,
    reason: str,
    marker: str,
) -> str:
    """Announce a connector whose graph introspection failed (BUG-50 / A-D2).

    Losing the domain profile costs the DESCRIPTIONS, never the TOOLS. The
    announcement says exactly that, in the lead position, so a consumer that
    captures `instructions` once can see that what it captured is a fallback —
    and never reads "with no entities yet" off a graph it simply could not probe.
    """
    parts = [
        marker,
        '',
        f'Connector: {CONNECTOR_BUILD}',
        '',
        f'This connector ({group_id}) could not introspect its graph at startup, so it has',
        'no live entity catalog, no relationship catalog, no counts and no sample values to',
        'announce. Every tool below is registered and functional -- only the DESCRIPTIONS',
        'are static fallbacks. This says NOTHING about whether the graph holds data.',
        '',
        'What to do:',
        '- Call get_schema first: it queries live and returns this graph\'s labels,',
        '  relationship types, counts and the backend `dialect_reference`.',
        '- Treat every example in a tool description as illustrative, not as this',
        '  graph\'s data.',
        '',
        f'Introspection failure: {reason}',
        '',
        'Resources: only `graphiti://schema` is served in this state. The',
        '`domain_summary`, `entity_catalog` and `relationship_types` resources are',
        'rendered from the domain profile and cannot be built without one.',
    ]
    parts += _key_tools_lines(flavour)
    parts += _analytical_queries_lines()
    parts += _census_caveat_lines(flavour)
    parts += _dialect_lines(flavour)
    parts += _other_primitives_lines()
    return '\n'.join(parts)


def build_instructions(profile: DomainProfile, flavour: 'Flavour | None' = None) -> str:
    """Build the MCP server instructions from a DomainProfile (and the backend flavour)."""
    # Which build wrote this guidance (A-D11). It travels with the announcement a
    # consumer captures and caches, so a stale cache is identifiable after the fact.
    parts = [f'Connector: {CONNECTOR_BUILD}', '']

    # Domain summary
    if profile.entity_types:
        entity_summary = ', '.join(
            f'{info.count} {info.label} entities'
            for info in sorted(profile.entity_types.values(), key=lambda x: -x.count)
        )
        parts.append(
            f'This is a knowledge graph ({profile.group_id}) containing {entity_summary}.'
        )
    else:
        parts.append(
            f'This is a knowledge graph ({profile.group_id}) with no entities yet.'
        )

    # Entity types. A hierarchy-only label is marked WHERE IT IS LISTED, not only
    # in a caveat further down: this list is what an agent reads to pick a label,
    # and `MATCH (n:Actor)` fails silently by returning zero rows (A-D4).
    if profile.entity_types:
        parts.append('')
        parts.append('Entity types in this graph:')
        for info in sorted(profile.entity_types.values(), key=lambda x: -x.count):
            desc = f' -- {info.description}' if info.description else ''
            flag = (
                ' [hierarchy label -- searchable, but NOT `(n:Label)`-matchable;'
                f" use `WHERE '{info.label}' IN n.labels`]"
                if info.hierarchy
                else ''
            )
            parts.append(f'- {info.label} ({info.count}){desc}{flag}')

    # Edge types
    if profile.edge_types:
        parts.append('')
        parts.append('Relationship types:')
        for info in sorted(profile.edge_types.values(), key=lambda x: -x.count):
            desc = f' -- {info.description}' if info.description else ''
            parts.append(f'- {info.name} ({info.count}){desc}')

    # Tool guidance (shared with the degraded announcement — one catalog)
    parts += _key_tools_lines(flavour)

    # Tips
    if profile.entity_types or profile.edge_types:
        parts.append('')
        parts.append('Tips:')
        if profile.entity_types:
            names = ', '.join(f'"{n}"' for n in profile.entity_type_names())
            parts.append(f'- Filter by entity_types with values like: {names}')
        if profile.edge_types:
            names = ', '.join(f'"{n}"' for n in profile.edge_type_names())
            parts.append(f'- Filter by edge_types with values like: {names}')
        parts.append('- Use valid_at for temporal queries (ISO date format, e.g. "2024-03-15")')

    # Dual access pattern guidance
    parts += _analytical_queries_lines()

    # How to READ the census above (ADR-019 R6) — governs every count and label
    parts += _census_caveat_lines(flavour)

    # Backend Cypher dialect — short form (ADR-019 R1/R6)
    parts += _dialect_lines(flavour)

    # Where else to look — the served primitives this catalog is not about (R1)
    parts += _other_primitives_lines()

    return '\n'.join(parts)


def build_search_description(
    profile: DomainProfile, flavour: Flavour | None = None
) -> str:
    """Build the search tool description from a DomainProfile (and the backend flavour).

    The flavour gates the episode half: see `_episode_leg_is_live`.
    """
    parts = [
        'Search this knowledge graph for entities, facts, and communities.',
        '',
        'RANKED TOP-K RETRIEVAL. Results are the best-scoring matches up to `limit` '
        '-- a relevance-ranked SAMPLE of the graph, never an enumeration of '
        'everything that matches. There is no offset or cursor, so repeating the '
        'same query returns the same sample rather than the next page: an answer '
        'this tool cannot reach in one call it cannot reach in ten.',
        '',
        'Note that intent="exhaustive" only widens the default sample (limit 50). A wider '
        'sample is still a sample -- it is not an exhaustive answer, and graph_query '
        'is what gives you one.',
        '',
        'Use when:',
        '- User asks a question or wants to find entities, facts, or relationships',
        '- User wants to filter by entity type, edge type, or date',
        '',
        'Do NOT use when:',
        '- You already know which entity to examine -- use explore_entity instead',
        '- You need schema or ontology definitions -- use search_ontology instead',
        '- You need a count, a ranking, a superlative (most/least/largest/first/'
        'longest) or any total computed exhaustively over the whole graph -- a '
        'ranked sample cannot answer these. Use graph_query instead.',
        '',
        'Returns: matching nodes (entities with name, summary, labels), '
        'edges (facts linking two entities), and community summaries.',
    ]

    if _episode_leg_is_live(flavour):
        parts[-1] = (
            'Returns: matching nodes (entities with name, summary, labels), '
            'edges (facts linking two entities), episodes (the ingested source '
            'documents, matched on their full text) and community summaries.'
        )
        parts += [
            '',
            'The episode leg matters: extraction lifts only part of a source document '
            'into entities and relationships, so a detail that no node or edge carries '
            'may still be in the document text. If nodes and edges look thin, read '
            '`episodes` before concluding the graph does not hold the answer.',
            '',
            f'Other search modes return at most {COMBINED_EPISODE_LIMIT} episodes. Use '
            'intent="narrative" (or search_mode="episodes") to search ONLY that source '
            'text and get the full limit back -- when the question is about what a '
            'document says rather than about an entity.',
            '',
            f'Episode content is served up to {EPISODE_CONTENT_CAP} characters; past '
            'that `content_truncated` is true and '
            'get_episode_context(episode_uuids=[uuid]) returns that episode\'s content '
            'in full.',
        ]

    if profile.entity_types:
        names = ', '.join(profile.entity_type_names())
        parts.append(f'\nAvailable entity_types: {names}')
    if profile.edge_types:
        names = ', '.join(profile.edge_type_names())
        parts.append(f'Available edge_types: {names}')

    # Example using real data
    if profile.entity_types:
        first_type = next(iter(sorted(profile.entity_types.values(), key=lambda x: -x.count)))
        parts.append(f'\nExample:')
        parts.append(
            f'  User: "Which {first_type.label} entities are most relevant to <topic>?"'
        )
        parts.append(
            f'  Call: search(query="<topic>", entity_types=["{first_type.label}"])'
        )

    return '\n'.join(parts)


def build_explore_entity_description(profile: DomainProfile) -> str:
    """Build the explore_entity tool description from a DomainProfile."""
    parts = [
        "Deep-dive on a specific entity -- shows its connections, facts, and community memberships.",
        '',
        'NEIGHBORHOOD EXPANSION around ONE entity, ranked by proximity to it and cut '
        'at `limit`. It is a sample of what surrounds that entity, not an exhaustive '
        'traversal, and it computes nothing over what it returns.',
        '',
        'Use when:',
        '- You have a specific entity name or UUID and want to see what connects to it',
        '- Following up on a search result to get more context about a specific entity',
        '',
        'Do NOT use when:',
        "- You don't know which entity to look at -- use search first",
        '- You need a count, a ranking, a superlative (most/least/largest/first/'
        'longest) or any total computed exhaustively over the whole graph -- use '
        'graph_query instead',
        '',
        'Returns: the center node, connected nodes, relationship edges, and community memberships.',
    ]

    # Example using a real sample entity
    sample_name = None
    for info in profile.entity_types.values():
        if info.sample_names:
            sample_name = info.sample_names[0]
            break
    if sample_name:
        parts.append(f'\nExample:')
        parts.append(f'  User: "Tell me about {sample_name}"')
        parts.append(f'  Call: explore_entity(node_name="{sample_name}", depth=2)')

    return '\n'.join(parts)


def build_search_ontology_description(profile: DomainProfile) -> str:
    """Build the search_ontology tool description."""
    parts = [
        'Search the companion ontology graph for schema definitions, entity types, and relationships.',
        '',
        'Use when:',
        '- You need to understand what types of entities and relationships are defined',
        '- You want to know what properties or attributes a type has',
        '- You need the formal schema behind the data',
        '',
        'Do NOT use when:',
        '- You want actual data (entities, facts) -- use search instead',
        '',
        'Returns: ontology class nodes with definitions, and relationship edges between classes.',
    ]

    if profile.entity_types:
        names = ', '.join(f'"{n}"' for n in profile.entity_type_names())
        parts.append(f'\nEntity types you can look up: {names}')

    parts.append(f'\nExample:')
    if profile.entity_types:
        first = profile.entity_type_names()[0]
        parts.append(f'  User: "What properties does {first} have?"')
        parts.append(f'  Call: search_ontology(query="{first} properties")')
    else:
        parts.append(f'  Call: search_ontology(query="entity types")')

    return '\n'.join(parts)


def build_explore_ontology_description(profile: DomainProfile) -> str:
    """Build the explore_ontology tool description."""
    parts = [
        'Explore a specific class in the companion ontology graph.',
        '',
        'Use when:',
        '- You want properties, relationships, and parent classes for a specific type',
        '- You want to understand the class hierarchy',
        '',
        'Do NOT use when:',
        '- You want actual data -- use explore_entity instead',
        '',
        'Returns: the ontology class node, connected property/relationship nodes, and parent classes.',
    ]

    if profile.entity_types:
        first = profile.entity_type_names()[0]
        parts.append(f'\nExample:')
        parts.append(f'  Call: explore_ontology(node_name="{first}")')

    return '\n'.join(parts)


def build_get_schema_description(profile: DomainProfile) -> str:
    """Build the get_schema tool description from a DomainProfile."""
    parts = [
        f'Retrieve the structural schema of the {profile.group_id} graph.',
        'Returns node labels with property keys, relationship types with',
        'source->target patterns, and counts.',
    ]

    # Current graph contents summary
    if profile.entity_types or profile.edge_types:
        entity_parts = []
        for info in sorted(profile.entity_types.values(), key=lambda x: -x.count):
            entity_parts.append(f'{info.label} ({info.count})')
        edge_parts = []
        for info in sorted(profile.edge_types.values(), key=lambda x: -x.count):
            edge_parts.append(f'{info.name} ({info.count})')

        contents = []
        if entity_parts:
            contents.append(', '.join(entity_parts) + ' nodes')
        if edge_parts:
            contents.append(', '.join(edge_parts) + ' relationships')

        if contents:
            parts.append('')
            parts.append(f'This graph contains: {" and ".join(contents)}.')

    parts.append('')
    parts.append(
        'Call this tool before writing Cypher queries to learn property'
    )
    parts.append(
        'names and relationship patterns. Results are cached -- only re-queries'
    )
    parts.append('after new data ingestion.')

    return '\n'.join(parts)


def _build_example_queries(profile: DomainProfile) -> list[str]:
    """Build deterministic example Cypher queries from templates.

    EVERY label here comes from `storage_entity_type_names()`, never from the full
    census (A-D4). On AGE the census counts the ontology hierarchy, so the
    unfiltered list leads with abstract supertypes: the templates used to emit
    `MATCH (s:Actor)-[:EJECUTADO_POR]->(t:Agente)` — three queries returning zero
    rows by construction, as the FIRST thing an agent reads about graph_query.

    A profile whose every label is hierarchy-only yields NO examples. That is the
    intended outcome: no example beats one that cannot match.
    """
    examples: list[str] = []
    entity_names = profile.storage_entity_type_names()
    edge_names = profile.edge_type_names()
    matchable = set(entity_names)

    def _endpoints(edge: str, fallback: tuple[str, str]) -> tuple[str, str] | None:
        """Source/target for an example, or None when they are not matchable.

        The announced pattern is data and can itself name a hierarchy label, so it
        is checked rather than trusted.
        """
        pattern = profile.edge_types[edge].source_target_pattern
        if pattern and '->' in pattern:
            src, tgt = (s.strip() for s in pattern.split('->'))
            if src in matchable and tgt in matchable:
                return src, tgt
            return None
        return fallback

    # Template 1: Count by relationship (needs 2+ matchable types, 1+ edge type)
    if len(entity_names) >= 2 and edge_names:
        edge = edge_names[0]
        ends = _endpoints(edge, (entity_names[0], entity_names[1]))
        if ends:
            examples.append(
                f'MATCH (s:{ends[0]})-[:{edge}]->(t:{ends[1]}) '
                f'RETURN t.name, count(s) AS cnt ORDER BY cnt DESC LIMIT 10'
            )

    # Template 2: Multi-occurrence aggregation (needs 1+ matchable type, 1+ edge type)
    if entity_names and edge_names:
        edge = edge_names[0]
        ends = _endpoints(edge, (entity_names[0], entity_names[0]))
        if ends:
            examples.append(
                f'MATCH (s:{ends[0]})-[:{edge}]->(t:{ends[1]}) '
                f'WITH t, count(s) AS total WHERE total > 1 '
                f'RETURN t.name, total ORDER BY total DESC'
            )

    # Template 3: WHERE...IN bridge for semantic-to-analytical
    # (needs 1+ matchable type with sample_names)
    for name in entity_names:
        info = profile.entity_types[name]
        if info.sample_names:
            sample = info.sample_names[0]
            examples.append(
                f'// Bridge: use entity names found via search\n'
                f'MATCH (n:{name}) WHERE n.name IN ["{sample}"] '
                f'RETURN n.name, labels(n)'
            )
            break

    return examples


def build_graph_query_description(profile: DomainProfile, flavour: 'Flavour | None' = None) -> str:
    """Build the graph_query tool description from a DomainProfile (and the backend flavour)."""
    parts = [
        f'Execute a read-only Cypher query against the {profile.group_id} graph.',
        '',
        'THE AGGREGATION SURFACE. Cypher is the only tool here that computes over '
        'the graph rather than retrieving from it: counts, rankings and superlatives '
        'are evaluated against EVERY matching row in the whole graph, not against a '
        'retrieved sample -- unless your own query limits its input first, since a '
        'LIMIT placed before the aggregation truncates what it sees. Written without '
        'one, a count is exact and an ORDER BY ... LIMIT n is the real top n. search '
        'and explore_entity return relevance-ranked samples and can answer none of '
        'these, however often they are called.',
        '',
        'Use when:',
        '- You need counts, aggregations, comparisons, or gap detection',
        '- You need a ranking or a superlative (most/least/largest/first/longest) '
        'over the whole graph',
        '- You need an exhaustive answer rather than the best few matches',
        '- You need path queries or pattern matching beyond what search provides',
        '- You want to compute metrics over the graph structure',
        '',
        'Do NOT use when:',
        '- You need semantic similarity search -- use search instead',
        '- You need to discover entities by natural language -- use search instead',
        '',
        'Guardrails: read-only (no CREATE/DELETE/SET), auto-limited to 200 rows '
        '(the cap bounds the rows RETURNED, not the rows aggregated over),',
        'LLM syntax auto-corrected (smart quotes, code blocks, missing RETURN).',
    ]

    # Backend dialect notes — short form from the flavour (ADR-019 R6); the full form lives in
    # get_schema's dialect_reference. No per-backend dialect is hardcoded here.
    if flavour is not None and flavour.dialect_summary:
        parts.append('')
        parts.append(f'Dialect notes: {flavour.dialect_summary}')
        parts.append('(See get_schema `dialect_reference` for the full notes.)')

    # Domain-specific examples
    examples = _build_example_queries(profile)
    if examples:
        parts.append('')
        parts.append('Example queries for this graph:')
        for i, ex in enumerate(examples, 1):
            parts.append(f'  {i}. {ex}')

    # Chained workflow guidance
    parts.append('')
    parts.append('Chained workflow: use search to discover entities semantically,')
    parts.append('then graph_query with WHERE n.name IN [...] to compute analytical')
    parts.append('metrics over the found entities.')

    return '\n'.join(parts)
