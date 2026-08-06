"""Dynamic tool descriptions and MCP instructions built from DomainProfile."""

from __future__ import annotations

from typing import TYPE_CHECKING

from domain_profile import DomainProfile

if TYPE_CHECKING:
    from flavours.base import Flavour


def _key_tools_lines() -> list[str]:
    """The capability catalog (ADR-019 R1).

    Profile-independent on purpose: the tools a connector serves do not depend on
    what its graph happens to contain, so the healthy and the DEGRADED
    announcements serve the same catalog and cannot drift apart.
    """
    return [
        '',
        'Key tools:',
        '',
        '1. search -- Find entities, facts, or communities by natural language query.',
        '   Use when: the user asks a question or wants to find something.',
        '   Use explore_node instead when: you already know which entity to examine.',
        '',
        "2. explore_node -- Expand a known entity's neighborhood.",
        '   Use when: you have a specific entity name and want its connections.',
        "   Use search instead when: you don't know which entity to start from.",
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
    ]


def _analytical_queries_lines() -> list[str]:
    """The two-family access-pattern guidance (ADR-019 R1)."""
    return [
        '',
        '## Analytical Queries',
        '',
        'Two complementary tool families for this graph:',
        '- **Semantic discovery** (search, explore_node): find entities, explore connections, community context',
        '- **Analytical queries** (get_schema, run_cypher): counts, aggregations, path queries, comparisons, gap detection',
        '',
        '**When to use which:**',
        '- Use search/explore_node when you need semantic similarity or entity discovery',
        '- Use get_schema + run_cypher when you need counts, aggregations, comparisons, or gap detection',
        '- Use search -> then run_cypher for chained workflows: discover entities semantically,',
        '  then compute metrics with Cypher using WHERE ... IN [...] to bridge results',
    ]


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
    ]
    parts += _key_tools_lines()
    parts += _analytical_queries_lines()
    parts += _dialect_lines(flavour)
    return '\n'.join(parts)


def build_instructions(profile: DomainProfile, flavour: 'Flavour | None' = None) -> str:
    """Build the MCP server instructions from a DomainProfile (and the backend flavour)."""
    parts = []

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

    # Entity types
    if profile.entity_types:
        parts.append('')
        parts.append('Entity types in this graph:')
        for info in sorted(profile.entity_types.values(), key=lambda x: -x.count):
            desc = f' -- {info.description}' if info.description else ''
            parts.append(f'- {info.label} ({info.count}){desc}')

    # Edge types
    if profile.edge_types:
        parts.append('')
        parts.append('Relationship types:')
        for info in sorted(profile.edge_types.values(), key=lambda x: -x.count):
            desc = f' -- {info.description}' if info.description else ''
            parts.append(f'- {info.name} ({info.count}){desc}')

    # Tool guidance (shared with the degraded announcement — one catalog)
    parts += _key_tools_lines()

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

    # Backend Cypher dialect — short form (ADR-019 R1/R6)
    parts += _dialect_lines(flavour)

    return '\n'.join(parts)


def build_search_description(profile: DomainProfile) -> str:
    """Build the search tool description from a DomainProfile."""
    parts = [
        'Search this knowledge graph for entities, facts, and communities.',
        '',
        'Use when:',
        '- User asks a question or wants to find entities, facts, or relationships',
        '- User wants to filter by entity type, edge type, or date',
        '',
        'Do NOT use when:',
        '- You already know which entity to examine -- use explore_node instead',
        '- You need schema or ontology definitions -- use search_ontology instead',
        '',
        'Returns: matching nodes (entities with name, summary, labels), '
        'edges (facts linking two entities), and community summaries.',
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
        parts.append(f'  User: "Find all {first_type.label} entities"')
        parts.append(f'  Call: search(query="{first_type.label}", entity_types=["{first_type.label}"])')

    return '\n'.join(parts)


def build_explore_node_description(profile: DomainProfile) -> str:
    """Build the explore_node tool description from a DomainProfile."""
    parts = [
        "Deep-dive on a specific entity -- shows its connections, facts, and community memberships.",
        '',
        'Use when:',
        '- You have a specific entity name or UUID and want to see everything connected to it',
        '- Following up on a search result to get more context about a specific entity',
        '',
        'Do NOT use when:',
        "- You don't know which entity to look at -- use search first",
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
        parts.append(f'  Call: explore_node(node_name="{sample_name}", depth=2)')

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
        '- You want actual data -- use explore_node instead',
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
    """Build deterministic domain-specific example Cypher queries from templates."""
    examples: list[str] = []
    entity_names = profile.entity_type_names()
    edge_names = profile.edge_type_names()

    # Template 1: Count by relationship (needs 2+ entity types, 1+ edge type)
    if len(entity_names) >= 2 and len(edge_names) >= 1:
        edge = edge_names[0]
        # Find the edge info to get source->target pattern
        edge_info = profile.edge_types[edge]
        pattern = edge_info.source_target_pattern
        if pattern and '->' in pattern:
            src_label, tgt_label = [s.strip() for s in pattern.split('->')]
        else:
            src_label, tgt_label = entity_names[0], entity_names[1]
        examples.append(
            f'MATCH (s:{src_label})-[:{edge}]->(t:{tgt_label}) '
            f'RETURN t.name, count(s) AS cnt ORDER BY cnt DESC LIMIT 10'
        )

    # Template 2: Multi-occurrence aggregation (needs 1+ entity type, 1+ edge type)
    if len(entity_names) >= 1 and len(edge_names) >= 1:
        edge = edge_names[0]
        edge_info = profile.edge_types[edge]
        pattern = edge_info.source_target_pattern
        if pattern and '->' in pattern:
            src_label, tgt_label = [s.strip() for s in pattern.split('->')]
        else:
            src_label = entity_names[0]
            tgt_label = entity_names[0]
        examples.append(
            f'MATCH (s:{src_label})-[:{edge}]->(t:{tgt_label}) '
            f'WITH t, count(s) AS total WHERE total > 1 '
            f'RETURN t.name, total ORDER BY total DESC'
        )

    # Template 3: WHERE...IN bridge for semantic-to-analytical
    # (needs 1+ entity type with sample_names)
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


def build_run_cypher_description(profile: DomainProfile, flavour: 'Flavour | None' = None) -> str:
    """Build the run_cypher tool description from a DomainProfile (and the backend flavour)."""
    parts = [
        f'Execute a read-only Cypher query against the {profile.group_id} graph.',
        '',
        'Use when:',
        '- You need counts, aggregations, comparisons, or gap detection',
        '- You need path queries or pattern matching beyond what search provides',
        '- You want to compute metrics over the graph structure',
        '',
        'Do NOT use when:',
        '- You need semantic similarity search -- use search instead',
        '- You need to discover entities by natural language -- use search instead',
        '',
        'Guardrails: read-only (no CREATE/DELETE/SET), auto-limited to 200 rows,',
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
    parts.append('then run_cypher with WHERE n.name IN [...] to compute analytical')
    parts.append('metrics over the found entities.')

    return '\n'.join(parts)
