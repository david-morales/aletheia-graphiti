"""Dynamic tool descriptions and MCP instructions built from DomainProfile."""

from __future__ import annotations

from domain_profile import DomainProfile


def build_instructions(profile: DomainProfile) -> str:
    """Build the MCP server instructions from a DomainProfile."""
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

    # Tool guidance
    parts.append('')
    parts.append('Key tools:')
    parts.append('')
    parts.append('1. search -- Find entities, facts, or communities by natural language query.')
    parts.append('   Use when: the user asks a question or wants to find something.')
    parts.append('   Use explore_node instead when: you already know which entity to examine.')
    parts.append('')
    parts.append('2. explore_node -- Expand a known entity\'s neighborhood.')
    parts.append('   Use when: you have a specific entity name and want its connections.')
    parts.append('   Use search instead when: you don\'t know which entity to start from.')
    parts.append('')
    parts.append('3. search_ontology -- Look up schema definitions in the companion ontology.')
    parts.append('   Use when: you need to understand what types or properties are defined.')
    parts.append('   Use search instead when: you want actual data, not schema definitions.')
    parts.append('')
    parts.append('4. explore_ontology -- Expand a specific ontology class.')
    parts.append('   Use when: you want properties and parent classes for a specific type.')

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
