"""Domain profile auto-discovered from graph data and ontology at startup."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from flavours.base import BaseFlavour

logger = logging.getLogger(__name__)


@dataclass
class EntityTypeInfo:
    label: str
    count: int
    description: str
    sample_names: list[str] = field(default_factory=list)


@dataclass
class EdgeTypeInfo:
    name: str
    count: int
    description: str
    source_target_pattern: str = ''


@dataclass
class DomainProfile:
    group_id: str
    entity_types: dict[str, EntityTypeInfo] = field(default_factory=dict)
    edge_types: dict[str, EdgeTypeInfo] = field(default_factory=dict)
    time_range: tuple[str, str] | None = None

    def entity_type_names(self) -> list[str]:
        return sorted(self.entity_types.keys())

    def edge_type_names(self) -> list[str]:
        return sorted(self.edge_types.keys())

    def render_domain_summary(self) -> str:
        lines = [f'# Knowledge Graph: {self.group_id}', '']

        if not self.entity_types:
            lines.append('This graph contains no entities yet.')
            return '\n'.join(lines)

        lines.append('## Entity Types')
        for info in sorted(self.entity_types.values(), key=lambda x: -x.count):
            desc = f' -- {info.description}' if info.description else ''
            lines.append(f'- {info.label} ({info.count}){desc}')

        lines.append('')
        lines.append('## Relationship Types')
        if self.edge_types:
            for info in sorted(self.edge_types.values(), key=lambda x: -x.count):
                desc = f' -- {info.description}' if info.description else ''
                lines.append(f'- {info.name} ({info.count}){desc}')
        else:
            lines.append('No relationships found.')

        if self.time_range:
            lines.append('')
            lines.append('## Time Range')
            lines.append(f'Earliest: {self.time_range[0]}')
            lines.append(f'Latest: {self.time_range[1]}')

        return '\n'.join(lines)

    def render_entity_catalog(self) -> str:
        lines = [f'# Entity Catalog: {self.group_id}', '']
        for info in sorted(self.entity_types.values(), key=lambda x: -x.count):
            lines.append(f'## {info.label}')
            if info.description:
                lines.append(info.description)
            lines.append(f'Count: {info.count}')
            if info.sample_names:
                lines.append(f'Examples: {", ".join(info.sample_names)}')
            lines.append('')
        return '\n'.join(lines)

    def render_relationship_types(self) -> str:
        lines = [f'# Relationship Types: {self.group_id}', '']
        for info in sorted(self.edge_types.values(), key=lambda x: -x.count):
            pattern = f' ({info.source_target_pattern})' if info.source_target_pattern else ''
            desc = f' -- {info.description}' if info.description else ''
            lines.append(f'- **{info.name}**{pattern}{desc} [{info.count} facts]')
        return '\n'.join(lines)


# Internal labels that should not appear as entity types
_INTERNAL_LABELS = frozenset({'Entity', 'Episodic', 'Community'})


def _log_probe_failure(probe: str, exc: Exception, flavour, query: str) -> None:
    """WARNING with the flavour's classification — startup must not die, but must be legible.

    The query is passed through: several classifier patterns can only be distinguished by
    corroborating against what was actually submitted, and without it they never fire.
    """
    err = flavour.classify_execution_error(str(exc), query=query)
    logger.warning(
        'Domain-profile probe %s failed [%s]: %s | hint: %s',
        probe, err.reason, exc, err.suggestion,
    )


async def _query_entity_types(driver, group_id: str, flavour=None) -> dict[str, EntityTypeInfo]:
    """Query data graph for entity labels and counts."""
    flavour = flavour or BaseFlavour()
    query = flavour.profile_queries()['entity_types']
    try:
        records, _, _ = await driver.execute_query(query, group_id=group_id)
    except Exception as e:
        _log_probe_failure('entity_types', e, flavour, query)
        return {}

    types: dict[str, EntityTypeInfo] = {}
    for record in records:
        # `or []`: AGE returns NULL for vertices with no stored labels list.
        labels = record.get('entity_type') or []
        count = record.get('cnt', 0) or 0
        for label in labels:
            if label in _INTERNAL_LABELS:
                continue
            if label in types:
                types[label].count += count
            else:
                types[label] = EntityTypeInfo(label=label, count=count, description='')
    return types


async def _query_edge_types(driver, group_id: str, flavour=None) -> dict[str, EdgeTypeInfo]:
    """Query data graph for relationship types and counts."""
    flavour = flavour or BaseFlavour()
    query = flavour.profile_queries()['edge_types']
    try:
        records, _, _ = await driver.execute_query(query, group_id=group_id)
    except Exception as e:
        _log_probe_failure('edge_types', e, flavour, query)
        return {}

    types: dict[str, EdgeTypeInfo] = {}
    for record in records:
        name = record.get('relationship_type', '')
        count = record.get('cnt', 0) or 0
        if name and name not in ('RELATES_TO',):
            types[name] = EdgeTypeInfo(name=name, count=count, description='')
    return types


async def _query_sample_names(
    driver, group_id: str, label: str, limit: int = 5, flavour=None
) -> list[str]:
    """Get sample entity names for a given label."""
    flavour = flavour or BaseFlavour()
    query = flavour.profile_queries()['sample_names']
    try:
        records, _, _ = await driver.execute_query(
            query, group_id=group_id, label=label, limit=limit
        )
    except Exception as e:
        _log_probe_failure(f'sample_names[{label}]', e, flavour, query)
        return []
    return [r['name'] for r in records if r.get('name')]


async def _query_time_range(driver, group_id: str, flavour=None) -> tuple[str, str] | None:
    """Get the earliest and latest fact dates in the graph."""
    flavour = flavour or BaseFlavour()
    query = flavour.profile_queries()['time_range']
    try:
        records, _, _ = await driver.execute_query(query, group_id=group_id)
    except Exception as e:
        _log_probe_failure('time_range', e, flavour, query)
        return None
    if records and records[0].get('earliest') and records[0].get('latest'):
        earliest = str(records[0]['earliest'])[:10]
        latest = str(records[0]['latest'])[:10]
        return (earliest, latest)
    return None


async def _enrich_from_ontology(
    ontology_driver,
    entity_types: dict[str, EntityTypeInfo],
    edge_types: dict[str, EdgeTypeInfo],
) -> None:
    """Enrich entity and edge type descriptions from ontology graph node summaries."""
    query = (
        'MATCH (n:Entity) '
        'WHERE n.summary IS NOT NULL '
        'RETURN n.name AS name, n.summary AS summary'
    )
    try:
        records, _, _ = await ontology_driver.execute_query(query)
    except Exception as e:
        logger.warning(f'Failed to query ontology descriptions: {e}')
        return

    for record in records:
        name = record.get('name', '')
        summary = record.get('summary', '')
        if not name or not summary:
            continue
        if name in entity_types:
            entity_types[name].description = summary
        # Edge types may match ontology relationship classes
        upper_name = name.upper().replace(' ', '_')
        if upper_name in edge_types:
            edge_types[upper_name].description = summary


async def build_domain_profile(
    client,
    group_id: str,
    ontology_client=None,
    flavour=None,
) -> DomainProfile:
    """Build a DomainProfile by introspecting data and ontology graphs.

    ``flavour`` supplies the backend-correct probe Cypher; omitted, the generic openCypher
    text is used (Neo4j / test callers).
    """
    driver = client.driver
    flavour = flavour or BaseFlavour()

    # Query data graph
    entity_types = await _query_entity_types(driver, group_id, flavour)
    edge_types = await _query_edge_types(driver, group_id, flavour)

    # Fetch sample names for each entity type
    for label, info in entity_types.items():
        info.sample_names = await _query_sample_names(driver, group_id, label, flavour=flavour)

    # Query time range
    time_range = await _query_time_range(driver, group_id, flavour)

    # Enrich from ontology if available
    if ontology_client is not None:
        await _enrich_from_ontology(
            ontology_client.driver,
            entity_types,
            edge_types,
        )

    profile = DomainProfile(
        group_id=group_id,
        entity_types=entity_types,
        edge_types=edge_types,
        time_range=time_range,
    )

    logger.info(
        f'Domain profile built: {len(entity_types)} entity types, '
        f'{len(edge_types)} edge types'
    )
    return profile
