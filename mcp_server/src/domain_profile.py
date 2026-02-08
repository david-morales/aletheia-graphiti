"""Domain profile auto-discovered from graph data and ontology at startup."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

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


# Internal labels that should not appear as entity types
_INTERNAL_LABELS = frozenset({'Entity', 'Episodic', 'Community'})


async def _query_entity_types(driver, group_id: str) -> dict[str, EntityTypeInfo]:
    """Query data graph for entity labels and counts."""
    query = (
        'MATCH (n:Entity) '
        'WHERE n.group_id = $group_id '
        'RETURN labels(n) AS entity_type, count(n) AS count '
        'ORDER BY count DESC'
    )
    try:
        records, _, _ = await driver.execute_query(query, group_id=group_id)
    except Exception as e:
        logger.warning(f'Failed to query entity types: {e}')
        return {}

    types: dict[str, EntityTypeInfo] = {}
    for record in records:
        labels = record.get('entity_type', [])
        count = record.get('count', 0)
        for label in labels:
            if label in _INTERNAL_LABELS:
                continue
            if label in types:
                types[label].count += count
            else:
                types[label] = EntityTypeInfo(label=label, count=count, description='')
    return types


async def _query_edge_types(driver, group_id: str) -> dict[str, EdgeTypeInfo]:
    """Query data graph for relationship types and counts."""
    query = (
        'MATCH (s:Entity)-[r]->(t:Entity) '
        'WHERE s.group_id = $group_id AND t.group_id = $group_id '
        'RETURN type(r) AS relationship_type, count(r) AS count '
        'ORDER BY count DESC'
    )
    try:
        records, _, _ = await driver.execute_query(query, group_id=group_id)
    except Exception as e:
        logger.warning(f'Failed to query edge types: {e}')
        return {}

    types: dict[str, EdgeTypeInfo] = {}
    for record in records:
        name = record.get('relationship_type', '')
        count = record.get('count', 0)
        if name and name not in ('RELATES_TO',):
            types[name] = EdgeTypeInfo(name=name, count=count, description='')
    return types


async def _query_sample_names(driver, group_id: str, label: str, limit: int = 5) -> list[str]:
    """Get sample entity names for a given label."""
    query = (
        'MATCH (n:Entity) '
        'WHERE $label IN labels(n) AND n.group_id = $group_id '
        'RETURN n.name AS name '
        'LIMIT $limit'
    )
    try:
        records, _, _ = await driver.execute_query(
            query, group_id=group_id, label=label, limit=limit
        )
    except Exception as e:
        logger.warning(f'Failed to query sample names for {label}: {e}')
        return []
    return [r['name'] for r in records if r.get('name')]


async def _query_time_range(driver, group_id: str) -> tuple[str, str] | None:
    """Get the earliest and latest fact dates in the graph."""
    query = (
        'MATCH (s:Entity)-[r]->(t:Entity) '
        'WHERE s.group_id = $group_id AND r.created_at IS NOT NULL '
        'RETURN min(r.created_at) AS earliest, max(r.created_at) AS latest'
    )
    try:
        records, _, _ = await driver.execute_query(query, group_id=group_id)
    except Exception as e:
        logger.warning(f'Failed to query time range: {e}')
        return None
    if records and records[0].get('earliest') and records[0].get('latest'):
        earliest = str(records[0]['earliest'])[:10]
        latest = str(records[0]['latest'])[:10]
        return (earliest, latest)
    return None


async def build_domain_profile(
    client,
    group_id: str,
    ontology_client=None,
) -> DomainProfile:
    """Build a DomainProfile by introspecting data and ontology graphs."""
    driver = client.driver

    # Query data graph
    entity_types = await _query_entity_types(driver, group_id)
    edge_types = await _query_edge_types(driver, group_id)

    # Fetch sample names for each entity type
    for label, info in entity_types.items():
        info.sample_names = await _query_sample_names(driver, group_id, label)

    # Query time range
    time_range = await _query_time_range(driver, group_id)

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
