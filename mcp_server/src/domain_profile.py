"""Domain profile auto-discovered from graph data and ontology at startup."""

from __future__ import annotations

from dataclasses import dataclass, field


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
