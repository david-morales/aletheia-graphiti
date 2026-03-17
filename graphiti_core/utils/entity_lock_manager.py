"""Per-entity lock manager for coordinating concurrent deduplication.

Provides asyncio.Lock instances keyed on (entity_type, normalized_name) so that
resolution of the same entity is serialized across concurrent add_episode_bulk() calls,
while unrelated entities proceed in parallel.

Also maintains a pending-resolution registry: once an entity is resolved (mapped to a
canonical UUID/node), subsequent callers get the cached answer without re-searching
the graph.
"""

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graphiti_core.nodes import EntityNode


class EntityLockManager:
    """Coordinates entity deduplication across concurrent bulk ingestion calls.

    Thread-safety: this class uses asyncio primitives and is designed for use within
    a single event loop. It is NOT safe across OS threads.
    """

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._resolved: dict[str, "EntityNode"] = {}

    @staticmethod
    def normalize_key(entity_type: str, name: str) -> str:
        """Produce a stable cache key from entity type and name.

        Public so that callers can build the same key for grouping without
        duplicating the normalization logic.
        """
        return f"{entity_type.strip().lower()}:{name.strip().lower()}"

    def get_lock(self, entity_type: str, name: str) -> asyncio.Lock:
        """Return the lock for a given entity, creating it on first access."""
        key = self.normalize_key(entity_type, name)
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]

    def register_resolved(self, entity_type: str, name: str, node: "EntityNode") -> None:
        """Record that *name* of *entity_type* resolved to *node*."""
        key = self.normalize_key(entity_type, name)
        self._resolved[key] = node

    def get_resolved(self, entity_type: str, name: str) -> "EntityNode | None":
        """Return a previously resolved node, or None."""
        key = self.normalize_key(entity_type, name)
        return self._resolved.get(key)

    def clear(self) -> None:
        """Reset all locks and cached resolutions."""
        self._locks.clear()
        self._resolved.clear()
