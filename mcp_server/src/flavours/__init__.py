"""Backend flavours for the Graphiti MCP server (dialect / quality / auto-fix as data)."""
from __future__ import annotations

from flavours.base import BaseFlavour, Flavour

__all__ = ["Flavour", "BaseFlavour", "build_flavour"]


def build_flavour(provider: str) -> Flavour:
    """Return the Flavour for a database provider. Unknown providers get BaseFlavour."""
    p = (provider or "").lower()
    if p == "falkordb":
        from flavours.falkordb import FalkorDbFlavour

        return FalkorDbFlavour()
    if p == "age":
        from flavours.age import AgeFlavour

        return AgeFlavour()
    # neo4j and anything else: the generic openCypher base.
    return BaseFlavour()
