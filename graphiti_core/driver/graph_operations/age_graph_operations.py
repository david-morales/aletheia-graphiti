"""GraphOperationsInterface implementation for the PostgreSQL + Apache AGE driver.

Phase 0 spike: methods are implemented incrementally, driven by the tests that
exercise `add_episode` + hybrid `search`. Anything not yet implemented inherits
the base `raise NotImplementedError` and is out of scope for this phase
(communities, saga nodes, BFS, next/has-episode edges beyond MENTIONS).
"""

from graphiti_core.driver.graph_operations.graph_operations import GraphOperationsInterface


class AGEGraphOperations(GraphOperationsInterface):
    """Native-SQL graph mutation/read operations against AGE + pgvector shadow tables."""

    pass
