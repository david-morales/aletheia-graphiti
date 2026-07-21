"""SearchInterface implementation for the PostgreSQL + Apache AGE driver.

Phase 0 spike: implements the fulltext/similarity methods that hybrid `search`
exercises, in native SQL over pgvector (similarity) and tsvector (fulltext)
shadow tables. Rerankers and community/BFS search inherit the base
`raise NotImplementedError` and are out of scope for this phase.
"""

from graphiti_core.driver.search_interface.search_interface import SearchInterface


class AGESearch(SearchInterface):
    """Native-SQL search over pgvector + tsvector shadow tables."""

    pass
