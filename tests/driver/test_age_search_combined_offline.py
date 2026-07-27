"""Offline regression guards for the AGE combined-search episode-fanout fix.

Bug: ``search`` / ``search_ontology`` in combined mode fan out a node + edge +
episode sub-search via ``asyncio.gather``. ``AGESearch`` never overrode
``episode_fulltext_search``, so the base class raised a bare (empty-message)
``NotImplementedError``, failing the whole call over the wire on AGE.

The fix returns ``[]`` (documented shadow-table gap: only Entity nodes and edges
are mirrored into a tsvector table on AGE, not episodic nodes).

Fixing the episode fanout unmasked a SECOND combined-mode failure: the recipe
also fans out community fulltext + similarity search. Their caller catches
``NotImplementedError`` and falls through to a FalkorDB/Neo4j ``YIELD node …``
Cypher that AGE cannot parse (``syntax error at or near "."``), so AGE must
likewise override both community searches to ``[]``. (Only an over-the-wire /
live combined search exercises this — offline structural guards do not — see the
tool-coverage matrix in mcp_server/tests.)

Runs without a live backend so it gates every CI run, not just live ones.
"""

import pytest

from graphiti_core.driver.search_interface.age_search import AGESearch
from graphiti_core.driver.search_interface.search_interface import SearchInterface


def test_age_search_overrides_episode_fulltext_search():
    # Regression: the base method raised a bare NotImplementedError, which surfaced
    # as the empty "Search error:" on every AGE combined search.
    assert AGESearch.episode_fulltext_search is not SearchInterface.episode_fulltext_search


@pytest.mark.asyncio
async def test_age_episode_fulltext_search_returns_empty_not_raises():
    res = await AGESearch().episode_fulltext_search(
        driver=None, query='anything', search_filter=None, group_ids=['g'], limit=10
    )
    assert res == []


def test_age_search_overrides_community_search():
    # The combined recipe also fans out community fulltext + similarity search; the
    # caller catches NotImplementedError and falls through to FalkorDB Cypher
    # ("YIELD node …") which AGE cannot parse (syntax error at or near ".").
    assert AGESearch.community_fulltext_search is not SearchInterface.community_fulltext_search
    assert AGESearch.community_similarity_search is not SearchInterface.community_similarity_search


@pytest.mark.asyncio
async def test_age_community_searches_return_empty_not_raise():
    s = AGESearch()
    assert await s.community_fulltext_search(driver=None, query='x', group_ids=['g']) == []
    assert await s.community_similarity_search(
        driver=None, search_vector=[0.1], group_ids=['g']
    ) == []
