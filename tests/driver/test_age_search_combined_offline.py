"""Offline regression guards for the AGE combined-search episode-fanout fix.

Bug: ``search`` / ``search_ontology`` in combined mode fan out a node + edge +
episode sub-search via ``asyncio.gather``. ``AGESearch`` never overrode
``episode_fulltext_search``, so the base class raised a bare (empty-message)
``NotImplementedError``, failing the whole call over the wire on AGE.

The fix returns ``[]`` (documented shadow-table gap: only Entity nodes and edges
are mirrored into a tsvector table on AGE, not episodic nodes).

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
