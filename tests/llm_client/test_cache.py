"""Tests for SQLite-based LLM cache (replacement for diskcache). Upstream #1238."""

import os
import tempfile

import pytest


@pytest.fixture
def cache_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def test_cache_get_miss(cache_dir):
    from graphiti_core.llm_client.cache import LLMCache
    cache = LLMCache(cache_dir)
    assert cache.get('nonexistent') is None
    cache.close()


def test_cache_set_and_get(cache_dir):
    from graphiti_core.llm_client.cache import LLMCache
    cache = LLMCache(cache_dir)
    cache.set('key1', {'result': 'hello', 'tokens': 42})
    assert cache.get('key1') == {'result': 'hello', 'tokens': 42}
    cache.close()


def test_cache_overwrite(cache_dir):
    from graphiti_core.llm_client.cache import LLMCache
    cache = LLMCache(cache_dir)
    cache.set('key1', {'v': 1})
    cache.set('key1', {'v': 2})
    assert cache.get('key1') == {'v': 2}
    cache.close()


def test_cache_persists_across_instances(cache_dir):
    from graphiti_core.llm_client.cache import LLMCache
    c1 = LLMCache(cache_dir)
    c1.set('persist', {'data': True})
    c1.close()
    c2 = LLMCache(cache_dir)
    assert c2.get('persist') == {'data': True}
    c2.close()


def test_cache_creates_directory():
    from graphiti_core.llm_client.cache import LLMCache
    with tempfile.TemporaryDirectory() as base:
        nested = os.path.join(base, 'sub', 'dir')
        cache = LLMCache(nested)
        cache.set('key', {'ok': True})
        assert cache.get('key') == {'ok': True}
        cache.close()


def test_cache_handles_non_serializable_value(cache_dir):
    from graphiti_core.llm_client.cache import LLMCache
    cache = LLMCache(cache_dir)
    cache.set('bad', {'fn': lambda x: x})
    assert cache.get('bad') is None
    cache.close()


def test_cache_handles_corrupted_entry(cache_dir):
    import sqlite3
    from graphiti_core.llm_client.cache import LLMCache
    db_path = os.path.join(cache_dir, 'cache.db')
    conn = sqlite3.connect(db_path)
    conn.execute('CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)')
    conn.execute("INSERT INTO cache (key, value) VALUES ('corrupt', 'not-valid-json{')")
    conn.commit()
    conn.close()

    cache = LLMCache(cache_dir)
    assert cache.get('corrupt') is None
    cache.close()
