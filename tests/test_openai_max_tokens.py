"""Tests for max_tokens resolution in BaseOpenAIClient / OpenAIClient.

Verifies that LLMConfig.max_tokens is respected when no explicit max_tokens
is passed to the constructor, and that an explicit max_tokens overrides the
config value.

Addresses: https://github.com/getzep/graphiti/issues/764
"""

from unittest.mock import MagicMock

import pytest

from graphiti_core.llm_client.config import DEFAULT_MAX_TOKENS, LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient


@pytest.fixture
def mock_openai_async_client():
    """Return a mock that stands in for AsyncOpenAI so no real API call is made."""
    return MagicMock()


# ---------- config.max_tokens respected ----------


def test_config_max_tokens_respected(mock_openai_async_client):
    """When config sets a custom max_tokens and the constructor does not
    override it, the client should use the config value."""
    config = LLMConfig(api_key="test-key", max_tokens=2000)
    client = OpenAIClient(config=config, client=mock_openai_async_client)
    assert client.max_tokens == 2000


def test_config_max_tokens_small_value(mock_openai_async_client):
    """Config max_tokens of 512 should be preserved."""
    config = LLMConfig(api_key="test-key", max_tokens=512)
    client = OpenAIClient(config=config, client=mock_openai_async_client)
    assert client.max_tokens == 512


# ---------- explicit max_tokens overrides config ----------


def test_explicit_max_tokens_overrides_config(mock_openai_async_client):
    """When both config and explicit max_tokens are provided, explicit wins."""
    config = LLMConfig(api_key="test-key", max_tokens=2000)
    client = OpenAIClient(config=config, client=mock_openai_async_client, max_tokens=4000)
    assert client.max_tokens == 4000


def test_explicit_max_tokens_without_config(mock_openai_async_client):
    """When no config is provided but explicit max_tokens is, explicit wins."""
    client = OpenAIClient(client=mock_openai_async_client, max_tokens=8000)
    assert client.max_tokens == 8000


# ---------- default behaviour (no config, no explicit) ----------


def test_default_max_tokens(mock_openai_async_client):
    """With no config and no explicit max_tokens, the default should apply."""
    client = OpenAIClient(client=mock_openai_async_client)
    assert client.max_tokens == DEFAULT_MAX_TOKENS


def test_default_config_equals_default_max_tokens(mock_openai_async_client):
    """A bare LLMConfig() should give DEFAULT_MAX_TOKENS."""
    config = LLMConfig(api_key="test-key")
    client = OpenAIClient(config=config, client=mock_openai_async_client)
    assert client.max_tokens == DEFAULT_MAX_TOKENS


# ---------- edge case: config explicitly set to DEFAULT_MAX_TOKENS ----------


def test_config_set_to_default_value(mock_openai_async_client):
    """If config explicitly sets max_tokens to DEFAULT_MAX_TOKENS (16384),
    the result should still be DEFAULT_MAX_TOKENS.  This is a degenerate case
    -- there is no observable difference from the true default."""
    config = LLMConfig(api_key="test-key", max_tokens=DEFAULT_MAX_TOKENS)
    client = OpenAIClient(config=config, client=mock_openai_async_client)
    assert client.max_tokens == DEFAULT_MAX_TOKENS
