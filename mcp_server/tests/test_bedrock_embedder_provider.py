"""Tests for the Bedrock embedder provider integration in the Graphiti MCP server."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# BUG-101: these tests `patch('langchain_aws....')`, and `mock.patch` IMPORTS the
# target module — so without langchain_aws installed the module does not merely
# skip, it errors. It is an OPTIONAL dependency here (the `bedrock` extra); a
# plain `uv sync` venv does not carry it, and this suite passed only on machines
# with an ad-hoc `uv pip install langchain-aws`. Skip cleanly instead.
pytest.importorskip('langchain_aws')


def _fake_bedrock_embeddings():
    fake_module = MagicMock()
    fake_emb = MagicMock()
    fake_module.BedrockEmbeddings = MagicMock(return_value=fake_emb)
    return fake_module, fake_emb


def test_embedder_providers_config_recognises_bedrock():
    """EmbedderProvidersConfig accepts a `bedrock` field reusing BedrockProviderConfig."""
    from mcp_server.src.config.schema import BedrockProviderConfig, EmbedderProvidersConfig

    cfg = EmbedderProvidersConfig(bedrock=BedrockProviderConfig(region='eu-west-1'))
    assert cfg.bedrock is not None
    assert cfg.bedrock.region == 'eu-west-1'


def test_embedder_providers_config_bedrock_defaults_to_none():
    """EmbedderProvidersConfig.bedrock is None by default (other providers may be set)."""
    from mcp_server.src.config.schema import EmbedderProvidersConfig

    cfg = EmbedderProvidersConfig()
    assert cfg.bedrock is None


def test_embedder_factory_dispatches_to_bedrock_embedder():
    """EmbedderFactory.create returns a BedrockEmbedder when provider='bedrock'."""
    with patch('langchain_aws.BedrockEmbeddings', MagicMock(return_value=MagicMock())):
        from graphiti_core.embedder.bedrock import BedrockEmbedder
        from mcp_server.src.config.schema import (
            BedrockProviderConfig,
            EmbedderConfig,
            EmbedderProvidersConfig,
        )
        from mcp_server.src.services.factories import EmbedderFactory

        cfg = EmbedderConfig(
            provider='bedrock',
            model='amazon.titan-embed-text-v2',
            dimensions=1024,
            providers=EmbedderProvidersConfig(bedrock=BedrockProviderConfig(region='us-east-1')),
        )
        embedder = EmbedderFactory.create(cfg)

    assert isinstance(embedder, BedrockEmbedder)


def test_embedder_factory_bedrock_raises_when_provider_config_missing():
    """If provider='bedrock' but providers.bedrock is None, factory raises ValueError."""
    from mcp_server.src.config.schema import EmbedderConfig, EmbedderProvidersConfig
    from mcp_server.src.services.factories import EmbedderFactory

    cfg = EmbedderConfig(
        provider='bedrock',
        model='amazon.titan-embed-text-v2',
        dimensions=1024,
        providers=EmbedderProvidersConfig(),
    )
    with pytest.raises(ValueError, match='Bedrock provider configuration'):
        EmbedderFactory.create(cfg)


def test_embedder_factory_bedrock_passes_dimensions_through():
    """The dimensions field in EmbedderConfig is forwarded to BedrockEmbedderConfig."""
    with patch('langchain_aws.BedrockEmbeddings', MagicMock(return_value=MagicMock())):
        from graphiti_core.embedder.bedrock import BedrockEmbedder
        from mcp_server.src.config.schema import (
            BedrockProviderConfig,
            EmbedderConfig,
            EmbedderProvidersConfig,
        )
        from mcp_server.src.services.factories import EmbedderFactory

        cfg = EmbedderConfig(
            provider='bedrock',
            model='cohere.embed-english-v3',
            dimensions=1024,
            providers=EmbedderProvidersConfig(bedrock=BedrockProviderConfig(region='us-east-1')),
        )
        embedder = EmbedderFactory.create(cfg)

    assert isinstance(embedder, BedrockEmbedder)
    assert embedder.config.embedding_model == 'cohere.embed-english-v3'
    assert embedder.config.embedding_dim == 1024
