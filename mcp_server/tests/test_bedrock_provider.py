"""Tests for the Bedrock provider integration in the Graphiti MCP server."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _fake_langchain_aws_module():
    """Helper: provide a stand-in for langchain_aws.ChatBedrockConverse."""
    fake_module = MagicMock()
    fake_chat = MagicMock()
    fake_module.ChatBedrockConverse = MagicMock(return_value=fake_chat)
    return fake_module, fake_chat


def test_bedrock_provider_config_accepts_region_and_max_retries():
    """BedrockProviderConfig is a valid pydantic model with the expected fields."""
    from mcp_server.src.config.schema import BedrockProviderConfig

    cfg = BedrockProviderConfig(region='eu-west-1', max_retries=5)
    assert cfg.region == 'eu-west-1'
    assert cfg.max_retries == 5


def test_bedrock_provider_config_has_sensible_defaults():
    """Region defaults to None (boto3 chain decides); max_retries defaults to 3."""
    from mcp_server.src.config.schema import BedrockProviderConfig

    cfg = BedrockProviderConfig()
    assert cfg.region is None
    assert cfg.max_retries == 3


def test_llm_providers_config_recognises_bedrock():
    """LLMProvidersConfig accepts a `bedrock` field."""
    from mcp_server.src.config.schema import BedrockProviderConfig, LLMProvidersConfig

    cfg = LLMProvidersConfig(bedrock=BedrockProviderConfig(region='us-east-1'))
    assert cfg.bedrock is not None
    assert cfg.bedrock.region == 'us-east-1'


def test_llm_client_factory_dispatches_to_bedrock():
    """LLMClientFactory.create returns a BedrockLLMClient when provider='bedrock'."""
    # langchain_aws is installed in this environment, so we can import the real
    # BedrockLLMClient. We still patch ChatBedrockConverse to avoid needing live AWS
    # credentials at test time, but we don't need to replace the whole langchain_aws module.
    fake_module, _ = _fake_langchain_aws_module()

    from graphiti_core.llm_client.bedrock_client import BedrockLLMClient
    from mcp_server.src.config.schema import (
        BedrockProviderConfig,
        LLMConfig,
        LLMProvidersConfig,
    )
    from mcp_server.src.services.factories import LLMClientFactory

    # Patch only ChatBedrockConverse so BedrockLLMClient.__init__ doesn't require
    # real AWS credentials. The rest of langchain_aws remains the real package.
    with patch('langchain_aws.ChatBedrockConverse', fake_module.ChatBedrockConverse):
        cfg = LLMConfig(
            provider='bedrock',
            model='amazon.nova-pro-v1:0',
            max_tokens=4096,
            providers=LLMProvidersConfig(bedrock=BedrockProviderConfig(region='us-east-1')),
        )
        client = LLMClientFactory.create(cfg)

    assert isinstance(client, BedrockLLMClient)


def test_llm_client_factory_bedrock_raises_when_provider_config_missing():
    """If provider='bedrock' but providers.bedrock is None, factory raises ValueError."""
    from mcp_server.src.config.schema import LLMConfig, LLMProvidersConfig
    from mcp_server.src.services.factories import LLMClientFactory

    cfg = LLMConfig(
        provider='bedrock',
        model='amazon.nova-pro-v1:0',
        providers=LLMProvidersConfig(),  # no bedrock entry
    )
    with pytest.raises(ValueError, match='Bedrock provider configuration'):
        LLMClientFactory.create(cfg)
