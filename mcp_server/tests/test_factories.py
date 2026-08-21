#!/usr/bin/env python3
"""Unit tests for service factory client routing.

The `is_non_openai_provider` / `reasoning_effort_for_model` tests that used to
live here were removed with BUG-101: neither symbol exists anywhere under
`src/` — `is_non_openai_provider` was last seen upstream in `2269d48` — so the
classes covering them tested nothing and took the whole module down at import.
The factory routing they sat beside is live and still covered below.
"""

import sys
from pathlib import Path

import pytest

# Add the src directory to the path (mirrors the other factory tests)
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from graphiti_core.llm_client import OpenAIClient
from graphiti_core.llm_client.azure_openai_client import AzureOpenAILLMClient

from config.schema import (
    AzureOpenAIProviderConfig,
    LLMConfig,
    LLMProvidersConfig,
    OpenAIProviderConfig,
)
from services.factories import LLMClientFactory


class TestLLMClientFactoryRouting:
    """Tests that the factory selects the right client based on base_url."""

    @staticmethod
    def _config(api_url: str) -> LLMConfig:
        return LLMConfig(
            provider='openai',
            model='gpt-5.5',
            providers=LLMProvidersConfig(
                openai=OpenAIProviderConfig(api_key='test-key', api_url=api_url)
            ),
        )

    def test_official_openai_uses_openai_client(self):
        client = LLMClientFactory.create(self._config('https://api.openai.com/v1'))
        assert isinstance(client, OpenAIClient)

    def test_a_compatible_endpoint_also_uses_the_openai_client(self):
        """BUG-101: base_url-based routing to `OpenAIGenericClient` is GONE.

        Upstream #1146 routed OpenAI-compatible endpoints to the generic client
        via `is_non_openai_provider`; that symbol and the branch that used it
        are both absent from `services/factories.py` here. This pins what the
        factory ACTUALLY does, so the next person reads the behaviour rather
        than a red assertion describing a deleted feature.
        """
        client = LLMClientFactory.create(self._config('http://localhost:11434/v1'))
        assert isinstance(client, OpenAIClient)


class TestLLMClientReasoningEffort:
    """The OpenAI factory selects reasoning effort by model family."""

    @staticmethod
    def _config(model: str) -> LLMConfig:
        return LLMConfig(
            provider='openai',
            model=model,
            providers=LLMProvidersConfig(
                openai=OpenAIProviderConfig(api_key='test-key', api_url='https://api.openai.com/v1')
            ),
        )

    def test_gpt_5_5_uses_the_reasoning_floor(self):
        """BUG-101: there is no per-model effort selection any more.

        `reasoning_effort_for_model` is absent from `services/factories.py`;
        the factory gives every model in the `('o1', 'o3', 'gpt-5')` family the
        same 'minimal' floor. The old assertion (`== 'none'`) described that
        deleted helper.
        """
        client = LLMClientFactory.create(self._config('gpt-5.5'))
        assert isinstance(client, OpenAIClient)
        assert client.reasoning == 'minimal'

    def test_earlier_reasoning_model_uses_minimal(self):
        """Earlier gpt-5 reasoning models keep the historical 'minimal' floor."""
        client = LLMClientFactory.create(self._config('gpt-5'))
        assert isinstance(client, OpenAIClient)
        assert client.reasoning == 'minimal'


class TestAzureReasoningEffort:
    """BUG-101: the Azure branch sets no reasoning effort AT ALL.

    `LLMClientFactory.create` builds `AzureOpenAILLMClient(azure_client, config,
    max_tokens)` and never passes `reasoning` — so it is `None` for every model,
    reasoning-family or not. The previous pair of tests read as a model-tied
    selection, and only the `gpt-4.1` one passed, by coincidence: it asserted
    the `None` that every model gets.
    """

    @staticmethod
    def _config(model: str) -> LLMConfig:
        return LLMConfig(
            provider='azure_openai',
            model=model,
            providers=LLMProvidersConfig(
                azure_openai=AzureOpenAIProviderConfig(
                    api_key='test-key',
                    api_url='https://example.openai.azure.com',
                )
            ),
        )

    @pytest.mark.parametrize('model', ['gpt-5.5', 'gpt-4.1'])
    def test_azure_sends_no_reasoning_effort_for_any_model(self, model):
        client = LLMClientFactory.create(self._config(model))
        assert isinstance(client, AzureOpenAILLMClient)
        assert client.reasoning is None
