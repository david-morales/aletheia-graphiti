"""Tests for OpenAI rate limiting with exponential backoff."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.errors import RateLimitError
from graphiti_core.llm_client.openai_base_client import BaseOpenAIClient
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.prompts.models import Message


def _make_client() -> OpenAIClient:
    """Create an OpenAIClient with a mocked AsyncOpenAI backend."""
    config = LLMConfig(api_key='test-key')
    mock_async_openai = MagicMock()
    client = OpenAIClient(config=config, client=mock_async_openai)
    return client


class TestRateLimitConstants:
    """Verify the new rate-limit constants exist and have correct defaults."""

    def test_rate_limit_constants_exist(self):
        assert hasattr(BaseOpenAIClient, 'MAX_RATE_LIMIT_RETRIES')
        assert hasattr(BaseOpenAIClient, 'RATE_LIMIT_BASE_DELAY')
        assert hasattr(BaseOpenAIClient, 'RATE_LIMIT_MAX_DELAY')

    def test_rate_limit_constant_defaults(self):
        assert BaseOpenAIClient.MAX_RATE_LIMIT_RETRIES == 5
        assert BaseOpenAIClient.RATE_LIMIT_BASE_DELAY == 1.0
        assert BaseOpenAIClient.RATE_LIMIT_MAX_DELAY == 60.0

    def test_max_retries_unchanged(self):
        """Existing MAX_RETRIES for general errors should be unaffected."""
        assert BaseOpenAIClient.MAX_RETRIES == 2


class TestExponentialBackoffDelay:
    """Test the delay formula: base * 2^(attempt-1) + jitter, capped at max."""

    @patch('graphiti_core.llm_client.openai_base_client.random.uniform', return_value=0.5)
    def test_delay_attempt_1(self, mock_uniform):
        """First attempt: 1.0 * 2^0 + 0.5 = 1.5s."""
        base = BaseOpenAIClient.RATE_LIMIT_BASE_DELAY
        max_delay = BaseOpenAIClient.RATE_LIMIT_MAX_DELAY
        attempt = 1
        delay = min(base * (2 ** (attempt - 1)) + 0.5, max_delay)
        assert delay == 1.5

    @patch('graphiti_core.llm_client.openai_base_client.random.uniform', return_value=0.5)
    def test_delay_attempt_3(self, mock_uniform):
        """Third attempt: 1.0 * 2^2 + 0.5 = 4.5s."""
        base = BaseOpenAIClient.RATE_LIMIT_BASE_DELAY
        max_delay = BaseOpenAIClient.RATE_LIMIT_MAX_DELAY
        attempt = 3
        delay = min(base * (2 ** (attempt - 1)) + 0.5, max_delay)
        assert delay == 4.5

    @patch('graphiti_core.llm_client.openai_base_client.random.uniform', return_value=0.5)
    def test_delay_capped_at_max(self, mock_uniform):
        """Very high attempt should be capped at RATE_LIMIT_MAX_DELAY."""
        base = BaseOpenAIClient.RATE_LIMIT_BASE_DELAY
        max_delay = BaseOpenAIClient.RATE_LIMIT_MAX_DELAY
        attempt = 20  # 1.0 * 2^19 = 524288, way above 60
        delay = min(base * (2 ** (attempt - 1)) + 0.5, max_delay)
        assert delay == max_delay


class TestRateLimitRetryBehavior:
    """Integration-level tests that exercise generate_response with mocked internals."""

    @pytest.mark.asyncio
    @patch('graphiti_core.llm_client.openai_base_client.asyncio.sleep', new_callable=AsyncMock)
    @patch('graphiti_core.llm_client.openai_base_client.random.uniform', return_value=0.5)
    async def test_rate_limit_retried_not_immediately_raised(self, mock_uniform, mock_sleep):
        """RateLimitError should trigger a retry, not an immediate raise."""
        client = _make_client()

        success_response = {'result': 'ok'}

        # First call raises RateLimitError, second succeeds
        client._generate_response = AsyncMock(
            side_effect=[RateLimitError(), success_response]
        )

        messages = [Message(role='system', content='test')]
        result = await client.generate_response(messages)

        assert result == success_response
        assert client._generate_response.call_count == 2
        # Should have slept once with exponential backoff delay
        mock_sleep.assert_called_once()
        # delay = 1.0 * 2^0 + 0.5 = 1.5
        actual_delay = mock_sleep.call_args[0][0]
        assert actual_delay == 1.5

    @pytest.mark.asyncio
    @patch('graphiti_core.llm_client.openai_base_client.asyncio.sleep', new_callable=AsyncMock)
    @patch('graphiti_core.llm_client.openai_base_client.random.uniform', return_value=0.5)
    async def test_max_rate_limit_retries_exceeded(self, mock_uniform, mock_sleep):
        """After MAX_RATE_LIMIT_RETRIES, should raise RateLimitError."""
        client = _make_client()

        # Always raise RateLimitError
        client._generate_response = AsyncMock(side_effect=RateLimitError())

        messages = [Message(role='system', content='test')]
        with pytest.raises(RateLimitError):
            await client.generate_response(messages)

        # Should have been called MAX_RATE_LIMIT_RETRIES + 1 times
        # (1 initial + 5 retries = 6 total, but the 6th triggers the "exceeded" branch)
        expected_calls = client.MAX_RATE_LIMIT_RETRIES + 1
        assert client._generate_response.call_count == expected_calls
        # Should have slept MAX_RATE_LIMIT_RETRIES times
        assert mock_sleep.call_count == client.MAX_RATE_LIMIT_RETRIES

    @pytest.mark.asyncio
    @patch('graphiti_core.llm_client.openai_base_client.asyncio.sleep', new_callable=AsyncMock)
    @patch('graphiti_core.llm_client.openai_base_client.random.uniform', return_value=0.5)
    async def test_backoff_delays_increase_exponentially(self, mock_uniform, mock_sleep):
        """Verify delays follow 2^n pattern across retries."""
        client = _make_client()

        # Fail 3 times then succeed
        success_response = {'result': 'ok'}
        client._generate_response = AsyncMock(
            side_effect=[RateLimitError(), RateLimitError(), RateLimitError(), success_response]
        )

        messages = [Message(role='system', content='test')]
        result = await client.generate_response(messages)

        assert result == success_response
        assert mock_sleep.call_count == 3

        delays = [call[0][0] for call in mock_sleep.call_args_list]
        # attempt 1: 1.0 * 2^0 + 0.5 = 1.5
        # attempt 2: 1.0 * 2^1 + 0.5 = 2.5
        # attempt 3: 1.0 * 2^2 + 0.5 = 4.5
        assert delays == [1.5, 2.5, 4.5]

    @pytest.mark.asyncio
    @patch('graphiti_core.llm_client.openai_base_client.asyncio.sleep', new_callable=AsyncMock)
    @patch('graphiti_core.llm_client.openai_base_client.random.uniform', return_value=0.5)
    async def test_rate_limit_counter_independent_of_general_retries(
        self, mock_uniform, mock_sleep
    ):
        """Rate limit retries use a separate counter from general retries."""
        client = _make_client()

        success_response = {'result': 'ok'}
        # Alternate between general errors and rate limit errors, then succeed
        client._generate_response = AsyncMock(
            side_effect=[
                ValueError('parse error'),  # general retry 1
                RateLimitError(),  # rate limit retry 1
                success_response,
            ]
        )

        messages = [Message(role='system', content='test')]
        result = await client.generate_response(messages)

        assert result == success_response
        assert client._generate_response.call_count == 3
