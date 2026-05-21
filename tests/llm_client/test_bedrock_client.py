"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Running tests: pytest -xvs tests/llm_client/test_bedrock_client.py

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helper: build a fake ChatBedrockConverse + its module
# ---------------------------------------------------------------------------


def _fake_langchain_aws_module(response_content):
    """Build a fake langchain_aws module exposing ChatBedrockConverse.

    The returned class, when instantiated, has an ``ainvoke`` AsyncMock that
    returns a response object whose ``content`` is ``response_content``.
    """
    fake_module = MagicMock()
    fake_chat = MagicMock()
    fake_response = MagicMock()
    fake_response.content = response_content
    fake_chat.ainvoke = AsyncMock(return_value=fake_response)
    fake_module.ChatBedrockConverse = MagicMock(return_value=fake_chat)
    return fake_module, fake_chat


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_bedrock_client_lazy_imports_langchain_aws():
    """The bedrock_client module must defer langchain_aws import to instance
    construction so importing graphiti_core.llm_client doesn't pull in AWS
    dependencies for users who don't use Bedrock."""
    fake_module, _ = _fake_langchain_aws_module(response_content='{"ok": true}')

    with patch.dict(sys.modules, {'langchain_aws': fake_module}):
        from graphiti_core.llm_client.bedrock_client import BedrockLLMClient
        from graphiti_core.llm_client.config import LLMConfig

        client = BedrockLLMClient(
            config=LLMConfig(
                api_key='not-required',
                model='amazon.nova-pro-v1:0',
                temperature=0.0,
                max_tokens=4096,
            )
        )
    assert client is not None
    # langchain_aws.ChatBedrockConverse was called once during construction
    fake_module.ChatBedrockConverse.assert_called_once()


@pytest.mark.asyncio
async def test_bedrock_client_generate_response_returns_parsed_json():
    """_generate_response wraps Graphiti Message[] in LangChain Human/System messages,
    awaits ChatBedrockConverse, and json.loads the content."""
    from graphiti_core.prompts.models import Message

    fake_module, fake_chat = _fake_langchain_aws_module(
        response_content='{"name": "Alice", "age": 30}'
    )

    with patch.dict(sys.modules, {'langchain_aws': fake_module}):
        from graphiti_core.llm_client.bedrock_client import BedrockLLMClient
        from graphiti_core.llm_client.config import LLMConfig

        client = BedrockLLMClient(
            config=LLMConfig(api_key='not-required', model='amazon.nova-pro-v1:0')
        )

        result = await client._generate_response(
            messages=[
                Message(role='system', content='You output JSON.'),
                Message(role='user', content='Extract name and age.'),
            ]
        )

    assert result == {'name': 'Alice', 'age': 30}
    fake_chat.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_bedrock_client_generate_response_handles_list_content():
    """Some Bedrock models return content as a list of {type: 'text', text: ...} blocks.
    The client should concatenate the text fields and parse as JSON."""
    from graphiti_core.prompts.models import Message

    list_content = [{'type': 'text', 'text': '{"answer":'}, {'type': 'text', 'text': ' 42}'}]
    fake_module, _ = _fake_langchain_aws_module(response_content=list_content)

    with patch.dict(sys.modules, {'langchain_aws': fake_module}):
        from graphiti_core.llm_client.bedrock_client import BedrockLLMClient
        from graphiti_core.llm_client.config import LLMConfig

        client = BedrockLLMClient(
            config=LLMConfig(api_key='not-required', model='amazon.nova-pro-v1:0')
        )

        result = await client._generate_response(
            messages=[Message(role='user', content='What is the answer?')]
        )

    assert result == {'answer': 42}


def test_bedrock_client_raises_helpful_error_when_langchain_aws_missing():
    """If langchain_aws is not installed, instantiating BedrockLLMClient
    raises ImportError pointing to the install command."""
    # Pretend langchain_aws is genuinely absent — pop it from sys.modules
    # and remove it from import paths via a None entry.
    from graphiti_core.llm_client.config import LLMConfig

    with patch.dict(sys.modules, {'langchain_aws': None}):
        # Force a fresh import of the bedrock_client module so the import
        # attempt happens under the patched sys.modules.
        sys.modules.pop('graphiti_core.llm_client.bedrock_client', None)
        from graphiti_core.llm_client.bedrock_client import BedrockLLMClient

        with pytest.raises(ImportError, match='langchain-aws'):
            BedrockLLMClient(config=LLMConfig(api_key='nr', model='amazon.nova-pro-v1:0'))
