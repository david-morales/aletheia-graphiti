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

Graphiti LLMClient for Amazon Bedrock via langchain-aws's ChatBedrockConverse.

Requires: pip install graphiti-core[bedrock]

AWS credentials resolved through the standard boto3 chain:
  1. AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY (environment variables)
  2. IAM role attached to the pod / instance (IRSA)
  3. ~/.aws/credentials
"""

from __future__ import annotations

import json
import logging
import os
import typing

from pydantic import BaseModel

from ..prompts.models import Message
from .client import LLMClient
from .config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize

logger = logging.getLogger(__name__)

# Conservative default for Bedrock invocation — half of Graphiti's
# DEFAULT_MAX_TOKENS (16384). Several Bedrock model families (older Titan,
# some Nova variants) advertise smaller output windows than the 16K Graphiti
# uses as a generic ceiling, and going above the model's actual limit raises
# at the AWS Converse API. Callers can always override via LLMConfig.max_tokens.
_BEDROCK_FALLBACK_MAX_TOKENS = 8192


class BedrockLLMClient(LLMClient):
    """Graphiti LLMClient that delegates to LangChain's ChatBedrockConverse.

    The actual AWS SDK call goes through langchain_aws.ChatBedrockConverse,
    which speaks the Bedrock Converse API and handles credential resolution
    via the standard boto3 chain.

    Install the bedrock extra to pull langchain-aws:
        pip install graphiti-core[bedrock]
    """

    def __init__(self, config: LLMConfig | None = None, cache: bool = False) -> None:
        try:
            from langchain_aws import ChatBedrockConverse
        except ImportError as e:
            raise ImportError(
                'langchain-aws is required for BedrockLLMClient. '
                'Install it with: pip install graphiti-core[bedrock]'
            ) from e

        super().__init__(config, cache)

        region = os.getenv('AWS_DEFAULT_REGION', os.getenv('AWS_REGION', 'us-east-1'))
        kwargs: dict[str, typing.Any] = dict(
            model_id=self.model,
            region_name=region,
            temperature=self.temperature,
            max_tokens=self.max_tokens or _BEDROCK_FALLBACK_MAX_TOKENS,
        )

        # Forward explicit AWS creds when set in the environment; otherwise
        # ChatBedrockConverse falls through to the boto3 chain (IRSA, ~/.aws/...).
        aws_key = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY')
        aws_session = os.getenv('AWS_SESSION_TOKEN')
        if aws_key and aws_secret:
            kwargs['aws_access_key_id'] = aws_key
            kwargs['aws_secret_access_key'] = aws_secret
            if aws_session:
                kwargs['aws_session_token'] = aws_session

        self._chat = ChatBedrockConverse(**kwargs)

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        """Bedrock-side generation: forward messages to ChatBedrockConverse and parse JSON.

        ``response_model`` and ``model_size`` are accepted to satisfy the
        :class:`LLMClient` ABC but are not consumed here. The base class
        :meth:`LLMClient.generate_response` performs JSON-schema injection
        before calling this method, and ``model_size`` routing is not yet
        implemented for Bedrock (single ``model_id`` per client instance).

        Returns the parsed JSON object. Raises ``json.JSONDecodeError`` if
        the model output is not valid JSON.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        lc_messages: list[typing.Any] = []
        for m in messages:
            m.content = self._clean_input(m.content)
            if m.role == 'system':
                lc_messages.append(SystemMessage(content=m.content))
            else:
                lc_messages.append(HumanMessage(content=m.content))

        response = await self._chat.ainvoke(lc_messages)
        content = response.content
        # Some Bedrock models return content as a list of {type, text} blocks;
        # concatenate the text fields before parsing.
        if isinstance(content, list):
            content = ''.join(
                block.get('text', '') if isinstance(block, dict) else str(block)
                for block in content
            )
        return json.loads(content)

    def _get_provider_type(self) -> str:
        """Override base-class heuristic to emit a clean provider name for tracing."""
        return 'bedrock'
