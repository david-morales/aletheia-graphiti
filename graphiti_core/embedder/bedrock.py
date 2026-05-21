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

Graphiti EmbedderClient for Amazon Bedrock via langchain-aws's
BedrockEmbeddings. Mirrors the BedrockLLMClient design — same boto3
credential chain, same lazy-import pattern, same `bedrock` extra.

Requires: pip install graphiti-core[bedrock]
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable

from .client import EmbedderClient, EmbedderConfig

logger = logging.getLogger(__name__)

DEFAULT_BEDROCK_EMBEDDING_MODEL = 'amazon.titan-embed-text-v2'
DEFAULT_BEDROCK_EMBEDDING_DIM = 1024


class BedrockEmbedderConfig(EmbedderConfig):
    """Configuration for the Bedrock embedder.

    embedding_model: Bedrock model id (e.g., amazon.titan-embed-text-v2,
        cohere.embed-english-v3). Defaults to amazon.titan-embed-text-v2.
    embedding_dim: dimensionality the model returns (1024 for titan-v2,
        1536 for titan-v1, 1024 for cohere-english-v3). Override per model.
    """

    embedding_model: str = DEFAULT_BEDROCK_EMBEDDING_MODEL
    embedding_dim: int = DEFAULT_BEDROCK_EMBEDDING_DIM


class BedrockEmbedder(EmbedderClient):
    """Embedder that delegates to langchain_aws.BedrockEmbeddings.

    AWS credentials resolved via the standard boto3 chain (env vars >
    IAM role / IRSA > ~/.aws/credentials). Region from AWS_DEFAULT_REGION
    or AWS_REGION (default us-east-1).
    """

    def __init__(self, config: BedrockEmbedderConfig | None = None) -> None:
        try:
            from langchain_aws import BedrockEmbeddings
        except ImportError as e:
            raise ImportError(
                'langchain-aws is required for BedrockEmbedder. '
                'Install it with: pip install graphiti-core[bedrock]'
            ) from e

        if config is None:
            config = BedrockEmbedderConfig()
        self.config = config

        region = os.getenv('AWS_DEFAULT_REGION', os.getenv('AWS_REGION', 'us-east-1'))
        kwargs: dict = {
            'model_id': config.embedding_model,
            'region_name': region,
        }
        # Forward explicit credentials when present; otherwise boto3 chain.
        aws_key = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY')
        aws_session = os.getenv('AWS_SESSION_TOKEN')
        if aws_key and aws_secret:
            kwargs['aws_access_key_id'] = aws_key
            kwargs['aws_secret_access_key'] = aws_secret
            if aws_session:
                kwargs['aws_session_token'] = aws_session

        self._embedder = BedrockEmbeddings(**kwargs)

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        """Embed a single input (str or list of strings concatenated) and return
        one vector. Truncates to `embedding_dim` for symmetry with LocalEmbedder."""
        if isinstance(input_data, str):
            text = input_data
        elif isinstance(input_data, list) and all(isinstance(x, str) for x in input_data):
            text = ' '.join(input_data)
        else:
            text = ' '.join(str(x) for x in input_data)

        vector = await self._embedder.aembed_query(text)
        return list(vector[: self.config.embedding_dim])

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        """Batch-embed a list of strings. Returns one vector per input."""
        vectors = await self._embedder.aembed_documents(input_data_list)
        return [list(v[: self.config.embedding_dim]) for v in vectors]
