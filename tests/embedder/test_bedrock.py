"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
"""

# Running tests: pytest -xvs tests/embedder/test_bedrock.py

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _fake_langchain_aws_module(vectors):
    """Build a fake langchain_aws module exposing BedrockEmbeddings."""
    fake_module = MagicMock()
    fake_emb = MagicMock()
    fake_emb.aembed_documents = AsyncMock(return_value=vectors)
    fake_emb.embed_query = MagicMock(return_value=vectors[0] if vectors else [])
    fake_emb.aembed_query = AsyncMock(return_value=vectors[0] if vectors else [])
    fake_module.BedrockEmbeddings = MagicMock(return_value=fake_emb)
    return fake_module, fake_emb


def test_bedrock_embedder_lazy_imports_langchain_aws():
    """BedrockEmbedder defers langchain_aws import to instance construction."""
    fake_module, _ = _fake_langchain_aws_module(vectors=[[0.1] * 1024])

    with patch.dict(sys.modules, {'langchain_aws': fake_module}):
        from graphiti_core.embedder.bedrock import (
            BedrockEmbedder,
            BedrockEmbedderConfig,
        )

        emb = BedrockEmbedder(
            config=BedrockEmbedderConfig(
                embedding_model='amazon.titan-embed-text-v2',
                embedding_dim=1024,
            )
        )

    assert emb is not None
    fake_module.BedrockEmbeddings.assert_called_once()


@pytest.mark.asyncio
async def test_bedrock_embedder_create_returns_single_vector():
    """BedrockEmbedder.create wraps embed_query and returns a list[float]."""
    fake_module, _ = _fake_langchain_aws_module(vectors=[[0.5] * 1024])

    with patch.dict(sys.modules, {'langchain_aws': fake_module}):
        from graphiti_core.embedder.bedrock import (
            BedrockEmbedder,
            BedrockEmbedderConfig,
        )

        emb = BedrockEmbedder(
            config=BedrockEmbedderConfig(
                embedding_model='amazon.titan-embed-text-v2',
                embedding_dim=1024,
            )
        )
        result = await emb.create('hello world')

    assert len(result) == 1024
    assert all(v == 0.5 for v in result)


@pytest.mark.asyncio
async def test_bedrock_embedder_create_batch_returns_per_input_vector():
    """BedrockEmbedder.create_batch returns one vector per input string."""
    vectors = [[0.1] * 1024, [0.2] * 1024, [0.3] * 1024]
    fake_module, _ = _fake_langchain_aws_module(vectors=vectors)

    with patch.dict(sys.modules, {'langchain_aws': fake_module}):
        from graphiti_core.embedder.bedrock import (
            BedrockEmbedder,
            BedrockEmbedderConfig,
        )

        emb = BedrockEmbedder(
            config=BedrockEmbedderConfig(
                embedding_model='amazon.titan-embed-text-v2',
                embedding_dim=1024,
            )
        )
        result = await emb.create_batch(['a', 'b', 'c'])

    assert len(result) == 3
    assert all(len(v) == 1024 for v in result)


def test_bedrock_embedder_raises_helpful_error_when_langchain_aws_missing():
    """ImportError points to graphiti-core[bedrock] when langchain_aws absent."""
    with patch.dict(sys.modules, {'langchain_aws': None}):
        sys.modules.pop('graphiti_core.embedder.bedrock', None)
        from graphiti_core.embedder.bedrock import (
            BedrockEmbedder,
            BedrockEmbedderConfig,
        )

        with pytest.raises(ImportError, match='langchain-aws'):
            BedrockEmbedder(
                config=BedrockEmbedderConfig(
                    embedding_model='amazon.titan-embed-text-v2', embedding_dim=1024
                )
            )
