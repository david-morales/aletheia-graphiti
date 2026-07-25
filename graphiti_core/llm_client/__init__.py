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

from .client import LLMClient  # noqa: F401
from .config import LLMConfig  # noqa: F401
from .errors import RateLimitError  # noqa: F401
from .openai_client import OpenAIClient  # noqa: F401
from .token_tracker import TokenUsage, TokenUsageTracker  # noqa: F401

# AnthropicClient is re-exported only when the `anthropic` package is
# installed. The underlying module raises ImportError at import time when
# the dependency is missing (lazy-dependency pattern), so we guard here
# to keep `from graphiti_core.llm_client import ...` working for the rest
# of the symbols in environments that don't install the anthropic extra.
try:
    from .anthropic_client import AnthropicClient  # noqa: F401

    _anthropic_available = True
except ImportError:
    _anthropic_available = False

# BedrockLLMClient is re-exported only when the `langchain-aws` package is
# installed (via the `bedrock` optional extra). Same lazy-dependency pattern
# as AnthropicClient.
try:
    from .bedrock_client import BedrockLLMClient  # noqa: F401

    _bedrock_available = True
except ImportError:
    _bedrock_available = False

_base = ['LLMClient', 'OpenAIClient', 'LLMConfig', 'RateLimitError', 'TokenUsage', 'TokenUsageTracker']
if _anthropic_available:
    _base.append('AnthropicClient')
if _bedrock_available:
    _base.append('BedrockLLMClient')

__all__ = _base
