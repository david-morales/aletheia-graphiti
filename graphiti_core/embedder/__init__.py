from .client import EmbedderClient  # noqa: F401
from .openai import OpenAIEmbedder, OpenAIEmbedderConfig  # noqa: F401

# BedrockEmbedder is re-exported only when the `langchain-aws` package is
# installed (via the `bedrock` optional extra). Same lazy-dependency pattern
# as BedrockLLMClient in graphiti_core.llm_client.
try:
    from .bedrock import BedrockEmbedder, BedrockEmbedderConfig  # noqa: F401

    _bedrock_available = True
except ImportError:
    _bedrock_available = False

_base = [
    'EmbedderClient',
    'OpenAIEmbedder',
    'OpenAIEmbedderConfig',
]
if _bedrock_available:
    _base.append('BedrockEmbedder')
    _base.append('BedrockEmbedderConfig')

__all__ = _base
