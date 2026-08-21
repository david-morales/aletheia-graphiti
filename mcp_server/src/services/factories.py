"""Factory classes for creating LLM, Embedder, and Database clients."""

import os

import httpx

from config.schema import (
    DEFAULT_CONNECT_TIMEOUT_SECONDS,
    DatabaseConfig,
    EmbedderConfig,
    LLMConfig,
)

# Try to import FalkorDriver if available
try:
    from graphiti_core.driver.falkordb_driver import FalkorDriver  # noqa: F401

    HAS_FALKOR = True
except ImportError:
    HAS_FALKOR = False

# Kuzu support removed - FalkorDB is now the default
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.embedder import EmbedderClient, OpenAIEmbedder
from graphiti_core.llm_client import LLMClient, OpenAIClient
from graphiti_core.llm_client.config import LLMConfig as GraphitiLLMConfig

# Try to import additional providers if available
try:
    from graphiti_core.embedder.azure_openai import AzureOpenAIEmbedderClient

    HAS_AZURE_EMBEDDER = True
except ImportError:
    HAS_AZURE_EMBEDDER = False

try:
    from graphiti_core.embedder.gemini import GeminiEmbedder

    HAS_GEMINI_EMBEDDER = True
except ImportError:
    HAS_GEMINI_EMBEDDER = False

try:
    from graphiti_core.embedder.voyage import VoyageAIEmbedder

    HAS_VOYAGE_EMBEDDER = True
except ImportError:
    HAS_VOYAGE_EMBEDDER = False

try:
    from graphiti_core.llm_client.azure_openai_client import AzureOpenAILLMClient

    HAS_AZURE_LLM = True
except ImportError:
    HAS_AZURE_LLM = False

try:
    from graphiti_core.llm_client.anthropic_client import AnthropicClient

    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from graphiti_core.llm_client.gemini_client import GeminiClient

    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    from graphiti_core.llm_client.groq_client import GroqClient

    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

try:
    from graphiti_core.llm_client.bedrock_client import BedrockLLMClient

    HAS_BEDROCK = True
except ImportError:
    BedrockLLMClient = None  # type: ignore[assignment,misc]
    HAS_BEDROCK = False

try:
    from graphiti_core.embedder.bedrock import BedrockEmbedder, BedrockEmbedderConfig

    HAS_BEDROCK_EMBEDDER = True
except ImportError:
    BedrockEmbedder = None  # type: ignore[assignment,misc]
    BedrockEmbedderConfig = None  # type: ignore[assignment,misc]
    HAS_BEDROCK_EMBEDDER = False


def request_timeout(seconds: float) -> httpx.Timeout:
    """Build the timeout every outbound provider client is constructed with (BUG-96).

    Returned as an `httpx.Timeout` rather than a bare float on purpose. A float
    sets all four legs to the same value, which would stretch `connect` from the
    SDK's 5 s out to `seconds` — the opposite of what this fix is for, since an
    unreachable edge should be reported immediately, not five minutes later.

    `read` is the leg that matters: it is the one a socket in CLOSE_WAIT sits on.
    `write` and `pool` follow it so a stall cannot simply move to a neighbouring
    leg — a request blocked forever waiting for a connection from the pool is the
    same outage as one blocked forever on the read.

    Both the openai and anthropic SDKs accept an `httpx.Timeout` directly, so the
    same object serves every provider path here.
    """
    return httpx.Timeout(seconds, connect=DEFAULT_CONNECT_TIMEOUT_SECONDS)


def _validate_api_key(provider_name: str, api_key: str | None, logger) -> str:
    """Validate API key is present.

    Args:
        provider_name: Name of the provider (e.g., 'OpenAI', 'Anthropic')
        api_key: The API key to validate
        logger: Logger instance for output

    Returns:
        The validated API key

    Raises:
        ValueError: If API key is None or empty
    """
    if not api_key:
        raise ValueError(
            f'{provider_name} API key is not configured. Please set the appropriate environment variable.'
        )

    logger.info(f'Creating {provider_name} client')

    return api_key


class LLMClientFactory:
    """Factory for creating LLM clients based on configuration."""

    @staticmethod
    def create(config: LLMConfig) -> LLMClient:
        """Create an LLM client based on the configured provider."""
        import logging

        logger = logging.getLogger(__name__)

        provider = config.provider.lower()

        match provider:
            case 'openai':
                if not config.providers.openai:
                    raise ValueError('OpenAI provider configuration not found')

                api_key = config.providers.openai.api_key
                _validate_api_key('OpenAI', api_key, logger)

                from graphiti_core.llm_client.config import LLMConfig as CoreLLMConfig

                # Use the same model for both main and small model slots
                small_model = config.model

                llm_config = CoreLLMConfig(
                    api_key=api_key,
                    model=config.model,
                    small_model=small_model,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                )

                # Check if this is a reasoning model (o1, o3, gpt-5 family)
                reasoning_prefixes = ('o1', 'o3', 'gpt-5')
                is_reasoning_model = config.model.startswith(reasoning_prefixes)

                # Build the HTTP client here rather than letting OpenAIClient build
                # its own (BUG-96): that path takes the SDK's 600 s default, and
                # `client=` is the seam that lets a chosen bound in. Constructed
                # with exactly the arguments OpenAIClient would have used — note
                # `llm_config.base_url` is None on this path, which preserves the
                # existing behaviour of the openai LLM branch ignoring `api_url`.
                from openai import AsyncOpenAI

                http_client = AsyncOpenAI(
                    api_key=llm_config.api_key,
                    base_url=llm_config.base_url,
                    timeout=request_timeout(config.request_timeout_seconds),
                )

                # Only pass reasoning/verbosity parameters for reasoning models (gpt-5 family)
                if is_reasoning_model:
                    return OpenAIClient(
                        config=llm_config,
                        client=http_client,
                        reasoning='minimal',
                        verbosity='low',
                    )
                else:
                    # For non-reasoning models, explicitly pass None to disable these parameters
                    return OpenAIClient(
                        config=llm_config,
                        client=http_client,
                        reasoning=None,
                        verbosity=None,
                    )

            case 'azure_openai':
                if not HAS_AZURE_LLM:
                    raise ValueError(
                        'Azure OpenAI LLM client not available in current graphiti-core version'
                    )
                if not config.providers.azure_openai:
                    raise ValueError('Azure OpenAI provider configuration not found')
                azure_config = config.providers.azure_openai

                if not azure_config.api_url:
                    raise ValueError('Azure OpenAI API URL is required')

                # Currently using API key authentication
                # TODO: Add Azure AD authentication support for v1 API compatibility
                api_key = azure_config.api_key
                _validate_api_key('Azure OpenAI', api_key, logger)

                # Azure OpenAI should use the standard AsyncOpenAI client with v1 compatibility endpoint
                # See: https://github.com/getzep/graphiti README Azure OpenAI section
                from openai import AsyncOpenAI

                # Ensure the base_url ends with /openai/v1/ for Azure v1 compatibility
                base_url = azure_config.api_url
                if not base_url.endswith('/'):
                    base_url += '/'
                if not base_url.endswith('openai/v1/'):
                    base_url += 'openai/v1/'

                azure_client = AsyncOpenAI(
                    base_url=base_url,
                    api_key=api_key,
                    timeout=request_timeout(config.request_timeout_seconds),
                )

                # Then create the LLMConfig
                from graphiti_core.llm_client.config import LLMConfig as CoreLLMConfig

                llm_config = CoreLLMConfig(
                    api_key=api_key,
                    base_url=base_url,
                    model=config.model,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                )

                return AzureOpenAILLMClient(
                    azure_client=azure_client,
                    config=llm_config,
                    max_tokens=config.max_tokens,
                )

            case 'anthropic':
                if not HAS_ANTHROPIC:
                    raise ValueError(
                        'Anthropic client not available in current graphiti-core version'
                    )
                if not config.providers.anthropic:
                    raise ValueError('Anthropic provider configuration not found')

                api_key = config.providers.anthropic.api_key
                _validate_api_key('Anthropic', api_key, logger)

                llm_config = GraphitiLLMConfig(
                    api_key=api_key,
                    model=config.model,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                )

                # Same seam as the openai branch (BUG-96). `max_retries=1` is not a
                # choice made here — it is what AnthropicClient sets when it builds
                # its own client, and building the client here would otherwise
                # silently restore the SDK's default of 2.
                from anthropic import AsyncAnthropic

                return AnthropicClient(
                    config=llm_config,
                    client=AsyncAnthropic(
                        api_key=api_key,
                        max_retries=1,
                        timeout=request_timeout(config.request_timeout_seconds),
                    ),
                )

            case 'bedrock':
                if not HAS_BEDROCK:
                    raise ValueError(
                        'Bedrock client not available. Install with: '
                        'pip install graphiti-core[bedrock]'
                    )
                if not config.providers.bedrock:
                    raise ValueError('Bedrock provider configuration not found')

                # No api_key — boto3 chain handles auth. Region falls through to
                # AWS_DEFAULT_REGION/AWS_REGION env vars when not set in config.
                logger.info('Creating Bedrock LLM client')
                bedrock_cfg = config.providers.bedrock
                if bedrock_cfg.region:
                    # Plumb region via env so BedrockLLMClient picks it up
                    # (the client reads AWS_DEFAULT_REGION/AWS_REGION at __init__).
                    os.environ.setdefault('AWS_REGION', bedrock_cfg.region)

                return BedrockLLMClient(
                    config=GraphitiLLMConfig(
                        api_key='not-required',
                        model=config.model,
                        temperature=config.temperature,
                        max_tokens=config.max_tokens,
                    ),
                )

            case 'gemini':
                if not HAS_GEMINI:
                    raise ValueError('Gemini client not available in current graphiti-core version')
                if not config.providers.gemini:
                    raise ValueError('Gemini provider configuration not found')

                api_key = config.providers.gemini.api_key
                _validate_api_key('Gemini', api_key, logger)

                llm_config = GraphitiLLMConfig(
                    api_key=api_key,
                    model=config.model,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                )
                return GeminiClient(config=llm_config)

            case 'groq':
                if not HAS_GROQ:
                    raise ValueError('Groq client not available in current graphiti-core version')
                if not config.providers.groq:
                    raise ValueError('Groq provider configuration not found')

                api_key = config.providers.groq.api_key
                _validate_api_key('Groq', api_key, logger)

                llm_config = GraphitiLLMConfig(
                    api_key=api_key,
                    base_url=config.providers.groq.api_url,
                    model=config.model,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                )
                return GroqClient(config=llm_config)

            case _:
                raise ValueError(f'Unsupported LLM provider: {provider}')


class EmbedderFactory:
    """Factory for creating Embedder clients based on configuration."""

    @staticmethod
    def create(config: EmbedderConfig) -> EmbedderClient:
        """Create an Embedder client based on the configured provider."""
        import logging

        logger = logging.getLogger(__name__)

        provider = config.provider.lower()

        match provider:
            case 'openai':
                if not config.providers.openai:
                    raise ValueError('OpenAI provider configuration not found')

                api_key = config.providers.openai.api_key
                _validate_api_key('OpenAI Embedder', api_key, logger)

                from graphiti_core.embedder.openai import OpenAIEmbedderConfig

                embedder_config = OpenAIEmbedderConfig(
                    api_key=api_key,
                    embedding_model=config.model,
                    base_url=config.providers.openai.api_url,  # Support custom endpoints like Ollama
                    embedding_dim=config.dimensions,  # Support custom embedding dimensions
                )

                # Pre-built so the call carries a chosen bound (BUG-96); same
                # arguments OpenAIEmbedder would have used, including the custom
                # base_url that makes Ollama-style endpoints work.
                from openai import AsyncOpenAI

                return OpenAIEmbedder(
                    config=embedder_config,
                    client=AsyncOpenAI(
                        api_key=embedder_config.api_key,
                        base_url=embedder_config.base_url,
                        timeout=request_timeout(config.request_timeout_seconds),
                    ),
                )

            case 'azure_openai':
                if not HAS_AZURE_EMBEDDER:
                    raise ValueError(
                        'Azure OpenAI embedder not available in current graphiti-core version'
                    )
                if not config.providers.azure_openai:
                    raise ValueError('Azure OpenAI provider configuration not found')
                azure_config = config.providers.azure_openai

                if not azure_config.api_url:
                    raise ValueError('Azure OpenAI API URL is required')

                # Currently using API key authentication
                # TODO: Add Azure AD authentication support for v1 API compatibility
                api_key = azure_config.api_key
                _validate_api_key('Azure OpenAI Embedder', api_key, logger)

                # Azure OpenAI should use the standard AsyncOpenAI client with v1 compatibility endpoint
                # See: https://github.com/getzep/graphiti README Azure OpenAI section
                from openai import AsyncOpenAI

                # Ensure the base_url ends with /openai/v1/ for Azure v1 compatibility
                base_url = azure_config.api_url
                if not base_url.endswith('/'):
                    base_url += '/'
                if not base_url.endswith('openai/v1/'):
                    base_url += 'openai/v1/'

                azure_client = AsyncOpenAI(
                    base_url=base_url,
                    api_key=api_key,
                    timeout=request_timeout(config.request_timeout_seconds),
                )

                return AzureOpenAIEmbedderClient(
                    azure_client=azure_client,
                    model=config.model or 'text-embedding-3-small',
                )

            case 'gemini':
                if not HAS_GEMINI_EMBEDDER:
                    raise ValueError(
                        'Gemini embedder not available in current graphiti-core version'
                    )
                if not config.providers.gemini:
                    raise ValueError('Gemini provider configuration not found')

                api_key = config.providers.gemini.api_key
                _validate_api_key('Gemini Embedder', api_key, logger)

                from graphiti_core.embedder.gemini import GeminiEmbedderConfig

                gemini_config = GeminiEmbedderConfig(
                    api_key=api_key,
                    embedding_model=config.model or 'models/text-embedding-004',
                    embedding_dim=config.dimensions or 768,
                )
                return GeminiEmbedder(config=gemini_config)

            case 'voyage':
                if not HAS_VOYAGE_EMBEDDER:
                    raise ValueError(
                        'Voyage embedder not available in current graphiti-core version'
                    )
                if not config.providers.voyage:
                    raise ValueError('Voyage provider configuration not found')

                api_key = config.providers.voyage.api_key
                _validate_api_key('Voyage Embedder', api_key, logger)

                from graphiti_core.embedder.voyage import VoyageAIEmbedderConfig

                voyage_config = VoyageAIEmbedderConfig(
                    api_key=api_key,
                    embedding_model=config.model or 'voyage-3',
                    embedding_dim=config.dimensions or 1024,
                )
                return VoyageAIEmbedder(config=voyage_config)

            case 'bedrock':
                if BedrockEmbedder is None:
                    raise ValueError(
                        'Bedrock embedder not available. Install with: '
                        'pip install graphiti-core[bedrock]'
                    )
                if not config.providers.bedrock:
                    raise ValueError('Bedrock provider configuration not found')

                logger.info('Creating Bedrock embedder')
                bedrock_cfg = config.providers.bedrock
                if bedrock_cfg.region:
                    os.environ.setdefault('AWS_REGION', bedrock_cfg.region)

                return BedrockEmbedder(
                    config=BedrockEmbedderConfig(
                        embedding_model=config.model,
                        embedding_dim=config.dimensions,
                    ),
                )

            case _:
                raise ValueError(f'Unsupported Embedder provider: {provider}')


class CrossEncoderFactory:
    """Factory for the reranker client — the outbound client no factory used to build.

    `Graphiti(...)` accepts `cross_encoder=`; every call site here omitted it, so
    graphiti_core supplied its own `OpenAIRerankerClient()` and that client took
    the openai SDK's 600 s default (BUG-96). It is not dormant code: it is reached
    from the live `search` surface whenever `reranker='cross_encoder'` is asked
    for, which the `precise` intent selects by default.

    Behaviour is otherwise preserved exactly. `config=` is left unset, so the
    reranker keeps the `LLMConfig()` defaults it has always run on (including its
    own model choice), and the HTTP client is built with `api_key=None` /
    `base_url=None` — the same arguments the implicit construction used, which
    means the api key still resolves from `OPENAI_API_KEY` in the environment
    exactly as before. The only delta is the timeout.
    """

    @staticmethod
    def create(config: LLMConfig) -> CrossEncoderClient:
        """Build a time-bounded reranker, using the LLM timeout (it is an LLM call)."""
        from openai import AsyncOpenAI

        return OpenAIRerankerClient(
            client=AsyncOpenAI(
                api_key=None,
                base_url=None,
                timeout=request_timeout(config.request_timeout_seconds),
            )
        )


class DatabaseDriverFactory:
    """Factory for creating Database drivers based on configuration.

    Note: This returns configuration dictionaries that can be passed to Graphiti(),
    not driver instances directly, as the drivers require complex initialization.
    """

    @staticmethod
    def create_config(config: DatabaseConfig) -> dict:
        """Create database configuration dictionary based on the configured provider."""
        provider = config.provider.lower()

        match provider:
            case 'neo4j':
                # Use Neo4j config if provided, otherwise use defaults
                if config.providers.neo4j:
                    neo4j_config = config.providers.neo4j
                else:
                    # Create default Neo4j configuration
                    from config.schema import Neo4jProviderConfig

                    neo4j_config = Neo4jProviderConfig()

                # Check for environment variable overrides (for CI/CD compatibility)
                import os

                uri = os.environ.get('NEO4J_URI', neo4j_config.uri)
                username = os.environ.get('NEO4J_USER', neo4j_config.username)
                password = os.environ.get('NEO4J_PASSWORD', neo4j_config.password)

                return {
                    'uri': uri,
                    'user': username,
                    'password': password,
                    # Note: database and use_parallel_runtime would need to be passed
                    # to the driver after initialization if supported
                }

            case 'falkordb':
                if not HAS_FALKOR:
                    raise ValueError(
                        'FalkorDB driver not available in current graphiti-core version'
                    )

                # Use FalkorDB config if provided, otherwise use defaults
                if config.providers.falkordb:
                    falkor_config = config.providers.falkordb
                else:
                    # Create default FalkorDB configuration
                    from config.schema import FalkorDBProviderConfig

                    falkor_config = FalkorDBProviderConfig()

                # Check for environment variable overrides (for CI/CD compatibility)
                import os
                from urllib.parse import urlparse

                uri = os.environ.get('FALKORDB_URI', falkor_config.uri)
                username = os.environ.get('FALKORDB_USERNAME', falkor_config.username)
                password = os.environ.get('FALKORDB_PASSWORD', falkor_config.password)

                # Parse the URI to extract host and port
                parsed = urlparse(uri)
                host = parsed.hostname or 'localhost'
                port = parsed.port or 6379

                return {
                    'driver': 'falkordb',
                    'host': host,
                    'port': port,
                    'username': username,
                    'password': password,
                    'database': falkor_config.database,
                }

            case 'age':
                # Use AGE config if provided, otherwise use defaults
                if config.providers.age:
                    age_config = config.providers.age
                else:
                    from config.schema import AgeProviderConfig

                    age_config = AgeProviderConfig()

                # Check for environment variable overrides (for CI/CD compatibility)
                import os

                return {
                    'driver': 'age',
                    'dsn': os.environ.get('AGE_DSN', age_config.dsn),
                    'graph_name': os.environ.get('AGE_GRAPH_NAME', age_config.graph_name),
                    'embedding_dim': int(
                        os.environ.get('AGE_EMBEDDING_DIM', age_config.embedding_dim)
                    ),
                    'text_search_config': os.environ.get(
                        'AGE_TEXT_SEARCH_CONFIG', age_config.text_search_config
                    ),
                }

            case _:
                raise ValueError(f'Unsupported Database provider: {provider}')
