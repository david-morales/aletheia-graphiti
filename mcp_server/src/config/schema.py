"""Configuration schemas with pydantic-settings and YAML support."""

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


def deep_merge(base: dict, overlay: dict) -> dict:
    """Deep-merge overlay onto base. Dicts merge recursively, all else replaces."""
    result = base.copy()
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class YamlSettingsSource(PydanticBaseSettingsSource):
    """Custom settings source for loading from YAML files."""

    def __init__(self, settings_cls: type[BaseSettings], config_path: Path | None = None):
        super().__init__(settings_cls)
        self.config_path = config_path or Path('config.yaml')

    def _expand_env_vars(self, value: Any) -> Any:
        """Recursively expand environment variables in configuration values."""
        if isinstance(value, str):
            # Support ${VAR} and ${VAR:default} syntax
            import re

            def replacer(match):
                var_name = match.group(1)
                default_value = match.group(3) if match.group(3) is not None else ''
                return os.environ.get(var_name, default_value)

            pattern = r'\$\{([^:}]+)(:([^}]*))?\}'

            # Check if the entire value is a single env var expression
            full_match = re.fullmatch(pattern, value)
            if full_match:
                result = replacer(full_match)
                # Convert boolean-like strings to actual booleans
                if isinstance(result, str):
                    lower_result = result.lower().strip()
                    if lower_result in ('true', '1', 'yes', 'on'):
                        return True
                    elif lower_result in ('false', '0', 'no', 'off'):
                        return False
                    elif lower_result == '':
                        # Empty string means env var not set - return None for optional fields
                        return None
                return result
            else:
                # Otherwise, do string substitution (keep as strings for partial replacements)
                return re.sub(pattern, replacer, value)
        elif isinstance(value, dict):
            return {k: self._expand_env_vars(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._expand_env_vars(item) for item in value]
        return value

    def get_field_value(self, field_name: str, field_info: Any) -> Any:
        """Get field value from YAML config."""
        return None

    def __call__(self) -> dict[str, Any]:
        """Load and parse YAML configuration."""
        if not self.config_path.exists():
            return {}

        with open(self.config_path) as f:
            raw_config = yaml.safe_load(f) or {}

        # Resolve base config if referenced
        if 'base' in raw_config:
            base_ref = raw_config.pop('base')
            base_path = (self.config_path.parent / base_ref).resolve()
            if not base_path.exists():
                raise FileNotFoundError(
                    f"Base config not found: {base_path} "
                    f"(referenced from {self.config_path})"
                )
            with open(base_path) as f:
                base_config = yaml.safe_load(f) or {}
            raw_config = deep_merge(base_config, raw_config)

        # Expand environment variables
        return self._expand_env_vars(raw_config)


# SDK 2.x caps streamable-HTTP POST bodies at 4 MiB; SDK 1.x had no cap at all.
# Taking the new default would silently start answering `413` to bulk `add_memory`
# calls that used to succeed — a transport failure, so not even an in-band
# ADR-015 R4 error the caller could read. This migration restores the 1.x
# behaviour instead.
#
# "No cap" cannot be requested: the SDK types the parameter `int` and rejects
# `<= 0` ("must be a positive number of bytes"), so it is expressed as a bound
# large enough not to be reachable by a legitimate episode payload while still
# bounding what a single request can make the process buffer.
DEFAULT_MAX_REQUEST_BODY_SIZE = 256 * 1024 * 1024


class ServerConfig(BaseModel):
    """Server configuration."""

    transport: str = Field(
        default='http',
        description='Transport type: http (default, recommended), stdio, or sse (deprecated)',
    )
    host: str = Field(default='0.0.0.0', description='Server host')
    port: int = Field(default=8000, description='Server port')
    max_request_body_size: int = Field(
        default=DEFAULT_MAX_REQUEST_BODY_SIZE,
        gt=0,
        description=(
            'Maximum streamable-HTTP POST body in bytes. The default restores the '
            'pre-SDK-2.x behaviour of not rejecting large bulk ingests; set it to '
            '4194304 to adopt the SDK 2.x default instead. Must be positive — the '
            'SDK has no unlimited sentinel.'
        ),
    )


# BUG-96 layer 1 — the bound on outbound provider HTTP calls.
#
# Applied on the openai, azure_openai and anthropic paths (LLM, embedder,
# reranker). NOT applied on bedrock/gemini/groq (LLM) or gemini/voyage/bedrock
# (embedder): those construct their transport internally with no seam this
# server can reach, so they keep their own SDK default. See the per-field
# descriptions below, and `services/factories.py` for the seam each path uses.
#
# Nothing here was literally unbounded: the `openai` SDK supplies
# `Timeout(connect=5.0, read=600, write=600, pool=600)` to any client built
# without one. What was missing is a bound the connector CHOSE. 600 s is twice
# the consumer's own `ALETHEIA_MCP_CALL_TIMEOUT_SECONDS` (300 s), so on a dead
# socket the consumer always gave up first while the connector went on holding
# the socket, the coroutine and its semaphore slot — which is the shape the
# 2026-08-17 network flip left the process in: zero CPU, sockets in CLOSE_WAIT,
# no answer ever.
#
# 300 s matches the consumer so the connector fails first and fails visibly.
DEFAULT_REQUEST_TIMEOUT_SECONDS = 300.0

# The connect leg keeps the SDK's own 5 s. Expressing the bound as a single
# float would set connect to 300 s as well — strictly worse for the failure
# being fixed, since an unreachable edge would then take five minutes to report
# instead of five seconds. Not a separate knob: one dial is the point.
DEFAULT_CONNECT_TIMEOUT_SECONDS = 5.0


class OpenAIProviderConfig(BaseModel):
    """OpenAI provider configuration."""

    api_key: str | None = None
    api_url: str = 'https://api.openai.com/v1'
    organization_id: str | None = None


class AzureOpenAIProviderConfig(BaseModel):
    """Azure OpenAI provider configuration."""

    api_key: str | None = None
    api_url: str | None = None
    api_version: str = '2024-10-21'
    deployment_name: str | None = None
    use_azure_ad: bool = False


class AnthropicProviderConfig(BaseModel):
    """Anthropic provider configuration."""

    api_key: str | None = None
    api_url: str = 'https://api.anthropic.com'
    max_retries: int = 3


class GeminiProviderConfig(BaseModel):
    """Gemini provider configuration."""

    api_key: str | None = None
    project_id: str | None = None
    location: str = 'us-central1'


class GroqProviderConfig(BaseModel):
    """Groq provider configuration."""

    api_key: str | None = None
    api_url: str = 'https://api.groq.com/openai/v1'


class VoyageProviderConfig(BaseModel):
    """Voyage AI provider configuration."""

    api_key: str | None = None
    api_url: str = 'https://api.voyageai.com/v1'
    model: str = 'voyage-3'


class BedrockProviderConfig(BaseModel):
    """AWS Bedrock provider configuration.

    No api_key field — Bedrock uses the standard boto3 credential chain
    (env vars AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY > IAM role/IRSA >
    ~/.aws/credentials). The `region` field maps to ChatBedrockConverse's
    region_name; when None, the underlying client reads AWS_DEFAULT_REGION
    or AWS_REGION (default us-east-1).
    """

    region: str | None = None
    max_retries: int = 3


class LLMProvidersConfig(BaseModel):
    """LLM providers configuration."""

    openai: OpenAIProviderConfig | None = None
    azure_openai: AzureOpenAIProviderConfig | None = None
    anthropic: AnthropicProviderConfig | None = None
    bedrock: BedrockProviderConfig | None = None
    gemini: GeminiProviderConfig | None = None
    groq: GroqProviderConfig | None = None


class LLMConfig(BaseModel):
    """LLM configuration."""

    provider: str = Field(default='openai', description='LLM provider')
    model: str = Field(default='gpt-4o-mini', description='Model name')
    temperature: float | None = Field(
        default=None, description='Temperature (optional, defaults to None for reasoning models)'
    )
    max_tokens: int = Field(default=4096, description='Max tokens')
    request_timeout_seconds: float = Field(
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        gt=0,
        description=(
            'Read/write/pool timeout in seconds for outbound LLM (and reranker) HTTP '
            'calls, on the openai, azure_openai and anthropic provider paths only. '
            'Bounds a dead socket so it raises instead of hanging (BUG-96). The '
            'bedrock, gemini and groq LLM paths are NOT bounded — they expose no '
            'timeout seam this server can reach, so they keep their own SDK default '
            'and this setting is silently inert for them. Override with '
            'LLM__REQUEST_TIMEOUT_SECONDS or `llm:` in the config file. Must be '
            'positive — 0 would disable the deadline, which is the bug.'
        ),
    )
    providers: LLMProvidersConfig = Field(default_factory=LLMProvidersConfig)


class EmbedderProvidersConfig(BaseModel):
    """Embedder providers configuration."""

    openai: OpenAIProviderConfig | None = None
    azure_openai: AzureOpenAIProviderConfig | None = None
    bedrock: BedrockProviderConfig | None = None
    gemini: GeminiProviderConfig | None = None
    voyage: VoyageProviderConfig | None = None


class EmbedderConfig(BaseModel):
    """Embedder configuration."""

    provider: str = Field(default='openai', description='Embedder provider')
    model: str = Field(default='text-embedding-3-small', description='Model name')
    dimensions: int = Field(default=1024, description='Embedding dimensions')
    request_timeout_seconds: float = Field(
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        gt=0,
        description=(
            'Read/write/pool timeout in seconds for outbound embedding HTTP calls, '
            'on the openai and azure_openai provider paths only. Bounds a dead socket '
            'so it raises instead of hanging (BUG-96). The gemini, voyage and bedrock '
            'embedder paths are NOT bounded — they expose no timeout seam this server '
            'can reach, so they keep their own SDK default and this setting is '
            'silently inert for them. Override with EMBEDDER__REQUEST_TIMEOUT_SECONDS '
            'or `embedder:` in the config file. Must be positive — 0 would disable '
            'the deadline, which is the bug.'
        ),
    )
    providers: EmbedderProvidersConfig = Field(default_factory=EmbedderProvidersConfig)


class Neo4jProviderConfig(BaseModel):
    """Neo4j provider configuration."""

    uri: str = 'bolt://localhost:7687'
    username: str = 'neo4j'
    password: str | None = None
    database: str = 'neo4j'
    use_parallel_runtime: bool = False


class FalkorDBProviderConfig(BaseModel):
    """FalkorDB provider configuration."""

    uri: str = 'redis://localhost:6379'
    username: str | None = None
    password: str | None = None
    database: str = 'default_db'


class AgeProviderConfig(BaseModel):
    """PostgreSQL + Apache AGE provider configuration.

    embedding_dim MUST equal the embedder's output dim (EmbedderConfig.dimensions,
    default 1024) and the dim the target graph was ingested at — a mismatch reproduces
    the DataError: expected 1536, got 1024. Default 1024 (NOT AGEDriver's own 1536 default).
    """

    dsn: str = 'postgresql://age:age@localhost:5433/age_test'
    graph_name: str = 'graphiti'
    embedding_dim: int = 1024
    # PostgreSQL text-search configuration for the keyword leg. `simple` folds
    # case and nothing else — no stemming, no stopwords — and is the only safe
    # default for a language-agnostic connector. Naming the corpus language
    # (`spanish`, `english`, …) buys stemming + stopword removal, i.e. parity
    # with the FalkorDB flavour, but it also changes the `tsv` generated columns:
    # an EXISTING graph must be rebuilt and re-ingested for it to take effect.
    text_search_config: str = 'simple'


class DatabaseProvidersConfig(BaseModel):
    """Database providers configuration."""

    neo4j: Neo4jProviderConfig | None = None
    falkordb: FalkorDBProviderConfig | None = None
    age: AgeProviderConfig | None = None


class DatabaseConfig(BaseModel):
    """Database configuration."""

    provider: str = Field(default='falkordb', description='Database provider')
    providers: DatabaseProvidersConfig = Field(default_factory=DatabaseProvidersConfig)


class EntityTypeConfig(BaseModel):
    """Entity type configuration."""

    name: str
    description: str


class GraphitiAppConfig(BaseModel):
    """Graphiti-specific configuration."""

    group_id: str = Field(default='main', description='Group ID')
    episode_id_prefix: str | None = Field(default='', description='Episode ID prefix')
    user_id: str = Field(default='mcp_user', description='User ID')
    entity_types: list[EntityTypeConfig] = Field(default_factory=list)
    ontology_graph: str | None = Field(
        default=None,
        description='Companion ontology graph for schema context',
    )

    def model_post_init(self, __context) -> None:
        """Convert None to empty string for episode_id_prefix."""
        if self.episode_id_prefix is None:
            self.episode_id_prefix = ''


class SearchDefaultsConfig(BaseModel):
    """Default search configuration, overridable per use case."""

    default_reranker: str = Field(default='rrf', description='Default reranker strategy')
    default_limit: int = Field(default=10, description='Default result limit')
    bfs_max_depth: int = Field(default=3, description='Default BFS depth for explore_entity')


class GraphitiConfig(BaseSettings):
    """Graphiti configuration with YAML and environment support."""

    server: ServerConfig = Field(default_factory=ServerConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    graphiti: GraphitiAppConfig = Field(default_factory=GraphitiAppConfig)
    search: SearchDefaultsConfig = Field(default_factory=SearchDefaultsConfig)

    # Additional server options
    destroy_graph: bool = Field(default=False, description='Clear graph on startup')

    model_config = SettingsConfigDict(
        env_prefix='',
        env_nested_delimiter='__',
        case_sensitive=False,
        extra='ignore',
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Customize settings sources to include YAML."""
        config_path = Path(os.environ.get('CONFIG_PATH', 'config/config.yaml'))
        yaml_settings = YamlSettingsSource(settings_cls, config_path)
        # Priority: CLI args (init) > env vars > yaml > defaults
        return (init_settings, env_settings, yaml_settings, dotenv_settings)

    def apply_cli_overrides(self, args) -> None:
        """Apply CLI argument overrides to configuration."""
        # Override server settings
        if hasattr(args, 'transport') and args.transport:
            self.server.transport = args.transport

        # Override LLM settings
        if hasattr(args, 'llm_provider') and args.llm_provider:
            self.llm.provider = args.llm_provider
        if hasattr(args, 'model') and args.model:
            self.llm.model = args.model
        if hasattr(args, 'temperature') and args.temperature is not None:
            self.llm.temperature = args.temperature

        # Override embedder settings
        if hasattr(args, 'embedder_provider') and args.embedder_provider:
            self.embedder.provider = args.embedder_provider
        if hasattr(args, 'embedder_model') and args.embedder_model:
            self.embedder.model = args.embedder_model

        # Override database settings
        if hasattr(args, 'database_provider') and args.database_provider:
            self.database.provider = args.database_provider

        # Override Graphiti settings
        if hasattr(args, 'group_id') and args.group_id:
            self.graphiti.group_id = args.group_id
        if hasattr(args, 'user_id') and args.user_id:
            self.graphiti.user_id = args.user_id
