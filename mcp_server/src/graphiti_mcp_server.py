#!/usr/bin/env python3
"""
Graphiti MCP Server - Exposes Graphiti functionality through the Model Context Protocol (MCP)
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

from dotenv import load_dotenv
from graphiti_core import Graphiti
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.edges import EntityEdge
from graphiti_core.errors import NodeNotFoundError
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.utils.bulk_utils import RawEpisode
from graphiti_core.search.search_config import (
    EdgeReranker,
    EdgeSearchConfig,
    EdgeSearchMethod,
    NodeReranker,
    NodeSearchConfig,
    NodeSearchMethod,
    SearchConfig,
)
from graphiti_core.search.search_config_recipes import (
    COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
    COMBINED_HYBRID_SEARCH_MMR,
    COMBINED_HYBRID_SEARCH_RRF,
    COMMUNITY_HYBRID_SEARCH_CROSS_ENCODER,
    COMMUNITY_HYBRID_SEARCH_MMR,
    COMMUNITY_HYBRID_SEARCH_RRF,
    EDGE_HYBRID_SEARCH_CROSS_ENCODER,
    EDGE_HYBRID_SEARCH_EPISODE_MENTIONS,
    EDGE_HYBRID_SEARCH_MMR,
    EDGE_HYBRID_SEARCH_NODE_DISTANCE,
    EDGE_HYBRID_SEARCH_RRF,
    NODE_HYBRID_SEARCH_CROSS_ENCODER,
    NODE_HYBRID_SEARCH_EPISODE_MENTIONS,
    NODE_HYBRID_SEARCH_MMR,
    NODE_HYBRID_SEARCH_NODE_DISTANCE,
    NODE_HYBRID_SEARCH_RRF,
)
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.utils.maintenance.graph_data_operations import clear_data
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.resources.types import TextResource
from pydantic import BaseModel
from starlette.responses import JSONResponse

from config.schema import GraphitiConfig, ServerConfig
from domain_profile import DomainProfile, build_domain_profile
from tool_descriptions import (
    build_degraded_instructions,
    build_instructions,
    build_search_description,
    build_explore_node_description,
    build_search_ontology_description,
    build_explore_ontology_description,
    build_get_schema_description,
    build_run_cypher_description,
)
from models.response_types import (
    AddMemoryResult,
    CommunityBuildResult,
    CypherResultResponse,
    EpisodeContextResult,
    EpisodeListResult,
    ExploreResult,
    MutationResult,
    OntologyClassContextResponse,
    OntologyDocumentationResponse,
    OntologyStructureResponse,
    ProfileGraphResponse,
    SchemaResponse,
    SearchResult,
    StatusResponse,
    SubgraphResponse,
)
from services.factories import DatabaseDriverFactory, EmbedderFactory, LLMClientFactory
from tool_annotations import annotations_for
from services.queue_service import QueueService
from graph_profiler import profile_graph as _run_profile_graph
from utils.cypher import (
    CypherError,
    format_error,
    format_result,
    validate_and_sanitize,
)
from flavours import build_flavour
from utils.formatting import format_community_result, format_edge_result

# Load .env file from mcp_server directory
mcp_server_dir = Path(__file__).parent.parent
env_file = mcp_server_dir / '.env'
if env_file.exists():
    load_dotenv(env_file)
else:
    # Try current working directory as fallback
    load_dotenv()


# Semaphore limit for concurrent Graphiti operations.
#
# This controls how many episodes can be processed simultaneously. Each episode
# processing involves multiple LLM calls (entity extraction, deduplication, etc.),
# so the actual number of concurrent LLM requests will be higher.
#
# TUNING GUIDELINES:
#
# LLM Provider Rate Limits (requests per minute):
# - OpenAI Tier 1 (free):     3 RPM   -> SEMAPHORE_LIMIT=1-2
# - OpenAI Tier 2:            60 RPM   -> SEMAPHORE_LIMIT=5-8
# - OpenAI Tier 3:           500 RPM   -> SEMAPHORE_LIMIT=10-15
# - OpenAI Tier 4:         5,000 RPM   -> SEMAPHORE_LIMIT=20-50
# - Anthropic (default):     50 RPM   -> SEMAPHORE_LIMIT=5-8
# - Anthropic (high tier): 1,000 RPM   -> SEMAPHORE_LIMIT=15-30
# - Azure OpenAI (varies):  Consult your quota -> adjust accordingly
#
# SYMPTOMS:
# - Too high: 429 rate limit errors, increased costs from parallel processing
# - Too low: Slow throughput, underutilized API quota
#
# MONITORING:
# - Watch logs for rate limit errors (429)
# - Monitor episode processing times
# - Check LLM provider dashboard for actual request rates
#
# DEFAULT: 10 (suitable for OpenAI Tier 3, mid-tier Anthropic)
SEMAPHORE_LIMIT = int(os.getenv('SEMAPHORE_LIMIT', 10))

# Resilience: bounded retry for FalkorDB connection failures during init/reconnect.
_RETRY_ATTEMPTS = 3
_RETRY_INITIAL_BACKOFF_S = 1.0
_RETRY_BACKOFF_MULTIPLIER = 2.0


# Configure structured logging with timestamps
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    stream=sys.stderr,
)

# Configure specific loggers
logging.getLogger('uvicorn').setLevel(logging.INFO)
logging.getLogger('uvicorn.access').setLevel(logging.WARNING)  # Reduce access log noise
logging.getLogger('mcp.server.streamable_http_manager').setLevel(
    logging.WARNING
)  # Reduce MCP noise


# Patch uvicorn's logging config to use our format
def configure_uvicorn_logging():
    """Configure uvicorn loggers to match our format after they're created."""
    for logger_name in ['uvicorn', 'uvicorn.error', 'uvicorn.access']:
        uvicorn_logger = logging.getLogger(logger_name)
        # Remove existing handlers and add our own with proper formatting
        uvicorn_logger.handlers.clear()
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        uvicorn_logger.addHandler(handler)
        uvicorn_logger.propagate = False


logger = logging.getLogger(__name__)

# Create global config instance - will be properly initialized later
config: GraphitiConfig

# MCP server instructions
GRAPHITI_MCP_INSTRUCTIONS = """
Graphiti is a knowledge graph memory service. It transforms information into a richly
connected network of entities (nodes), facts (edges), and communities, organized by
group_id for separate knowledge domains.

Key tools:

1. search — Unified search with control over strategy (nodes/edges/communities/combined),
   reranking (rrf/mmr/cross_encoder/node_distance), temporal filters, type filters,
   BFS traversal from known nodes, and cross-graph queries via group_ids.

2. explore_node — Deep dive on a specific entity. Provide a name or UUID and get the
   full neighborhood: connected nodes, relationships, and community memberships.

3. add_memory — Add episodes (text, JSON, or messages) to the graph. Supports single
   async episodes and bulk synchronous ingestion.

4. build_communities — Cluster entities into communities for high-level overview queries.
   Run after ingestion, then search with search_mode="communities".

5. get_episode_context — Inspect what was extracted from specific episodes (nodes + edges).

6. get_episodes — List recent episodes by group_id.

7. delete_entity_edge / delete_episode — Remove specific relationships or episodes.

8. clear_graph / get_status — Graph management and health checks.

9. search_ontology — Search the companion ontology graph for schema definitions,
   entity types, properties, and relationships. Use this to understand what types
   of entities and relationships exist in the knowledge graph.

10. explore_ontology — Deep dive on a specific ontology class. Shows properties,
    relationships, and parent classes for a given type.

Note: Ontology tools (9-10) are only available when an ontology graph is configured.

Tips:
- Use group_ids to search across multiple graphs simultaneously.
- Use center_node_uuid with reranker="node_distance" to find nearby entities.
- Use bfs_origin_node_uuids to traverse the graph from known starting points.
- Use valid_at to filter for temporally valid facts.
"""

# MCP server instance — read host from env to set DNS rebinding policy at init time.
# When FASTMCP_HOST=0.0.0.0 (Docker), FastMCP skips DNS rebinding protection so
# inter-container requests with Docker hostnames in the Host header are accepted.
_init_host = os.environ.get('FASTMCP_HOST', '127.0.0.1')
_init_port = int(os.environ.get('FASTMCP_PORT', '8000'))
mcp = FastMCP(
    'Graphiti Agent Memory',
    instructions=GRAPHITI_MCP_INSTRUCTIONS,
    host=_init_host,
    port=_init_port,
)

# Global services
graphiti_service: Optional['GraphitiService'] = None
queue_service: QueueService | None = None

# Global client for backward compatibility
graphiti_client: Graphiti | None = None
semaphore: asyncio.Semaphore


class GraphitiService:
    """Graphiti service using the unified configuration system."""

    def __init__(self, config: GraphitiConfig, semaphore_limit: int = 10):
        self.config = config
        self.flavour = build_flavour(config.database.provider)
        self.semaphore_limit = semaphore_limit
        self.semaphore = asyncio.Semaphore(semaphore_limit)
        self.client: Graphiti | None = None
        self.ontology_client: Graphiti | None = None
        self.entity_types = None
        self._schema_cache: dict | None = None
        self._schema_dirty: bool = True
        self.domain_profile: 'DomainProfile | None' = None
        self._cached_db_config: dict | None = None
        self._cached_embedder_client: Any = None

    async def _connect_ontology_client(self, db_config: dict, embedder_client) -> 'Graphiti | None':
        """Build and return an ontology Graphiti client.

        Retries up to _RETRY_ATTEMPTS times with exponential backoff to survive
        transient FalkorDB connection failures.

        Returns None if the configured database provider has no ontology support.
        Raises on persistent connection failure after all retries exhausted.
        """
        provider = self.config.database.provider.lower()
        ontology_graph_name = self.config.graphiti.ontology_graph

        if provider == 'falkordb':
            ontology_driver = FalkorDriver(
                host=db_config['host'],
                port=db_config['port'],
                username=db_config.get('username'),
                password=db_config['password'],
                database=ontology_graph_name,
            )
        elif provider == 'age':
            # AGE ontology lives in a companion graph (<graph>_ontology) in the same Postgres/AGE
            # instance — same DSN + embedding_dim, different graph_name.
            from graphiti_core.driver.age_driver import AGEDriver

            ontology_driver = AGEDriver(
                dsn=db_config['dsn'],
                graph_name=ontology_graph_name,
                embedding_dim=db_config['embedding_dim'],
            )
        else:
            logger.warning(f'Ontology graph not supported for {provider} provider')
            return None

        client = Graphiti(
            graph_driver=ontology_driver,
            llm_client=None,
            embedder=embedder_client,
        )

        backoff = _RETRY_INITIAL_BACKOFF_S
        for attempt in range(1, _RETRY_ATTEMPTS + 1):
            try:
                await client.build_indices_and_constraints()
                logger.info(f'Ontology graph connected: {ontology_graph_name}')
                return client
            except Exception as e:  # transient FalkorDB connect/protocol errors come in many flavors
                if attempt < _RETRY_ATTEMPTS:
                    logger.warning(
                        f'Ontology connect attempt {attempt}/{_RETRY_ATTEMPTS} failed: {e}. '
                        f'Retrying in {backoff:.1f}s...'
                    )
                    await asyncio.sleep(backoff)
                    backoff *= _RETRY_BACKOFF_MULTIPLIER
                else:
                    logger.warning(
                        f'Ontology connect gave up after {_RETRY_ATTEMPTS} attempts: {e}'
                    )
                    raise

    async def _ensure_ontology_client(self) -> bool:
        """Ensure self.ontology_client is connected; lazy-reconnect if it dropped.

        Returns True if the client is usable (existing or freshly reconnected);
        False if no ontology graph is configured, or if reconnect failed.
        """
        if self.ontology_client is not None:
            return True

        if not self.config.graphiti.ontology_graph:
            return False

        if self._cached_db_config is None or self._cached_embedder_client is None:
            # initialize() never ran successfully; nothing to retry from
            return False

        try:
            self.ontology_client = await self._connect_ontology_client(
                self._cached_db_config,
                self._cached_embedder_client,
            )
            return self.ontology_client is not None
        except Exception as e:
            logger.warning(f'Lazy ontology reconnect failed: {e}')
            self.ontology_client = None
            return False

    async def initialize(self) -> None:
        """Initialize the Graphiti client with factory-created components."""
        try:
            # Create clients using factories
            llm_client = None
            embedder_client = None

            # Create LLM client based on configured provider
            try:
                llm_client = LLMClientFactory.create(self.config.llm)
            except Exception as e:
                logger.warning(f'Failed to create LLM client: {e}')

            # Create embedder client based on configured provider
            try:
                embedder_client = EmbedderFactory.create(self.config.embedder)
            except Exception as e:
                logger.warning(f'Failed to create embedder client: {e}')

            # Get database configuration
            db_config = DatabaseDriverFactory.create_config(self.config.database)
            self._cached_db_config = db_config
            self._cached_embedder_client = embedder_client

            # Build entity types from configuration
            custom_types = None
            if self.config.graphiti.entity_types:
                custom_types = {}
                for entity_type in self.config.graphiti.entity_types:
                    # Create a dynamic Pydantic model for each entity type
                    # Note: Don't use 'name' as it's a protected Pydantic attribute
                    entity_model = type(
                        entity_type.name,
                        (BaseModel,),
                        {
                            '__doc__': entity_type.description,
                        },
                    )
                    custom_types[entity_type.name] = entity_model

            # Store entity types for later use
            self.entity_types = custom_types

            # Initialize Graphiti client with appropriate driver
            try:
                if self.config.database.provider.lower() == 'falkordb':
                    # For FalkorDB, create a FalkorDriver instance directly
                    from graphiti_core.driver.falkordb_driver import FalkorDriver

                    falkor_driver = FalkorDriver(
                        host=db_config['host'],
                        port=db_config['port'],
                        username=db_config.get('username'),
                        password=db_config['password'],
                        database=db_config['database'],
                    )

                    self.client = Graphiti(
                        graph_driver=falkor_driver,
                        llm_client=llm_client,
                        embedder=embedder_client,
                        max_coroutines=self.semaphore_limit,
                    )
                elif self.config.database.provider.lower() == 'age':
                    # For AGE (PostgreSQL + Apache AGE), create an AGEDriver instance directly.
                    from graphiti_core.driver.age_driver import AGEDriver

                    age_driver = AGEDriver(
                        dsn=db_config['dsn'],
                        graph_name=db_config['graph_name'],
                        embedding_dim=db_config['embedding_dim'],
                    )

                    self.client = Graphiti(
                        graph_driver=age_driver,
                        llm_client=llm_client,
                        embedder=embedder_client,
                        max_coroutines=self.semaphore_limit,
                    )
                else:
                    # For Neo4j (default), use the original approach
                    self.client = Graphiti(
                        uri=db_config['uri'],
                        user=db_config['user'],
                        password=db_config['password'],
                        llm_client=llm_client,
                        embedder=embedder_client,
                        max_coroutines=self.semaphore_limit,
                    )
            except Exception as db_error:
                # Check for connection errors
                error_msg = str(db_error).lower()
                if 'connection refused' in error_msg or 'could not connect' in error_msg:
                    db_provider = self.config.database.provider
                    if db_provider.lower() == 'falkordb':
                        raise RuntimeError(
                            f'\n{"=" * 70}\n'
                            f'Database Connection Error: FalkorDB is not running\n'
                            f'{"=" * 70}\n\n'
                            f'FalkorDB at {db_config["host"]}:{db_config["port"]} is not accessible.\n\n'
                            f'To start FalkorDB:\n'
                            f'  - Using Docker Compose: cd mcp_server && docker compose up\n'
                            f'  - Or run FalkorDB manually: docker run -p 6379:6379 falkordb/falkordb\n\n'
                            f'{"=" * 70}\n'
                        ) from db_error
                    elif db_provider.lower() == 'neo4j':
                        raise RuntimeError(
                            f'\n{"=" * 70}\n'
                            f'Database Connection Error: Neo4j is not running\n'
                            f'{"=" * 70}\n\n'
                            f'Neo4j at {db_config.get("uri", "unknown")} is not accessible.\n\n'
                            f'To start Neo4j:\n'
                            f'  - Using Docker Compose: cd mcp_server && docker compose -f docker/docker-compose-neo4j.yml up\n'
                            f'  - Or install Neo4j Desktop from: https://neo4j.com/download/\n'
                            f'  - Or run Neo4j manually: docker run -p 7474:7474 -p 7687:7687 neo4j:latest\n\n'
                            f'{"=" * 70}\n'
                        ) from db_error
                    else:
                        raise RuntimeError(
                            f'\n{"=" * 70}\n'
                            f'Database Connection Error: {db_provider} is not running\n'
                            f'{"=" * 70}\n\n'
                            f'{db_provider} at {db_config.get("uri", "unknown")} is not accessible.\n\n'
                            f'Please ensure {db_provider} is running and accessible.\n\n'
                            f'{"=" * 70}\n'
                        ) from db_error
                # Re-raise other errors
                raise

            # Retry build_indices to survive transient FalkorDB blips during cold start
            backoff = _RETRY_INITIAL_BACKOFF_S
            for attempt in range(1, _RETRY_ATTEMPTS + 1):
                try:
                    await self.client.build_indices_and_constraints()
                    break
                except Exception as e:  # transient FalkorDB connect/protocol errors come in many flavors
                    if attempt < _RETRY_ATTEMPTS:
                        logger.warning(
                            f'Main client build_indices attempt {attempt}/{_RETRY_ATTEMPTS} '
                            f'failed: {e}. Retrying in {backoff:.1f}s...'
                        )
                        await asyncio.sleep(backoff)
                        backoff *= _RETRY_BACKOFF_MULTIPLIER
                    else:
                        logger.error(
                            f'Main client build_indices gave up after {_RETRY_ATTEMPTS} attempts: {e}'
                        )
                        raise

            # Initialize ontology client if configured
            if self.config.graphiti.ontology_graph:
                try:
                    self.ontology_client = await self._connect_ontology_client(db_config, embedder_client)
                except Exception as e:
                    logger.warning(f'Failed to connect to ontology graph: {e}')
                    self.ontology_client = None

            logger.info('Successfully initialized Graphiti client')

            # Log configuration details
            if llm_client:
                logger.info(
                    f'Using LLM provider: {self.config.llm.provider} / {self.config.llm.model}'
                )
            else:
                logger.info('No LLM client configured - entity extraction will be limited')

            if embedder_client:
                logger.info(f'Using Embedder provider: {self.config.embedder.provider}')
            else:
                logger.info('No Embedder client configured - search will be limited')

            if self.entity_types:
                entity_type_names = list(self.entity_types.keys())
                logger.info(f'Using custom entity types: {", ".join(entity_type_names)}')
            else:
                logger.info('Using default entity types')

            logger.info(f'Using database: {self.config.database.provider}')
            logger.info(f'Using group_id: {self.config.graphiti.group_id}')

        except Exception as e:
            logger.error(f'Failed to initialize Graphiti client: {e}')
            raise

    async def get_client(self) -> Graphiti:
        """Get the Graphiti client, initializing if necessary."""
        if self.client is None:
            await self.initialize()
        if self.client is None:
            raise RuntimeError('Failed to initialize Graphiti client')
        return self.client


# Recipe lookup: (search_mode, reranker) -> SearchConfig
SEARCH_RECIPES: dict[tuple[str, str], SearchConfig] = {
    ('combined', 'rrf'): COMBINED_HYBRID_SEARCH_RRF,
    ('combined', 'mmr'): COMBINED_HYBRID_SEARCH_MMR,
    ('combined', 'cross_encoder'): COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
    ('edges', 'rrf'): EDGE_HYBRID_SEARCH_RRF,
    ('edges', 'mmr'): EDGE_HYBRID_SEARCH_MMR,
    ('edges', 'node_distance'): EDGE_HYBRID_SEARCH_NODE_DISTANCE,
    ('edges', 'episode_mentions'): EDGE_HYBRID_SEARCH_EPISODE_MENTIONS,
    ('edges', 'cross_encoder'): EDGE_HYBRID_SEARCH_CROSS_ENCODER,
    ('nodes', 'rrf'): NODE_HYBRID_SEARCH_RRF,
    ('nodes', 'mmr'): NODE_HYBRID_SEARCH_MMR,
    ('nodes', 'node_distance'): NODE_HYBRID_SEARCH_NODE_DISTANCE,
    ('nodes', 'episode_mentions'): NODE_HYBRID_SEARCH_EPISODE_MENTIONS,
    ('nodes', 'cross_encoder'): NODE_HYBRID_SEARCH_CROSS_ENCODER,
    ('communities', 'rrf'): COMMUNITY_HYBRID_SEARCH_RRF,
    ('communities', 'mmr'): COMMUNITY_HYBRID_SEARCH_MMR,
    ('communities', 'cross_encoder'): COMMUNITY_HYBRID_SEARCH_CROSS_ENCODER,
}

INTENT_STRATEGIES: dict[str, dict] = {
    'exhaustive':   {'search_mode': 'combined', 'reranker': 'rrf',              'limit': 50},
    'precise':      {'search_mode': 'nodes',    'reranker': 'cross_encoder',    'limit': 5},
    'neighborhood': {'search_mode': 'edges',    'reranker': 'node_distance',    'limit': 20},
    'diverse':      {'search_mode': 'combined', 'reranker': 'mmr',             'limit': 10},
    'temporal':     {'search_mode': 'combined', 'reranker': 'rrf',              'limit': 10},
    'path':         {'search_mode': 'edges',    'reranker': 'rrf',              'limit': 10},
    'importance':   {'search_mode': 'nodes',    'reranker': 'episode_mentions', 'limit': 10},
}


def resolve_search_config(search_mode: str, reranker: str, limit: int) -> SearchConfig:
    """Map search_mode + reranker to a SearchConfig recipe."""
    key = (search_mode.lower(), reranker.lower())
    recipe = SEARCH_RECIPES.get(key)
    if recipe is None:
        raise ValueError(
            f"Invalid search_mode='{search_mode}' + reranker='{reranker}'. "
            f"Valid combinations: {list(SEARCH_RECIPES.keys())}"
        )
    config = recipe.model_copy(deep=True)
    config.limit = limit
    return config


def format_node_result(node: EntityNode) -> dict[str, Any]:
    """An EntityNode as a wire dict, with every embedding key stripped.

    Companion to `format_edge_result` / `format_community_result`. Embeddings are
    thousands of floats an analyst never reads and a context window cannot
    afford, so any key containing 'embedding' is dropped.
    """
    return {
        'uuid': node.uuid,
        'name': node.name,
        'labels': node.labels or [],
        'created_at': node.created_at.isoformat() if node.created_at else None,
        'summary': node.summary,
        'group_id': node.group_id,
        'attributes': {
            k: v for k, v in (node.attributes or {}).items() if 'embedding' not in k.lower()
        },
    }


@mcp.tool(annotations=annotations_for('add_memory'))
async def add_memory(
    name: str | None = None,
    episode_body: str | None = None,
    group_id: str | None = None,
    source: Literal['text', 'json', 'message'] = 'text',
    source_description: str = '',
    uuid: str | None = None,
    sync: bool = False,
    episodes: list[dict] | None = None,
) -> AddMemoryResult:
    """Add information to the knowledge graph.

    Use when:
    - Ingesting new data (text documents, JSON records, or conversation messages)
    - Bulk loading multiple documents at once

    For single episodes, provide name + episode_body (queued for async processing).
    For bulk ingestion, provide episodes list (processed synchronously, returns when done).

    Args:
        name: Name of the episode (single mode).
        episode_body: Content to persist (single mode). When source='json', must be a JSON string.
        group_id: Graph partition ID. Uses default if omitted.
        source: Source type -- 'text' (default), 'json', or 'message'.
        source_description: Description of the source.
        uuid: Optional UUID for the episode (single mode).
        sync: If True, bypass the async queue and call Graphiti directly (single mode only).
              Returns EpisodeAddedResponse with extracted node/edge UUIDs.
        episodes: List of episodes for bulk ingestion (bulk mode).
                  Each dict: {"name": str, "content": str, "source": str, "source_description": str}

    Examples:
        # Single episode
        add_memory(name="Report", episode_body="Aircraft PH-KZB experienced...", source="text")

        # Bulk ingestion
        add_memory(episodes=[
            {"name": "Doc 1", "content": "...", "source": "text", "source_description": "report"},
        ])

    Used by downstream services (e.g., aletheia-extraction) to persist extracted
    records into the knowledge graph after ontology mapping and human approval.

    See docs/extraction-integration.md for the integration contract:
    response shape, adapter recipe (where applicable), and error modes.
    """
    global graphiti_service, queue_service

    if graphiti_service is None or queue_service is None:
        return AddMemoryResult(error='Services not initialized')

    effective_group_id = group_id or config.graphiti.group_id

    try:
        # Bulk mode
        if episodes is not None:
            if not episodes:
                return AddMemoryResult(error='Episodes list is empty')

            client = await graphiti_service.get_client()

            raw_episodes = []
            for i, ep in enumerate(episodes):
                if 'name' not in ep or 'content' not in ep:
                    missing = [k for k in ('name', 'content') if k not in ep]
                    return AddMemoryResult(
                        error=f"Episode at index {i} missing required key(s): {', '.join(missing)}"
                    )

                ep_source = ep.get('source', 'text')
                try:
                    episode_type = EpisodeType[ep_source.lower()]
                except (KeyError, AttributeError):
                    episode_type = EpisodeType.text

                raw_episodes.append(RawEpisode(
                    name=ep['name'],
                    content=ep['content'],
                    source_description=ep.get('source_description', ''),
                    source=episode_type,
                    reference_time=datetime.now(),
                ))

            results = await client.add_episode_bulk(
                bulk_episodes=raw_episodes,
                group_id=effective_group_id,
                entity_types=graphiti_service.entity_types,
            )

            # Invalidate schema cache after ingestion
            graphiti_service._schema_dirty = True

            return AddMemoryResult(
                message=f"Bulk ingested {len(raw_episodes)} episodes into '{effective_group_id}': "
                        f"{len(results.nodes)} nodes, {len(results.edges)} edges created"
            )

        # Single mode (existing behavior)
        if not name or not episode_body:
            return AddMemoryResult(error='Provide name + episode_body for single mode, or episodes for bulk mode')

        episode_type = EpisodeType.text
        if source:
            try:
                episode_type = EpisodeType[source.lower()]
            except (KeyError, AttributeError):
                logger.warning(f"Unknown source type '{source}', using 'text'")
                episode_type = EpisodeType.text

        if sync:
            # Synchronous: bypass queue, call Graphiti directly, return UUIDs
            client = await graphiti_service.get_client()
            results = await client.add_episode(
                name=name,
                episode_body=episode_body,
                source_description=source_description,
                source=episode_type,
                group_id=effective_group_id,
                reference_time=datetime.now(),
                entity_types=graphiti_service.entity_types,
                uuid=uuid or None,
            )

            graphiti_service._schema_dirty = True

            return AddMemoryResult(
                message=f"Episode '{name}' processed synchronously in '{effective_group_id}': "
                        f"{len(results.nodes)} nodes, {len(results.edges)} edges",
                node_uuids=[n.uuid for n in results.nodes],
                edge_uuids=[e.uuid for e in results.edges],
            )

        await queue_service.add_episode(
            group_id=effective_group_id,
            name=name,
            content=episode_body,
            source_description=source_description,
            episode_type=episode_type,
            entity_types=graphiti_service.entity_types,
            uuid=uuid or None,
        )

        # Invalidate schema cache after ingestion
        graphiti_service._schema_dirty = True

        return AddMemoryResult(
            message=f"Episode '{name}' queued for processing in group '{effective_group_id}'"
        )

    except Exception as e:
        logger.error(f'Error in add_memory: {e}')
        return AddMemoryResult(error=f'Error adding memory: {e}')


async def search(
    query: str,
    intent: Literal['exhaustive', 'precise', 'neighborhood', 'diverse', 'temporal', 'path', 'importance'] | None = None,
    group_ids: list[str] | None = None,
    search_mode: Literal['nodes', 'edges', 'communities', 'combined'] = 'combined',
    reranker: Literal['rrf', 'mmr', 'cross_encoder', 'node_distance', 'episode_mentions'] = 'rrf',
    center_node_uuid: str | None = None,
    bfs_origin_node_uuids: list[str] | None = None,
    entity_types: list[str] | None = None,
    edge_types: list[str] | None = None,
    valid_at: str | None = None,
    limit: int = 10,
) -> SearchResult:
    """Search the knowledge graph using a semantic intent or explicit parameters.

    Prefer passing `intent` to let the server choose the best strategy.
    Pass `search_mode`/`reranker` directly only when you need explicit control.

    Args:
        query: Natural language search query.
        intent: Search intent — the server maps this to the best search_mode + reranker.
                One of: exhaustive, precise, neighborhood, diverse, temporal, path, importance.
                When provided, overrides search_mode and reranker defaults.
        group_ids: Search across these graph partitions. Omit to use the default.
        search_mode: What to search — "nodes", "edges", "communities", or "combined" (default).
        reranker: Reranking strategy — "rrf" (default), "mmr", "cross_encoder",
                  "node_distance" (requires center_node_uuid), or "episode_mentions".
        center_node_uuid: Rerank results by proximity to this node.
        bfs_origin_node_uuids: Start BFS graph traversal from these nodes.
        entity_types: Only return nodes with these labels (e.g. ["Person", "Organization"]).
        edge_types: Only return edges of these types (e.g. ["OWNERSHIP", "SANCTION"]).
        valid_at: ISO date string — only return facts valid at this date.
        limit: Maximum results to return (default 10).
    """
    global graphiti_service

    if graphiti_service is None:
        return SearchResult(error='Graphiti service not initialized')

    try:
        start_time = time.time()
        client = await graphiti_service.get_client()

        effective_group_ids = (
            group_ids
            if group_ids is not None
            else [config.graphiti.group_id]
            if config.graphiti.group_id
            else []
        )

        # Resolve intent to search_mode + reranker
        effective_search_mode = search_mode
        effective_reranker = reranker
        effective_limit = limit
        if intent:
            strategy = INTENT_STRATEGIES.get(intent)
            if strategy is None:
                return SearchResult(
                    error=f"Unknown intent '{intent}'. "
                    f"Valid intents: {list(INTENT_STRATEGIES.keys())}"
                )
            effective_search_mode = strategy['search_mode']
            effective_reranker = strategy['reranker']
            if limit == 10:  # default value — use intent's limit
                effective_limit = strategy['limit']

        search_config = resolve_search_config(effective_search_mode, effective_reranker, effective_limit)

        search_filters = SearchFilters()
        if entity_types:
            search_filters.node_labels = entity_types
        if edge_types:
            search_filters.edge_types = edge_types
        if valid_at:
            from graphiti_core.search.search_filters import DateFilter, ComparisonOperator

            valid_date = datetime.fromisoformat(valid_at)
            search_filters.valid_at = [
                [
                    DateFilter(
                        date=valid_date,
                        comparison_operator=ComparisonOperator.less_than_equal,
                    ),
                ]
            ]
            search_filters.invalid_at = [
                [
                    DateFilter(
                        date=valid_date,
                        comparison_operator=ComparisonOperator.greater_than,
                    ),
                ],
                [
                    DateFilter(
                        date=None,
                        comparison_operator=ComparisonOperator.is_null,
                    ),
                ],
            ]

        results = await client.search_(
            query=query,
            config=search_config,
            group_ids=effective_group_ids,
            center_node_uuid=center_node_uuid,
            bfs_origin_node_uuids=bfs_origin_node_uuids,
            search_filter=search_filters,
        )

        node_results = [
            {
                'uuid': n.uuid,
                'name': n.name,
                'labels': n.labels or [],
                'created_at': n.created_at.isoformat() if n.created_at else None,
                'summary': n.summary,
                'group_id': n.group_id,
                'attributes': {
                    k: v
                    for k, v in (n.attributes or {}).items()
                    if 'embedding' not in k.lower()
                },
            }
            for n in (results.nodes or [])
        ]

        edge_results = [format_edge_result(e) for e in (results.edges or [])]
        community_results = [format_community_result(c) for c in (results.communities or [])]

        return SearchResult(
            message=f'Found {len(node_results)} nodes, {len(edge_results)} edges, {len(community_results)} communities',
            nodes=node_results,
            edges=edge_results,
            communities=community_results,
            execution_ms=round((time.time() - start_time) * 1000, 1),
        )

    except ValueError as e:
        return SearchResult(error=str(e))
    except Exception as e:
        logger.error(f'Error in search: {e}')
        return SearchResult(error=f'Search error: {e}')


async def explore_node(
    node_name: str | None = None,
    node_uuid: str | None = None,
    group_ids: list[str] | None = None,
    depth: Literal[1, 2, 3, 4] = 2,
    edge_types: list[str] | None = None,
    limit: int = 20,
) -> ExploreResult:
    """Explore everything connected to a specific entity in the knowledge graph.

    Resolves a node by name or UUID, then expands outward via graph traversal.
    Results are ranked by proximity to the center node.

    Args:
        node_name: Find the node by name (performs a quick search). Provide this or node_uuid.
        node_uuid: Expand directly from this node UUID. Provide this or node_name.
        group_ids: Which graph partitions to explore. Omit for default.
        depth: How many hops to traverse (1-4, default 2).
        edge_types: Only traverse these relationship types (e.g. ["OWNERSHIP"]).
        limit: Maximum results to return (default 20).
    """
    global graphiti_service

    if graphiti_service is None:
        return ExploreResult(error='Graphiti service not initialized')

    if not node_name and not node_uuid:
        return ExploreResult(error='Provide either node_name or node_uuid')

    try:
        client = await graphiti_service.get_client()

        effective_group_ids = (
            group_ids
            if group_ids is not None
            else [config.graphiti.group_id]
            if config.graphiti.group_id
            else []
        )

        # Resolve the center node — by name (search) or by UUID (direct lookup).
        # The UUID branch MUST look the node up rather than hope it turns up in
        # the neighbourhood results: a center node is not necessarily its own
        # neighbour, so relying on the results left `center_node: null` on every
        # uuid call whose traversal came back thin.
        resolved_uuid = node_uuid
        center_node = None

        if node_name and not node_uuid:
            resolve_results = await client.search_(
                query=node_name,
                config=NODE_HYBRID_SEARCH_RRF,
                group_ids=effective_group_ids,
            )
            if not resolve_results.nodes:
                return ExploreResult(
                    message=f'No node found matching "{node_name}"',
                    center_node=None,
                    nodes=[],
                    edges=[],
                    communities=[],
                )
            center_node = resolve_results.nodes[0]
            resolved_uuid = center_node.uuid
        else:
            try:
                center_node = await EntityNode.get_by_uuid(client.driver, node_uuid)
            except NodeNotFoundError:
                # ONLY "that uuid is not in the graph" is an answer. A dropped
                # connection pool or a backend error must NOT be dressed up as a
                # missing node — it falls through to the outer handler and comes
                # back as an ErrorResponse the caller can act on.
                #
                # Note (recorded divergence, not fixed here): on AGE
                # `node_get_by_uuid` is not label-scoped, so an EPISODIC uuid
                # resolves as a centre instead of raising, where Neo4j/FalkorDB
                # match `(n:Entity …)` and raise. Changing that lookup has other
                # callers and belongs to its own lane.
                logger.info(f'explore_node: no node with uuid {node_uuid}')
                return ExploreResult(
                    message=f'No node found with UUID "{node_uuid}"',
                    center_node=None,
                    nodes=[],
                    edges=[],
                    communities=[],
                )

        center_node_result = format_node_result(center_node)

        # Build a node_distance config with BFS
        explore_config = SearchConfig(
            edge_config=EdgeSearchConfig(
                search_methods=[
                    EdgeSearchMethod.bm25,
                    EdgeSearchMethod.cosine_similarity,
                    EdgeSearchMethod.bfs,
                ],
                reranker=EdgeReranker.node_distance,
                bfs_max_depth=min(depth, 4),
            ),
            node_config=NodeSearchConfig(
                search_methods=[
                    NodeSearchMethod.bm25,
                    NodeSearchMethod.cosine_similarity,
                    NodeSearchMethod.bfs,
                ],
                reranker=NodeReranker.node_distance,
                bfs_max_depth=min(depth, 4),
            ),
            limit=limit,
        )

        search_filters = SearchFilters()
        if edge_types:
            search_filters.edge_types = edge_types

        results = await client.search_(
            query=node_name or '',
            config=explore_config,
            group_ids=effective_group_ids,
            center_node_uuid=resolved_uuid,
            bfs_origin_node_uuids=[resolved_uuid] if resolved_uuid else None,
            search_filter=search_filters,
        )

        node_results = [format_node_result(n) for n in (results.nodes or [])]
        edge_results = [format_edge_result(e) for e in (results.edges or [])]
        community_results = [format_community_result(c) for c in (results.communities or [])]

        return ExploreResult(
            message=f'Explored "{node_name or node_uuid}": {len(node_results)} nodes, {len(edge_results)} edges',
            center_node=center_node_result,
            nodes=node_results,
            edges=edge_results,
            communities=community_results,
        )

    except Exception as e:
        logger.error(f'Error in explore_node: {e}')
        return ExploreResult(error=f'Explore error: {e}')


@mcp.tool(annotations=annotations_for('get_episode_context'))
async def get_episode_context(
    episode_uuids: list[str],
) -> EpisodeContextResult:
    """Get all entities and relationships extracted from specific episodes.

    Use when:
    - You want to see what nodes and edges were created from a specific document
    - Inspecting extraction quality for a particular episode

    Pair with get_episodes to first list episodes, then inspect what was extracted.

    Args:
        episode_uuids: List of episode UUIDs to inspect.
    """
    global graphiti_service

    if graphiti_service is None:
        return EpisodeContextResult(error='Graphiti service not initialized')

    if not episode_uuids:
        return EpisodeContextResult(error='Provide at least one episode UUID')

    try:
        client = await graphiti_service.get_client()

        results = await client.get_nodes_and_edges_by_episode(episode_uuids)

        node_results = [
            {
                'uuid': n.uuid,
                'name': n.name,
                'labels': n.labels or [],
                'created_at': n.created_at.isoformat() if n.created_at else None,
                'summary': n.summary,
                'group_id': n.group_id,
                'attributes': {
                    k: v
                    for k, v in (n.attributes or {}).items()
                    if 'embedding' not in k.lower()
                },
            }
            for n in (results.nodes or [])
        ]

        edge_results = [format_edge_result(e) for e in (results.edges or [])]

        return EpisodeContextResult(
            message=f'Found {len(node_results)} nodes and {len(edge_results)} edges from {len(episode_uuids)} episodes',
            nodes=node_results,
            edges=edge_results,
        )

    except Exception as e:
        logger.error(f'Error in get_episode_context: {e}')
        return EpisodeContextResult(error=f'Episode context error: {e}')


@mcp.tool(annotations=annotations_for('build_communities'))
async def build_communities(
    group_ids: list[str],
) -> CommunityBuildResult:
    """Build communities by clustering entities in the knowledge graph.

    Use when:
    - You want to search with search_mode="communities" (must be built first)
    - You want high-level summaries of entity clusters

    Do NOT use when:
    - Communities have already been built for these group_ids

    Args:
        group_ids: Which graph partitions to cluster.
    """
    global graphiti_service

    if graphiti_service is None:
        return CommunityBuildResult(error='Graphiti service not initialized')

    if not group_ids:
        return CommunityBuildResult(error='Provide at least one group_id')

    try:
        client = await graphiti_service.get_client()

        community_nodes, community_edges = await client.build_communities(
            group_ids=group_ids,
        )

        community_results = [
            format_community_result(c, member_count=0)
            for c in community_nodes
        ]

        return CommunityBuildResult(
            message=f'Built {len(community_nodes)} communities across {len(group_ids)} graphs',
            community_count=len(community_nodes),
            communities=community_results,
        )

    except Exception as e:
        logger.error(f'Error building communities: {e}')
        return CommunityBuildResult(error=f'Community build error: {e}')


@mcp.tool(annotations=annotations_for('delete_entity_edge'))
async def delete_entity_edge(uuid: str) -> MutationResult:
    """Delete a relationship (edge) from the knowledge graph.

    Use when:
    - A specific fact or relationship needs to be removed
    - Correcting incorrect information in the graph

    Args:
        uuid: UUID of the edge to delete.
    """
    global graphiti_service

    if graphiti_service is None:
        return MutationResult(error='Graphiti service not initialized')

    try:
        client = await graphiti_service.get_client()

        # Get the entity edge by UUID
        entity_edge = await EntityEdge.get_by_uuid(client.driver, uuid)
        # Delete the edge using its delete method
        await entity_edge.delete(client.driver)
        graphiti_service._schema_dirty = True
        return MutationResult(message=f'Entity edge with UUID {uuid} deleted successfully')
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error deleting entity edge: {error_msg}')
        return MutationResult(error=f'Error deleting entity edge: {error_msg}')


@mcp.tool(annotations=annotations_for('delete_episode'))
async def delete_episode(uuid: str) -> MutationResult:
    """Delete an episode and its extracted data from the knowledge graph.

    Use when:
    - An ingested document should be removed along with its extracted entities and relationships

    Args:
        uuid: UUID of the episode to delete.
    """
    global graphiti_service

    if graphiti_service is None:
        return MutationResult(error='Graphiti service not initialized')

    try:
        client = await graphiti_service.get_client()

        # Get the episodic node by UUID
        episodic_node = await EpisodicNode.get_by_uuid(client.driver, uuid)
        # Delete the node using its delete method
        await episodic_node.delete(client.driver)
        graphiti_service._schema_dirty = True
        return MutationResult(message=f'Episode with UUID {uuid} deleted successfully')
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error deleting episode: {error_msg}')
        return MutationResult(error=f'Error deleting episode: {error_msg}')


@mcp.tool(annotations=annotations_for('get_episodes'))
async def get_episodes(
    group_ids: list[str] | None = None,
    max_episodes: int = 10,
) -> EpisodeListResult:
    """List recent episodes (ingested documents) from the knowledge graph.

    Use when:
    - You want to see what data has been ingested
    - You need episode UUIDs for get_episode_context

    Args:
        group_ids: Optional list of group IDs to filter results.
        max_episodes: Maximum number of episodes to return (default: 10).
    """
    global graphiti_service

    if graphiti_service is None:
        return EpisodeListResult(error='Graphiti service not initialized')

    try:
        client = await graphiti_service.get_client()

        # Use the provided group_ids or fall back to the default from config if none provided
        effective_group_ids = (
            group_ids
            if group_ids is not None
            else [config.graphiti.group_id]
            if config.graphiti.group_id
            else []
        )

        # Get episodes from the driver directly
        from graphiti_core.nodes import EpisodicNode

        if effective_group_ids:
            episodes = await EpisodicNode.get_by_group_ids(
                client.driver, effective_group_ids, limit=max_episodes
            )
        else:
            # If no group IDs, we need to use a different approach
            # For now, return empty list when no group IDs specified
            episodes = []

        if not episodes:
            return EpisodeListResult(message='No episodes found', episodes=[])

        # Format the results
        episode_results = []
        for episode in episodes:
            episode_dict = {
                'uuid': episode.uuid,
                'name': episode.name,
                'content': episode.content,
                'created_at': episode.created_at.isoformat() if episode.created_at else None,
                'source': episode.source.value
                if hasattr(episode.source, 'value')
                else str(episode.source),
                'source_description': episode.source_description,
                'group_id': episode.group_id,
            }
            episode_results.append(episode_dict)

        return EpisodeListResult(
            message='Episodes retrieved successfully', episodes=episode_results
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error getting episodes: {error_msg}')
        return EpisodeListResult(error=f'Error getting episodes: {error_msg}')


@mcp.tool(annotations=annotations_for('clear_graph'))
async def clear_graph(group_ids: list[str] | None = None) -> MutationResult:
    """Delete all data from the knowledge graph for specified group IDs.

    Use when:
    - You need to completely reset a knowledge graph partition
    - Starting fresh with new data

    WARNING: This is destructive and cannot be undone.

    Args:
        group_ids: Group IDs to clear. If not provided, clears the default group.
    """
    global graphiti_service

    if graphiti_service is None:
        return MutationResult(error='Graphiti service not initialized')

    try:
        client = await graphiti_service.get_client()

        # Use the provided group_ids or fall back to the default from config if none provided
        effective_group_ids = (
            group_ids or [config.graphiti.group_id] if config.graphiti.group_id else []
        )

        if not effective_group_ids:
            return MutationResult(error='No group IDs specified for clearing')

        # Clear data for the specified group IDs
        await clear_data(client.driver, group_ids=effective_group_ids)

        graphiti_service._schema_dirty = True

        return MutationResult(
            message=f'Graph data cleared successfully for group IDs: {", ".join(effective_group_ids)}'
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error clearing graph: {error_msg}')
        return MutationResult(error=f'Error clearing graph: {error_msg}')


@mcp.tool(annotations=annotations_for('get_status'))
async def get_status() -> StatusResponse:
    """Check if the MCP server and database connection are healthy.

    Use when:
    - Verifying the server is running and database is reachable
    - Troubleshooting connection issues
    """
    global graphiti_service

    if graphiti_service is None:
        return StatusResponse(status='error', message='Graphiti service not initialized')

    try:
        client = await graphiti_service.get_client()

        # Test database connection with a simple query
        async with client.driver.session() as session:
            result = await session.run('MATCH (n) RETURN count(n) as count')
            # Consume the result to verify query execution
            if result:
                _ = [record async for record in result]

        # Use the provider from the service's config, not the global
        provider_name = graphiti_service.config.database.provider
        return StatusResponse(
            status='ok',
            message=f'Graphiti MCP server is running and connected to {provider_name} database',
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error checking database connection: {error_msg}')
        return StatusResponse(
            status='error',
            message=f'Graphiti MCP server is running but database connection failed: {error_msg}',
        )


async def search_ontology(
    query: str,
    search_mode: Literal['nodes', 'edges', 'communities', 'combined'] = 'combined',
    reranker: Literal['rrf', 'mmr', 'cross_encoder'] = 'rrf',
    limit: int = 10,
) -> SearchResult:
    """Search the companion ontology graph for schema definitions, entity types, and relationships.

    Use this to understand what types of entities and relationships exist in the knowledge graph,
    what properties they have, and how they relate to each other.

    Ontology access tiers: this tool is semantic RECALL — it finds candidate
    classes by meaning when you don't know the exact name. For the
    whole-ontology surface map use get_ontology_structure; to study ONE class
    in full context use explore_ontology; for the complete reference (full
    prose + property definitions, large) use get_ontology_documentation.

    Args:
        query: Natural language search query (e.g., "AirworthinessDirective", "what properties does Aircraft have").
        search_mode: What to search — "nodes", "edges", "communities", or "combined" (default).
        reranker: Reranking strategy — "rrf" (default), "mmr", or "cross_encoder".
        limit: Maximum results to return (default 10).
    """
    global graphiti_service

    if graphiti_service is None:
        return SearchResult(error='Graphiti service not initialized')

    if not await graphiti_service._ensure_ontology_client():
        return SearchResult(error='No ontology graph configured for this server')
    assert graphiti_service.ontology_client is not None  # type narrowing — _ensure_ontology_client guarantees non-None on True

    try:
        start_time = time.time()
        search_config = resolve_search_config(search_mode, reranker, limit)

        ontology_group_id = config.graphiti.ontology_graph

        results = await graphiti_service.ontology_client.search_(
            query=query,
            config=search_config,
            group_ids=[ontology_group_id],
        )

        node_results = [
            {
                'uuid': n.uuid,
                'name': n.name,
                'labels': n.labels or [],
                'created_at': n.created_at.isoformat() if n.created_at else None,
                'summary': n.summary,
                'group_id': n.group_id,
                'attributes': {
                    k: v
                    for k, v in (n.attributes or {}).items()
                    if 'embedding' not in k.lower()
                },
            }
            for n in (results.nodes or [])
        ]

        edge_results = [format_edge_result(e) for e in (results.edges or [])]
        community_results = [format_community_result(c) for c in (results.communities or [])]

        return SearchResult(
            message=f'Ontology: {len(node_results)} nodes, {len(edge_results)} edges, {len(community_results)} communities',
            nodes=node_results,
            edges=edge_results,
            communities=community_results,
            execution_ms=round((time.time() - start_time) * 1000, 1),
        )

    except ValueError as e:
        return SearchResult(error=str(e))
    except Exception as e:
        logger.error(f'Error in search_ontology: {e}')
        return SearchResult(error=f'Ontology search error: {e}')


# The three ontology tiers read their rows through `flavour.ontology_queries()`
# (keys: class_context / structure / relates). The FLAVOUR owns the query text
# because the storage shape is dialect-specific — FalkorDB keeps every
# descriptive ontology field as a top-level node property and stores
# relationships as RELATES_TO edges; Apache AGE nests the fields in an
# `attributes` map and materializes the relation name as the edge LABEL. The
# parsing below is shared, because every variant projects the same aliases.


def _parse_properties(raw: str | None) -> list[dict[str, Any]]:
    """Parse the OntologyClass 'properties' JSON attribute; [] on absent/bad."""
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _first_sentence(text: str) -> str:
    """First sentence (or line) of a summary — the one-line preview form."""
    text = (text or '').strip()
    for sep in ('. ', '.\n', '\n'):
        idx = text.find(sep)
        if idx > 0:
            return text[: idx + 1].strip()
    return text


def _edge_relationship_entry(rec: dict[str, Any]) -> dict[str, Any]:
    """Relationship entry derived from a RELATES_TO edge row.

    The summary is the edge `fact` with its leading
    "<source> <name> <target>: " prefix stripped when present. Caveat:
    edge facts may be shorter than the full ontology relationship comment —
    full-fidelity relationship prose is a builder-side follow-up.
    """
    source = rec.get('source') or ''
    name = rec.get('name') or ''
    target = rec.get('target') or ''
    fact = rec.get('fact') or ''
    prefix = f'{source} {name} {target}: '
    summary = fact[len(prefix):] if fact.startswith(prefix) else fact
    return {
        'name': name,
        'ontology_type': 'relationship_class',
        'summary': summary,
        'alt_labels': [],
        'inherits_from': [],
        'examples': [],
        'identity': False,
        'properties': [],
        'source_entity': source,
        'target_entity': target,
    }


def _combine_relationship_entries(
    node_derived: list[dict[str, Any]],
    edge_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Union node-derived relationship entries with edge-derived (RELATES_TO)
    entries, deduped by (name, source_entity, target_entity). Node-derived
    entries win — reified-class ontologies keep their richer prose."""
    combined = list(node_derived)
    seen = {
        (e.get('name'), e.get('source_entity'), e.get('target_entity'))
        for e in node_derived
    }
    for rec in edge_records:
        # Builders store class hierarchy as RELATES_TO edges named
        # SUBCLASS_OF. Hierarchy travels via inherits_from, not as a
        # relationship entry — exclude these to avoid duplicating the
        # inheritance links in relationship listings.
        if rec.get('name') == 'SUBCLASS_OF':
            continue
        entry = _edge_relationship_entry(rec)
        key = (entry['name'], entry['source_entity'], entry['target_entity'])
        if key in seen:
            continue
        seen.add(key)
        combined.append(entry)
    return combined


def _ontology_full_entry(rec: dict[str, Any]) -> dict[str, Any]:
    """Full-detail entry for one OntologyClass row: structure-tool fields plus
    full summary, parsed properties, and identity flag."""
    ontology_type = rec.get('ontology_type', '') or ''
    entry = {
        'name': rec.get('name', '') or '',
        'ontology_type': ontology_type,
        'summary': rec.get('summary', '') or '',  # full text, never truncated
        'alt_labels': rec.get('alt_labels') or [],
        'inherits_from': rec.get('inherits_from') or [],
        'examples': rec.get('examples') or [],
        'identity': bool(rec.get('identity') or False),
        'properties': _parse_properties(rec.get('properties')),
    }
    if ontology_type == 'relationship_class':
        entry['source_entity'] = rec.get('source_entity', '') or ''
        entry['target_entity'] = rec.get('target_entity', '') or ''
    return entry


async def explore_ontology(
    node_name: str | None = None,
    node_uuid: str | None = None,
    depth: Literal[1, 2, 3, 4] = 2,
    limit: int = 20,
) -> OntologyClassContextResponse:
    """Gather the full context of ONE ontology class.

    Returns its complete documentation, properties, and typed surroundings:
    relationships with their meanings, parents/children/siblings in the class
    hierarchy, and the classes it connects to.

    Ontology access tiers: use get_ontology_structure for the whole-ontology
    map, search_ontology to find a class semantically, and
    get_ontology_documentation for the complete reference (large).

    Relationships come from reified relationship_class nodes and from
    RELATES_TO edges (object-property ontologies). Caveat: edge-derived
    relationship summaries come from the stored edge fact and may be shorter
    than the full ontology relationship comment.

    Args:
        node_name: Find the ontology class by name (e.g., "AirworthinessDirective"). Provide this or node_uuid.
        node_uuid: Resolve the class by its node UUID. Provide this or node_name.
        depth: 1 = direct neighbors only; >=2 adds a name-only second hop (default 2).
        limit: Maximum entries per surrounding list (default 20).

    Returns:
        {center, relationships: {outgoing, incoming},
         hierarchy: {parents, children, siblings}, neighbors}
    """
    global graphiti_service

    if graphiti_service is None:
        return OntologyClassContextResponse(error='Graphiti service not initialized')

    if not await graphiti_service._ensure_ontology_client():
        return OntologyClassContextResponse(error='No ontology graph configured for this server')
    assert graphiti_service.ontology_client is not None  # type narrowing — _ensure_ontology_client guarantees non-None on True

    if not node_name and not node_uuid:
        return OntologyClassContextResponse(error='Provide either node_name or node_uuid')

    try:
        ontology_client = graphiti_service.ontology_client
        ontology_group_id = graphiti_service.config.graphiti.ontology_graph
        ontology_queries = graphiti_service.flavour.ontology_queries()

        # ONE query: every ontology class, full projection, held in memory
        # (ontology graphs are small).
        rows, _, _ = await ontology_client.driver.execute_query(
            ontology_queries['class_context']
        )
        by_name: dict[str, dict[str, Any]] = {
            r.get('name'): r for r in rows if r.get('name')
        }

        # Resolve the center row: uuid, exact name, then semantic fallback.
        center_row: dict[str, Any] | None = None
        if node_uuid:
            center_row = next((r for r in rows if r.get('uuid') == node_uuid), None)
        if center_row is None and node_name:
            center_row = by_name.get(node_name)
            if center_row is None:
                lowered = {n.lower(): r for n, r in by_name.items()}
                center_row = lowered.get(node_name.lower())
            if center_row is None:
                # Fuzzy name: resolve via ontology semantic search.
                resolve_results = await ontology_client.search_(
                    query=node_name,
                    config=NODE_HYBRID_SEARCH_RRF,
                    group_ids=[ontology_group_id],
                )
                for match in resolve_results.nodes or []:
                    if match.name in by_name:
                        center_row = by_name[match.name]
                        break
        if center_row is None:
            return OntologyClassContextResponse(
                error=f'No ontology class found matching "{node_name or node_uuid}"'
            )

        center_name = center_row.get('name') or ''
        center = _ontology_full_entry(center_row)

        # Relationships touching the center (full summaries — their meanings).
        # Two sources, deduped by (name, source, target) with node-derived
        # winning: reified relationship_class nodes AND RELATES_TO edges
        # (object-property ontologies store relationships only as edges).
        node_rel_rows = [
            r for r in rows if (r.get('ontology_type') or '') == 'relationship_class'
        ]
        edge_records, _, _ = await ontology_client.driver.execute_query(
            ontology_queries['relates']
        )
        rel_rows = _combine_relationship_entries(node_rel_rows, edge_records)
        outgoing = [
            {
                'name': r.get('name'),
                'target': r.get('target_entity'),
                'summary': r.get('summary') or '',
            }
            for r in rel_rows
            if r.get('source_entity') == center_name
        ]
        incoming = [
            {
                'name': r.get('name'),
                'source': r.get('source_entity'),
                'summary': r.get('summary') or '',
            }
            for r in rel_rows
            if r.get('target_entity') == center_name
        ]

        # Class hierarchy: parents, children, siblings (one-line previews).
        parents = [
            {'name': p, 'summary_line': _first_sentence(by_name[p].get('summary') or '')}
            for p in (center_row.get('inherits_from') or [])
            if p in by_name
        ]
        children = [
            {
                'name': r.get('name'),
                'summary_line': _first_sentence(r.get('summary') or ''),
            }
            for r in rows
            if center_name in (r.get('inherits_from') or [])
        ][:limit]
        center_parents = set(center_row.get('inherits_from') or [])
        siblings = [
            {
                'name': r.get('name'),
                'summary_line': _first_sentence(r.get('summary') or ''),
            }
            for r in rows
            if r.get('name') != center_name
            and center_parents & set(r.get('inherits_from') or [])
        ][:limit]

        # Connected classes: direct neighbors with previews, plus (depth>=2)
        # a name-only second hop.
        seen: set[str] = {center_name}
        neighbors: list[dict[str, Any]] = []
        for rel in outgoing + incoming:
            other = rel.get('target') or rel.get('source')
            if not other or other in seen or other not in by_name:
                continue
            seen.add(other)
            neighbors.append(
                {
                    'name': other,
                    'summary_line': _first_sentence(by_name[other].get('summary') or ''),
                    'via': rel['name'],
                }
            )
        if depth >= 2:
            for n in list(neighbors):
                for r in rel_rows:
                    if len(neighbors) >= limit:
                        break
                    if (
                        r.get('source_entity') == n['name']
                        and r.get('target_entity') not in seen
                        and r.get('target_entity') in by_name
                    ):
                        seen.add(r['target_entity'])
                        neighbors.append({'name': r['target_entity'], 'via': r.get('name')})
        neighbors = neighbors[:limit]

        return {
            'center': center,
            'relationships': {'outgoing': outgoing, 'incoming': incoming},
            'hierarchy': {'parents': parents, 'children': children, 'siblings': siblings},
            'neighbors': neighbors,
        }

    except Exception as e:
        logger.error(f'Error in explore_ontology: {e}')
        return OntologyClassContextResponse(error=f'Ontology explore error: {e}')


@mcp.custom_route('/health', methods=['GET'])
async def health_check(request) -> JSONResponse:
    """Health check endpoint for Docker and load balancers."""
    return JSONResponse({'status': 'healthy', 'service': 'graphiti-mcp'})


def _pick_endpoint(rec: dict[str, Any], side: str, internal: frozenset[str] | set[str]) -> str | None:
    """The endpoint label to advertise for one side of a relationship-pattern row.

    Prefer the flavour's announced LEAF column when it is present: on AGE
    `source_labels` is the full ontology hierarchy, so picking positionally lands
    on an abstract supertype — which no vertex carries as its stored label (a
    label-scoped probe against it matches nothing), and which merges genuinely
    distinct patterns wherever the caller dedups on the advertised pair.

    Flavours that announce no leaf column (FalkorDB / openCypher, where
    `labels(s)` already lists matchable labels) keep the positional pick over the
    filtered list — behaviour unchanged.

    NOTE: graph_profiler.py carries a textually parallel copy. The two consumers
    are deliberately independent (no cross-module import between the server module
    and the tool module); keep them in step.
    """
    leaf = rec.get(f'{side}_leaf')
    if leaf and leaf not in internal:
        return leaf
    labels = [l for l in rec.get(f'{side}_labels') or [] if l not in internal]
    return labels[0] if labels else None


async def get_schema() -> SchemaResponse:
    """Retrieve the structural schema of the knowledge graph.

    Returns node labels with property keys, relationship types with
    source->target patterns, and counts. Results are cached until
    new data is ingested via add_memory.

    Used by downstream services (e.g., aletheia-extraction) to discover
    available entity and relationship types before generating extraction strategies.

    See docs/extraction-integration.md for the integration contract:
    response shape, adapter recipe (where applicable), and error modes.
    """
    if graphiti_service is None:
        return {'error': 'Service not initialized. Please wait for startup to complete.'}

    try:
        # Return cache if clean
        if graphiti_service._schema_cache is not None and not graphiti_service._schema_dirty:
            return graphiti_service._schema_cache

        client = await graphiti_service.get_client()
        driver = client.driver
        group_id = graphiti_service.config.graphiti.group_id
        internal_labels = {'Entity', 'Episodic', 'Community'}
        # The FLAVOUR owns every dialect-sensitive query text below (censuses,
        # attribute_keys); the parsing is shared because the variants project the
        # same aliases. On AGE `labels(n)` returns only the ontology leaf, so a
        # hardcoded base census hid every abstract supertype from the schema.
        flavour = graphiti_service.flavour
        census = flavour.census_queries()

        # 1. Label counts (single-pass)
        label_records, _, _ = await driver.execute_query(census['label_counts'])
        label_counts: dict[str, int] = {}
        for rec in label_records:
            for label in rec.get('lbls') or []:
                if label not in internal_labels:
                    label_counts[label] = label_counts.get(label, 0) + rec.get('cnt', 0)

        # 1b. Storage labels — the ones a vertex is actually STORED under. The
        #     census above counts the full hierarchy on AGE, so the difference is
        #     the hierarchy-only labels: searchable, but `MATCH (n:Actor)` reaches
        #     nothing and they can appear in no pattern. Without saying so, a
        #     schema view draws them as disconnected nodes (operator finding).
        #
        #     DEGRADE DIRECTION MATTERS: a census that cannot answer yields an
        #     EMPTY set, and diffing against empty would flag EVERY label as
        #     hierarchy-only — blanking the whole view. So a failure (or an empty
        #     answer) disables the flag entirely rather than inverting the schema.
        storage_labels: set[str] | None = None
        try:
            storage_records, _, _ = await driver.execute_query(census['storage_labels'])
            found = {
                rec.get('storage_label')
                for rec in storage_records
                if rec.get('storage_label')
            }
            storage_labels = found or None
        except Exception as census_error:  # noqa: BLE001 — never invert the schema
            logger.warning(
                f'get_schema: storage-label census failed, leaving every label '
                f'unflagged: {census_error}'
            )
        if storage_labels is None:
            logger.debug('get_schema: no storage-label census; hierarchy flag disabled')

        # 2. Properties + attribute_keys per label (sample 50)
        #    `properties` = full top-level keys — feeds cypher_quality's schema_match and the
        #    documented aletheia-extraction contract (kept unchanged). `attribute_keys` = the
        #    canonical ADR-019 R5 domain-queryable keys, flavour-specific (FalkorDB: top-level
        #    minus reserved bookkeeping; AGE: keys of the nested `attributes` agtype map).
        node_labels: dict[str, dict] = {}
        for label in label_counts:
            # A censused label need not be MATCHable. On AGE the hierarchy
            # (abstract) labels have no label table at all, so a label-scoped probe
            # against one answers with no rows — or raises. Neither may take the
            # whole schema down: an unhandled raise here returns {'error': ...} for
            # the entire call, i.e. a total get_schema outage on AGE. Same rule as
            # AgeFlavour.attribute_keys ("a non-map label must not break schema") —
            # a probe that cannot answer degrades ONE entry, and says so via
            # `sampled`. The census count is independent and always survives.
            try:
                # RETURN DISTINCT key AS key: AGE names an unaliased projection `col0`
                # (openCypher variable projections lose their name), so `r['key']`
                # would KeyError on AGE. The explicit alias makes the column `key` on
                # both flavours (FalkorDB already returns `key`). Regression: AGE
                # get_schema live test.
                prop_records, _, _ = await driver.execute_query(
                    f'MATCH (n:`{label}`) WITH keys(n) AS k LIMIT 50 UNWIND k AS key RETURN DISTINCT key AS key'
                )
                attribute_keys = await flavour.attribute_keys(driver, label)
            except Exception as probe_error:  # noqa: BLE001 — one label must not break schema
                logger.warning(
                    f'get_schema: label-scoped probe failed for `{label}`, '
                    f'reporting it unsampled: {probe_error}'
                )
                prop_records, attribute_keys = [], []
            props = [r['key'] for r in prop_records if r.get('key') not in ('name_embedding',)]
            node_labels[label] = {
                'count': label_counts[label],
                'attribute_keys': attribute_keys,
                'properties': sorted(props),
                # Honest: an entry whose probe raised or returned nothing was never
                # sampled, and a consumer must not read its empty `properties` as
                # "this type has no properties".
                'sampled': bool(prop_records),
            }
            # Set ONLY when true, so "storage type" reads as absent/None — the
            # shape every existing consumer already sees on FalkorDB.
            if storage_labels is not None and label not in storage_labels:
                node_labels[label]['hierarchy'] = True

        # 3. Relationship counts (single-pass)
        rel_records, _, _ = await driver.execute_query(
            'MATCH ()-[r]->() RETURN type(r) AS rel_type, count(r) AS cnt'
        )
        rel_counts: dict[str, int] = {}
        for rec in rel_records:
            rel_type = rec.get('rel_type', '')
            if rel_type:
                rel_counts[rel_type] = rec.get('cnt', 0)

        # 4. Relationship patterns (source->target)
        relationship_types: dict[str, dict] = {}
        for rel_type in rel_counts:
            pattern_records, _, _ = await driver.execute_query(
                census['rel_patterns'].format(rel_type=rel_type)
            )
            patterns = []
            for rec in pattern_records:
                # Endpoint choice only — this loop never deduped, and it still
                # does not (the census is already DISTINCT).
                src = _pick_endpoint(rec, 'source', internal_labels)
                tgt = _pick_endpoint(rec, 'target', internal_labels)
                if src and tgt:
                    patterns.append([src, tgt])

            relationship_types[rel_type] = {
                'count': rel_counts[rel_type],
                'patterns': patterns,
            }

        schema = {
            'type': 'schema',
            'graph_name': group_id,
            'domain': group_id.replace('_', ' ').title(),
            'dialect': flavour.dialect_id,
            'dialect_reference': flavour.dialect_reference,
            'node_labels': node_labels,
            'relationship_types': relationship_types,
        }

        # Enrich from domain profile if available
        if graphiti_service is not None and graphiti_service.domain_profile is not None:
            dp = graphiti_service.domain_profile
            for label, info in node_labels.items():
                if label in dp.entity_types:
                    et = dp.entity_types[label]
                    if et.description:
                        info['description'] = et.description
                    if et.sample_names:
                        info['sample_names'] = et.sample_names
            for rel_type, info in relationship_types.items():
                if rel_type in dp.edge_types:
                    et = dp.edge_types[rel_type]
                    if et.description:
                        info['description'] = et.description

        # Extract IMPORTANT: notes from entity/relationship descriptions
        # into a top-level field so they're prominent, not buried in type details.
        # Seeded with the flavour's census caveats (ADR-019 R6): those govern how to
        # read EVERY entry above, so they lead.
        analysis_notes: list[str] = list(flavour.census_notes())
        for label, info in node_labels.items():
            desc = info.get('description', '')
            for line in desc.split('\n'):
                stripped = line.strip()
                if stripped.startswith('IMPORTANT:'):
                    analysis_notes.append(f"[{label}] {stripped}")
        for rel_type, info in relationship_types.items():
            desc = info.get('description', '')
            for line in desc.split('\n'):
                stripped = line.strip()
                if stripped.startswith('IMPORTANT:'):
                    analysis_notes.append(f"[{rel_type}] {stripped}")
        if analysis_notes:
            schema['analysis_notes'] = analysis_notes

        # Tool capability metadata for reasoning engine discovery
        schema['tool_capabilities'] = {
            'search': {
                'search_methods': [
                    {'index': 'name_embedding', 'type': 'cosine_similarity',
                     'matches': 'entity names'},
                    {'index': 'summary_embedding', 'type': 'cosine_similarity',
                     'matches': 'entity summaries — contextual descriptions including event details and roles'},
                    {'index': 'summary', 'type': 'bm25_fulltext',
                     'matches': 'exact keyword matches in entity names and summaries'},
                    {'index': 'fact_embedding', 'type': 'cosine_similarity',
                     'matches': 'relationship facts'},
                ],
                'strategies': INTENT_STRATEGIES,
                'rerankers': ['rrf', 'mmr', 'cross_encoder', 'node_distance', 'episode_mentions'],
                'covers': {
                    'entity_fields': ['name', 'summary'],
                    'edge_fields': ['fact'],
                    'communities': True,
                },
                'does_not_cover': {
                    'entity_fields': ['domain_attribute_properties'],
                },
                'best_for': 'semantic discovery — concept searches match entity summaries '
                            'and relationship facts, not just entity names',
            },
            'run_cypher': {
                'search_methods': [{'type': 'property_match'}],
                'covers': {
                    'entity_fields': ['all_properties'],
                    'relationships': True,
                    'aggregations': True,
                },
                'requires': ['schema_knowledge'],
                'best_for': 'property filtering, counts, aggregations, path queries',
            },
            'explore_node': {
                'search_methods': [{'type': 'graph_traversal'}],
                'covers': {
                    'neighborhood': True,
                    'connected_edges': True,
                    'communities': True,
                },
                'best_for': 'deep dive on a known entity',
            },
        }

        # Backward-compatible alias of `dialect_reference` (emitted canonically above) under the
        # fork's historical field name, for consumers not yet reading `dialect_reference`
        # (ADR-019 R5 rename; drop in a future release). Now sourced from the flavour — correct
        # per-backend (AGE gets AGE guidance), no longer FalkorDB-hardcoded.
        schema['cypher_reference'] = flavour.dialect_reference

        # Cache the result
        graphiti_service._schema_cache = schema
        graphiti_service._schema_dirty = False

        return schema

    except Exception as e:
        logger.error(f'Error in get_schema: {e}')
        return {'error': f'Failed to retrieve schema: {e}'}


def _dedup_rows_by_uuid(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop repeated uuids, first row wins.

    Two independent sources of duplicates: AGE can hold duplicate-uuid sibling
    vertices (BUG-38 reintroduces them under concurrent ingestion), and an edge
    query can return the same edge row twice when both endpoints match more than
    once. A graph view would draw each twice. Rows with no uuid are not
    addressable, so they cannot be deduplicated by key and pass through.
    """
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        uuid = row.get('uuid')
        if uuid is None:
            out.append(row)
            continue
        if uuid in seen:
            continue
        seen.add(uuid)
        out.append(row)
    return out


def _iso_or_raw(value: Any) -> Any:
    """Datetime-like values to ISO strings; everything else through untouched.

    FalkorDB and AGE deliver `created_at` as a string, but the generic/Neo4j path
    returns a neo4j.time.DateTime, which FastMCP's output validation rejects. That
    failure happens in convert_result — OUTSIDE this tool's try/except — so it escapes
    the ADR-015 error envelope as a protocol error instead of an `error` payload.
    Every sibling tool normalizes the same way (format_node_result, explore_node,
    get_episode_context). None has no isoformat and passes straight through.
    """
    return value.isoformat() if hasattr(value, 'isoformat') else value


# Labels that describe Graphiti's own bookkeeping, never a domain type.
_SUBGRAPH_INTERNAL_LABELS = frozenset({'Entity', 'Episodic', 'Community'})


def _subgraph_node_row(r: dict[str, Any]) -> dict[str, Any]:
    """One SubgraphNode from one driver row, including the announced `leaf`.

    `leaf` is the node's single most specific label — what a consumer should type
    and colour by. It exists because `labels` is UNORDERED: a UI taking its first
    element typed AGE nodes by whatever the hierarchy happened to start with,
    collapsing 16 leaf types into 3 supertypes and colouring the two backends
    differently for the same graph.

    Two sources, in order:

    * the row's own `leaf` column when the flavour ANNOUNCES one (AGE projects
      `label(n)`, which is exactly the one stored leaf) — taken verbatim, never
      re-derived, because re-deriving from an unordered list is the bug;
    * otherwise derived from `labels`. Sound on base/FalkorDB only because their
      writer stores exactly [Entity, <leaf>], so the single non-internal label IS
      the leaf regardless of order — which is why that arm needs no query change.

    Null when neither source yields a non-internal label; consumers fall back to
    set-filtering `labels`.
    """
    labels = r.get('labels') or []
    announced = r.get('leaf')
    if announced and announced not in _SUBGRAPH_INTERNAL_LABELS:
        leaf = announced
    else:
        domain = [x for x in labels if x and x not in _SUBGRAPH_INTERNAL_LABELS]
        leaf = domain[0] if domain else None
    return {
        'uuid': r.get('uuid'), 'name': r.get('name'),
        'labels': labels,
        'leaf': leaf,
        'created_at': _iso_or_raw(r.get('created_at')), 'summary': r.get('summary'),
        'group_id': r.get('group_id'),
    }


@mcp.tool(annotations=annotations_for('sample_subgraph'))
async def sample_subgraph(limit: int = 100) -> SubgraphResponse:
    """Flavour-normalized node/edge sample of the knowledge graph.

    Returns up to `limit` entity nodes (uuid, name, labels — the FULL logical
    hierarchy on every backend — leaf, created_at, summary, group_id) and the
    edges among them (edge limit = 2.5x node limit). Intended for graph-view
    consumers; replaces client-composed Cypher, which cannot be
    dialect-correct across backends.

    `leaf` is the node's most specific label: consumers should type and colour by
    `leaf`, falling back to set-filtering `labels` only when it is null.

    `labels` is UNORDERED on every backend: set-filter the internal labels
    (Entity, Episodic, ...) to find the domain type, never index positionally.

    Args:
        limit: Maximum entity nodes to sample (clamped to 1..1000, default 100).
    """
    if graphiti_service is None:
        return {'error': 'Service not initialized. Please wait for startup to complete.'}
    try:
        client = await graphiti_service.get_client()
        driver = client.driver
        flavour = graphiti_service.flavour
        limit = max(1, min(int(limit), 1000))

        node_q = flavour.subgraph_node_query().replace('$limit', str(limit))
        node_records, _ = await flavour.execute_graph_query(driver, node_q)
        nodes = [_subgraph_node_row(r) for r in _dedup_rows_by_uuid(list(node_records))]

        edges = []
        uuids = [n['uuid'] for n in nodes if n['uuid']]
        if uuids:
            edge_q = flavour.subgraph_edge_query(uuids).replace('$limit', str(round(limit * 2.5)))
            edge_records, _ = await flavour.execute_graph_query(driver, edge_q)
            edges = [
                {
                    'uuid': r.get('uuid'), 'name': r.get('name'), 'fact': r.get('fact'),
                    'source_node_uuid': r.get('source_node_uuid'),
                    'target_node_uuid': r.get('target_node_uuid'),
                    'created_at': _iso_or_raw(r.get('created_at')),
                }
                for r in _dedup_rows_by_uuid(list(edge_records))
            ]

        return {
            'type': 'subgraph',
            'graph_name': graphiti_service.config.graphiti.group_id,
            'nodes': nodes,
            'edges': edges,
        }
    except Exception as e:
        logger.error(f'Error in sample_subgraph: {e}')
        return {'error': str(e)}


async def get_ontology_structure() -> OntologyStructureResponse:
    """Return the full ontology class hierarchy in one call.

    Returns entity classes (with inheritance, alt_labels, descriptions)
    and relationship classes (with source/target constraints).

    Used by downstream services (e.g., aletheia-extraction) to map raw data
    fields to ontology-typed entities and relationships before ingestion.

    Ontology access tiers: this tool is the SURFACE map — one compact entry
    per class, cheap to read whole. For full documentation prose and
    per-class property definitions use get_ontology_documentation (large);
    to study ONE class in full context use explore_ontology; to find classes
    by meaning use search_ontology (semantic recall).

    See docs/extraction-integration.md for the integration contract:
    response shape, adapter recipe (where applicable), and error modes.
    """
    if graphiti_service is None:
        return {'error': 'Service not initialized. Please wait for startup to complete.'}

    if not await graphiti_service._ensure_ontology_client():
        return {'error': 'No ontology graph configured for this connector.'}
    assert graphiti_service.ontology_client is not None  # type narrowing — _ensure_ontology_client guarantees non-None on True

    try:
        ontology_client = graphiti_service.ontology_client
        driver = ontology_client.driver
        ontology_graph_name = graphiti_service.config.graphiti.ontology_graph or ''

        # Query 1: all ontology classes. Flavour-owned text, frozen SHAPE: the
        # map tier stays lightweight (no uuid, no properties/identity) on every
        # flavour — see Flavour.ontology_queries().
        class_records, _, _ = await driver.execute_query(
            graphiti_service.flavour.ontology_queries()['structure']
        )

        entity_classes = []
        relationship_classes = []
        for rec in class_records:
            ontology_type = rec.get('ontology_type', '')
            entry = {
                'name': rec.get('name', ''),
                'ontology_type': ontology_type,
                'summary': rec.get('summary', ''),
                'alt_labels': rec.get('alt_labels', ''),
                'inherits_from': rec.get('inherits_from', ''),
                'examples': rec.get('examples', ''),
            }
            if ontology_type == 'relationship_class':
                entry['source_entity'] = rec.get('source_entity', '')
                entry['target_entity'] = rec.get('target_entity', '')
                relationship_classes.append(entry)
            else:
                # class, abstract_class, or any other entity-level type
                entity_classes.append(entry)

        return {
            'ontology_graph': ontology_graph_name,
            'entity_classes': entity_classes,
            'relationship_classes': relationship_classes,
        }

    except Exception as e:
        logger.error(f'Error in get_ontology_structure: {e}')
        return {'error': f'Failed to retrieve ontology structure: {e}'}


async def get_ontology_documentation() -> OntologyDocumentationResponse:
    """Complete ontology reference: every class with full documentation prose
    and per-class property definitions.

    LARGE — intended for UIs, exports, and batch consumers. For agent use
    prefer get_ontology_structure (the map), search_ontology (semantic
    recall), or explore_ontology (one class in full context).

    Returns {ontology_graph, entity_classes, relationship_classes}. Each
    entry carries the structure-tool fields plus the FULL `summary`, parsed
    `properties` (list of {name, label, range, comment, required}), and the
    `identity` flag. Relationship entries also carry source/target
    constraints.

    Relationship entries come from reified relationship_class nodes AND from
    RELATES_TO edges (ontologies that model relationships as object
    properties store them only as edges). Caveat: edge-derived summaries
    come from the stored edge fact and may be shorter than the full ontology
    relationship comment.
    """
    if graphiti_service is None:
        return {'error': 'Service not initialized. Please wait for startup to complete.'}

    if not await graphiti_service._ensure_ontology_client():
        return {'error': 'No ontology graph configured for this connector.'}
    assert graphiti_service.ontology_client is not None  # type narrowing — _ensure_ontology_client guarantees non-None on True

    try:
        driver = graphiti_service.ontology_client.driver
        ontology_graph_name = graphiti_service.config.graphiti.ontology_graph or ''
        ontology_queries = graphiti_service.flavour.ontology_queries()

        class_records, _, _ = await driver.execute_query(
            ontology_queries['class_context']
        )

        entity_classes = []
        relationship_classes = []
        for rec in class_records:
            entry = _ontology_full_entry(rec)
            if entry['ontology_type'] == 'relationship_class':
                relationship_classes.append(entry)
            else:
                # class, abstract_class, or any other entity-level type
                entity_classes.append(entry)

        # Object-property ontologies store relationships as edges between
        # OntologyClass nodes instead of reified relationship_class nodes.
        # Union both sources; node-derived entries win on duplicates.
        edge_records, _, _ = await driver.execute_query(ontology_queries['relates'])
        relationship_classes = _combine_relationship_entries(
            relationship_classes, edge_records
        )

        return {
            'ontology_graph': ontology_graph_name,
            'entity_classes': entity_classes,
            'relationship_classes': relationship_classes,
        }

    except Exception as e:
        logger.error(f'Error in get_ontology_documentation: {e}')
        return {'error': f'Failed to retrieve ontology documentation: {e}'}


async def run_cypher(query: str) -> CypherResultResponse:
    """Execute a read-only Cypher query against the knowledge graph.

    The query is validated and sanitized before execution.
    Write operations are blocked. LIMIT 200 is auto-injected if missing;
    explicit LIMIT values are respected.  Returns typed JSON (scalar,
    tabular, graph, path) with metadata.

    Cypher dialect is backend-specific: the registered tool description and
    get_schema's `dialect_reference` carry this graph's exact dialect notes
    (FalkorDB openCypher vs Apache AGE openCypher differ) — follow those.
    """
    if graphiti_service is None:
        return format_error(query, CypherError(
            stage='initialization',
            reason='service_not_ready',
            found='',
            explanation='Service not initialized. Please wait for startup to complete.',
            suggestion='Try again in a few seconds.',
        ))

    # Validate and sanitize (flavour drives the dialect reject + auto-fix)
    flavour = graphiti_service.flavour
    result = validate_and_sanitize(query, flavour)
    if isinstance(result, CypherError):
        return format_error(query, result)

    sanitized = result
    limit = sanitized.effective_limit

    try:
        client = await graphiti_service.get_client()
        driver = client.driver

        # Read-only execution is a flavour concern: FalkorDB uses DB-enforced ro_query;
        # AGE/base rely on the pipeline whitelist + execute_query. Returns (records, header).
        start_time = time.time()
        records, header = await flavour.execute_graph_query(driver, sanitized.query)
        execution_ms = round((time.time() - start_time) * 1000, 1)

        _cache = getattr(graphiti_service, '_schema_cache', None) if graphiti_service else None
        schema = _cache if isinstance(_cache, dict) else None
        return format_result(records, header, sanitized.query, sanitized.auto_fixes, execution_ms, limit, schema=schema)

    except Exception as e:
        logger.error(f'Cypher execution error: {e}')
        error = flavour.classify_execution_error(str(e), query=sanitized.query)
        result = format_error(sanitized.query, error)
        result['auto_fixes'] = sanitized.auto_fixes
        return result


async def profile_graph(sample_size: int = 5) -> ProfileGraphResponse:
    """Profile entity properties and relationship patterns in this knowledge graph.

    Returns property coverage, sample values, detected languages, relationship
    cardinality, and sample traversal paths. Use this to understand data quality,
    multilingual content, and graph structure before querying.

    Results are cached until new data is ingested via add_memory.

    Args:
        sample_size: Number of sample values per property (default 5).
    """
    if graphiti_service is None:
        return {'error': 'Service not initialized. Please wait for startup to complete.'}

    try:
        client = await graphiti_service.get_client()
        # The flavour owns the census texts (same seam as get_schema): without it
        # the profiler runs FalkorDB-shaped `labels(n)` censuses on every backend.
        return await _run_profile_graph(
            client.driver, sample_size=sample_size, flavour=graphiti_service.flavour
        )
    except Exception as e:
        logger.error(f'Error in profile_graph: {e}')
        return {'error': f'Failed to profile graph: {e}'}


# The nine tools registered at startup rather than by decorator, because their
# descriptions are rendered from the live DomainProfile. ONE list: the dynamic
# path and the degraded fallback both walk it, so a tool can never be served by
# one and forgotten by the other (the bug behind A-D2).
_DYNAMIC_TOOLS = (
    search,
    explore_node,
    search_ontology,
    explore_ontology,
    get_schema,
    get_ontology_structure,
    get_ontology_documentation,
    run_cypher,
    profile_graph,
)


def register_dynamic_tools(profile: DomainProfile) -> None:
    """Register the main tools with dynamic descriptions from the DomainProfile."""
    # Remove any existing registrations (e.g., if called multiple times)
    for fn in _DYNAMIC_TOOLS:
        if fn.__name__ in mcp._tool_manager._tools:
            del mcp._tool_manager._tools[fn.__name__]

    # Backend flavour drives the per-backend Cypher dialect surfaced in the run_cypher
    # description + server instructions (ADR-019 R1/R6).
    flavour = graphiti_service.flavour if graphiti_service is not None else None

    # Annotations (ADR-019 R3) come from the one table in tool_annotations.py, the same
    # source the static @mcp.tool() decorators read — the two paths cannot drift.
    mcp.add_tool(
        search,
        description=build_search_description(profile),
        annotations=annotations_for('search'),
    )
    mcp.add_tool(
        explore_node,
        description=build_explore_node_description(profile),
        annotations=annotations_for('explore_node'),
    )
    mcp.add_tool(
        search_ontology,
        description=build_search_ontology_description(profile),
        annotations=annotations_for('search_ontology'),
    )
    mcp.add_tool(
        explore_ontology,
        description=build_explore_ontology_description(profile),
        annotations=annotations_for('explore_ontology'),
    )
    mcp.add_tool(
        get_schema,
        description=build_get_schema_description(profile),
        annotations=annotations_for('get_schema'),
    )
    mcp.add_tool(get_ontology_structure, annotations=annotations_for('get_ontology_structure'))
    mcp.add_tool(
        get_ontology_documentation,
        annotations=annotations_for('get_ontology_documentation'),
    )
    mcp.add_tool(
        run_cypher,
        description=build_run_cypher_description(profile, flavour),
        annotations=annotations_for('run_cypher'),
    )
    mcp.add_tool(profile_graph, annotations=annotations_for('profile_graph'))

    # Update MCP instructions
    mcp._mcp_server.instructions = build_instructions(profile, flavour)

    logger.info('Registered tools with dynamic descriptions')


# Leads the served `instructions` whenever graph introspection failed at startup
# (BUG-50 / A-D2). A consumer that captures the announcement once — which is the
# common shape — must be able to SEE that what it captured is a fallback.
DEGRADED_INSTRUCTIONS_MARKER = (
    '!! DEGRADED: domain profile unavailable !!'
)


def register_fallback_tools(reason: str) -> None:
    """Serve the FULL surface with static descriptions when introspection fails.

    The old fallback re-registered four tools and left the other five profile-driven
    ones unregistered, so `get_schema` (the canonical ADR-019 R5 payload consumers
    discover this connector through), `run_cypher`, `profile_graph` and both
    ontology-bulk tools disappeared from `tools/list` while `/health` stayed green.
    Losing the profile costs the DESCRIPTIONS, never the TOOLS: every tool still
    works, it just describes itself from its docstring instead of from live data.

    The announcement is rebuilt to say so, in the lead position, and keeps the
    backend dialect (the flavour is known even when the graph cannot be read).
    """
    flavour = graphiti_service.flavour if graphiti_service is not None else None
    # `config` is declared but not assigned at import time — read it defensively so a
    # very early failure degrades honestly instead of raising NameError on the way out.
    cfg = globals().get('config')
    group_id = cfg.graphiti.group_id if cfg is not None else 'unknown'

    for fn in _DYNAMIC_TOOLS:
        name = fn.__name__
        if name in mcp._tool_manager._tools:
            del mcp._tool_manager._tools[name]
        mcp.add_tool(fn, annotations=annotations_for(name))

    mcp._mcp_server.instructions = build_degraded_instructions(
        group_id=group_id,
        flavour=flavour,
        reason=reason,
        marker=DEGRADED_INSTRUCTIONS_MARKER,
    )

    logger.error(
        'Serving a DEGRADED surface for %s: all %d tools registered with static '
        'descriptions, announcement marked degraded. Cause: %s',
        group_id,
        len(mcp._tool_manager._tools),
        reason,
    )


async def _build_and_register_domain_surface() -> None:
    """Introspect the graph and register the profile-driven tools and resources.

    Split out of `initialize_server` so the failure path is reachable from a test:
    it is the branch that used to collapse the served surface in silence.
    """
    try:
        profile_client = await graphiti_service.get_client()
        ontology_client = graphiti_service.ontology_client
        domain_profile = await build_domain_profile(
            profile_client,
            group_id=config.graphiti.group_id,
            ontology_client=ontology_client,
            flavour=graphiti_service.flavour,
        )
        graphiti_service.domain_profile = domain_profile
        register_dynamic_tools(domain_profile)
        register_resources(domain_profile)
    except Exception as e:
        # ERROR, not warning: the connector is now answering with guidance it did
        # not derive from this graph, and nothing else in the stack will say so.
        logger.error(f'Failed to build domain profile, degrading to static descriptions: {e}')
        register_fallback_tools(reason=str(e))


def register_resources(profile: DomainProfile) -> None:
    """Register MCP resources with rendered content from the DomainProfile."""
    domain_summary = TextResource(
        uri='graphiti://domain_summary',
        name='Domain Summary',
        description='Overview of entity types, relationship types, and data in this knowledge graph',
        text=profile.render_domain_summary(),
    )
    entity_catalog = TextResource(
        uri='graphiti://entity_catalog',
        name='Entity Catalog',
        description='Detailed listing of all entity types with descriptions and sample entities',
        text=profile.render_entity_catalog(),
    )
    relationship_types = TextResource(
        uri='graphiti://relationship_types',
        name='Relationship Types',
        description='All relationship types with descriptions and source/target patterns',
        text=profile.render_relationship_types(),
    )

    mcp.add_resource(domain_summary)
    mcp.add_resource(entity_catalog)
    mcp.add_resource(relationship_types)

    logger.info(f'Registered 3 MCP resources for {profile.group_id}')


async def initialize_server() -> ServerConfig:
    """Parse CLI arguments and initialize the Graphiti server configuration."""
    global config, graphiti_service, queue_service, graphiti_client, semaphore

    parser = argparse.ArgumentParser(
        description='Run the Graphiti MCP server with YAML configuration support'
    )

    # Configuration file argument
    # Default to config/config.yaml relative to the mcp_server directory
    default_config = Path(__file__).parent.parent / 'config' / 'config.yaml'
    parser.add_argument(
        '--config',
        type=Path,
        default=default_config,
        help='Path to YAML configuration file (default: config/config.yaml)',
    )

    # Transport arguments
    parser.add_argument(
        '--transport',
        choices=['sse', 'stdio', 'http'],
        help='Transport to use: http (recommended, default), stdio (standard I/O), or sse (deprecated)',
    )
    parser.add_argument(
        '--host',
        help='Host to bind the MCP server to',
    )
    parser.add_argument(
        '--port',
        type=int,
        help='Port to bind the MCP server to',
    )

    # Provider selection arguments
    parser.add_argument(
        '--llm-provider',
        choices=['openai', 'azure_openai', 'anthropic', 'gemini', 'groq'],
        help='LLM provider to use',
    )
    parser.add_argument(
        '--embedder-provider',
        choices=['openai', 'azure_openai', 'gemini', 'voyage'],
        help='Embedder provider to use',
    )
    parser.add_argument(
        '--database-provider',
        choices=['neo4j', 'falkordb', 'age'],
        help='Database provider to use',
    )

    # LLM configuration arguments
    parser.add_argument('--model', help='Model name to use with the LLM client')
    parser.add_argument('--small-model', help='Small model name to use with the LLM client')
    parser.add_argument(
        '--temperature', type=float, help='Temperature setting for the LLM (0.0-2.0)'
    )

    # Embedder configuration arguments
    parser.add_argument('--embedder-model', help='Model name to use with the embedder')

    # Graphiti-specific arguments
    parser.add_argument(
        '--group-id',
        help='Namespace for the graph. If not provided, uses config file or generates random UUID.',
    )
    parser.add_argument(
        '--user-id',
        help='User ID for tracking operations',
    )
    parser.add_argument(
        '--destroy-graph',
        action='store_true',
        help='Destroy all Graphiti graphs on startup',
    )

    args = parser.parse_args()

    # Set config path in environment for the settings to pick up
    if args.config:
        os.environ['CONFIG_PATH'] = str(args.config)

    # Load configuration with environment variables and YAML
    config = GraphitiConfig()

    # Apply CLI overrides
    config.apply_cli_overrides(args)

    # Also apply legacy CLI args for backward compatibility
    if hasattr(args, 'destroy_graph'):
        config.destroy_graph = args.destroy_graph

    # Log configuration details
    logger.info('Using configuration:')
    logger.info(f'  - LLM: {config.llm.provider} / {config.llm.model}')
    logger.info(f'  - Embedder: {config.embedder.provider} / {config.embedder.model}')
    logger.info(f'  - Database: {config.database.provider}')
    logger.info(f'  - Group ID: {config.graphiti.group_id}')
    if config.graphiti.ontology_graph:
        logger.info(f'  - Ontology graph: {config.graphiti.ontology_graph}')
    logger.info(f'  - Transport: {config.server.transport}')

    # Set dynamic MCP server name based on group_id
    mcp._mcp_server.name = f'Graphiti - {config.graphiti.group_id}'

    # Log graphiti-core version
    try:
        import graphiti_core

        graphiti_version = getattr(graphiti_core, '__version__', 'unknown')
        logger.info(f'  - Graphiti Core: {graphiti_version}')
    except Exception:
        # Check for Docker-stored version file
        version_file = Path('/app/.graphiti-core-version')
        if version_file.exists():
            graphiti_version = version_file.read_text().strip()
            logger.info(f'  - Graphiti Core: {graphiti_version}')
        else:
            logger.info('  - Graphiti Core: version unavailable')

    # Handle graph destruction if requested
    if hasattr(config, 'destroy_graph') and config.destroy_graph:
        logger.warning('Destroying all Graphiti graphs as requested...')
        temp_service = GraphitiService(config, SEMAPHORE_LIMIT)
        await temp_service.initialize()
        client = await temp_service.get_client()
        await clear_data(client.driver)
        logger.info('All graphs destroyed')

    # Initialize services
    graphiti_service = GraphitiService(config, SEMAPHORE_LIMIT)
    queue_service = QueueService()
    await graphiti_service.initialize()

    # Build domain profile from graph introspection. On failure the surface stays
    # complete and the announcement declares itself degraded (BUG-50 / A-D2).
    await _build_and_register_domain_surface()

    # Set global client for backward compatibility
    graphiti_client = await graphiti_service.get_client()
    semaphore = graphiti_service.semaphore

    # Initialize queue service with the client
    await queue_service.initialize(graphiti_client)

    # Set MCP server settings (only for HTTP/SSE — stdio doesn't bind a port)
    if config.server.transport != 'stdio':
        if config.server.host:
            mcp.settings.host = config.server.host
        if config.server.port:
            mcp.settings.port = config.server.port

    # Return MCP configuration for transport
    return config.server


async def run_mcp_server():
    """Run the MCP server in the current event loop."""
    # Initialize the server
    mcp_config = await initialize_server()

    # Run the server with configured transport
    logger.info(f'Starting MCP server with transport: {mcp_config.transport}')
    if mcp_config.transport == 'stdio':
        await mcp.run_stdio_async()
    elif mcp_config.transport == 'sse':
        logger.info(
            f'Running MCP server with SSE transport on {mcp.settings.host}:{mcp.settings.port}'
        )
        logger.info(f'Access the server at: http://{mcp.settings.host}:{mcp.settings.port}/sse')
        await mcp.run_sse_async()
    elif mcp_config.transport == 'http':
        # Use localhost for display if binding to 0.0.0.0
        display_host = 'localhost' if mcp.settings.host == '0.0.0.0' else mcp.settings.host
        logger.info(
            f'Running MCP server with streamable HTTP transport on {mcp.settings.host}:{mcp.settings.port}'
        )
        logger.info('=' * 60)
        logger.info('MCP Server Access Information:')
        logger.info(f'  Base URL: http://{display_host}:{mcp.settings.port}/')
        logger.info(f'  MCP Endpoint: http://{display_host}:{mcp.settings.port}/mcp/')
        logger.info('  Transport: HTTP (streamable)')

        # Show FalkorDB Browser UI access if enabled
        if os.environ.get('BROWSER', '1') == '1':
            logger.info(f'  FalkorDB Browser UI: http://{display_host}:3000/')

        logger.info('=' * 60)
        logger.info('For MCP clients, connect to the /mcp/ endpoint above')

        # Configure uvicorn logging to match our format
        configure_uvicorn_logging()

        await mcp.run_streamable_http_async()
    else:
        raise ValueError(
            f'Unsupported transport: {mcp_config.transport}. Use "sse", "stdio", or "http"'
        )


def main():
    """Main function to run the Graphiti MCP server."""
    try:
        # Run everything in a single event loop
        asyncio.run(run_mcp_server())
    except KeyboardInterrupt:
        logger.info('Server shutting down...')
    except Exception as e:
        logger.error(f'Error initializing Graphiti MCP server: {str(e)}')
        raise


if __name__ == '__main__':
    main()
