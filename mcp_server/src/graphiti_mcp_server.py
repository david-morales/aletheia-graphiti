#!/usr/bin/env python3
"""
Graphiti MCP Server - Exposes Graphiti functionality through the Model Context Protocol (MCP)
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from graphiti_core import Graphiti
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodeType, EpisodicNode
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
from pydantic import BaseModel
from starlette.responses import JSONResponse

from config.schema import GraphitiConfig, ServerConfig
from models.response_types import (
    CommunityBuildResponse,
    EpisodeContextResponse,
    EpisodeSearchResponse,
    ErrorResponse,
    ExploreResponse,
    SearchResponse,
    StatusResponse,
    SuccessResponse,
)
from services.factories import DatabaseDriverFactory, EmbedderFactory, LLMClientFactory
from services.queue_service import QueueService
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

# MCP server instance
mcp = FastMCP(
    'Graphiti Agent Memory',
    instructions=GRAPHITI_MCP_INSTRUCTIONS,
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
        self.semaphore_limit = semaphore_limit
        self.semaphore = asyncio.Semaphore(semaphore_limit)
        self.client: Graphiti | None = None
        self.ontology_client: Graphiti | None = None
        self.entity_types = None

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

            # Build indices
            await self.client.build_indices_and_constraints()

            # Initialize ontology client if configured
            if self.config.graphiti.ontology_graph:
                try:
                    ontology_graph_name = self.config.graphiti.ontology_graph
                    if self.config.database.provider.lower() == 'falkordb':
                        from graphiti_core.driver.falkordb_driver import FalkorDriver

                        ontology_driver = FalkorDriver(
                            host=db_config['host'],
                            port=db_config['port'],
                            username=db_config.get('username'),
                            password=db_config['password'],
                            database=ontology_graph_name,
                        )

                        self.ontology_client = Graphiti(
                            graph_driver=ontology_driver,
                            llm_client=None,
                            embedder=embedder_client,
                        )
                    else:
                        logger.warning(
                            f'Ontology graph not supported for {self.config.database.provider} provider'
                        )

                    if self.ontology_client:
                        await self.ontology_client.build_indices_and_constraints()
                        logger.info(f'Ontology graph connected: {ontology_graph_name}')
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


@mcp.tool()
async def add_memory(
    name: str | None = None,
    episode_body: str | None = None,
    group_id: str | None = None,
    source: Literal['text', 'json', 'message'] = 'text',
    source_description: str = '',
    uuid: str | None = None,
    episodes: list[dict] | None = None,
) -> SuccessResponse | ErrorResponse:
    """Add information to the knowledge graph.

    For single episodes, provide name + episode_body (queued for async processing).
    For bulk ingestion, provide episodes list (processed synchronously, returns when done).

    Args:
        name: Name of the episode (single mode).
        episode_body: Content to persist (single mode). When source='json', must be a JSON string.
        group_id: Graph partition ID. Uses default if omitted.
        source: Source type — 'text' (default), 'json', or 'message'.
        source_description: Description of the source.
        uuid: Optional UUID for the episode (single mode).
        episodes: List of episodes for bulk ingestion (bulk mode).
                  Each dict: {"name": str, "content": str, "source": str, "source_description": str}

    Examples:
        # Single episode
        add_memory(name="News", episode_body="Acme Corp announced a new product.", source="text")

        # Bulk ingestion
        add_memory(episodes=[
            {"name": "Doc 1", "content": "...", "source": "text", "source_description": "report"},
            {"name": "Doc 2", "content": "...", "source": "json", "source_description": "data"},
        ], group_id="my_graph")
    """
    global graphiti_service, queue_service

    if graphiti_service is None or queue_service is None:
        return ErrorResponse(error='Services not initialized')

    effective_group_id = group_id or config.graphiti.group_id

    try:
        # Bulk mode
        if episodes is not None:
            if not episodes:
                return ErrorResponse(error='Episodes list is empty')

            client = await graphiti_service.get_client()

            raw_episodes = []
            for i, ep in enumerate(episodes):
                if 'name' not in ep or 'content' not in ep:
                    missing = [k for k in ('name', 'content') if k not in ep]
                    return ErrorResponse(
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

            return SuccessResponse(
                message=f"Bulk ingested {len(raw_episodes)} episodes into '{effective_group_id}': "
                        f"{len(results.nodes)} nodes, {len(results.edges)} edges created"
            )

        # Single mode (existing behavior)
        if not name or not episode_body:
            return ErrorResponse(error='Provide name + episode_body for single mode, or episodes for bulk mode')

        episode_type = EpisodeType.text
        if source:
            try:
                episode_type = EpisodeType[source.lower()]
            except (KeyError, AttributeError):
                logger.warning(f"Unknown source type '{source}', using 'text'")
                episode_type = EpisodeType.text

        await queue_service.add_episode(
            group_id=effective_group_id,
            name=name,
            content=episode_body,
            source_description=source_description,
            episode_type=episode_type,
            entity_types=graphiti_service.entity_types,
            uuid=uuid or None,
        )

        return SuccessResponse(
            message=f"Episode '{name}' queued for processing in group '{effective_group_id}'"
        )

    except Exception as e:
        logger.error(f'Error in add_memory: {e}')
        return ErrorResponse(error=f'Error adding memory: {e}')


@mcp.tool()
async def search(
    query: str,
    group_ids: list[str] | None = None,
    search_mode: Literal['nodes', 'edges', 'communities', 'combined'] = 'combined',
    reranker: Literal['rrf', 'mmr', 'cross_encoder', 'node_distance', 'episode_mentions'] = 'rrf',
    center_node_uuid: str | None = None,
    bfs_origin_node_uuids: list[str] | None = None,
    entity_types: list[str] | None = None,
    edge_types: list[str] | None = None,
    valid_at: str | None = None,
    limit: int = 10,
) -> SearchResponse | ErrorResponse:
    """Search the knowledge graph with full control over search strategy.

    Supports multiple search modes (nodes, edges, communities, or combined), reranking
    strategies, graph traversal from known starting nodes, and temporal/type filtering.

    Args:
        query: Natural language search query.
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
        return ErrorResponse(error='Graphiti service not initialized')

    try:
        client = await graphiti_service.get_client()

        effective_group_ids = (
            group_ids
            if group_ids is not None
            else [config.graphiti.group_id]
            if config.graphiti.group_id
            else []
        )

        search_config = resolve_search_config(search_mode, reranker, limit)

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

        return SearchResponse(
            message=f'Found {len(node_results)} nodes, {len(edge_results)} edges, {len(community_results)} communities',
            nodes=node_results,
            edges=edge_results,
            communities=community_results,
        )

    except ValueError as e:
        return ErrorResponse(error=str(e))
    except Exception as e:
        logger.error(f'Error in search: {e}')
        return ErrorResponse(error=f'Search error: {e}')


@mcp.tool()
async def explore_node(
    node_name: str | None = None,
    node_uuid: str | None = None,
    group_ids: list[str] | None = None,
    depth: Literal[1, 2, 3, 4] = 2,
    edge_types: list[str] | None = None,
    limit: int = 20,
) -> ExploreResponse | ErrorResponse:
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
        return ErrorResponse(error='Graphiti service not initialized')

    if not node_name and not node_uuid:
        return ErrorResponse(error='Provide either node_name or node_uuid')

    try:
        client = await graphiti_service.get_client()

        effective_group_ids = (
            group_ids
            if group_ids is not None
            else [config.graphiti.group_id]
            if config.graphiti.group_id
            else []
        )

        # Resolve node UUID from name if needed
        resolved_uuid = node_uuid
        center_node_result = None

        if node_name and not node_uuid:
            resolve_results = await client.search_(
                query=node_name,
                config=NODE_HYBRID_SEARCH_RRF,
                group_ids=effective_group_ids,
            )
            if not resolve_results.nodes:
                return ExploreResponse(
                    message=f'No node found matching "{node_name}"',
                    center_node=None,
                    nodes=[],
                    edges=[],
                    communities=[],
                )
            best_match = resolve_results.nodes[0]
            resolved_uuid = best_match.uuid
            center_node_result = {
                'uuid': best_match.uuid,
                'name': best_match.name,
                'labels': best_match.labels or [],
                'created_at': best_match.created_at.isoformat() if best_match.created_at else None,
                'summary': best_match.summary,
                'group_id': best_match.group_id,
                'attributes': {
                    k: v
                    for k, v in (best_match.attributes or {}).items()
                    if 'embedding' not in k.lower()
                },
            }

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

        # If we only have a UUID, try to find center node in results
        if node_uuid and not center_node_result:
            for n in results.nodes or []:
                if n.uuid == node_uuid:
                    center_node_result = {
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
                    break

        return ExploreResponse(
            message=f'Explored "{node_name or node_uuid}": {len(node_results)} nodes, {len(edge_results)} edges',
            center_node=center_node_result,
            nodes=node_results,
            edges=edge_results,
            communities=community_results,
        )

    except Exception as e:
        logger.error(f'Error in explore_node: {e}')
        return ErrorResponse(error=f'Explore error: {e}')


@mcp.tool()
async def get_episode_context(
    episode_uuids: list[str],
) -> EpisodeContextResponse | ErrorResponse:
    """Get all entities and relationships extracted from specific episodes.

    Returns the nodes and edges that were created when these episodes were ingested.
    Pair with get_episodes to first list episodes, then inspect what was extracted.

    Args:
        episode_uuids: List of episode UUIDs to inspect.
    """
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    if not episode_uuids:
        return ErrorResponse(error='Provide at least one episode UUID')

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

        return EpisodeContextResponse(
            message=f'Found {len(node_results)} nodes and {len(edge_results)} edges from {len(episode_uuids)} episodes',
            nodes=node_results,
            edges=edge_results,
        )

    except Exception as e:
        logger.error(f'Error in get_episode_context: {e}')
        return ErrorResponse(error=f'Episode context error: {e}')


@mcp.tool()
async def build_communities(
    group_ids: list[str],
) -> CommunityBuildResponse | ErrorResponse:
    """Build communities by clustering entities in the knowledge graph.

    Uses label propagation to detect communities of related entities, then
    generates summaries for each community. Communities must be built before
    they can be searched with search(search_mode="communities").

    Args:
        group_ids: Which graph partitions to cluster.
    """
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    if not group_ids:
        return ErrorResponse(error='Provide at least one group_id')

    try:
        client = await graphiti_service.get_client()

        community_nodes, community_edges = await client.build_communities(
            group_ids=group_ids,
        )

        community_results = [
            format_community_result(c, member_count=0)
            for c in community_nodes
        ]

        return CommunityBuildResponse(
            message=f'Built {len(community_nodes)} communities across {len(group_ids)} graphs',
            community_count=len(community_nodes),
            communities=community_results,
        )

    except Exception as e:
        logger.error(f'Error building communities: {e}')
        return ErrorResponse(error=f'Community build error: {e}')


@mcp.tool()
async def delete_entity_edge(uuid: str) -> SuccessResponse | ErrorResponse:
    """Delete an entity edge from the graph memory.

    Args:
        uuid: UUID of the entity edge to delete
    """
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    try:
        client = await graphiti_service.get_client()

        # Get the entity edge by UUID
        entity_edge = await EntityEdge.get_by_uuid(client.driver, uuid)
        # Delete the edge using its delete method
        await entity_edge.delete(client.driver)
        return SuccessResponse(message=f'Entity edge with UUID {uuid} deleted successfully')
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error deleting entity edge: {error_msg}')
        return ErrorResponse(error=f'Error deleting entity edge: {error_msg}')


@mcp.tool()
async def delete_episode(uuid: str) -> SuccessResponse | ErrorResponse:
    """Delete an episode from the graph memory.

    Args:
        uuid: UUID of the episode to delete
    """
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    try:
        client = await graphiti_service.get_client()

        # Get the episodic node by UUID
        episodic_node = await EpisodicNode.get_by_uuid(client.driver, uuid)
        # Delete the node using its delete method
        await episodic_node.delete(client.driver)
        return SuccessResponse(message=f'Episode with UUID {uuid} deleted successfully')
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error deleting episode: {error_msg}')
        return ErrorResponse(error=f'Error deleting episode: {error_msg}')


@mcp.tool()
async def get_episodes(
    group_ids: list[str] | None = None,
    max_episodes: int = 10,
) -> EpisodeSearchResponse | ErrorResponse:
    """Get episodes from the graph memory.

    Args:
        group_ids: Optional list of group IDs to filter results
        max_episodes: Maximum number of episodes to return (default: 10)
    """
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

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
            return EpisodeSearchResponse(message='No episodes found', episodes=[])

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

        return EpisodeSearchResponse(
            message='Episodes retrieved successfully', episodes=episode_results
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error getting episodes: {error_msg}')
        return ErrorResponse(error=f'Error getting episodes: {error_msg}')


@mcp.tool()
async def clear_graph(group_ids: list[str] | None = None) -> SuccessResponse | ErrorResponse:
    """Clear all data from the graph for specified group IDs.

    Args:
        group_ids: Optional list of group IDs to clear. If not provided, clears the default group.
    """
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    try:
        client = await graphiti_service.get_client()

        # Use the provided group_ids or fall back to the default from config if none provided
        effective_group_ids = (
            group_ids or [config.graphiti.group_id] if config.graphiti.group_id else []
        )

        if not effective_group_ids:
            return ErrorResponse(error='No group IDs specified for clearing')

        # Clear data for the specified group IDs
        await clear_data(client.driver, group_ids=effective_group_ids)

        return SuccessResponse(
            message=f'Graph data cleared successfully for group IDs: {", ".join(effective_group_ids)}'
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error clearing graph: {error_msg}')
        return ErrorResponse(error=f'Error clearing graph: {error_msg}')


@mcp.tool()
async def get_status() -> StatusResponse:
    """Get the status of the Graphiti MCP server and database connection."""
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


@mcp.tool()
async def search_ontology(
    query: str,
    search_mode: Literal['nodes', 'edges', 'communities', 'combined'] = 'combined',
    reranker: Literal['rrf', 'mmr', 'cross_encoder'] = 'rrf',
    limit: int = 10,
) -> SearchResponse | ErrorResponse:
    """Search the companion ontology graph for schema definitions, entity types, and relationships.

    Use this to understand what types of entities and relationships exist in the knowledge graph,
    what properties they have, and how they relate to each other.

    Args:
        query: Natural language search query (e.g., "AirworthinessDirective", "what properties does Aircraft have").
        search_mode: What to search — "nodes", "edges", "communities", or "combined" (default).
        reranker: Reranking strategy — "rrf" (default), "mmr", or "cross_encoder".
        limit: Maximum results to return (default 10).
    """
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    if graphiti_service.ontology_client is None:
        return ErrorResponse(error='No ontology graph configured for this server')

    try:
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

        return SearchResponse(
            message=f'Ontology: {len(node_results)} nodes, {len(edge_results)} edges, {len(community_results)} communities',
            nodes=node_results,
            edges=edge_results,
            communities=community_results,
        )

    except ValueError as e:
        return ErrorResponse(error=str(e))
    except Exception as e:
        logger.error(f'Error in search_ontology: {e}')
        return ErrorResponse(error=f'Ontology search error: {e}')


@mcp.tool()
async def explore_ontology(
    node_name: str | None = None,
    node_uuid: str | None = None,
    depth: Literal[1, 2, 3, 4] = 2,
    limit: int = 20,
) -> ExploreResponse | ErrorResponse:
    """Explore a specific class in the companion ontology graph.

    Shows properties, relationships, and parent classes for a given ontology type.
    Use this to understand the structure of a specific entity or relationship type.

    Args:
        node_name: Find the ontology class by name (e.g., "AirworthinessDirective"). Provide this or node_uuid.
        node_uuid: Expand directly from this node UUID. Provide this or node_name.
        depth: How many hops to traverse (1-4, default 2).
        limit: Maximum results to return (default 20).
    """
    global graphiti_service

    if graphiti_service is None:
        return ErrorResponse(error='Graphiti service not initialized')

    if graphiti_service.ontology_client is None:
        return ErrorResponse(error='No ontology graph configured for this server')

    if not node_name and not node_uuid:
        return ErrorResponse(error='Provide either node_name or node_uuid')

    try:
        ontology_client = graphiti_service.ontology_client
        ontology_group_id = config.graphiti.ontology_graph

        # Resolve node UUID from name if needed
        resolved_uuid = node_uuid
        center_node_result = None

        if node_name and not node_uuid:
            resolve_results = await ontology_client.search_(
                query=node_name,
                config=NODE_HYBRID_SEARCH_RRF,
                group_ids=[ontology_group_id],
            )
            if not resolve_results.nodes:
                return ExploreResponse(
                    message=f'No ontology class found matching "{node_name}"',
                    center_node=None,
                    nodes=[],
                    edges=[],
                    communities=[],
                )
            best_match = resolve_results.nodes[0]
            resolved_uuid = best_match.uuid
            center_node_result = {
                'uuid': best_match.uuid,
                'name': best_match.name,
                'labels': best_match.labels or [],
                'created_at': best_match.created_at.isoformat() if best_match.created_at else None,
                'summary': best_match.summary,
                'group_id': best_match.group_id,
                'attributes': {
                    k: v
                    for k, v in (best_match.attributes or {}).items()
                    if 'embedding' not in k.lower()
                },
            }

        # Build explore config
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

        results = await ontology_client.search_(
            query=node_name or '',
            config=explore_config,
            group_ids=[ontology_group_id],
            center_node_uuid=resolved_uuid,
            bfs_origin_node_uuids=[resolved_uuid] if resolved_uuid else None,
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

        # If we only have a UUID, try to find center node in results
        if node_uuid and not center_node_result:
            for n in results.nodes or []:
                if n.uuid == node_uuid:
                    center_node_result = {
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
                    break

        return ExploreResponse(
            message=f'Ontology: explored "{node_name or node_uuid}": {len(node_results)} nodes, {len(edge_results)} edges',
            center_node=center_node_result,
            nodes=node_results,
            edges=edge_results,
            communities=community_results,
        )

    except Exception as e:
        logger.error(f'Error in explore_ontology: {e}')
        return ErrorResponse(error=f'Ontology explore error: {e}')


@mcp.custom_route('/health', methods=['GET'])
async def health_check(request) -> JSONResponse:
    """Health check endpoint for Docker and load balancers."""
    return JSONResponse({'status': 'healthy', 'service': 'graphiti-mcp'})


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
        choices=['neo4j', 'falkordb'],
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

    # Set global client for backward compatibility
    graphiti_client = await graphiti_service.get_client()
    semaphore = graphiti_service.semaphore

    # Initialize queue service with the client
    await queue_service.initialize(graphiti_client)

    # Set MCP server settings
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
