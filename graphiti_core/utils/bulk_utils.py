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

import asyncio
import json
import logging
import typing
from datetime import datetime

import numpy as np
from pydantic import BaseModel, Field
from typing_extensions import Any

from graphiti_core.driver.driver import (
    GraphDriver,
    GraphDriverSession,
    GraphProvider,
)
from graphiti_core.edges import Edge, EntityEdge, EpisodicEdge, create_entity_edge_embeddings
from graphiti_core.embedder import EmbedderClient
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.helpers import normalize_l2, semaphore_gather
from graphiti_core.models.edges.edge_db_queries import (
    get_entity_edge_save_bulk_query,
    get_episodic_edge_save_bulk_query,
)
from graphiti_core.models.nodes.node_db_queries import (
    get_entity_node_save_bulk_query,
    get_episode_node_save_bulk_query,
)
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.utils.datetime_utils import convert_datetimes_to_strings
from graphiti_core.utils.maintenance.dedup_helpers import (
    DedupResolutionState,
    _build_candidate_indexes,
    _normalize_string_exact,
    _resolve_with_similarity,
)
from graphiti_core.utils.maintenance.edge_operations import (
    extract_edges,
    resolve_extracted_edge,
)
from graphiti_core.utils.maintenance.graph_data_operations import (
    EPISODE_WINDOW_LEN,
    retrieve_episodes,
)
from graphiti_core.utils.entity_lock_manager import EntityLockManager
from graphiti_core.utils.maintenance.node_operations import (
    extract_nodes,
    resolve_extracted_nodes,
)

logger = logging.getLogger(__name__)

CHUNK_SIZE = 10


def _build_directed_uuid_map(pairs: list[tuple[str, str]]) -> dict[str, str]:
    """Collapse alias -> canonical chains while preserving direction.

    The incoming pairs represent directed mappings discovered during node dedupe. We use a simple
    union-find with iterative path compression to ensure every source UUID resolves to its ultimate
    canonical target, even if aliases appear lexicographically smaller than the canonical UUID.
    """

    parent: dict[str, str] = {}

    def find(uuid: str) -> str:
        """Directed union-find lookup using iterative path compression."""
        parent.setdefault(uuid, uuid)
        root = uuid
        while parent[root] != root:
            root = parent[root]

        while parent[uuid] != root:
            next_uuid = parent[uuid]
            parent[uuid] = root
            uuid = next_uuid

        return root

    for source_uuid, target_uuid in pairs:
        parent.setdefault(source_uuid, source_uuid)
        parent.setdefault(target_uuid, target_uuid)
        parent[find(source_uuid)] = find(target_uuid)

    return {uuid: find(uuid) for uuid in parent}


class RawEpisode(BaseModel):
    name: str
    uuid: str | None = Field(default=None)
    content: str
    source_description: str
    source: EpisodeType
    reference_time: datetime


async def retrieve_previous_episodes_bulk(
    driver: GraphDriver, episodes: list[EpisodicNode]
) -> list[tuple[EpisodicNode, list[EpisodicNode]]]:
    previous_episodes_list = await semaphore_gather(
        *[
            retrieve_episodes(
                driver, episode.valid_at, last_n=EPISODE_WINDOW_LEN, group_ids=[episode.group_id]
            )
            for episode in episodes
        ]
    )
    episode_tuples: list[tuple[EpisodicNode, list[EpisodicNode]]] = [
        (episode, previous_episodes_list[i]) for i, episode in enumerate(episodes)
    ]

    return episode_tuples


async def add_nodes_and_edges_bulk(
    driver: GraphDriver,
    episodic_nodes: list[EpisodicNode],
    episodic_edges: list[EpisodicEdge],
    entity_nodes: list[EntityNode],
    entity_edges: list[EntityEdge],
    embedder: EmbedderClient,
):
    session = driver.session()
    try:
        await session.execute_write(
            add_nodes_and_edges_bulk_tx,
            episodic_nodes,
            episodic_edges,
            entity_nodes,
            entity_edges,
            embedder,
            driver=driver,
        )
    finally:
        await session.close()


async def add_nodes_and_edges_bulk_tx(
    tx: GraphDriverSession,
    episodic_nodes: list[EpisodicNode],
    episodic_edges: list[EpisodicEdge],
    entity_nodes: list[EntityNode],
    entity_edges: list[EntityEdge],
    embedder: EmbedderClient,
    driver: GraphDriver,
):
    episodes = [dict(episode) for episode in episodic_nodes]
    for episode in episodes:
        episode['source'] = str(episode['source'].value)
        episode.pop('labels', None)

    nodes = []

    for node in entity_nodes:
        if node.name_embedding is None:
            await node.generate_name_embedding(embedder)

        entity_data: dict[str, Any] = {
            'uuid': node.uuid,
            'name': node.name,
            'group_id': node.group_id,
            'summary': node.summary,
            'created_at': node.created_at,
            'name_embedding': node.name_embedding,
            'labels': list(set(node.labels + ['Entity'])),
        }

        if driver.provider == GraphProvider.KUZU:
            attributes = convert_datetimes_to_strings(node.attributes) if node.attributes else {}
            entity_data['attributes'] = json.dumps(attributes)
        elif driver.provider == GraphProvider.FALKORDB:
            # FalkorDB needs complex types JSON-serialized
            if node.attributes:
                for k, v in node.attributes.items():
                    if isinstance(v, (dict, list)):
                        entity_data[k] = json.dumps(v)
                    else:
                        entity_data[k] = v
        else:
            entity_data.update(node.attributes or {})

        nodes.append(entity_data)

    edges = []
    for edge in entity_edges:
        if edge.fact_embedding is None:
            await edge.generate_embedding(embedder)
        edge_data: dict[str, Any] = {
            'uuid': edge.uuid,
            'source_node_uuid': edge.source_node_uuid,
            'target_node_uuid': edge.target_node_uuid,
            'name': edge.name,
            'fact': edge.fact,
            'group_id': edge.group_id,
            'episodes': edge.episodes,
            'created_at': edge.created_at,
            'expired_at': edge.expired_at,
            'valid_at': edge.valid_at,
            'invalid_at': edge.invalid_at,
            'fact_embedding': edge.fact_embedding,
        }

        if driver.provider == GraphProvider.KUZU:
            attributes = convert_datetimes_to_strings(edge.attributes) if edge.attributes else {}
            edge_data['attributes'] = json.dumps(attributes)
        elif driver.provider == GraphProvider.FALKORDB:
            # FalkorDB needs complex types JSON-serialized
            if edge.attributes:
                for k, v in edge.attributes.items():
                    if isinstance(v, (dict, list)):
                        edge_data[k] = json.dumps(v)
                    else:
                        edge_data[k] = v
        else:
            edge_data.update(edge.attributes or {})

        edges.append(edge_data)

    if driver.graph_operations_interface:
        await driver.graph_operations_interface.episodic_node_save_bulk(None, driver, tx, episodes)
        await driver.graph_operations_interface.node_save_bulk(None, driver, tx, nodes)
        await driver.graph_operations_interface.episodic_edge_save_bulk(
            None, driver, tx, [edge.model_dump() for edge in episodic_edges]
        )
        await driver.graph_operations_interface.edge_save_bulk(None, driver, tx, edges)

    elif driver.provider == GraphProvider.KUZU:
        # FIXME: Kuzu's UNWIND does not currently support STRUCT[] type properly, so we insert the data one by one instead for now.
        episode_query = get_episode_node_save_bulk_query(driver.provider)
        for episode in episodes:
            await tx.run(episode_query, **episode)
        entity_node_query = get_entity_node_save_bulk_query(driver.provider, nodes)
        for node in nodes:
            await tx.run(entity_node_query, **node)
        entity_edge_query = get_entity_edge_save_bulk_query(driver.provider)
        for edge in edges:
            await tx.run(entity_edge_query, **edge)
        episodic_edge_query = get_episodic_edge_save_bulk_query(driver.provider)
        for edge in episodic_edges:
            await tx.run(episodic_edge_query, **edge.model_dump())
    else:
        await tx.run(get_episode_node_save_bulk_query(driver.provider), episodes=episodes)
        await tx.run(
            get_entity_node_save_bulk_query(driver.provider, nodes),
            nodes=nodes,
        )
        await tx.run(
            get_episodic_edge_save_bulk_query(driver.provider),
            episodic_edges=[edge.model_dump() for edge in episodic_edges],
        )
        # FalkorDB: group edges by type and run separate queries to support custom edge types
        if driver.provider == GraphProvider.FALKORDB:
            from collections import defaultdict

            edges_by_type: dict[str, list] = defaultdict(list)
            for edge in edges:
                edge_type = edge.get('name', 'RELATES_TO') or 'RELATES_TO'
                edges_by_type[edge_type].append(edge)

            for edge_type, typed_edges in edges_by_type.items():
                # Sanitize edge type to prevent injection
                safe_edge_type = ''.join(c for c in edge_type if c.isalnum() or c == '_')
                if not safe_edge_type:
                    safe_edge_type = 'RELATES_TO'
                query = f"""
                    UNWIND $entity_edges AS edge
                    MATCH (source:Entity {{uuid: edge.source_node_uuid}})
                    MATCH (target:Entity {{uuid: edge.target_node_uuid}})
                    MERGE (source)-[r:{safe_edge_type}]->(target)
                    SET r = edge
                    SET r.fact_embedding = vecf32(edge.fact_embedding)
                    RETURN edge.uuid AS uuid
                """
                await tx.run(query, entity_edges=typed_edges)
        else:
            await tx.run(
                get_entity_edge_save_bulk_query(driver.provider),
                entity_edges=edges,
            )


async def extract_nodes_and_edges_bulk(
    clients: GraphitiClients,
    episode_tuples: list[tuple[EpisodicNode, list[EpisodicNode]]],
    edge_type_map: dict[tuple[str, str], list[str]],
    entity_types: dict[str, type[BaseModel]] | None = None,
    excluded_entity_types: list[str] | None = None,
    edge_types: dict[str, type[BaseModel]] | None = None,
    custom_extraction_instructions: str | None = None,
) -> tuple[list[list[EntityNode]], list[list[EntityEdge]], list[int]]:
    """Extract nodes and edges from episodes with per-episode isolation.

    If any single episode's extraction fails, the error is caught and the episode
    is recorded in ``failed_indices`` instead of aborting the entire batch.

    Returns
    -------
    tuple[list[list[EntityNode]], list[list[EntityEdge]], list[int]]
        (nodes_per_episode, edges_per_episode, failed_indices) where the node/edge
        lists contain only successful episodes and failed_indices lists the original
        indices of episodes that failed during extraction.
    """
    failed_indices: list[int] = []

    # --- Phase 1: Node extraction with per-episode isolation ---

    async def _safe_extract_nodes(
        idx: int, episode: EpisodicNode, previous_episodes: list[EpisodicNode]
    ) -> tuple[int, list[EntityNode] | None, Exception | None]:
        try:
            nodes = await extract_nodes(
                clients,
                episode,
                previous_episodes,
                entity_types=entity_types,
                excluded_entity_types=excluded_entity_types,
                custom_extraction_instructions=custom_extraction_instructions,
            )
            return (idx, nodes, None)
        except Exception as e:
            logger.warning(
                'Node extraction failed for episode %d (%s): %s',
                idx, episode.name, e,
            )
            return (idx, None, e)

    node_results: list[tuple[int, list[EntityNode] | None, Exception | None]] = (
        await semaphore_gather(
            *[
                _safe_extract_nodes(i, episode, previous_episodes)
                for i, (episode, previous_episodes) in enumerate(episode_tuples)
            ]
        )
    )

    # Separate successes from failures
    node_failed_set: set[int] = set()
    successful_node_results: list[tuple[int, list[EntityNode]]] = []
    for idx, nodes, error in node_results:
        if error is not None or nodes is None:
            node_failed_set.add(idx)
        else:
            successful_node_results.append((idx, nodes))

    failed_indices.extend(sorted(node_failed_set))

    # If all episodes failed node extraction, skip edge extraction entirely
    if not successful_node_results:
        return [], [], failed_indices

    # --- Phase 2: Edge extraction only for successful episodes ---

    async def _safe_extract_edges(
        original_idx: int, episode: EpisodicNode, nodes: list[EntityNode],
        previous_episodes: list[EpisodicNode],
    ) -> tuple[int, list[EntityEdge] | None, Exception | None]:
        try:
            edges = await extract_edges(
                clients,
                episode,
                nodes,
                previous_episodes,
                edge_type_map=edge_type_map,
                group_id=episode.group_id,
                edge_types=edge_types,
                custom_extraction_instructions=custom_extraction_instructions,
            )
            return (original_idx, edges, None)
        except Exception as e:
            logger.warning(
                'Edge extraction failed for episode %d (%s): %s',
                original_idx, episode.name, e,
            )
            return (original_idx, None, e)

    edge_results: list[tuple[int, list[EntityEdge] | None, Exception | None]] = (
        await semaphore_gather(
            *[
                _safe_extract_edges(
                    idx, episode_tuples[idx][0], nodes, episode_tuples[idx][1],
                )
                for idx, nodes in successful_node_results
            ]
        )
    )

    # Collect edge failures (episodes that passed node extraction but failed edge extraction)
    edge_failed_set: set[int] = set()
    for idx, edges, error in edge_results:
        if error is not None or edges is None:
            edge_failed_set.add(idx)

    if edge_failed_set:
        failed_indices.extend(sorted(edge_failed_set))
        failed_indices.sort()

    # Build final results: only fully successful episodes
    all_failed = node_failed_set | edge_failed_set
    extracted_nodes_bulk: list[list[EntityNode]] = []
    extracted_edges_bulk: list[list[EntityEdge]] = []

    # Build a lookup for edge results by original index
    edge_by_idx: dict[int, list[EntityEdge]] = {}
    for idx, edges, error in edge_results:
        if error is None and edges is not None:
            edge_by_idx[idx] = edges

    for idx, nodes in successful_node_results:
        if idx in all_failed:
            continue
        extracted_nodes_bulk.append(nodes)
        extracted_edges_bulk.append(edge_by_idx[idx])

    return extracted_nodes_bulk, extracted_edges_bulk, failed_indices


async def dedupe_nodes_bulk(
    clients: GraphitiClients,
    extracted_nodes: list[list[EntityNode]],
    episode_tuples: list[tuple[EpisodicNode, list[EpisodicNode]]],
    entity_types: dict[str, type[BaseModel]] | None = None,
    lock_manager: EntityLockManager | None = None,
) -> tuple[dict[str, list[EntityNode]], dict[str, str]]:
    """Resolve entity duplicates across an in-memory batch using a two-pass strategy.

    1. Run :func:`resolve_extracted_nodes` for every episode in parallel so each batch item is
       reconciled against the live graph just like the non-batch flow.
    2. Re-run the deterministic similarity heuristics across the union of resolved nodes to catch
       duplicates that only co-occur inside this batch, emitting a canonical UUID map that callers
       can apply to edges and persistence.
    """

    # First pass: resolve per-episode against the graph, using entity locks if available
    first_pass_nodes_by_episode: dict[str, list[EntityNode]] = {
        episode_tuples[i][0].uuid: nodes
        for i, nodes in enumerate(extracted_nodes)
    }

    first_pass_resolved, first_pass_uuid_map = await resolve_nodes_with_locks(
        clients=clients,
        nodes_by_episode=first_pass_nodes_by_episode,
        episode_context=episode_tuples,
        entity_types=entity_types,
        lock_manager=lock_manager,
    )

    # Build episode resolutions from the lane-based results
    # Map each resolved node back to its originating episode(s)
    resolved_by_uuid: dict[str, EntityNode] = {n.uuid: n for n in first_pass_resolved}

    episode_resolutions: list[tuple[str, list[EntityNode]]] = []
    per_episode_uuid_maps: list[dict[str, str]] = []
    duplicate_pairs: list[tuple[str, str]] = []

    for i, (episode, _) in enumerate(episode_tuples):
        episode_nodes: list[EntityNode] = []
        ep_uuid_map: dict[str, str] = {}
        for node in extracted_nodes[i]:
            canonical_uuid = first_pass_uuid_map.get(node.uuid, node.uuid)
            ep_uuid_map[node.uuid] = canonical_uuid
            canonical = resolved_by_uuid.get(canonical_uuid)
            if canonical is not None:
                episode_nodes.append(canonical)
            else:
                episode_nodes.append(node)
                ep_uuid_map[node.uuid] = node.uuid
            if canonical_uuid != node.uuid:
                duplicate_pairs.append((node.uuid, canonical_uuid))
        episode_resolutions.append((episode.uuid, episode_nodes))
        per_episode_uuid_maps.append(ep_uuid_map)

    canonical_nodes: dict[str, EntityNode] = {}
    for _, resolved_nodes in episode_resolutions:
        for node in resolved_nodes:
            # NOTE: this loop is O(n^2) in the number of nodes inside the batch because we rebuild
            # the MinHash index for the accumulated canonical pool each time. The LRU-backed
            # shingle cache keeps the constant factors low for typical batch sizes (≤ CHUNK_SIZE),
            # but if batches grow significantly we should switch to an incremental index or chunked
            # processing.
            if not canonical_nodes:
                canonical_nodes[node.uuid] = node
                continue

            existing_candidates = list(canonical_nodes.values())
            normalized = _normalize_string_exact(node.name)
            exact_match = next(
                (
                    candidate
                    for candidate in existing_candidates
                    if _normalize_string_exact(candidate.name) == normalized
                ),
                None,
            )
            if exact_match is not None:
                if exact_match.uuid != node.uuid:
                    duplicate_pairs.append((node.uuid, exact_match.uuid))
                continue

            indexes = _build_candidate_indexes(existing_candidates)
            state = DedupResolutionState(
                resolved_nodes=[None],
                uuid_map={},
                unresolved_indices=[],
            )
            _resolve_with_similarity([node], indexes, state)

            resolved = state.resolved_nodes[0]
            if resolved is None:
                canonical_nodes[node.uuid] = node
                continue

            canonical_uuid = resolved.uuid
            canonical_nodes.setdefault(canonical_uuid, resolved)
            if canonical_uuid != node.uuid:
                duplicate_pairs.append((node.uuid, canonical_uuid))

    union_pairs: list[tuple[str, str]] = []
    for uuid_map in per_episode_uuid_maps:
        union_pairs.extend(uuid_map.items())
    union_pairs.extend(duplicate_pairs)

    compressed_map: dict[str, str] = _build_directed_uuid_map(union_pairs)

    nodes_by_episode: dict[str, list[EntityNode]] = {}
    for episode_uuid, resolved_nodes in episode_resolutions:
        deduped_nodes: list[EntityNode] = []
        seen: set[str] = set()
        for node in resolved_nodes:
            canonical_uuid = compressed_map.get(node.uuid, node.uuid)
            if canonical_uuid in seen:
                continue
            seen.add(canonical_uuid)
            canonical_node = canonical_nodes.get(canonical_uuid)
            if canonical_node is None:
                logger.error(
                    'Canonical node %s missing during batch dedupe; falling back to %s',
                    canonical_uuid,
                    node.uuid,
                )
                canonical_node = node
            deduped_nodes.append(canonical_node)

        nodes_by_episode[episode_uuid] = deduped_nodes

    return nodes_by_episode, compressed_map


async def resolve_nodes_with_locks(
    clients: GraphitiClients,
    nodes_by_episode: dict[str, list[EntityNode]],
    episode_context: list[tuple[EpisodicNode, list[EpisodicNode]]],
    entity_types: dict[str, type[BaseModel]] | None,
    lock_manager: EntityLockManager | None = None,
) -> tuple[list[EntityNode], dict[str, str]]:
    """Resolve extracted nodes against the graph with per-entity locking.

    When *lock_manager* is provided, nodes are grouped by (primary_label, normalized_name)
    into "entity lanes". Each lane acquires its lock before searching the graph, and
    registers the resolved node in the pending registry so that concurrent calls for the
    same entity skip the graph search entirely.

    When *lock_manager* is None, falls back to the original per-episode parallel resolution
    (backward compatible).

    Returns
    -------
    tuple[list[EntityNode], dict[str, str]]
        (all_resolved_nodes, uuid_map) — same contract as the caller expects.
    """
    if lock_manager is None:
        # Fallback: original per-episode parallel resolution
        results = await semaphore_gather(
            *[
                resolve_extracted_nodes(
                    clients,
                    [node for node in nodes_by_episode.get(episode.uuid, [])],
                    episode,
                    previous_episodes,
                    entity_types,
                )
                for episode, previous_episodes in episode_context
            ]
        )
        all_nodes: list[EntityNode] = []
        uuid_map: dict[str, str] = {}
        for resolved_nodes, ep_uuid_map, _ in results:
            all_nodes.extend(resolved_nodes)
            uuid_map.update(ep_uuid_map)
        return all_nodes, uuid_map

    # --- Entity-lane resolution ---

    # Build episode lookup for finding the right episode context per node
    episode_lookup: dict[str, tuple[EpisodicNode, list[EpisodicNode]]] = {
        episode.uuid: (episode, previous) for episode, previous in episode_context
    }

    # 1. Collect all unique nodes, group by entity key, track episode origin
    entity_lanes: dict[str, list[EntityNode]] = {}  # key → [nodes with same identity]
    node_episode_origin: dict[str, str] = {}  # node_uuid → episode_uuid (first seen)
    seen_uuids: set[str] = set()

    for episode_uuid, nodes in nodes_by_episode.items():
        for node in nodes:
            if node.uuid in seen_uuids:
                continue
            seen_uuids.add(node.uuid)
            primary_label = node.labels[0] if node.labels else 'Entity'
            key = lock_manager.normalize_key(primary_label, node.name)
            entity_lanes.setdefault(key, []).append(node)
            node_episode_origin.setdefault(node.uuid, episode_uuid)

    # 2. Resolve each lane with its lock — each returns its own results
    async def _resolve_lane(
        entity_key: str, nodes: list[EntityNode]
    ) -> tuple[str, EntityNode, dict[str, str]]:
        primary_label, _, entity_name = entity_key.partition(':')

        async with lock_manager.get_lock(primary_label, entity_name):
            # Check pending registry first
            cached = lock_manager.get_resolved(primary_label, entity_name)
            if cached is not None:
                lane_uuid_map = {node.uuid: cached.uuid for node in nodes}
                return entity_key, cached, lane_uuid_map

            # Not cached — resolve the representative node against the graph
            representative = nodes[0]

            # Use the episode that first mentioned this entity for context
            origin_ep_uuid = node_episode_origin[representative.uuid]
            ep_context = episode_lookup.get(origin_ep_uuid)
            if ep_context:
                episode, previous = ep_context
            else:
                episode, previous = episode_context[0]

            resolved, rep_uuid_map, _ = await resolve_extracted_nodes(
                clients,
                [representative],
                episode,
                previous,
                entity_types,
            )

            canonical = resolved[0] if resolved else representative

            # Build lane uuid map: all nodes in lane → canonical
            lane_uuid_map = {node.uuid: canonical.uuid for node in nodes}
            # Also include the representative's own mapping from resolve_extracted_nodes
            lane_uuid_map.update(rep_uuid_map)

            # Register in pending registry for concurrent callers
            lock_manager.register_resolved(primary_label, entity_name, canonical)

            return entity_key, canonical, lane_uuid_map

    # Use asyncio.gather (NOT semaphore_gather) to avoid deadlock: each lane
    # holds an entity lock and internally calls resolve_extracted_nodes which
    # uses semaphore_gather for graph searches and LLM calls. If we also wrap
    # the outer lanes with the global semaphore, nested acquisition deadlocks
    # when all slots are held by lane wrappers waiting for inner searches.
    # Entity locks already serialize per-entity; LLM throttling happens inside.
    lane_results: list[tuple[str, EntityNode, dict[str, str]]] = await asyncio.gather(
        *[_resolve_lane(key, nodes) for key, nodes in entity_lanes.items()]
    )

    # 3. Merge results sequentially (no shared mutable state during concurrent execution)
    all_resolved: list[EntityNode] = []
    uuid_map: dict[str, str] = {}
    for entity_key, canonical, lane_uuid_map in lane_results:
        all_resolved.append(canonical)
        uuid_map.update(lane_uuid_map)

    return all_resolved, uuid_map


async def dedupe_edges_bulk(
    clients: GraphitiClients,
    extracted_edges: list[list[EntityEdge]],
    episode_tuples: list[tuple[EpisodicNode, list[EpisodicNode]]],
    _entities: list[EntityNode],
    edge_types: dict[str, type[BaseModel]],
    _edge_type_map: dict[tuple[str, str], list[str]],
) -> dict[str, list[EntityEdge]]:
    embedder = clients.embedder
    min_score = 0.6

    # generate embeddings
    await semaphore_gather(
        *[create_entity_edge_embeddings(embedder, edges) for edges in extracted_edges]
    )

    # Find similar results
    dedupe_tuples: list[tuple[EpisodicNode, EntityEdge, list[EntityEdge]]] = []
    for i, edges_i in enumerate(extracted_edges):
        existing_edges: list[EntityEdge] = []
        for edges_j in extracted_edges:
            existing_edges += edges_j

        for edge in edges_i:
            candidates: list[EntityEdge] = []
            for existing_edge in existing_edges:
                # Skip self-comparison
                if edge.uuid == existing_edge.uuid:
                    continue
                # Approximate BM25 by checking for word overlaps (this is faster than creating many in-memory indices)
                # This approach will cast a wider net than BM25, which is ideal for this use case
                if (
                    edge.source_node_uuid != existing_edge.source_node_uuid
                    or edge.target_node_uuid != existing_edge.target_node_uuid
                ):
                    continue

                edge_words = set(edge.fact.lower().split())
                existing_edge_words = set(existing_edge.fact.lower().split())
                has_overlap = not edge_words.isdisjoint(existing_edge_words)
                if has_overlap:
                    candidates.append(existing_edge)
                    continue

                # Check for semantic similarity even if there is no overlap
                similarity = np.dot(
                    normalize_l2(edge.fact_embedding or []),
                    normalize_l2(existing_edge.fact_embedding or []),
                )
                if similarity >= min_score:
                    candidates.append(existing_edge)

            dedupe_tuples.append((episode_tuples[i][0], edge, candidates))

    bulk_edge_resolutions: list[
        tuple[EntityEdge, EntityEdge, list[EntityEdge]]
    ] = await semaphore_gather(
        *[
            resolve_extracted_edge(
                clients.llm_client,
                edge,
                candidates,
                candidates,
                episode,
                edge_types,
            )
            for episode, edge, candidates in dedupe_tuples
        ]
    )

    # For now we won't track edge invalidation
    duplicate_pairs: list[tuple[str, str]] = []
    for i, (_, _, duplicates) in enumerate(bulk_edge_resolutions):
        episode, edge, candidates = dedupe_tuples[i]
        for duplicate in duplicates:
            duplicate_pairs.append((edge.uuid, duplicate.uuid))

    # Now we compress the duplicate_map, so that 3 -> 2 and 2 -> becomes 3 -> 1 (sorted by uuid)
    compressed_map: dict[str, str] = compress_uuid_map(duplicate_pairs)

    edge_uuid_map: dict[str, EntityEdge] = {
        edge.uuid: edge for edges in extracted_edges for edge in edges
    }

    edges_by_episode: dict[str, list[EntityEdge]] = {}
    for i, edges in enumerate(extracted_edges):
        episode = episode_tuples[i][0]

        edges_by_episode[episode.uuid] = [
            edge_uuid_map[compressed_map.get(edge.uuid, edge.uuid)] for edge in edges
        ]

    return edges_by_episode


class UnionFind:
    def __init__(self, elements):
        # start each element in its own set
        self.parent = {e: e for e in elements}

    def find(self, x):
        # path‐compression
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # attach the lexicographically larger root under the smaller
        if ra < rb:
            self.parent[rb] = ra
        else:
            self.parent[ra] = rb


def compress_uuid_map(duplicate_pairs: list[tuple[str, str]]) -> dict[str, str]:
    """
    all_ids: iterable of all entity IDs (strings)
    duplicate_pairs: iterable of (id1, id2) pairs
    returns: dict mapping each id -> lexicographically smallest id in its duplicate set
    """
    all_uuids = set()
    for pair in duplicate_pairs:
        all_uuids.add(pair[0])
        all_uuids.add(pair[1])

    uf = UnionFind(all_uuids)
    for a, b in duplicate_pairs:
        uf.union(a, b)
    # ensure full path‐compression before mapping
    return {uuid: uf.find(uuid) for uuid in all_uuids}


E = typing.TypeVar('E', bound=Edge)


def resolve_edge_pointers(edges: list[E], uuid_map: dict[str, str]):
    for edge in edges:
        source_uuid = edge.source_node_uuid
        target_uuid = edge.target_node_uuid
        edge.source_node_uuid = uuid_map.get(source_uuid, source_uuid)
        edge.target_node_uuid = uuid_map.get(target_uuid, target_uuid)

    return edges
