import asyncio
import logging
from collections import defaultdict

from pydantic import BaseModel

from graphiti_core.driver.driver import GraphDriver, GraphProvider
from graphiti_core.edges import CommunityEdge
from graphiti_core.embedder import EmbedderClient
from graphiti_core.helpers import semaphore_gather
from graphiti_core.llm_client import LLMClient
from graphiti_core.models.nodes.node_db_queries import COMMUNITY_NODE_RETURN
from graphiti_core.nodes import CommunityNode, EntityNode, get_community_node_from_record
from graphiti_core.prompts import prompt_library
from graphiti_core.prompts.summarize_nodes import Summary, SummaryDescription
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.utils.maintenance.edge_operations import build_community_edges
from graphiti_core.utils.text_utils import MAX_SUMMARY_CHARS, truncate_at_sentence

MAX_COMMUNITY_BUILD_CONCURRENCY = 10
MAX_ITERATIONS = 100
OSCILLATION_WINDOW = 5

logger = logging.getLogger(__name__)


class Neighbor(BaseModel):
    node_uuid: str
    edge_count: int


async def get_community_clusters(
    driver: GraphDriver, group_ids: list[str] | None
) -> list[list[EntityNode]]:
    if driver.graph_operations_interface:
        try:
            return await driver.graph_operations_interface.get_community_clusters(driver, group_ids)
        except NotImplementedError:
            pass

    community_clusters: list[list[EntityNode]] = []

    if group_ids is None:
        group_id_values, _, _ = await driver.execute_query(
            """
            MATCH (n:Entity)
            WHERE n.group_id IS NOT NULL
            RETURN
                collect(DISTINCT n.group_id) AS group_ids
            """
        )

        group_ids = group_id_values[0]['group_ids'] if group_id_values else []

    for group_id in group_ids:
        projection: dict[str, list[Neighbor]] = {}
        nodes = await EntityNode.get_by_group_ids(driver, [group_id])

        if not nodes:
            continue

        # Batch query: get all neighbors for all nodes in this group at once
        batch_query = """
            MATCH (n:Entity {group_id: $group_id})-[e:RELATES_TO]-(m:Entity {group_id: $group_id})
            WITH n.uuid AS source_uuid, count(e) AS edge_count, m.uuid AS neighbor_uuid
            RETURN source_uuid, neighbor_uuid, edge_count
        """
        if driver.provider == GraphProvider.KUZU:
            batch_query = """
                MATCH (n:Entity {group_id: $group_id})-[:RELATES_TO]-(e:RelatesToNode_)-[:RELATES_TO]-(m:Entity {group_id: $group_id})
                WITH n.uuid AS source_uuid, count(e) AS edge_count, m.uuid AS neighbor_uuid
                RETURN source_uuid, neighbor_uuid, edge_count
            """

        records, _, _ = await driver.execute_query(batch_query, group_id=group_id)

        # Initialize all nodes with empty neighbor lists (including isolated nodes)
        for node in nodes:
            projection[node.uuid] = []

        # Populate neighbor lists from batch results
        for record in records:
            source = record['source_uuid']
            if source in projection:
                projection[source].append(
                    Neighbor(node_uuid=record['neighbor_uuid'], edge_count=record['edge_count'])
                )

        cluster_uuids = label_propagation(projection)

        community_clusters.extend(
            list(
                await semaphore_gather(
                    *[EntityNode.get_by_uuids(driver, cluster) for cluster in cluster_uuids]
                )
            )
        )

    return community_clusters


def label_propagation(projection: dict[str, list[Neighbor]]) -> list[list[str]]:
    # Implement the label propagation community detection algorithm.
    # 1. Start with each node being assigned its own community
    # 2. Each node will take on the community of the plurality of its neighbors
    # 3. Ties are broken by going to the largest community
    # 4. Continue until no communities change during propagation
    # 5. Safety: cap iterations and detect oscillation via state hashing

    community_map = {uuid: i for i, uuid in enumerate(projection.keys())}
    state_history: list[int] = []

    for _iteration in range(MAX_ITERATIONS):
        no_change = True
        new_community_map: dict[str, int] = {}

        for uuid, neighbors in projection.items():
            curr_community = community_map[uuid]

            community_candidates: dict[int, int] = defaultdict(int)
            for neighbor in neighbors:
                community_candidates[community_map[neighbor.node_uuid]] += neighbor.edge_count
            community_lst = [
                (count, community) for community, count in community_candidates.items()
            ]

            community_lst.sort(reverse=True)
            candidate_rank, community_candidate = community_lst[0] if community_lst else (0, -1)
            if community_candidate != -1 and candidate_rank > 1:
                new_community = community_candidate
            else:
                new_community = max(community_candidate, curr_community)

            new_community_map[uuid] = new_community

            if new_community != curr_community:
                no_change = False

        if no_change:
            break

        community_map = new_community_map

        # Oscillation detection: hash current state and check recent history
        state_hash = hash(tuple(sorted(community_map.items())))
        if state_hash in state_history[-OSCILLATION_WINDOW:]:
            logger.warning(
                'Label propagation detected oscillation at iteration %d, stopping.',
                _iteration + 1,
            )
            break
        state_history.append(state_hash)
    else:
        logger.warning(
            'Label propagation reached maximum iterations (%d) without converging.',
            MAX_ITERATIONS,
        )

    community_cluster_map = defaultdict(list)
    for uuid, community in community_map.items():
        community_cluster_map[community].append(uuid)

    clusters = [cluster for cluster in community_cluster_map.values()]
    return clusters


async def summarize_pair(llm_client: LLMClient, summary_pair: tuple[str, str]) -> str:
    # Prepare context for LLM
    context = {
        'node_summaries': [{'summary': summary} for summary in summary_pair],
    }

    llm_response = await llm_client.generate_response(
        prompt_library.summarize_nodes.summarize_pair(context),
        response_model=Summary,
        prompt_name='summarize_nodes.summarize_pair',
    )

    pair_summary = llm_response.get('summary', '')

    return truncate_at_sentence(pair_summary, MAX_SUMMARY_CHARS)


async def generate_summary_description(llm_client: LLMClient, summary: str) -> str:
    context = {
        'summary': summary,
    }

    llm_response = await llm_client.generate_response(
        prompt_library.summarize_nodes.summary_description(context),
        response_model=SummaryDescription,
        prompt_name='summarize_nodes.summary_description',
    )

    description = llm_response.get('description', '')

    return description


async def build_community(
    llm_client: LLMClient, community_cluster: list[EntityNode]
) -> tuple[CommunityNode, list[CommunityEdge]]:
    summaries = [entity.summary for entity in community_cluster]
    length = len(summaries)
    while length > 1:
        odd_one_out: str | None = None
        if length % 2 == 1:
            odd_one_out = summaries.pop()
            length -= 1
        new_summaries: list[str] = list(
            await semaphore_gather(
                *[
                    summarize_pair(llm_client, (str(left_summary), str(right_summary)))
                    for left_summary, right_summary in zip(
                        summaries[: int(length / 2)], summaries[int(length / 2) :], strict=False
                    )
                ]
            )
        )
        if odd_one_out is not None:
            new_summaries.append(odd_one_out)
        summaries = new_summaries
        length = len(summaries)

    summary = truncate_at_sentence(summaries[0], MAX_SUMMARY_CHARS)
    name = await generate_summary_description(llm_client, summary)
    now = utc_now()
    community_node = CommunityNode(
        name=name,
        group_id=community_cluster[0].group_id,
        labels=['Community'],
        created_at=now,
        summary=summary,
    )
    community_edges = build_community_edges(community_cluster, community_node, now)

    logger.debug(f'Built community {community_node.uuid} with {len(community_edges)} edges')

    return community_node, community_edges


async def build_communities(
    driver: GraphDriver,
    llm_client: LLMClient,
    group_ids: list[str] | None,
) -> tuple[list[CommunityNode], list[CommunityEdge]]:
    community_clusters = await get_community_clusters(driver, group_ids)

    semaphore = asyncio.Semaphore(MAX_COMMUNITY_BUILD_CONCURRENCY)

    async def limited_build_community(cluster):
        async with semaphore:
            return await build_community(llm_client, cluster)

    communities: list[tuple[CommunityNode, list[CommunityEdge]]] = list(
        await semaphore_gather(
            *[limited_build_community(cluster) for cluster in community_clusters]
        )
    )

    community_nodes: list[CommunityNode] = []
    community_edges: list[CommunityEdge] = []
    for community in communities:
        community_nodes.append(community[0])
        community_edges.extend(community[1])

    return community_nodes, community_edges


async def remove_communities(driver: GraphDriver, group_ids: list[str] | None = None):
    """Delete community nodes, scoped to `group_ids`.

    Three cases, and the distinction between the last two is load-bearing:

    * `group_ids=['a']` — delete the communities of partition 'a'. Unscoped, this took
      every OTHER partition's community layer with it and never rebuilt it (BUG-57).
    * `group_ids=None` — no scope was given, so the whole graph is cleared. This is
      what a full rebuild needs and what the `build_communities` docstring promises.
    * `group_ids=[]` — an EMPTY list of partitions. Delete nothing.

    `[]` was briefly treated as "no scope" (i.e. as `None`), justified as protecting a
    full rebuild from becoming a silent no-op. That justification was false, and in the
    dangerous direction: `get_community_clusters` iterates `for group_id in group_ids`,
    so it builds nothing from an empty list. `build_communities(group_ids=[])` therefore
    deleted every community in the graph and rebuilt none of them — the exact
    destroy-and-do-not-restore failure BUG-57 was raised about, reintroduced through the
    argument meant to fix it.

    An empty collection means "no elements" everywhere else in this module, and it means
    that here: `[] ` deletes nothing, which is also the only reading under which the
    caller's intent ("these partitions") is impossible to misread as its opposite
    ("all partitions").
    """
    if driver.graph_operations_interface:
        try:
            return await driver.graph_operations_interface.remove_communities(driver, group_ids)
        except NotImplementedError:
            pass

    if group_ids is not None:
        # `IN []` matches nothing, which is precisely the semantics an empty partition
        # list should have. No special case, and no way for it to widen.
        await driver.execute_query(
            """
            MATCH (c:Community)
            WHERE c.group_id IN $group_ids
            DETACH DELETE c
            """,
            group_ids=group_ids,
        )
        return

    await driver.execute_query(
        """
        MATCH (c:Community)
        DETACH DELETE c
        """
    )


async def determine_entity_community(
    driver: GraphDriver, entity: EntityNode
) -> tuple[CommunityNode | None, bool]:
    if driver.graph_operations_interface:
        try:
            return await driver.graph_operations_interface.determine_entity_community(
                driver, entity
            )
        except NotImplementedError:
            pass

    # Check if the node is already part of a community
    records, _, _ = await driver.execute_query(
        """
        MATCH (c:Community)-[:HAS_MEMBER]->(n:Entity {uuid: $entity_uuid})
        RETURN
        """
        + COMMUNITY_NODE_RETURN,
        entity_uuid=entity.uuid,
    )

    if len(records) > 0:
        return get_community_node_from_record(records[0]), False

    # If the node has no community, add it to the mode community of surrounding entities
    match_query = """
        MATCH (c:Community)-[:HAS_MEMBER]->(m:Entity)-[:RELATES_TO]-(n:Entity {uuid: $entity_uuid})
    """
    if driver.provider == GraphProvider.KUZU:
        match_query = """
            MATCH (c:Community)-[:HAS_MEMBER]->(m:Entity)-[:RELATES_TO]-(e:RelatesToNode_)-[:RELATES_TO]-(n:Entity {uuid: $entity_uuid})
        """
    records, _, _ = await driver.execute_query(
        match_query
        + """
        RETURN
        """
        + COMMUNITY_NODE_RETURN,
        entity_uuid=entity.uuid,
    )

    communities: list[CommunityNode] = [
        get_community_node_from_record(record) for record in records
    ]

    community_map: dict[str, int] = defaultdict(int)
    for community in communities:
        community_map[community.uuid] += 1

    community_uuid = None
    max_count = 0
    for uuid, count in community_map.items():
        if count > max_count:
            community_uuid = uuid
            max_count = count

    if max_count == 0:
        return None, False

    for community in communities:
        if community.uuid == community_uuid:
            return community, True

    return None, False


async def update_community(
    driver: GraphDriver,
    llm_client: LLMClient,
    embedder: EmbedderClient,
    entity: EntityNode,
) -> tuple[list[CommunityNode], list[CommunityEdge]]:
    community, is_new = await determine_entity_community(driver, entity)

    if community is None:
        return [], []

    new_summary = await summarize_pair(llm_client, (entity.summary, community.summary))
    new_name = await generate_summary_description(llm_client, new_summary)

    community.summary = new_summary
    community.name = new_name

    community_edges = []
    if is_new:
        community_edge = (build_community_edges([entity], community, utc_now()))[0]
        await community_edge.save(driver)
        community_edges.append(community_edge)

    await community.generate_name_embedding(embedder)

    await community.save(driver)

    return [community], community_edges
