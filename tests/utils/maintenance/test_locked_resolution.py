"""Tests for entity-lane locked resolution in bulk_utils."""
import asyncio
from unittest.mock import MagicMock, patch

import pytest

from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.utils.bulk_utils import resolve_nodes_with_locks
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.utils.entity_lock_manager import EntityLockManager


def _make_episode(suffix: str) -> EpisodicNode:
    return EpisodicNode(
        name=f"ep-{suffix}",
        group_id="group",
        labels=[],
        source=EpisodeType.message,
        content="content",
        source_description="test",
        created_at=utc_now(),
        valid_at=utc_now(),
    )


def _make_clients() -> GraphitiClients:
    return GraphitiClients.model_construct(
        driver=MagicMock(),
        embedder=MagicMock(),
        cross_encoder=MagicMock(),
        llm_client=MagicMock(),
    )


@pytest.mark.asyncio
async def test_shared_entity_resolved_once_across_episodes(monkeypatch):
    """When two episodes share 'Madrid', it is resolved against the graph only once."""
    clients = _make_clients()
    lock_mgr = EntityLockManager()

    madrid_a = EntityNode(uuid="uuid-a", name="Madrid", group_id="group", labels=["Location"])
    madrid_b = EntityNode(uuid="uuid-b", name="Madrid", group_id="group", labels=["Location"])
    barcelona = EntityNode(uuid="uuid-c", name="Barcelona", group_id="group", labels=["Location"])

    ep1 = _make_episode("1")
    ep2 = _make_episode("2")

    nodes_by_episode = {
        ep1.uuid: [madrid_a, barcelona],
        ep2.uuid: [madrid_b],
    }

    resolve_calls = []

    async def fake_resolve(clients_arg, nodes_arg, episode_arg, previous_arg, entity_types_arg,
                           existing_nodes_override=None):
        resolve_calls.append([n.name for n in nodes_arg])
        resolved = []
        uuid_map = {}
        for n in nodes_arg:
            resolved.append(n)
            uuid_map[n.uuid] = n.uuid
        return resolved, uuid_map, []

    monkeypatch.setattr(
        "graphiti_core.utils.bulk_utils.resolve_extracted_nodes",
        fake_resolve,
    )

    result_nodes, result_uuid_map = await resolve_nodes_with_locks(
        clients=clients,
        nodes_by_episode=nodes_by_episode,
        episode_context=[(ep1, []), (ep2, [])],
        entity_types=None,
        lock_manager=lock_mgr,
    )

    # Madrid should appear only once in resolve calls (the second was served from registry)
    all_resolved_names = [name for call in resolve_calls for name in call]
    assert all_resolved_names.count("Madrid") == 1
    assert all_resolved_names.count("Barcelona") == 1

    # uuid_map should map madrid_b → madrid_a (first one wins)
    assert result_uuid_map.get(madrid_b.uuid) == madrid_a.uuid


@pytest.mark.asyncio
async def test_different_types_same_name_get_separate_lanes():
    """'Madrid' as Location and 'Madrid' as Organization are separate entities."""
    clients = _make_clients()
    lock_mgr = EntityLockManager()

    madrid_loc = EntityNode(uuid="uuid-loc", name="Madrid", group_id="group", labels=["Location"])
    madrid_org = EntityNode(uuid="uuid-org", name="Madrid", group_id="group", labels=["Organization"])

    ep = _make_episode("1")
    nodes_by_episode = {ep.uuid: [madrid_loc, madrid_org]}

    async def fake_resolve(clients_arg, nodes_arg, episode_arg, previous_arg, entity_types_arg,
                           existing_nodes_override=None):
        return nodes_arg, {n.uuid: n.uuid for n in nodes_arg}, []

    with patch("graphiti_core.utils.bulk_utils.resolve_extracted_nodes", side_effect=fake_resolve):
        result_nodes, result_uuid_map = await resolve_nodes_with_locks(
            clients=clients,
            nodes_by_episode=nodes_by_episode,
            episode_context=[(ep, [])],
            entity_types=None,
            lock_manager=lock_mgr,
        )

    assert len(result_nodes) == 2
    assert result_uuid_map[madrid_loc.uuid] == madrid_loc.uuid
    assert result_uuid_map[madrid_org.uuid] == madrid_org.uuid


@pytest.mark.asyncio
async def test_no_lock_manager_falls_back_to_per_episode():
    """When lock_manager is None, behavior is the same as before (per-episode parallel)."""
    clients = _make_clients()

    node = EntityNode(uuid="uuid-x", name="Solo", group_id="group", labels=["Entity"])
    ep = _make_episode("1")
    nodes_by_episode = {ep.uuid: [node]}

    async def fake_resolve(clients_arg, nodes_arg, episode_arg, previous_arg, entity_types_arg,
                           existing_nodes_override=None):
        return nodes_arg, {n.uuid: n.uuid for n in nodes_arg}, []

    with patch("graphiti_core.utils.bulk_utils.resolve_extracted_nodes", side_effect=fake_resolve):
        result_nodes, result_uuid_map = await resolve_nodes_with_locks(
            clients=clients,
            nodes_by_episode=nodes_by_episode,
            episode_context=[(ep, [])],
            entity_types=None,
            lock_manager=None,
        )

    assert len(result_nodes) == 1
    assert result_uuid_map[node.uuid] == node.uuid


@pytest.mark.asyncio
async def test_concurrent_batches_share_lock_manager(monkeypatch):
    """Two concurrent resolve_nodes_with_locks calls sharing a lock_manager
    serialize resolution of shared entities."""
    clients = _make_clients()
    lock_mgr = EntityLockManager()

    madrid_1 = EntityNode(uuid="uuid-m1", name="Madrid", group_id="group", labels=["Location"])
    madrid_2 = EntityNode(uuid="uuid-m2", name="Madrid", group_id="group", labels=["Location"])

    ep1 = _make_episode("1")
    ep2 = _make_episode("2")

    resolve_call_uuids = []

    async def fake_resolve(clients_arg, nodes_arg, episode_arg, previous_arg, entity_types_arg,
                           existing_nodes_override=None):
        for n in nodes_arg:
            resolve_call_uuids.append(n.uuid)
        await asyncio.sleep(0.02)
        return nodes_arg, {n.uuid: n.uuid for n in nodes_arg}, []

    monkeypatch.setattr(
        "graphiti_core.utils.bulk_utils.resolve_extracted_nodes",
        fake_resolve,
    )

    task1 = resolve_nodes_with_locks(
        clients=clients,
        nodes_by_episode={ep1.uuid: [madrid_1]},
        episode_context=[(ep1, [])],
        entity_types=None,
        lock_manager=lock_mgr,
    )
    task2 = resolve_nodes_with_locks(
        clients=clients,
        nodes_by_episode={ep2.uuid: [madrid_2]},
        episode_context=[(ep2, [])],
        entity_types=None,
        lock_manager=lock_mgr,
    )

    await asyncio.gather(task1, task2)

    # Only one Madrid should have hit fake_resolve
    assert len(resolve_call_uuids) == 1


@pytest.mark.asyncio
async def test_uses_correct_episode_context(monkeypatch):
    """Each lane resolves using the episode that first mentioned the entity."""
    clients = _make_clients()
    lock_mgr = EntityLockManager()

    madrid = EntityNode(uuid="uuid-m", name="Madrid", group_id="group", labels=["Location"])
    ep1 = _make_episode("1")
    ep1.content = "Episode about Madrid incident"

    nodes_by_episode = {ep1.uuid: [madrid]}

    episode_used = []

    async def fake_resolve(clients_arg, nodes_arg, episode_arg, previous_arg, entity_types_arg,
                           existing_nodes_override=None):
        episode_used.append(episode_arg.uuid)
        return nodes_arg, {n.uuid: n.uuid for n in nodes_arg}, []

    monkeypatch.setattr(
        "graphiti_core.utils.bulk_utils.resolve_extracted_nodes",
        fake_resolve,
    )

    await resolve_nodes_with_locks(
        clients=clients,
        nodes_by_episode=nodes_by_episode,
        episode_context=[(ep1, [])],
        entity_types=None,
        lock_manager=lock_mgr,
    )

    assert episode_used == [ep1.uuid]
