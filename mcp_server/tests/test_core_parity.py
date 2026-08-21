#!/usr/bin/env python3
"""Unit tests for the MCP <-> graphiti-core parity wiring.

These tests exercise the queue-service argument threading and the core API
surface it depends on, without requiring a live database or LLM. They run as
part of the default (non-integration) suite.

The helper-function tests that used to live here went with `utils/type_config.py`
(BUG-101): that module was orphaned — nothing under `src/` imported it, and the
`EdgeTypeConfig`/`EdgeTypeMapEntry` config classes it read had already been
dropped from `config/schema.py`. The entity-type feature it once served is built
inline in `graphiti_mcp_server.py` today.
"""

import inspect
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from graphiti_core import Graphiti
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode

# Add the src directory to the path (mirrors the other unit tests)
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.edge_types import EDGE_TYPES  # noqa: E402
from models.entity_types import ENTITY_TYPES  # noqa: E402
from services.queue_service import QueueService  # noqa: E402


class TestQueueServiceThreading:
    """The queue service must forward every parity param to Graphiti.add_episode."""

    @pytest.mark.asyncio
    async def test_add_episode_forwards_all_params(self):
        client = AsyncMock(spec=Graphiti)
        service = QueueService()
        await service.initialize(client)

        ref_time = datetime(2024, 6, 1, tzinfo=timezone.utc)
        edge_types = {'WorksFor': EDGE_TYPES['WorksFor']}
        edge_type_map = {('Entity', 'Entity'): ['WorksFor']}

        await service.add_episode(
            group_id='g1',
            name='ep',
            content='body',
            source_description='desc',
            episode_type='text',
            entity_types={'Preference': ENTITY_TYPES['Preference']},
            uuid='ep-uuid',
            reference_time=ref_time,
            edge_types=edge_types,
            edge_type_map=edge_type_map,
            excluded_entity_types=['Object'],
            previous_episode_uuids=['prev-uuid'],
            custom_extraction_instructions='extra',
            update_communities=True,
            saga='my-saga',
            saga_previous_episode_uuid='saga-prev',
        )

        # The worker runs the queued coroutine in the background; wait for it.
        await service._episode_queues['g1'].join()

        client.add_episode.assert_awaited_once()
        kwargs = client.add_episode.await_args.kwargs
        assert kwargs['reference_time'] == ref_time
        assert kwargs['edge_types'] == edge_types
        assert kwargs['edge_type_map'] == edge_type_map
        assert kwargs['excluded_entity_types'] == ['Object']
        assert kwargs['previous_episode_uuids'] == ['prev-uuid']
        assert kwargs['custom_extraction_instructions'] == 'extra'
        assert kwargs['update_communities'] is True
        assert kwargs['saga'] == 'my-saga'
        assert kwargs['saga_previous_episode_uuid'] == 'saga-prev'
        assert kwargs['uuid'] == 'ep-uuid'

    @pytest.mark.asyncio
    async def test_add_episode_defaults_reference_time_to_now(self):
        client = AsyncMock(spec=Graphiti)
        service = QueueService()
        await service.initialize(client)

        before = datetime.now(timezone.utc)
        await service.add_episode(
            group_id='g2',
            name='ep',
            content='body',
            source_description='desc',
            episode_type='text',
            entity_types=None,
            uuid=None,
        )
        await service._episode_queues['g2'].join()
        after = datetime.now(timezone.utc)

        kwargs = client.add_episode.await_args.kwargs
        assert before <= kwargs['reference_time'] <= after


class TestCoreSignatureCompatibility:
    """Guard against drift between the params we send and graphiti-core's API."""

    def test_queue_service_kwargs_are_accepted_by_add_episode(self):
        params = set(inspect.signature(Graphiti.add_episode).parameters)
        sent = {
            'name',
            'episode_body',
            'source_description',
            'source',
            'group_id',
            'reference_time',
            'entity_types',
            'edge_types',
            'edge_type_map',
            'excluded_entity_types',
            'previous_episode_uuids',
            'custom_extraction_instructions',
            'update_communities',
            'saga',
            'saga_previous_episode_uuid',
            'uuid',
        }
        assert sent <= params

    def test_core_exposes_parity_methods(self):
        # The new tools depend on these methods; guard against being pointed at a
        # graphiti-core too old to support them (e.g. pre-0.29 lacks summarize_saga).
        for method in (
            'remove_episode',
            'summarize_saga',
            'build_communities',
            'add_triplet',
            'get_nodes_and_edges_by_episode',
        ):
            assert hasattr(Graphiti, method), f'graphiti-core is missing {method}'

    def test_triplet_objects_construct(self):
        """The shapes add_triplet builds must satisfy EntityNode/EntityEdge."""
        now = datetime.now(timezone.utc)
        source = EntityNode(uuid='s', name='Alice', group_id='g', created_at=now)
        target = EntityNode(uuid='t', name='Acme', group_id='g', created_at=now)
        edge = EntityEdge(
            name='WORKS_FOR',
            fact='Alice works for Acme',
            group_id='g',
            source_node_uuid=source.uuid,
            target_node_uuid=target.uuid,
            created_at=now,
        )
        assert edge.source_node_uuid == 's'
        assert edge.target_node_uuid == 't'


class TestEntityTypeRegistration:
    """Configured entity types must be registerable with graphiti-core."""

    def test_configured_entity_types_avoid_reserved_field_names(self):
        # graphiti-core rejects custom entity-type fields that collide with
        # EntityNode's own fields (e.g. 'name'); such a clash silently fails every
        # episode ingest. Guard the registered models against reintroducing one.
        from models.entity_types import ENTITY_TYPES

        reserved = set(EntityNode.model_fields.keys())
        for type_name, model in ENTITY_TYPES.items():
            clashes = set(model.model_fields.keys()) & reserved
            assert not clashes, (
                f'entity type {type_name} uses reserved EntityNode field(s): {sorted(clashes)}'
            )


