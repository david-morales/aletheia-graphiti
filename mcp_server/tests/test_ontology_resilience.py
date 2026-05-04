"""Tests for ontology client resilience: retry-on-init and lazy-reconnect."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestConnectOntologyClient:
    """Tests for the GraphitiService._connect_ontology_client extracted method."""

    @pytest.mark.asyncio
    async def test_returns_graphiti_instance_on_success(self):
        """A successful connect returns a Graphiti instance with build_indices called."""
        from graphiti_mcp_server import GraphitiService
        from config.schema import GraphitiConfig

        cfg = GraphitiConfig()
        cfg.graphiti.ontology_graph = 'test_ontology'
        cfg.database.provider = 'falkordb'
        svc = GraphitiService(cfg)

        db_config = {'host': 'h', 'port': 6379, 'password': 'p'}
        embedder = MagicMock()

        with patch('graphiti_mcp_server.FalkorDriver'), \
             patch('graphiti_mcp_server.Graphiti') as mock_graphiti_cls:
            mock_graphiti = AsyncMock()
            mock_graphiti_cls.return_value = mock_graphiti

            client = await svc._connect_ontology_client(db_config, embedder)

        assert client is mock_graphiti
        mock_graphiti.build_indices_and_constraints.assert_awaited_once()
