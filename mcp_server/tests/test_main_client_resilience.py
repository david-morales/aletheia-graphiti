"""Tests for main client resilience: retry on transient FalkorDB failure during init."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestMainClientConnectRetry:
    """Bounded retry on main client build_indices_and_constraints."""

    @pytest.mark.asyncio
    async def test_recovers_after_one_transient_failure(self, monkeypatch):
        """A single failure on main client build_indices is retried successfully."""
        from graphiti_mcp_server import GraphitiService
        from config.schema import GraphitiConfig

        monkeypatch.setattr('graphiti_mcp_server._RETRY_INITIAL_BACKOFF_S', 0.01)

        cfg = GraphitiConfig()
        cfg.database.provider = 'falkordb'
        cfg.graphiti.ontology_graph = None  # skip ontology path for this test
        svc = GraphitiService(cfg)

        attempt_log: list[int] = []

        async def flaky_build_indices():
            attempt_log.append(len(attempt_log) + 1)
            if len(attempt_log) == 1:
                raise ConnectionResetError('peer reset')

        with patch('graphiti_mcp_server.FalkorDriver'), \
             patch('graphiti_core.driver.falkordb_driver.FalkorDriver'), \
             patch('graphiti_mcp_server.Graphiti') as mock_graphiti_cls, \
             patch('graphiti_mcp_server.LLMClientFactory'), \
             patch('graphiti_mcp_server.EmbedderFactory'), \
             patch('graphiti_mcp_server.DatabaseDriverFactory') as mock_db_factory:
            mock_db_factory.create_config.return_value = {
                'host': 'h', 'port': 6379, 'password': 'p', 'database': 'main'
            }
            mock_graphiti = MagicMock()
            mock_graphiti.build_indices_and_constraints = flaky_build_indices
            mock_graphiti_cls.return_value = mock_graphiti

            await svc.initialize()

        assert svc.client is mock_graphiti
        assert len(attempt_log) == 2

    @pytest.mark.asyncio
    async def test_main_client_gives_up_after_max_attempts(self, monkeypatch):
        """Persistent failure on main client init raises after _RETRY_ATTEMPTS attempts."""
        from graphiti_mcp_server import GraphitiService
        from config.schema import GraphitiConfig

        monkeypatch.setattr('graphiti_mcp_server._RETRY_INITIAL_BACKOFF_S', 0.01)

        cfg = GraphitiConfig()
        cfg.database.provider = 'falkordb'
        cfg.graphiti.ontology_graph = None
        svc = GraphitiService(cfg)

        attempt_log: list[int] = []

        async def always_failing():
            attempt_log.append(len(attempt_log) + 1)
            raise ConnectionResetError('persistent failure')

        with patch('graphiti_mcp_server.FalkorDriver'), \
             patch('graphiti_core.driver.falkordb_driver.FalkorDriver'), \
             patch('graphiti_mcp_server.Graphiti') as mock_graphiti_cls, \
             patch('graphiti_mcp_server.LLMClientFactory'), \
             patch('graphiti_mcp_server.EmbedderFactory'), \
             patch('graphiti_mcp_server.DatabaseDriverFactory') as mock_db_factory:
            mock_db_factory.create_config.return_value = {
                'host': 'h', 'port': 6379, 'password': 'p', 'database': 'main'
            }
            mock_graphiti = MagicMock()
            mock_graphiti.build_indices_and_constraints = always_failing
            mock_graphiti_cls.return_value = mock_graphiti

            with pytest.raises(ConnectionResetError):
                await svc.initialize()

        assert len(attempt_log) == 3
