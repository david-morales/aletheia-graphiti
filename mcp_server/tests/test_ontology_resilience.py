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


class TestOntologyConnectRetry:
    """Bounded retry on ontology client connect."""

    @pytest.mark.asyncio
    async def test_recovers_after_one_transient_failure(self, monkeypatch):
        """A single failure on build_indices_and_constraints is retried successfully."""
        from graphiti_mcp_server import GraphitiService
        from config.schema import GraphitiConfig

        # Speed up: tighter backoff for the test
        monkeypatch.setattr('graphiti_mcp_server._RETRY_INITIAL_BACKOFF_S', 0.01)

        cfg = GraphitiConfig()
        cfg.graphiti.ontology_graph = 'test_ontology'
        cfg.database.provider = 'falkordb'
        svc = GraphitiService(cfg)

        attempt_log: list[int] = []

        async def flaky_build_indices():
            attempt_log.append(len(attempt_log) + 1)
            if len(attempt_log) == 1:
                raise ConnectionResetError('peer reset')

        with patch('graphiti_mcp_server.FalkorDriver'), \
             patch('graphiti_mcp_server.Graphiti') as mock_graphiti_cls:
            mock_graphiti = MagicMock()
            mock_graphiti.build_indices_and_constraints = flaky_build_indices
            mock_graphiti_cls.return_value = mock_graphiti

            client = await svc._connect_ontology_client(
                {'host': 'h', 'port': 6379, 'password': 'p'},
                embedder_client=MagicMock(),
            )

        assert client is mock_graphiti
        assert len(attempt_log) == 2  # one failure, one success

    @pytest.mark.asyncio
    async def test_gives_up_after_max_attempts(self, monkeypatch):
        """Persistent failure raises after exactly _RETRY_ATTEMPTS attempts."""
        from graphiti_mcp_server import GraphitiService
        from config.schema import GraphitiConfig

        monkeypatch.setattr('graphiti_mcp_server._RETRY_INITIAL_BACKOFF_S', 0.01)

        cfg = GraphitiConfig()
        cfg.graphiti.ontology_graph = 'test_ontology'
        cfg.database.provider = 'falkordb'
        svc = GraphitiService(cfg)

        attempt_log: list[int] = []

        async def always_failing():
            attempt_log.append(len(attempt_log) + 1)
            raise ConnectionResetError('persistent failure')

        with patch('graphiti_mcp_server.FalkorDriver'), \
             patch('graphiti_mcp_server.Graphiti') as mock_graphiti_cls:
            mock_graphiti = MagicMock()
            mock_graphiti.build_indices_and_constraints = always_failing
            mock_graphiti_cls.return_value = mock_graphiti

            with pytest.raises(ConnectionResetError):
                await svc._connect_ontology_client(
                    {'host': 'h', 'port': 6379, 'password': 'p'},
                    embedder_client=MagicMock(),
                )

        assert len(attempt_log) == 3  # _RETRY_ATTEMPTS


class TestEnsureOntologyClient:
    """Lazy reconnect when ontology_client is None at runtime."""

    @pytest.mark.asyncio
    async def test_returns_true_when_client_already_exists(self):
        from graphiti_mcp_server import GraphitiService
        from config.schema import GraphitiConfig

        cfg = GraphitiConfig()
        cfg.graphiti.ontology_graph = 'test_ontology'
        svc = GraphitiService(cfg)
        svc.ontology_client = MagicMock()  # already connected

        ok = await svc._ensure_ontology_client()
        assert ok is True

    @pytest.mark.asyncio
    async def test_returns_false_when_no_ontology_configured(self):
        from graphiti_mcp_server import GraphitiService
        from config.schema import GraphitiConfig

        cfg = GraphitiConfig()
        cfg.graphiti.ontology_graph = None  # not configured
        svc = GraphitiService(cfg)
        svc.ontology_client = None

        ok = await svc._ensure_ontology_client()
        assert ok is False

    @pytest.mark.asyncio
    async def test_reconnects_when_client_is_none_and_config_set(self, monkeypatch):
        """When ontology_graph is set but client is None, attempt reconnect."""
        from graphiti_mcp_server import GraphitiService
        from config.schema import GraphitiConfig

        cfg = GraphitiConfig()
        cfg.graphiti.ontology_graph = 'test_ontology'
        cfg.database.provider = 'falkordb'
        svc = GraphitiService(cfg)
        svc.ontology_client = None

        # Stub the db_config + embedder cache that initialize() sets up
        svc._cached_db_config = {'host': 'h', 'port': 6379, 'password': 'p'}
        svc._cached_embedder_client = MagicMock()

        rebuilt = MagicMock()

        async def fake_connect(db_config, embedder_client):
            return rebuilt

        monkeypatch.setattr(svc, '_connect_ontology_client', fake_connect)

        ok = await svc._ensure_ontology_client()
        assert ok is True
        assert svc.ontology_client is rebuilt

    @pytest.mark.asyncio
    async def test_returns_false_when_reconnect_fails(self, monkeypatch):
        """If reconnect raises, client stays None and helper returns False."""
        from graphiti_mcp_server import GraphitiService
        from config.schema import GraphitiConfig

        cfg = GraphitiConfig()
        cfg.graphiti.ontology_graph = 'test_ontology'
        cfg.database.provider = 'falkordb'
        svc = GraphitiService(cfg)
        svc.ontology_client = None
        svc._cached_db_config = {'host': 'h', 'port': 6379, 'password': 'p'}
        svc._cached_embedder_client = MagicMock()

        async def failing_connect(db_config, embedder_client):
            raise ConnectionResetError('still down')

        monkeypatch.setattr(svc, '_connect_ontology_client', failing_connect)

        ok = await svc._ensure_ontology_client()
        assert ok is False
        assert svc.ontology_client is None

    @pytest.mark.asyncio
    async def test_returns_false_when_cache_unset(self):
        """If initialize() never ran, cache is None and reconnect can't happen."""
        from graphiti_mcp_server import GraphitiService
        from config.schema import GraphitiConfig

        cfg = GraphitiConfig()
        cfg.graphiti.ontology_graph = 'test_ontology'
        cfg.database.provider = 'falkordb'
        svc = GraphitiService(cfg)
        svc.ontology_client = None  # explicit
        # _cached_db_config and _cached_embedder_client stay at their __init__ defaults (None)

        ok = await svc._ensure_ontology_client()
        assert ok is False
        assert svc.ontology_client is None
