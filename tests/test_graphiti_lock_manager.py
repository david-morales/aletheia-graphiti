"""Test that Graphiti instance has an entity lock manager."""
import pytest
from unittest.mock import MagicMock, patch
from graphiti_core.graphiti import Graphiti
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.utils.entity_lock_manager import EntityLockManager


def test_graphiti_has_entity_lock_manager():
    """Graphiti.__init__ should create an EntityLockManager on the instance."""
    # Patch GraphitiClients to accept MagicMock arguments (bypass Pydantic validation)
    with patch.object(GraphitiClients, '__init__', lambda self, **kw: None):
        graphiti = Graphiti(
            graph_driver=MagicMock(),
            llm_client=MagicMock(),
            embedder=MagicMock(),
            cross_encoder=MagicMock(),
        )
    assert hasattr(graphiti, '_entity_lock_manager')
    assert isinstance(graphiti._entity_lock_manager, EntityLockManager)
