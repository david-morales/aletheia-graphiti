"""Offline test: the ontology gate builds an AGE ontology client for provider=age."""
import pytest

import graphiti_mcp_server as srv


@pytest.mark.asyncio
async def test_connect_ontology_client_age_builds_client(monkeypatch):
    """For provider=age with an ontology_graph set, _connect_ontology_client builds a client
    (not None), constructing an AGEDriver on the companion <graph>_ontology. AGEDriver + Graphiti
    are stubbed so no live DB is needed."""
    built: dict = {}

    class _FakeAge:
        def __init__(self, **kw):
            built.update(kw)

    class _FakeGraphiti:
        def __init__(self, **kw):
            self._kw = kw

        async def build_indices_and_constraints(self):
            return None

    # _connect_ontology_client does a lazy `from graphiti_core.driver.age_driver import AGEDriver`.
    import graphiti_core.driver.age_driver as age_mod

    monkeypatch.setattr(age_mod, "AGEDriver", _FakeAge)
    monkeypatch.setattr(srv, "Graphiti", _FakeGraphiti)

    cfg = srv.GraphitiConfig()
    cfg.database.provider = "age"
    cfg.graphiti.ontology_graph = "policia_age_poc_ontology"
    svc = srv.GraphitiService(config=cfg)

    db_config = {
        "dsn": "postgresql://age:age@localhost:5433/age_test",
        "graph_name": "policia_age_poc",
        "embedding_dim": 1024,
    }
    client = await svc._connect_ontology_client(db_config, embedder_client=None)

    assert client is not None
    assert built["graph_name"] == "policia_age_poc_ontology"  # companion ontology graph
    assert built["dsn"] == "postgresql://age:age@localhost:5433/age_test"
    assert built["embedding_dim"] == 1024


@pytest.mark.asyncio
async def test_connect_ontology_client_unknown_provider_returns_none(monkeypatch):
    cfg = srv.GraphitiConfig()
    cfg.database.provider = "neo4j"
    cfg.graphiti.ontology_graph = "whatever"
    svc = srv.GraphitiService(config=cfg)
    client = await svc._connect_ontology_client({}, embedder_client=None)
    assert client is None
