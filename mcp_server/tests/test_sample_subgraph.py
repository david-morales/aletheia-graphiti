"""Offline tests for the `sample_subgraph` tool (Step 2 UI alignment).

The tool orchestrates the flavour subgraph hooks (Task 2) into one
flavour-normalized payload for graph-view consumers. These tests stub the
service/client/driver — same fixture idiom as test_get_schema_canonical.py — so
the wire contract is pinned without a live backend.
"""
import inspect

import pytest

import graphiti_mcp_server as srv
from flavours.base import BaseFlavour


# Real uuid shapes on purpose: the flavour's edge-query literal sanitizer only
# inlines the hex+dash charset, so a toy id like "n1" would be dropped whole and
# the uuid-scoping assertion below would silently pass against an empty list.
N1 = "11111111-1111-4111-8111-111111111111"
N2 = "22222222-2222-4222-8222-222222222222"
E1 = "eeeeeeee-0001-4000-8000-000000000001"

NODE_ROW = {
    "uuid": N1,
    "name": "Ada",
    "labels": ["Entity", "Actor", "Persona"],
    "created_at": "2026-01-01T00:00:00Z",
    "summary": "a person",
    "group_id": "policia",
}
EDGE_ROW = {
    "uuid": E1,
    "name": "ES_DETENIDO",
    "fact": "Ada was detained",
    "source_node_uuid": N1,
    "target_node_uuid": N2,
    "created_at": "2026-01-02T00:00:00Z",
}


class _StubDriver:
    """Records every query it is handed and answers node/edge probes from a script.

    Rows come back in the FalkorDB/AGE driver shape: ``(records, header, summary)``
    with dict rows — exactly what `Flavour.execute_graph_query` normalizes.
    """

    def __init__(self, node_rows=None, edge_rows=None, raises: Exception | None = None):
        self.queries: list[str] = []
        self._node_rows = [] if node_rows is None else node_rows
        self._edge_rows = [] if edge_rows is None else edge_rows
        self._raises = raises

    async def execute_query(self, query: str, *a, **k):
        self.queries.append(query)
        if self._raises is not None:
            raise self._raises
        # The node query is the one that has no source/target projection.
        if "source_node_uuid" in query:
            rows = self._edge_rows
        else:
            rows = self._node_rows
        header = list(rows[0].keys()) if rows else []
        return rows, header, None


class _StubClient:
    def __init__(self, driver):
        self.driver = driver


class _StubService:
    def __init__(self, flavour, driver):
        self.flavour = flavour
        self._driver = driver
        self.config = srv.GraphitiConfig()
        self.config.graphiti.group_id = "policia"

    async def get_client(self):
        return _StubClient(self._driver)


def _service(node_rows=None, edge_rows=None, raises=None):
    driver = _StubDriver(node_rows=node_rows, edge_rows=edge_rows, raises=raises)
    return _StubService(BaseFlavour(), driver), driver


@pytest.mark.asyncio
async def test_sample_subgraph_returns_normalized_nodes_and_edges(monkeypatch):
    svc, _ = _service(node_rows=[dict(NODE_ROW)], edge_rows=[dict(EDGE_ROW)])
    monkeypatch.setattr(srv, "graphiti_service", svc)

    payload = await srv.sample_subgraph()

    assert payload["type"] == "subgraph"
    assert payload["graph_name"] == "policia"

    assert len(payload["nodes"]) == 1
    node = payload["nodes"][0]
    assert node == {
        "uuid": N1,
        "name": "Ada",
        "labels": ["Entity", "Actor", "Persona"],
        "created_at": "2026-01-01T00:00:00Z",
        "summary": "a person",
        "group_id": "policia",
    }

    assert len(payload["edges"]) == 1
    edge = payload["edges"][0]
    assert edge == {
        "uuid": E1,
        "name": "ES_DETENIDO",
        "fact": "Ada was detained",
        "source_node_uuid": N1,
        "target_node_uuid": N2,
        "created_at": "2026-01-02T00:00:00Z",
    }


@pytest.mark.asyncio
async def test_sample_subgraph_substitutes_limit_and_scales_edge_limit(monkeypatch):
    svc, driver = _service(node_rows=[dict(NODE_ROW)], edge_rows=[dict(EDGE_ROW)])
    monkeypatch.setattr(srv, "graphiti_service", svc)

    await srv.sample_subgraph(limit=40)

    assert len(driver.queries) == 2
    node_q, edge_q = driver.queries
    assert "LIMIT 40" in node_q
    assert "$limit" not in node_q  # substituted, no placeholder residue
    # Edge budget is 2.5x the node budget so a dense sample keeps its edges.
    assert "LIMIT 100" in edge_q
    assert "$limit" not in edge_q
    # The edge query is scoped to the sampled uuids, not re-matched over the graph.
    assert f'"{N1}"' in edge_q


@pytest.mark.asyncio
async def test_sample_subgraph_with_no_nodes_skips_the_edge_query(monkeypatch):
    svc, driver = _service(node_rows=[], edge_rows=[dict(EDGE_ROW)])
    monkeypatch.setattr(srv, "graphiti_service", svc)

    payload = await srv.sample_subgraph()

    assert payload["nodes"] == []
    assert payload["edges"] == []
    # An empty sample has nothing to join on — no second round-trip.
    assert len(driver.queries) == 1


@pytest.mark.asyncio
async def test_sample_subgraph_error_path_returns_ADR015_error(monkeypatch):
    # 1. Service not yet initialized.
    monkeypatch.setattr(srv, "graphiti_service", None)
    payload = await srv.sample_subgraph()
    assert isinstance(payload.get("error"), str)
    assert payload["error"]
    assert "nodes" not in payload and "edges" not in payload

    # 2. Driver blows up mid-sample — the error is the whole payload, never partial data.
    svc, _ = _service(raises=RuntimeError("boom"))
    monkeypatch.setattr(srv, "graphiti_service", svc)
    payload = await srv.sample_subgraph()
    assert payload["error"] == "boom"
    assert "nodes" not in payload and "edges" not in payload


@pytest.mark.asyncio
async def test_sample_subgraph_dedups_nodes_and_edges_by_uuid(monkeypatch):
    """AGE duplicate-uuid sibling vertices (BUG-38) and repeated edge rows must not
    reach the client: a graph view would draw the same node/edge twice."""
    dup_node = dict(NODE_ROW)
    dup_node["name"] = "Ada (sibling)"  # same uuid, different row — first row wins
    svc, _ = _service(
        node_rows=[dict(NODE_ROW), dup_node],
        edge_rows=[dict(EDGE_ROW), dict(EDGE_ROW)],
    )
    monkeypatch.setattr(srv, "graphiti_service", svc)

    payload = await srv.sample_subgraph()

    assert [n["uuid"] for n in payload["nodes"]] == [N1]
    assert payload["nodes"][0]["name"] == "Ada"  # first row wins
    assert [e["uuid"] for e in payload["edges"]] == [E1]


@pytest.mark.asyncio
async def test_sample_subgraph_passes_labels_through_unordered(monkeypatch):
    """`labels` order is NOT a contract on either flavour (response_types.py:183-188):
    the tool must pass the list through untouched and never index it positionally."""
    domain_first = dict(NODE_ROW)
    domain_first["uuid"] = N2
    domain_first["labels"] = ["Droga", "Entity"]  # live bench shape: domain label first
    svc, _ = _service(node_rows=[dict(NODE_ROW), domain_first])
    monkeypatch.setattr(srv, "graphiti_service", svc)

    payload = await srv.sample_subgraph()

    assert payload["nodes"][0]["labels"] == ["Entity", "Actor", "Persona"]
    assert payload["nodes"][1]["labels"] == ["Droga", "Entity"]

    # Structural guard: no positional label access anywhere in the tool.
    source = inspect.getsource(srv.sample_subgraph)
    assert "labels[0]" not in source
    assert "labels[-1]" not in source


def test_sample_subgraph_is_registered_and_announced():
    """Task 7's UI route calls the tool by this exact name, and the server is the
    single source of truth for what it is (ADR-019)."""
    assert "sample_subgraph" in srv.mcp._tool_manager._tools

    from domain_profile import DomainProfile
    from tool_descriptions import build_instructions

    instructions = build_instructions(DomainProfile(group_id="policia"))
    assert "sample_subgraph" in instructions
