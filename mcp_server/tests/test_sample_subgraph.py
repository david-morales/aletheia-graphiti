"""Offline tests for the `sample_subgraph` tool (Step 2 UI alignment).

The tool orchestrates the flavour subgraph hooks (Task 2) into one
flavour-normalized payload for graph-view consumers. These tests stub the
service/client/driver — same fixture idiom as test_get_schema_canonical.py — so
the wire contract is pinned without a live backend.
"""
import inspect
from datetime import datetime, timezone

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
        # Derived: the base/FalkorDB writer stores exactly [Entity, <leaf>], so
        # the first non-internal label IS the leaf. (This fixture carries an
        # AGE-shaped 3-element hierarchy, which is why the derived answer here is
        # the supertype — see the row-builder tests for the contract.)
        "leaf": "Actor",
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
async def test_sample_subgraph_clamps_the_limit_into_range(monkeypatch):
    """The clamp is what makes textual `$limit` substitution safe — it is the reason
    the value inlined into the query text can only ever be a bounded int."""
    svc, driver = _service(node_rows=[dict(NODE_ROW)], edge_rows=[dict(EDGE_ROW)])
    monkeypatch.setattr(srv, "graphiti_service", svc)

    await srv.sample_subgraph(limit=5000)
    node_q, edge_q = driver.queries
    # endswith, not `in`: "LIMIT 5000" also contains "LIMIT 5", so a substring
    # assertion would pass against an unclamped value.
    assert node_q.endswith("LIMIT 1000")
    assert edge_q.endswith("LIMIT 2500")

    driver.queries.clear()
    await srv.sample_subgraph(limit=0)
    assert driver.queries[0].endswith("LIMIT 1")


@pytest.mark.asyncio
async def test_sample_subgraph_normalizes_datetime_created_at(monkeypatch):
    """FalkorDB and AGE hand back strings, but the generic/Neo4j path returns a
    neo4j.time.DateTime. Unnormalized it dies in FastMCP output validation —
    OUTSIDE this tool's try/except, so it escapes the ADR-015 error envelope as a
    protocol error rather than an `error` payload."""
    dt = datetime(2026, 1, 1, 12, 30, tzinfo=timezone.utc)
    node = dict(NODE_ROW)
    node["created_at"] = dt
    edge = dict(EDGE_ROW)
    edge["created_at"] = dt
    svc, _ = _service(node_rows=[node], edge_rows=[edge])
    monkeypatch.setattr(srv, "graphiti_service", svc)

    payload = await srv.sample_subgraph()

    assert payload["nodes"][0]["created_at"] == "2026-01-01T12:30:00+00:00"
    assert payload["edges"][0]["created_at"] == "2026-01-01T12:30:00+00:00"
    assert isinstance(payload["nodes"][0]["created_at"], str)
    assert isinstance(payload["edges"][0]["created_at"], str)


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


class TestLeafRowBuilder:
    """`leaf` — the producer-announced most-specific label.

    The consumer hazard this closes: `labels` is UNORDERED, so a UI taking the
    first element types AGE nodes by whatever the hierarchy happens to start with.
    Live on the bench graph that collapsed 16 leaf types into three supertypes
    (Actor 94 / Event 67 / Ubicacion 39) and coloured the two arms differently.
    """

    def test_falkordb_shaped_row_derives_the_leaf_from_labels(self):
        """The base/FalkorDB writer stores exactly [Entity, <leaf>], so the first
        non-internal label IS the leaf — no query change needed on that arm."""
        assert srv._subgraph_node_row({"labels": ["Entity", "Persona"]})["leaf"] == "Persona"
        # Order is not a contract: the domain label may come first.
        assert srv._subgraph_node_row({"labels": ["Droga", "Entity"]})["leaf"] == "Droga"

    def test_an_explicit_leaf_column_is_passed_through_verbatim(self):
        """AGE announces it — never re-derive, because deriving from an UNORDERED
        hierarchy is exactly the bug."""
        row = {"labels": ["Entity", "Actor", "Persona"], "leaf": "Persona"}
        assert srv._subgraph_node_row(row)["leaf"] == "Persona"

    def test_the_announced_leaf_wins_over_the_derived_one(self):
        """The derivation would answer `Actor` here — the announced value must win."""
        row = {"labels": ["Entity", "Actor", "Persona"], "leaf": "Persona"}
        derived = srv._subgraph_node_row({"labels": row["labels"]})["leaf"]
        assert derived == "Actor"
        assert srv._subgraph_node_row(row)["leaf"] == "Persona"

    def test_leaf_is_null_when_it_cannot_be_established(self):
        for row in ({}, {"labels": None}, {"labels": []}, {"labels": ["Entity"]},
                    {"labels": ["Entity", "Episodic", "Community"]},
                    {"labels": None, "leaf": ""},
                    {"labels": ["Entity"], "leaf": None}):
            assert srv._subgraph_node_row(row)["leaf"] is None, row

    def test_an_empty_announced_leaf_falls_back_to_the_derivation(self):
        """A null/empty `leaf` column must not SUPPRESS the fallback — a backend
        that announces the column but cannot fill it for one row still has a
        derivable answer whenever a non-internal label is present."""
        for row in ({"labels": ["Entity", "Persona"], "leaf": None},
                    {"labels": ["Entity", "Persona"], "leaf": ""}):
            assert srv._subgraph_node_row(row)["leaf"] == "Persona", row

    def test_an_internal_label_is_never_announced_as_the_leaf(self):
        """A backend that answered `label(n) = 'Entity'` must not type the node
        as internal — fall back to the derivation, then to null."""
        row = {"labels": ["Entity", "Persona"], "leaf": "Entity"}
        assert srv._subgraph_node_row(row)["leaf"] == "Persona"

    @pytest.mark.asyncio
    async def test_the_tool_emits_leaf_for_every_node(self, monkeypatch):
        age_shaped = dict(NODE_ROW)
        age_shaped["uuid"] = N2
        age_shaped["leaf"] = "Persona"
        svc, _ = _service(node_rows=[dict(NODE_ROW), age_shaped])
        monkeypatch.setattr(srv, "graphiti_service", svc)

        payload = await srv.sample_subgraph()

        assert [n["leaf"] for n in payload["nodes"]] == ["Actor", "Persona"]


def test_sample_subgraph_is_registered_and_announced():
    """Task 7's UI route calls the tool by this exact name, and the server is the
    single source of truth for what it is (ADR-019)."""
    assert "sample_subgraph" in srv.mcp._tool_manager._tools

    from domain_profile import DomainProfile
    from tool_descriptions import build_instructions

    instructions = build_instructions(DomainProfile(group_id="policia"))
    assert "sample_subgraph" in instructions
