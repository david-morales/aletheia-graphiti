"""GraphOperationsInterface implementation for the PostgreSQL + Apache AGE driver.

Phase 0 spike: methods are implemented incrementally, driven by the tests that
exercise `add_episode` + hybrid `search`. Anything not yet implemented inherits
the base `raise NotImplementedError` and is out of scope for this phase
(communities, saga nodes, BFS, next/has-episode edges beyond MENTIONS).

Write strategy (avoids AGE's cypher() parameter friction):
  * Graph mutations inline a safely-escaped Cypher map literal (`_map`/`_cy`).
    Node/edge labels collapse to AGE's single-label model — the ontology type
    is kept in a `labels` property and the base label is `:Entity`/`:Episodic`;
    multi-label traversal is a later (Phase 1) concern.
  * Embeddings + keyword content go to the uuid-keyed pgvector/tsvector shadow
    tables via typed asyncpg parameters.
"""

import json
import re
from datetime import datetime
from typing import Any

from graphiti_core.driver.graph_operations.graph_operations import GraphOperationsInterface

# Property keys handled explicitly during hydration; anything else in a node's
# stored map is treated as a custom attribute.
_KNOWN_NODE_KEYS = {
    'uuid', 'name', 'group_id', 'summary', 'created_at', 'labels', 'attributes',
    'name_embedding', 'summary_embedding',
}


def _cy(value: Any) -> str:
    """Serialize a Python scalar/list to a safe AGE Cypher literal."""
    if value is None:
        return 'null'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, datetime):
        value = value.isoformat()
    if isinstance(value, (list, tuple)):
        return '[' + ', '.join(_cy(v) for v in value) + ']'
    s = str(value)
    s = (
        s.replace('\\', '\\\\')
        .replace("'", "\\'")
        .replace('\n', '\\n')
        .replace('\r', '\\r')
        .replace('\t', '\\t')
    )
    return "'" + s + "'"


def _map(props: dict[str, Any]) -> str:
    """Build a Cypher map literal from a dict with identifier-safe keys."""
    return '{' + ', '.join(f'{k}: {_cy(v)}' for k, v in props.items()) + '}'


_IDENT_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


def _node_label(labels: Any) -> str:
    """AGE vertex label = the most-specific (leaf) ontology class.

    Graphiti stores a label list like ['Entity', 'RolInvolucramiento', 'Detencion'];
    the last identifier-safe, non-'Entity' label is the leaf. Falls back to 'Entity'
    (Graphiti's base label, and the narrative-extraction default). The full list is
    still persisted in the `labels` property for abstract-tier filtering.
    """
    for lbl in reversed(list(labels or [])):
        if lbl and lbl != 'Entity' and _IDENT_RE.match(str(lbl)):
            return str(lbl)
    return 'Entity'


def _edge_label(name: Any) -> str:
    """AGE edge label = the relationship type when identifier-safe, else RELATES_TO.

    Deterministic-projection edges use exact ontology edge types (ES_DETENIDO,
    EN_PARTE, INVOLUCRA_ARMA, …) → typed labels. Narrative-extracted edges carry
    free-text predicates that are not valid labels → RELATES_TO fallback. The
    original name is always preserved in the `name` property.
    """
    return str(name) if name and _IDENT_RE.match(str(name)) else 'RELATES_TO'


def _vec(embedding: Any) -> str | None:
    """Format an embedding list as a pgvector text literal, or None."""
    if not embedding:
        return None
    return '[' + ','.join(str(float(x)) for x in embedding) + ']'


def _parse_vec(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [float(x) for x in value]
    return [float(x) for x in json.loads(value)]  # pgvector text '[..]' is valid JSON


def _parse_dt(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return value
    return value


class AGEGraphOperations(GraphOperationsInterface):
    """Native-SQL graph mutation/read operations against AGE + pgvector shadow tables."""

    # ---------------------------------------------------------------- maintenance
    async def clear_data(self, driver: Any, group_ids: list[str] | None = None) -> None:
        if group_ids:
            quoted = ', '.join(_cy(str(g)) for g in group_ids)
            await driver.execute_query(f'MATCH (n) WHERE n.group_id IN [{quoted}] DETACH DELETE n')
            await driver.execute_sql(
                f'DELETE FROM {driver._node_tbl} WHERE group_id = ANY($1::text[])', group_ids
            )
            await driver.execute_sql(
                f'DELETE FROM {driver._edge_tbl} WHERE group_id = ANY($1::text[])', group_ids
            )
        else:
            await driver.execute_query('MATCH (n) DETACH DELETE n')
            await driver.execute_sql(f'TRUNCATE {driver._node_tbl}, {driver._edge_tbl}')

    # --------------------------------------------------------------- entity nodes
    async def node_save(self, node: Any, driver: Any) -> None:
        props = {
            'uuid': node.uuid,
            'name': node.name,
            'group_id': node.group_id,
            'summary': getattr(node, 'summary', '') or '',
            'created_at': node.created_at.isoformat(),
            'labels': list(node.labels or []),
            'attributes': json.dumps(getattr(node, 'attributes', {}) or {}),
        }
        await driver.execute_query(
            f'MERGE (n:{_node_label(node.labels)} {{uuid: {_cy(node.uuid)}}}) SET n += {_map(props)}'
        )
        content = (node.name or '') + '\n' + (getattr(node, 'summary', '') or '')
        await driver.execute_sql(
            f"""INSERT INTO {driver._node_tbl} (uuid, group_id, content, name_embedding)
                VALUES ($1, $2, $3, $4::vector)
                ON CONFLICT (uuid) DO UPDATE SET
                    group_id = EXCLUDED.group_id,
                    content = EXCLUDED.content,
                    name_embedding = EXCLUDED.name_embedding""",
            node.uuid, node.group_id, content, _vec(getattr(node, 'name_embedding', None)),
        )

    def _hydrate_entity(self, cls: Any, props: dict[str, Any]) -> Any:
        raw_attrs = props.get('attributes')
        attributes = json.loads(raw_attrs) if isinstance(raw_attrs, str) else (raw_attrs or {})
        return cls(
            uuid=props['uuid'],
            name=props['name'],
            group_id=props['group_id'],
            labels=props.get('labels') or [],
            created_at=_parse_dt(props.get('created_at')),
            summary=props.get('summary') or '',
            attributes=attributes,
        )

    async def node_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> Any:
        records, _, _ = await driver.execute_query(
            f'MATCH (n) WHERE n.uuid = {_cy(uuid)} RETURN properties(n) AS props'
        )
        if not records:
            from graphiti_core.errors import NodeNotFoundError

            raise NodeNotFoundError(uuid)
        return self._hydrate_entity(_cls, records[0]['props'])

    async def node_get_by_uuids(self, _cls: Any, driver: Any, uuids: list[str]) -> list[Any]:
        if not uuids:
            return []
        in_list = ', '.join(_cy(u) for u in uuids)
        records, _, _ = await driver.execute_query(
            f'MATCH (n) WHERE n.uuid IN [{in_list}] RETURN properties(n) AS props'
        )
        return [self._hydrate_entity(_cls, r['props']) for r in records]

    async def node_load_embeddings(self, node: Any, driver: Any) -> None:
        rows = await driver.execute_sql(
            f'SELECT name_embedding FROM {driver._node_tbl} WHERE uuid = $1', node.uuid
        )
        if rows:
            node.name_embedding = _parse_vec(rows[0]['name_embedding'])

    async def node_load_embeddings_bulk(
        self, driver: Any, nodes: list[Any], batch_size: int = 100
    ) -> dict[str, list[float]]:
        uuids = [n.uuid for n in nodes]
        if not uuids:
            return {}
        rows = await driver.execute_sql(
            f'SELECT uuid, name_embedding FROM {driver._node_tbl} WHERE uuid = ANY($1::text[])',
            uuids,
        )
        by_uuid = {r['uuid']: _parse_vec(r['name_embedding']) for r in rows}
        for n in nodes:
            if n.uuid in by_uuid and by_uuid[n.uuid] is not None:
                n.name_embedding = by_uuid[n.uuid]
        return {k: v for k, v in by_uuid.items() if v is not None}

    # ------------------------------------------------------------- episodic nodes
    async def episodic_node_save(self, node: Any, driver: Any) -> None:
        props = {
            'uuid': node.uuid,
            'name': node.name,
            'group_id': node.group_id,
            'source': node.source.value if hasattr(node.source, 'value') else str(node.source),
            'source_description': getattr(node, 'source_description', '') or '',
            'content': getattr(node, 'content', '') or '',
            'entity_edges': list(getattr(node, 'entity_edges', []) or []),
            'created_at': node.created_at.isoformat(),
            'valid_at': node.valid_at.isoformat(),
        }
        await driver.execute_query(
            f'MERGE (e:Episodic {{uuid: {_cy(node.uuid)}}}) SET e += {_map(props)}'
        )

    def _hydrate_episodic(self, cls: Any, props: dict[str, Any]) -> Any:
        from graphiti_core.nodes import EpisodeType

        source = props['source']
        return cls(
            uuid=props['uuid'],
            name=props['name'],
            group_id=props['group_id'],
            labels=props.get('labels') or [],
            created_at=_parse_dt(props['created_at']),
            source=source if isinstance(source, EpisodeType) else EpisodeType(source),
            source_description=props.get('source_description') or '',
            content=props.get('content') or '',
            valid_at=_parse_dt(props['valid_at']),
            entity_edges=props.get('entity_edges') or [],
        )

    async def episodic_node_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> Any:
        records, _, _ = await driver.execute_query(
            f'MATCH (e:Episodic {{uuid: {_cy(uuid)}}}) RETURN properties(e) AS props'
        )
        if not records:
            from graphiti_core.errors import NodeNotFoundError

            raise NodeNotFoundError(uuid)
        return self._hydrate_episodic(_cls, records[0]['props'])

    async def episodic_node_get_by_uuids(self, _cls: Any, driver: Any, uuids: list[str]) -> list[Any]:
        if not uuids:
            return []
        in_list = ', '.join(_cy(u) for u in uuids)
        records, _, _ = await driver.execute_query(
            f'MATCH (e:Episodic) WHERE e.uuid IN [{in_list}] RETURN properties(e) AS props'
        )
        return [self._hydrate_episodic(_cls, r['props']) for r in records]

    async def retrieve_episodes(
        self,
        driver: Any,
        reference_time: Any,
        last_n: int = 3,
        group_ids: list[str] | None = None,
        source: Any | None = None,
        saga: str | None = None,
    ) -> list[Any]:
        from graphiti_core.nodes import EpisodicNode

        where = ''
        if group_ids:
            where = f'WHERE e.group_id IN [{", ".join(_cy(g) for g in group_ids)}]'
        records, _, _ = await driver.execute_query(
            f'MATCH (e:Episodic) {where} RETURN properties(e) AS props'
        )
        episodes = [self._hydrate_episodic(EpisodicNode, r['props']) for r in records]
        # Point-in-time filter + most-recent-n, done in Python (AGE datetime
        # support is limited); return oldest-first as the interface specifies.
        if reference_time is not None:
            episodes = [e for e in episodes if e.valid_at <= reference_time]
        if source is not None:
            episodes = [e for e in episodes if e.source == source]
        episodes.sort(key=lambda e: e.valid_at)
        return episodes[-last_n:] if last_n else episodes

    # -------------------------------------------------------------- entity edges
    async def edge_save(self, edge: Any, driver: Any) -> None:
        def _iso(v: Any) -> Any:
            return v.isoformat() if isinstance(v, datetime) else v

        props = {
            'uuid': edge.uuid,
            'group_id': edge.group_id,
            'source_node_uuid': edge.source_node_uuid,
            'target_node_uuid': edge.target_node_uuid,
            'name': getattr(edge, 'name', '') or '',
            'fact': getattr(edge, 'fact', '') or '',
            'episodes': list(getattr(edge, 'episodes', []) or []),
            'created_at': edge.created_at.isoformat(),
            'valid_at': _iso(getattr(edge, 'valid_at', None)),
            'invalid_at': _iso(getattr(edge, 'invalid_at', None)),
            'expired_at': _iso(getattr(edge, 'expired_at', None)),
            'attributes': json.dumps(getattr(edge, 'attributes', {}) or {}),
        }
        await driver.execute_query(
            f'MATCH (a), (b) WHERE a.uuid = {_cy(edge.source_node_uuid)} '
            f'AND b.uuid = {_cy(edge.target_node_uuid)} '
            f'MERGE (a)-[r:{_edge_label(getattr(edge, "name", ""))} {{uuid: {_cy(edge.uuid)}}}]->(b) '
            f'SET r += {_map(props)}'
        )
        content = (getattr(edge, 'name', '') or '') + '\n' + (getattr(edge, 'fact', '') or '')
        await driver.execute_sql(
            f"""INSERT INTO {driver._edge_tbl}
                (uuid, group_id, source_node_uuid, target_node_uuid, content, fact_embedding)
                VALUES ($1, $2, $3, $4, $5, $6::vector)
                ON CONFLICT (uuid) DO UPDATE SET
                    group_id = EXCLUDED.group_id,
                    source_node_uuid = EXCLUDED.source_node_uuid,
                    target_node_uuid = EXCLUDED.target_node_uuid,
                    content = EXCLUDED.content,
                    fact_embedding = EXCLUDED.fact_embedding""",
            edge.uuid, edge.group_id, edge.source_node_uuid, edge.target_node_uuid,
            content, _vec(getattr(edge, 'fact_embedding', None)),
        )

    def _hydrate_edge(self, cls: Any, props: dict[str, Any]) -> Any:
        raw_attrs = props.get('attributes')
        attributes = json.loads(raw_attrs) if isinstance(raw_attrs, str) else (raw_attrs or {})
        return cls(
            uuid=props['uuid'],
            group_id=props['group_id'],
            source_node_uuid=props['source_node_uuid'],
            target_node_uuid=props['target_node_uuid'],
            name=props.get('name') or '',
            fact=props.get('fact') or '',
            episodes=props.get('episodes') or [],
            created_at=_parse_dt(props.get('created_at')),
            valid_at=_parse_dt(props.get('valid_at')),
            invalid_at=_parse_dt(props.get('invalid_at')),
            expired_at=_parse_dt(props.get('expired_at')),
            attributes=attributes,
        )

    async def edge_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> Any:
        records, _, _ = await driver.execute_query(
            f'MATCH ()-[r]->() WHERE r.uuid = {_cy(uuid)} RETURN properties(r) AS props'
        )
        if not records:
            from graphiti_core.errors import EdgeNotFoundError

            raise EdgeNotFoundError(uuid)
        return self._hydrate_edge(_cls, records[0]['props'])

    async def edge_get_by_uuids(self, _cls: Any, driver: Any, uuids: list[str]) -> list[Any]:
        if not uuids:
            return []
        in_list = ', '.join(_cy(u) for u in uuids)
        records, _, _ = await driver.execute_query(
            f'MATCH ()-[r]->() WHERE r.uuid IN [{in_list}] RETURN properties(r) AS props'
        )
        return [self._hydrate_edge(_cls, r['props']) for r in records]

    async def edge_get_between_nodes(
        self, _cls: Any, driver: Any, source_node_uuid: str, target_node_uuid: str
    ) -> list[Any]:
        records, _, _ = await driver.execute_query(
            f'MATCH (a)-[r]->(b) WHERE a.uuid = {_cy(source_node_uuid)} '
            f'AND b.uuid = {_cy(target_node_uuid)} RETURN properties(r) AS props'
        )
        return [self._hydrate_edge(_cls, r['props']) for r in records]

    async def edge_get_by_node_uuid(self, _cls: Any, driver: Any, node_uuid: str) -> list[Any]:
        records, _, _ = await driver.execute_query(
            f'MATCH (a)-[r]->(b) '
            f'WHERE a.uuid = {_cy(node_uuid)} OR b.uuid = {_cy(node_uuid)} '
            f'RETURN properties(r) AS props'
        )
        return [self._hydrate_edge(_cls, r['props']) for r in records]

    async def edge_load_embeddings(self, edge: Any, driver: Any) -> None:
        rows = await driver.execute_sql(
            f'SELECT fact_embedding FROM {driver._edge_tbl} WHERE uuid = $1', edge.uuid
        )
        if rows:
            edge.fact_embedding = _parse_vec(rows[0]['fact_embedding'])

    async def edge_load_embeddings_bulk(
        self, driver: Any, edges: list[Any], batch_size: int = 100
    ) -> dict[str, list[float]]:
        uuids = [e.uuid for e in edges]
        if not uuids:
            return {}
        rows = await driver.execute_sql(
            f'SELECT uuid, fact_embedding FROM {driver._edge_tbl} WHERE uuid = ANY($1::text[])',
            uuids,
        )
        by_uuid = {r['uuid']: _parse_vec(r['fact_embedding']) for r in rows}
        for e in edges:
            if by_uuid.get(e.uuid) is not None:
                e.fact_embedding = by_uuid[e.uuid]
        return {k: v for k, v in by_uuid.items() if v is not None}

    # ------------------------------------------------------------ episodic edges
    async def episodic_edge_save(self, edge: Any, driver: Any) -> None:
        props = {
            'uuid': edge.uuid,
            'group_id': edge.group_id,
            'source_node_uuid': edge.source_node_uuid,
            'target_node_uuid': edge.target_node_uuid,
            'created_at': edge.created_at.isoformat(),
        }
        await driver.execute_query(
            f'MATCH (e:Episodic {{uuid: {_cy(edge.source_node_uuid)}}}), '
            f'(n:Entity {{uuid: {_cy(edge.target_node_uuid)}}}) '
            f'MERGE (e)-[r:MENTIONS {{uuid: {_cy(edge.uuid)}}}]->(n) SET r += {_map(props)}'
        )

    async def get_mentioned_nodes(self, driver: Any, episodes: list[Any]) -> list[Any]:
        from graphiti_core.nodes import EntityNode

        uuids = [e.uuid for e in episodes]
        if not uuids:
            return []
        in_list = ', '.join(_cy(u) for u in uuids)
        records, _, _ = await driver.execute_query(
            f'MATCH (e:Episodic)-[:MENTIONS]->(n) WHERE e.uuid IN [{in_list}] '
            f'RETURN DISTINCT properties(n) AS props'
        )
        return [self._hydrate_entity(EntityNode, r['props']) for r in records]

    # ----------------------------------------------------------------- bulk save
    # add_episode persists via the bulk path, which passes plain dicts (attributes
    # flattened, embeddings included) rather than node/edge objects. These
    # field-writers persist a single dict; the bulk methods loop over them
    # (per-item; batched COPY is a later perf refinement).
    @staticmethod
    def _iso(v: Any) -> Any:
        return v.isoformat() if isinstance(v, datetime) else v

    async def _upsert_node_shadow(self, driver: Any, d: dict[str, Any]) -> None:
        content = (d.get('name') or '') + '\n' + (d.get('summary') or '')
        await driver.execute_sql(
            f"""INSERT INTO {driver._node_tbl} (uuid, group_id, content, name_embedding)
                VALUES ($1, $2, $3, $4::vector)
                ON CONFLICT (uuid) DO UPDATE SET
                    group_id = EXCLUDED.group_id, content = EXCLUDED.content,
                    name_embedding = EXCLUDED.name_embedding""",
            d['uuid'], d['group_id'], content, _vec(d.get('name_embedding')),
        )

    async def _write_entity_from_fields(self, driver: Any, d: dict[str, Any]) -> None:
        attributes = {k: v for k, v in d.items() if k not in _KNOWN_NODE_KEYS}
        props = {
            'uuid': d['uuid'],
            'name': d.get('name') or '',
            'group_id': d['group_id'],
            'summary': d.get('summary') or '',
            'created_at': self._iso(d.get('created_at')),
            'labels': list(d.get('labels') or []),
            'attributes': json.dumps(attributes, default=str),
        }
        await driver.execute_query(
            f'MERGE (n:{_node_label(d.get("labels"))} {{uuid: {_cy(d["uuid"])}}}) SET n += {_map(props)}'
        )
        await self._upsert_node_shadow(driver, d)

    async def _write_episode_from_fields(self, driver: Any, d: dict[str, Any]) -> None:
        source = d.get('source')
        props = {
            'uuid': d['uuid'],
            'name': d.get('name') or '',
            'group_id': d['group_id'],
            'source': source.value if hasattr(source, 'value') else str(source),
            'source_description': d.get('source_description') or '',
            'content': d.get('content') or '',
            'entity_edges': list(d.get('entity_edges') or []),
            'created_at': self._iso(d.get('created_at')),
            'valid_at': self._iso(d.get('valid_at')),
        }
        await driver.execute_query(
            f'MERGE (e:Episodic {{uuid: {_cy(d["uuid"])}}}) SET e += {_map(props)}'
        )

    _KNOWN_EDGE_KEYS = {
        'uuid', 'source_node_uuid', 'target_node_uuid', 'name', 'fact', 'group_id',
        'episodes', 'created_at', 'expired_at', 'valid_at', 'invalid_at', 'fact_embedding',
    }

    async def _write_entity_edge_from_fields(self, driver: Any, d: dict[str, Any]) -> None:
        attributes = {k: v for k, v in d.items() if k not in self._KNOWN_EDGE_KEYS}
        props = {
            'uuid': d['uuid'],
            'group_id': d['group_id'],
            'source_node_uuid': d['source_node_uuid'],
            'target_node_uuid': d['target_node_uuid'],
            'name': d.get('name') or '',
            'fact': d.get('fact') or '',
            'episodes': list(d.get('episodes') or []),
            'created_at': self._iso(d.get('created_at')),
            'valid_at': self._iso(d.get('valid_at')),
            'invalid_at': self._iso(d.get('invalid_at')),
            'expired_at': self._iso(d.get('expired_at')),
            'attributes': json.dumps(attributes, default=str),
        }
        await driver.execute_query(
            f'MATCH (a), (b) WHERE a.uuid = {_cy(d["source_node_uuid"])} '
            f'AND b.uuid = {_cy(d["target_node_uuid"])} '
            f'MERGE (a)-[r:{_edge_label(d.get("name"))} {{uuid: {_cy(d["uuid"])}}}]->(b) '
            f'SET r += {_map(props)}'
        )
        content = (d.get('name') or '') + '\n' + (d.get('fact') or '')
        await driver.execute_sql(
            f"""INSERT INTO {driver._edge_tbl}
                (uuid, group_id, source_node_uuid, target_node_uuid, content, fact_embedding)
                VALUES ($1, $2, $3, $4, $5, $6::vector)
                ON CONFLICT (uuid) DO UPDATE SET
                    group_id = EXCLUDED.group_id,
                    source_node_uuid = EXCLUDED.source_node_uuid,
                    target_node_uuid = EXCLUDED.target_node_uuid,
                    content = EXCLUDED.content, fact_embedding = EXCLUDED.fact_embedding""",
            d['uuid'], d['group_id'], d['source_node_uuid'], d['target_node_uuid'],
            content, _vec(d.get('fact_embedding')),
        )

    async def _write_episodic_edge_from_fields(self, driver: Any, d: dict[str, Any]) -> None:
        props = {
            'uuid': d['uuid'],
            'group_id': d['group_id'],
            'source_node_uuid': d['source_node_uuid'],
            'target_node_uuid': d['target_node_uuid'],
            'created_at': self._iso(d.get('created_at')),
        }
        await driver.execute_query(
            f'MATCH (e:Episodic {{uuid: {_cy(d["source_node_uuid"])}}}), '
            f'(n:Entity {{uuid: {_cy(d["target_node_uuid"])}}}) '
            f'MERGE (e)-[r:MENTIONS {{uuid: {_cy(d["uuid"])}}}]->(n) SET r += {_map(props)}'
        )

    async def node_save_bulk(
        self, _cls: Any, driver: Any, transaction: Any, nodes: list[Any], batch_size: int = 100
    ) -> None:
        for d in nodes:
            await self._write_entity_from_fields(driver, d)

    async def episodic_node_save_bulk(
        self, _cls: Any, driver: Any, transaction: Any, nodes: list[Any], batch_size: int = 100
    ) -> None:
        for d in nodes:
            await self._write_episode_from_fields(driver, d)

    async def edge_save_bulk(
        self, _cls: Any, driver: Any, transaction: Any, edges: list[Any], batch_size: int = 100
    ) -> None:
        for d in edges:
            await self._write_entity_edge_from_fields(driver, d)

    async def episodic_edge_save_bulk(
        self,
        _cls: Any,
        driver: Any,
        transaction: Any,
        episodic_edges: list[Any],
        batch_size: int = 100,
    ) -> None:
        for d in episodic_edges:
            await self._write_episodic_edge_from_fields(driver, d)
