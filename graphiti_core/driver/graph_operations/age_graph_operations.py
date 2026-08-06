"""GraphOperationsInterface implementation for the PostgreSQL + Apache AGE driver.

Phase 0 spike: methods are implemented incrementally, driven by the tests that
exercise `add_episode` + hybrid `search`. Anything not yet implemented inherits
the base `raise NotImplementedError` and is out of scope for this phase
(communities, saga nodes, BFS, next/has-episode edges beyond MENTIONS).

Write strategy (avoids AGE's cypher() parameter friction):
  * Graph mutations inline a safely-escaped Cypher map literal (`_map`/`_cy`).
    Node/edge labels collapse to AGE's single-label model — the full ontology
    list is kept in a `labels` property, and the ONE stored label is the LEAF
    class (`_node_label`), NOT `:Entity`; `:Entity` is only the fallback for a
    node with no typed class. Episodes are the exception: they are always
    `:Episodic`. So a pattern may pin an episode by label but must reach an
    entity by uuid alone — matching a typed entity as `(n:Entity {uuid: ...})`
    silently finds nothing. Multi-label traversal is a later (Phase 1) concern.
  * Embeddings + keyword content go to the uuid-keyed pgvector/tsvector shadow
    tables via typed asyncpg parameters.
"""

import json
import logging
import re
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
from typing import Any

from graphiti_core.driver.graph_operations.graph_operations import GraphOperationsInterface

logger = logging.getLogger(__name__)

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
    if isinstance(value, dict):
        return _map(value)
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


_IDENT_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


def _map(props: dict[str, Any]) -> str:
    """Build an agtype map literal from a dict — nested maps and lists round-trip
    through `_cy`, so a value that is itself a dict becomes a queryable nested map
    (e.g. `n.attributes.<field>`) rather than an opaque JSON string. Keys that are
    not identifier-safe are double-quoted."""
    parts = []
    for k, v in props.items():
        ks = str(k)
        key = ks if _IDENT_RE.match(ks) else '"' + ks.replace('\\', '\\\\').replace('"', '\\"') + '"'
        parts.append(f'{key}: {_cy(v)}')
    return '{' + ', '.join(parts) + '}'


def _merged_attributes(stored: dict[str, Any] | None, incoming: dict[str, Any] | None) -> dict[str, Any]:
    """Per-key union of the stored and incoming attribute maps; an incoming key
    wins only with a non-empty value. Storage-layer guard for the 2026-08-01
    attribute-loss bug: AGE keeps attributes as ONE nested map and `SET n +=`
    replaces a nested map wholesale, so a re-save carrying empty in-memory
    attributes (entity resolution does this) must never erase stored ones.

    Limitation: an attribute cannot be cleared by saving an empty value —
    clearing needs an explicit future API, not this write path. The cross-type
    consequence of that: when a uuid-stable node is re-typed, its map becomes the
    union of every type's attributes it has ever carried, since nothing prunes the
    keys the old type contributed."""
    merged = dict(stored or {})
    for k, v in (incoming or {}).items():
        if v not in (None, '', [], {}):
            merged[k] = v
    return merged


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

    # ---------------------------------------------- merge-on-write stored lookup
    # Nodes and edges both keep their custom attributes in ONE nested `attributes`
    # map, and both are written with `SET x += {...}`, which replaces a nested map
    # wholesale. Every writer therefore reads the stored map first and merges —
    # see `_merged_attributes`.
    #
    # The read MUST be scoped to the same label the write MERGEs on. AGE's MERGE
    # is label-scoped, so one uuid can legitimately exist as several vertices under
    # different leaf labels. `bulk_utils` now preserves label order (dict.fromkeys,
    # 2026-08-01), but the multi-vertex state remains reachable: legacy graphs
    # written before the fix and not yet repaired, `add_triplet`'s set-union label
    # merge (graphiti.py, unordered), and repeated non-Entity labels whose first
    # and last occurrences differ. A label-blind read would return some other
    # vertex's map and the write
    # would then stamp it onto this one. Scoping also makes the read address
    # exactly one vertex, so there is no first-row/last-row ambiguity to resolve.
    # ------------------------------------------------------- label-bearing writes
    @staticmethod
    async def _write(
        driver: Any,
        cypher: str,
        *,
        vertex_labels: tuple[str, ...] = (),
        edge_labels: tuple[str, ...] = (),
    ):
        """Run a write that NAMES labels, materialising those labels first.

        AGE creates a label's backing relations on first use, implicitly and
        without a lock, so concurrent writers sharing a brand-new label collide on
        the DDL and the losers lose their save entirely (BUG-38 — see
        `AGEDriver._ensure_label`). Declaring the labels here means the graph
        already has them by the time the MERGE runs.

        Every MERGE/CREATE in this class goes through this method; that is what
        keeps the guarantee from depending on each writer remembering. Read
        patterns do NOT need it: AGE's MATCH on an unknown label creates nothing.
        """
        for label in vertex_labels:
            await driver.ensure_vertex_label(label)
        for label in edge_labels:
            await driver.ensure_edge_label(label)
        return await driver.execute_query(cypher)

    @staticmethod
    def _node_read_pattern(label: str) -> tuple[str, str]:
        """Read pattern for the vertex `node_save`/`_write_entity_from_fields` MERGE."""
        return f'MATCH (n:{label})', 'n'

    @staticmethod
    def _edge_read_pattern(label: str) -> tuple[str, str]:
        """Read pattern for the edge `edge_save`/`_write_entity_edge_from_fields` MERGE."""
        return f'MATCH ()-[r:{label}]->()', 'r'

    @staticmethod
    def _as_attr_map(raw: Any) -> dict[str, Any]:
        """Coerce a returned `attributes` value to a dict (agtype maps decode to
        dicts; a JSON string is tolerated for older rows; anything else = {})."""
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except ValueError:
                return {}
        return raw if isinstance(raw, dict) else {}

    async def _stored_attributes_by_label(
        self,
        driver: Any,
        pattern_for: Callable[[str], tuple[str, str]],
        uuids_by_label: dict[str, list[str]],
    ) -> dict[tuple[str, str], dict[str, Any]]:
        """{(label, uuid): stored attributes}, ONE query per label group — so a
        bulk write costs O(distinct labels) reads, not O(rows). Absent (label,
        uuid) pairs are simply missing from the result (callers treat that as {}).
        Keyed by label AND uuid because one uuid may exist under several labels."""
        out: dict[tuple[str, str], dict[str, Any]] = {}
        for label, uuids in uuids_by_label.items():
            if not uuids:
                continue
            match, var = pattern_for(label)
            in_list = ', '.join(_cy(u) for u in uuids)
            records, _, _ = await driver.execute_query(
                f'{match} WHERE {var}.uuid IN [{in_list}] '
                f'RETURN {var}.uuid AS uuid, {var}.attributes AS attributes'
            )
            for r in records:
                key = r.get('uuid')
                if key is not None:
                    out[(label, str(key))] = self._as_attr_map(r.get('attributes'))
        return out

    async def _stored_attributes(
        self,
        driver: Any,
        pattern_for: Callable[[str], tuple[str, str]],
        label: str,
        uuid: str,
    ) -> dict[str, Any]:
        """Attributes stored on the (label, uuid) vertex/edge ({} when absent)."""
        found = await self._stored_attributes_by_label(driver, pattern_for, {label: [uuid]})
        return found.get((label, uuid), {})

    # --------------------------------------------------------------- entity nodes
    async def node_save(self, node: Any, driver: Any) -> None:
        label = _node_label(node.labels)
        stored = await self._stored_attributes(driver, self._node_read_pattern, label, node.uuid)
        props = {
            'uuid': node.uuid,
            'name': node.name,
            'group_id': node.group_id,
            'summary': getattr(node, 'summary', '') or '',
            'created_at': node.created_at.isoformat(),
            'labels': list(node.labels or []),
            'attributes': _merged_attributes(stored, getattr(node, 'attributes', {}) or {}),
        }
        await self._write(
            driver,
            f'MERGE (n:{label} {{uuid: {_cy(node.uuid)}}}) SET n += {_map(props)}',
            vertex_labels=(label,),
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

    async def node_get_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str], group_id: str | None = None
    ) -> list[Any]:
        # group_id added to the GraphOperationsInterface in graphiti-core v0.29.2;
        # honor it as an optional partition filter when provided.
        if not uuids:
            return []
        in_list = ', '.join(_cy(u) for u in uuids)
        where = f'n.uuid IN [{in_list}]'
        if group_id is not None:
            where += f' AND n.group_id = {_cy(group_id)}'
        records, _, _ = await driver.execute_query(
            f'MATCH (n) WHERE {where} RETURN properties(n) AS props'
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
        await self._write(
            driver,
            f'MERGE (e:Episodic {{uuid: {_cy(node.uuid)}}}) SET e += {_map(props)}',
            vertex_labels=('Episodic',),
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

        label = _edge_label(getattr(edge, 'name', ''))
        stored = await self._stored_attributes(
            driver, self._edge_read_pattern, label, edge.uuid
        )
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
            'attributes': _merged_attributes(stored, getattr(edge, 'attributes', {}) or {}),
        }
        await self._write(
            driver,
            f'MATCH (a), (b) WHERE a.uuid = {_cy(edge.source_node_uuid)} '
            f'AND b.uuid = {_cy(edge.target_node_uuid)} '
            f'MERGE (a)-[r:{label} {{uuid: {_cy(edge.uuid)}}}]->(b) '
            f'SET r += {_map(props)}',
            edge_labels=(label,),
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
    async def _merge_mentions_edge(self, driver: Any, props: dict[str, Any]) -> None:
        """MERGE one MENTIONS edge from `props`; shared by both episodic-edge writers.

        The target is matched LABEL-FREE by uuid, mirroring the entity-edge writer
        (`_write_entity_edge_from_fields`). It used to be `(n:Entity {uuid: ...})`,
        which no typed entity can satisfy: AGE stores exactly one label per vertex
        and `_node_label` makes it the LEAF ontology class, so the MATCH found
        nothing and the MERGE no-opped for every typed target. The leaf label is
        not a stable fact about a node — it varies with the node's ontology class —
        so it cannot be part of a lookup key; the uuid is what identifies a node
        here, and the label constraint only ever excluded valid targets. (uuids are
        *intended* to be unique but are not guaranteed so in practice: duplicate-uuid
        siblings are a known integrity defect with their own repair CLI. That affects
        HOW MANY vertices this MERGE can reach, not whether the label belongs in the
        pattern.)

        `n.labels IS NOT NULL` keeps the widened pattern from reaching Graphiti's own
        bookkeeping vertices: every entity write persists a `labels` list (verified on
        the live bed for typed, untyped, empty and no-Entity-base label lists, through
        both the projection and bulk writers) while `episodic_node_save` never writes
        one. Without it a corrupt `target_node_uuid` naming an episode — or the source
        episode itself — would MERGE an episode-to-episode MENTIONS edge, which the old
        `:Entity` pattern made structurally impossible. Same discriminator the MCP AGE
        flavour uses to exclude bookkeeping from its profile probes.

        The RETURN + warning exist because that failure was SILENT — a MERGE whose
        MATCH is empty writes nothing and raises nothing. A genuine miss (an edge
        pointing at an episode or entity that was never persisted) is a real
        integrity problem, so it belongs in the logs rather than in a later count.
        """
        records, _, _ = await self._write(
            driver,
            f'MATCH (e:Episodic), (n) '
            f'WHERE e.uuid = {_cy(props["source_node_uuid"])} '
            f'AND n.uuid = {_cy(props["target_node_uuid"])} '
            f'AND n.labels IS NOT NULL '
            f'MERGE (e)-[r:MENTIONS {{uuid: {_cy(props["uuid"])}}}]->(n) '
            f'SET r += {_map(props)} RETURN r.uuid AS uuid',
            edge_labels=('MENTIONS',),
        )
        if not records:
            logger.warning(
                'MENTIONS edge %s not written: no match for episode %s -> entity %s',
                props['uuid'], props['source_node_uuid'], props['target_node_uuid'],
            )

    async def episodic_edge_save(self, edge: Any, driver: Any) -> None:
        props = {
            'uuid': edge.uuid,
            'group_id': edge.group_id,
            'source_node_uuid': edge.source_node_uuid,
            'target_node_uuid': edge.target_node_uuid,
            'created_at': edge.created_at.isoformat(),
        }
        await self._merge_mentions_edge(driver, props)

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

    async def _write_entity_from_fields(
        self, driver: Any, d: dict[str, Any], stored_attributes: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Persist one flat entity dict; returns the attribute map actually written.

        `stored_attributes` is the map currently stored on the (label, uuid) vertex
        this write MERGEs on; pass it when the caller has already batch-fetched
        (see `node_save_bulk`), leave it None to have this method look it up. Either
        way the write merges rather than replaces — see `_merged_attributes`. The
        return value lets a batching caller fold the result forward, so a uuid
        repeated inside one batch accumulates instead of dropping earlier writes.
        """
        label = _node_label(d.get('labels'))
        incoming = {k: v for k, v in d.items() if k not in _KNOWN_NODE_KEYS}
        if stored_attributes is None:
            stored_attributes = await self._stored_attributes(
                driver, self._node_read_pattern, label, d['uuid']
            )
        merged = _merged_attributes(stored_attributes, incoming)
        props = {
            'uuid': d['uuid'],
            'name': d.get('name') or '',
            'group_id': d['group_id'],
            'summary': d.get('summary') or '',
            'created_at': self._iso(d.get('created_at')),
            'labels': list(d.get('labels') or []),
            'attributes': merged,
        }
        await self._write(
            driver,
            f'MERGE (n:{label} {{uuid: {_cy(d["uuid"])}}}) SET n += {_map(props)}',
            vertex_labels=(label,),
        )
        await self._upsert_node_shadow(driver, d)
        return merged

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
        await self._write(
            driver,
            f'MERGE (e:Episodic {{uuid: {_cy(d["uuid"])}}}) SET e += {_map(props)}',
            vertex_labels=('Episodic',),
        )

    _KNOWN_EDGE_KEYS = {
        'uuid', 'source_node_uuid', 'target_node_uuid', 'name', 'fact', 'group_id',
        'episodes', 'created_at', 'expired_at', 'valid_at', 'invalid_at', 'fact_embedding',
    }

    async def _write_entity_edge_from_fields(
        self, driver: Any, d: dict[str, Any], stored_attributes: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Persist one flat edge dict; returns the attribute map actually written.

        `stored_attributes` is the map currently stored on the (label, uuid) edge
        this write MERGEs on; pass it when the caller has already batch-fetched
        (see `edge_save_bulk`), leave it None to have this method look it up. Either
        way the write merges rather than replaces — see `_merged_attributes`. The
        return value lets a batching caller fold the result forward, so a uuid
        repeated inside one batch accumulates instead of dropping earlier writes.
        """
        label = _edge_label(d.get('name'))
        incoming = {k: v for k, v in d.items() if k not in self._KNOWN_EDGE_KEYS}
        if stored_attributes is None:
            stored_attributes = await self._stored_attributes(
                driver, self._edge_read_pattern, label, d['uuid']
            )
        merged = _merged_attributes(stored_attributes, incoming)
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
            'attributes': merged,
        }
        await self._write(
            driver,
            f'MATCH (a), (b) WHERE a.uuid = {_cy(d["source_node_uuid"])} '
            f'AND b.uuid = {_cy(d["target_node_uuid"])} '
            f'MERGE (a)-[r:{label} {{uuid: {_cy(d["uuid"])}}}]->(b) '
            f'SET r += {_map(props)}',
            edge_labels=(label,),
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
        return merged

    async def _write_episodic_edge_from_fields(self, driver: Any, d: dict[str, Any]) -> None:
        props = {
            'uuid': d['uuid'],
            'group_id': d['group_id'],
            'source_node_uuid': d['source_node_uuid'],
            'target_node_uuid': d['target_node_uuid'],
            'created_at': self._iso(d.get('created_at')),
        }
        await self._merge_mentions_edge(driver, props)

    async def node_save_bulk(
        self, _cls: Any, driver: Any, transaction: Any, nodes: list[Any], batch_size: int = 100
    ) -> None:
        # One round-trip per distinct leaf label, so the merge-on-write guard costs
        # O(labels) extra queries for the whole batch. Each row's merged result is
        # folded back in so a uuid repeated within the batch accumulates rather
        # than resetting to the pre-batch snapshot.
        uuids_by_label: dict[str, list[str]] = defaultdict(list)
        for d in nodes:
            uuids_by_label[_node_label(d.get('labels'))].append(d['uuid'])
        stored = await self._stored_attributes_by_label(
            driver, self._node_read_pattern, uuids_by_label
        )
        for d in nodes:
            key = (_node_label(d.get('labels')), d['uuid'])
            stored[key] = await self._write_entity_from_fields(driver, d, stored.get(key, {}))

    async def episodic_node_save_bulk(
        self, _cls: Any, driver: Any, transaction: Any, nodes: list[Any], batch_size: int = 100
    ) -> None:
        for d in nodes:
            await self._write_episode_from_fields(driver, d)

    async def edge_save_bulk(
        self, _cls: Any, driver: Any, transaction: Any, edges: list[Any], batch_size: int = 100
    ) -> None:
        # One round-trip per distinct relationship label, and each row's merged
        # result folded back in — same shape as `node_save_bulk`.
        uuids_by_label: dict[str, list[str]] = defaultdict(list)
        for d in edges:
            uuids_by_label[_edge_label(d.get('name'))].append(d['uuid'])
        stored = await self._stored_attributes_by_label(
            driver, self._edge_read_pattern, uuids_by_label
        )
        for d in edges:
            key = (_edge_label(d.get('name')), d['uuid'])
            stored[key] = await self._write_entity_edge_from_fields(driver, d, stored.get(key, {}))

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
