"""Formatting utilities for Graphiti MCP Server."""

from typing import Any

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodicNode

from models.response_types import EdgeResult, NodeResult


def to_node_result(node: EntityNode) -> NodeResult:
    """Build a NodeResult TypedDict from an EntityNode, dropping embeddings."""
    attrs = node.attributes if node.attributes else {}
    attrs = {k: v for k, v in attrs.items() if 'embedding' not in k.lower()}
    return NodeResult(
        uuid=node.uuid,
        name=node.name,
        labels=node.labels if node.labels else [],
        created_at=node.created_at.isoformat() if node.created_at else None,
        summary=node.summary,
        group_id=node.group_id,
        attributes=attrs,
    )


def to_edge_result(edge: EntityEdge) -> EdgeResult:
    """Build an EdgeResult TypedDict from an EntityEdge."""
    return EdgeResult(
        uuid=edge.uuid,
        name=edge.name,
        fact=edge.fact,
        source_node_uuid=edge.source_node_uuid,
        target_node_uuid=edge.target_node_uuid,
        group_id=edge.group_id,
        created_at=edge.created_at.isoformat() if edge.created_at else None,
        valid_at=edge.valid_at.isoformat() if edge.valid_at else None,
        invalid_at=edge.invalid_at.isoformat() if edge.invalid_at else None,
    )


def format_node_result(node: EntityNode) -> dict[str, Any]:
    """Format an entity node into a readable result.

    Since EntityNode is a Pydantic BaseModel, we can use its built-in serialization capabilities.
    Excludes embedding vectors to reduce payload size and avoid exposing internal representations.

    Args:
        node: The EntityNode to format

    Returns:
        A dictionary representation of the node with serialized dates and excluded embeddings
    """
    result = node.model_dump(
        mode='json',
        exclude={
            'name_embedding',
        },
    )
    # Remove any embedding that might be in attributes
    result.get('attributes', {}).pop('name_embedding', None)
    return result


def format_fact_result(edge: EntityEdge) -> dict[str, Any]:
    """Format an entity edge into a readable result.

    Since EntityEdge is a Pydantic BaseModel, we can use its built-in serialization capabilities.

    Args:
        edge: The EntityEdge to format

    Returns:
        A dictionary representation of the edge with serialized dates and excluded embeddings
    """
    result = edge.model_dump(
        mode='json',
        exclude={
            'fact_embedding',
        },
    )
    result.get('attributes', {}).pop('fact_embedding', None)
    return result


def format_edge_result(edge: EntityEdge) -> dict[str, Any]:
    """Format an entity edge into an EdgeResult dict."""
    return {
        'uuid': edge.uuid,
        'name': edge.name,
        'fact': edge.fact,
        'source_node_uuid': edge.source_node_uuid,
        'target_node_uuid': edge.target_node_uuid,
        'created_at': edge.created_at.isoformat() if edge.created_at else None,
        'valid_at': edge.valid_at.isoformat() if edge.valid_at else None,
        'invalid_at': edge.invalid_at.isoformat() if edge.invalid_at else None,
        'group_id': edge.group_id,
    }


def format_community_result(community: CommunityNode, member_count: int = 0) -> dict[str, Any]:
    """Format a community node into a CommunityResult dict."""
    return {
        'uuid': community.uuid,
        'name': community.name,
        'summary': community.summary or '',
        'member_count': member_count,
        'group_id': community.group_id,
    }


EPISODE_CONTENT_CAP = 6000
"""How much episode text `search` serves per hit before it truncates.

Deliberately generous. For nodes and edges the wire payload is a NAME or a
distilled fact and the body lives elsewhere; for an episode the free text IS
what matched, so a stub would defeat the leg that found it. The bound that
matters is the search `limit` — episode count is capped by it — not the length
of any one narrative.
"""


def format_episode_result(episode: EpisodicNode) -> dict[str, Any]:
    """An episode search hit as a wire dict.

    Companion to `format_node_result` / `format_edge_result` /
    `format_community_result`, and the same embedding rule: any key containing
    'embedding' is dropped. Nothing here produces one today — the fields are
    named explicitly rather than dumped from the model — and the filter stays so
    that adding a field cannot quietly ship thousands of floats into a context
    window.

    `content_truncated` is ALWAYS emitted and always a bool. A consumer must not
    have to infer from a length whether it is holding the whole narrative or a
    prefix: when it is a prefix, the follow-up is `get_episode_context(uuid)`.
    """
    content = episode.content or ''
    truncated = len(content) > EPISODE_CONTENT_CAP
    result = {
        'uuid': episode.uuid,
        'name': episode.name,
        'content': content[:EPISODE_CONTENT_CAP] if truncated else content,
        'content_truncated': truncated,
        'source': episode.source.value
        if hasattr(episode.source, 'value')
        else str(episode.source),
        'source_description': episode.source_description,
        'group_id': episode.group_id,
        'created_at': episode.created_at.isoformat() if episode.created_at else None,
        'valid_at': episode.valid_at.isoformat() if episode.valid_at else None,
    }
    return {k: v for k, v in result.items() if 'embedding' not in k.lower()}
