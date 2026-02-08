"""Tests for entity type context in the node deduplication prompt.

The batch dedup prompt (nodes()) must surface entity type descriptions for
BOTH extracted nodes AND existing entities so the dedup LLM can recognise
shared identifiers (e.g. ICAO codes, registration numbers) that make two
differently-named entities the same real-world object.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from graphiti_core.nodes import EntityNode
from graphiti_core.prompts.dedupe_nodes import nodes
from graphiti_core.utils.maintenance.node_operations import _resolve_with_llm


def _render_nodes_prompt(
    extracted_nodes: list[dict] | None = None,
    existing_nodes: list[dict] | None = None,
) -> str:
    """Render the batch nodes() prompt with given context and return user message text."""
    if extracted_nodes is None:
        extracted_nodes = [
            {
                'id': 0,
                'name': 'Foo',
                'entity_type': ['Entity', 'Airport'],
                'entity_type_description': 'An airport identified by its ICAO code.',
            }
        ]
    if existing_nodes is None:
        existing_nodes = [
            {
                'name': 'Bar',
                'entity_types': ['Entity', 'Airport'],
                'entity_type_description': 'An airport identified by its ICAO code.',
            }
        ]
    context = {
        'extracted_nodes': extracted_nodes,
        'existing_nodes': existing_nodes,
        'episode_content': 'test content',
        'previous_episodes': [],
    }
    messages = nodes(context)
    return messages[1].content


# ---------------------------------------------------------------------------
# Existing entities structure must document entity_type_description
# ---------------------------------------------------------------------------

class TestExistingEntitiesStructure:
    """The prompt's description of EXISTING ENTITIES structure must include
    entity_type_description so the LLM knows it's available."""

    def test_existing_entities_structure_mentions_entity_type_description(self):
        text = _render_nodes_prompt()
        # The structure documentation for EXISTING ENTITIES should mention
        # entity_type_description alongside name and entity_types.
        # Find the part describing existing entities structure.
        lower = text.lower()
        existing_section_start = lower.find('existing entities is an object')
        assert existing_section_start != -1, (
            "Prompt should describe EXISTING ENTITIES structure"
        )
        # The structure description should mention entity_type_description
        # somewhere after the structure heading
        structure_section = lower[existing_section_start:existing_section_start + 500]
        assert 'entity_type_description' in structure_section, (
            "EXISTING ENTITIES structure description must include entity_type_description field"
        )

    def test_existing_entities_data_included_in_prompt(self):
        """entity_type_description from existing nodes context should appear in the rendered prompt."""
        text = _render_nodes_prompt(
            existing_nodes=[
                {
                    'name': 'Paris CDG',
                    'entity_types': ['Entity', 'Airport'],
                    'entity_type_description': 'Airport with ICAO code.',
                }
            ],
        )
        assert 'Airport with ICAO code.' in text, (
            "The entity_type_description value from existing nodes must appear in the prompt"
        )


# ---------------------------------------------------------------------------
# Dedup guidance must reference entity type properties
# ---------------------------------------------------------------------------

class TestDedupGuidanceForSharedIdentifiers:
    """The prompt must guide the LLM to use entity type descriptions and
    their identifying properties when judging duplicates."""

    def test_prompt_mentions_shared_identifiers(self):
        text = _render_nodes_prompt()
        lower = text.lower()
        has_identifier = 'identifier' in lower
        has_property = 'propert' in lower  # property/properties
        assert has_identifier or has_property, (
            "Prompt should mention identifiers or properties for duplicate recognition"
        )

    def test_prompt_references_entity_type_for_dedup_judgment(self):
        """The prompt should tell the LLM to consider entity type descriptions
        when determining if entities are duplicates."""
        text = _render_nodes_prompt()
        lower = text.lower()
        has_type_desc_ref = 'entity_type_description' in lower or 'entity type description' in lower
        assert has_type_desc_ref, (
            "Prompt guidance should reference entity type descriptions for dedup decisions"
        )


# ---------------------------------------------------------------------------
# Wiring: _resolve_with_llm passes entity type descriptions for existing nodes
# ---------------------------------------------------------------------------

def _make_node(name: str, uuid: str = '', labels: list[str] | None = None) -> EntityNode:
    return EntityNode(
        name=name,
        uuid=uuid or f'uuid-{name.lower().replace(" ", "-")}',
        labels=labels or ['Entity'],
        group_id='test',
    )


def _make_state(n: int):
    return SimpleNamespace(
        resolved_nodes=[None] * n,
        uuid_map={},
        unresolved_indices=list(range(n)),
        duplicate_pairs=[],
    )


def _make_indexes(existing_nodes: list[EntityNode]):
    return SimpleNamespace(existing_nodes=existing_nodes)


class AerodromeGeneral(BaseModel):
    """An airport where an incident occurred. Key attributes: name_value (The name and ICAO code)."""
    pass


class TestResolveLlmPassesEntityTypeDescriptions:
    """_resolve_with_llm must include entity_type_description in the
    existing_nodes_context so the dedup prompt has full type context."""

    @pytest.mark.asyncio
    async def test_existing_nodes_context_includes_entity_type_description(self):
        """When entity_types dict is provided, existing nodes context must
        carry the matching entity_type_description."""
        existing = _make_node('Barcelona-El Prat Airport (LEBL)',
                              uuid='existing-lebl',
                              labels=['Entity', 'AerodromeGeneral'])
        extracted = _make_node('En route near LEBL',
                               uuid='new-lebl',
                               labels=['Entity', 'AerodromeGeneral'])

        state = _make_state(1)
        indexes = _make_indexes([existing])

        captured_context: dict = {}

        llm_client = MagicMock()

        async def capture_and_respond(prompt, **kwargs):
            # The prompt is a list of Messages; capture the context that was
            # used to build it by inspecting existing_nodes_context in the
            # rendered prompt text.
            captured_context['prompt_text'] = prompt[1].content
            return {
                'entity_resolutions': [
                    {'id': 0, 'name': 'En route near LEBL', 'duplicate_name': ''},
                ]
            }

        llm_client.generate_response = capture_and_respond

        entity_types = {'AerodromeGeneral': AerodromeGeneral}

        await _resolve_with_llm(
            llm_client, [extracted], indexes, state,
            episode=None, previous_episodes=None, entity_types=entity_types,
        )

        prompt_text = captured_context.get('prompt_text', '')
        assert 'Key attributes' in prompt_text, (
            "Existing node's entity_type_description (from AerodromeGeneral.__doc__) "
            "must appear in the prompt sent to the LLM"
        )

    @pytest.mark.asyncio
    async def test_existing_nodes_without_entity_types_get_default_description(self):
        """When no entity_types dict is provided, existing nodes should get a
        default entity type description."""
        existing = _make_node('Some Entity', uuid='existing-1')
        extracted = _make_node('Another Entity', uuid='new-1')

        state = _make_state(1)
        indexes = _make_indexes([existing])

        captured_context: dict = {}

        llm_client = MagicMock()

        async def capture_and_respond(prompt, **kwargs):
            captured_context['prompt_text'] = prompt[1].content
            return {
                'entity_resolutions': [
                    {'id': 0, 'name': 'Another Entity', 'duplicate_name': ''},
                ]
            }

        llm_client.generate_response = capture_and_respond

        await _resolve_with_llm(
            llm_client, [extracted], indexes, state,
            episode=None, previous_episodes=None, entity_types=None,
        )

        prompt_text = captured_context.get('prompt_text', '')
        assert 'Default Entity Type' in prompt_text, (
            "When no entity_types dict is provided, existing nodes should "
            "get 'Default Entity Type' as description"
        )
