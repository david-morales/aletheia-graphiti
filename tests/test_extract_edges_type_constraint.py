"""Tests for edge type constraint in the edge extraction prompt.

When FACT_TYPES are provided, the prompt must constrain the LLM to ONLY
use predefined types — no freeform invention. This prevents non-deterministic
edge type naming between runs.
"""

from graphiti_core.prompts.extract_edges import edge


def _render_edge_prompt(
    edge_types: list[dict] | None = None,
) -> str:
    """Render the edge extraction prompt and return user message text."""
    context = {
        'episode_content': 'test content',
        'nodes': [{'name': 'A', 'entity_types': ['Entity']}],
        'previous_episodes': [],
        'reference_time': '2024-01-01T00:00:00Z',
        'edge_types': edge_types or [],
        'custom_extraction_instructions': '',
    }
    messages = edge(context)
    return messages[1].content


class TestEdgeTypeConstraintWhenFactTypesProvided:
    """When FACT_TYPES are provided, the LLM must be constrained to only
    use those types — not invent new relation_type values."""

    def test_prompt_requires_using_provided_types(self):
        """The prompt must instruct the LLM to use ONLY the provided FACT_TYPES."""
        text = _render_edge_prompt(edge_types=[
            {'fact_type_name': 'WORKS_AT', 'fact_type_signatures': [('Person', 'Company')],
             'fact_type_description': 'Employment relationship'},
        ])
        lower = text.lower()
        # Should say to use only/must use the provided types
        has_must = 'must' in lower and 'fact_type' in lower
        has_only = 'only' in lower and 'fact_type' in lower
        assert has_must or has_only, (
            "When FACT_TYPES are provided, prompt must instruct LLM to use ONLY those types"
        )

    def test_prompt_prohibits_inventing_types_when_fact_types_present(self):
        """The prompt must NOT tell the LLM to derive/invent types when FACT_TYPES exist."""
        text = _render_edge_prompt(edge_types=[
            {'fact_type_name': 'WORKS_AT', 'fact_type_signatures': [('Person', 'Company')],
             'fact_type_description': 'Employment relationship'},
        ])
        lower = text.lower()
        # The "otherwise derive" escape hatch should not be present when
        # FACT_TYPES are provided
        has_otherwise_derive = 'otherwise' in lower and 'derive' in lower
        assert not has_otherwise_derive, (
            "Prompt must NOT offer 'otherwise derive' escape hatch when FACT_TYPES are provided"
        )


class TestEdgeTypeConstraintWhenNoFactTypes:
    """When no FACT_TYPES are provided, the LLM should still be able to
    derive relation types freely."""

    def test_prompt_allows_freeform_types_without_fact_types(self):
        """Without FACT_TYPES, the prompt should allow deriving relation types."""
        text = _render_edge_prompt(edge_types=[])
        lower = text.lower()
        has_derive = 'derive' in lower or 'screaming_snake_case' in lower
        assert has_derive, (
            "Without FACT_TYPES, prompt should allow deriving relation types"
        )
