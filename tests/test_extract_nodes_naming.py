"""Tests for entity extraction prompt naming guidance.

Ensures extraction prompts guide the LLM to use entity type definitions
and properties to identify specific entities, stripping contextual noise.
"""

from graphiti_core.prompts.extract_nodes import extract_text, extract_message


def _text_prompt(custom_instructions: str = '') -> str:
    """Render extract_text prompt and return the user message content."""
    context = {
        'entity_types': '[]',
        'episode_content': 'test content',
        'custom_extraction_instructions': custom_instructions,
    }
    messages = extract_text(context)
    return messages[1].content


def _message_prompt(custom_instructions: str = '') -> str:
    """Render extract_message prompt and return the user message content."""
    context = {
        'entity_types': '[]',
        'previous_episodes': [],
        'episode_content': 'test content',
        'custom_extraction_instructions': custom_instructions,
    }
    messages = extract_message(context)
    return messages[1].content


class TestTextPromptNamingGuidance:
    """extract_text prompt should guide LLM to use type definitions for naming."""

    def test_guides_toward_specific_entity(self):
        """Prompt should tell LLM to identify the specific entity, not copy context."""
        text = _text_prompt().lower()
        assert 'specific entity' in text or 'identify the entity' in text

    def test_references_type_definition_for_naming(self):
        """Prompt should tell LLM to use entity type definition for naming."""
        text = _text_prompt().lower()
        assert 'type definition' in text or 'type description' in text or 'entity type' in text

    def test_discourages_contextual_noise(self):
        """Prompt should tell LLM to strip contextual noise from names."""
        text = _text_prompt().lower()
        has_context = 'context' in text or 'description' in text or 'surrounding' in text
        assert has_context

    def test_no_verbose_encouragement(self):
        """Should NOT tell LLM to 'use full names' — that encourages verbose descriptions."""
        text = _text_prompt().lower()
        assert 'using full names' not in text


class TestMessagePromptNamingGuidance:
    """extract_message prompt should guide LLM to use type definitions for naming."""

    def test_guides_toward_specific_entity(self):
        """Prompt should tell LLM to identify the specific entity, not copy context."""
        text = _message_prompt().lower()
        assert 'specific entity' in text or 'identify the entity' in text

    def test_references_type_definition_for_naming(self):
        """Prompt should tell LLM to use entity type definition for naming."""
        text = _message_prompt().lower()
        assert 'type definition' in text or 'type description' in text or 'entity type' in text

    def test_discourages_contextual_noise(self):
        """Prompt should tell LLM to strip contextual noise from names."""
        text = _message_prompt().lower()
        has_context = 'context' in text or 'description' in text or 'surrounding' in text
        assert has_context

    def test_no_verbose_encouragement(self):
        """Should NOT tell LLM to 'use full names' — that encourages verbose descriptions."""
        text = _message_prompt().lower()
        assert 'use full names' not in text
