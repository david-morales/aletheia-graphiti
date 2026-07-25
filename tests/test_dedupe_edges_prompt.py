"""Tests for edge deduplication prompt guidelines."""

from graphiti_core.prompts.dedupe_edges import resolve_edge


def _get_prompt_text() -> str:
    """Render the resolve_edge prompt with dummy context and return the user message content."""
    context = {
        'existing_edges': '[]',
        'edge_invalidation_candidates': '[]',
        'new_edge': '{}',
    }
    messages = resolve_edge(context)
    # The user message is the second message (index 1)
    return messages[1].content


def test_prompt_mentions_voice_equivalence():
    text = _get_prompt_text()
    assert 'active' in text.lower(), "Prompt should mention active voice"
    assert 'passive' in text.lower(), "Prompt should mention passive voice"


def test_prompt_mentions_numeric_format():
    text = _get_prompt_text()
    assert 'numeric format' in text.lower(), "Prompt should mention numeric format variations"


def test_prompt_mentions_alias_matching():
    text = _get_prompt_text()
    has_alias = 'alias' in text.lower()
    has_alternate = 'alternate name' in text.lower()
    assert has_alias or has_alternate, "Prompt should mention aliases or alternate names"


def test_existing_guidelines_preserved():
    text = _get_prompt_text()
    # Guideline 1: numeric key differences should not be marked as duplicates.
    # v0.29.2 rephrased the conclusion ("Do not mark these facts as duplicates"
    # -> "NEVER mark facts as duplicates if they have key differences"); the
    # intent is preserved.
    assert 'key differences' in text, "Original guideline about key differences must be preserved"
    assert 'numeric values' in text, "Original guideline about numeric values must be preserved"
    assert 'never mark facts as duplicates' in text.lower(), (
        "Guideline that facts with key differences are not duplicates must be preserved"
    )
