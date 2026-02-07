"""Tests for the attributes field on ExtractedEntity and prompt coverage."""

from graphiti_core.prompts.extract_nodes import ExtractedEntity, extract_message, extract_json, extract_text


# ---------------------------------------------------------------------------
# Model-level tests
# ---------------------------------------------------------------------------


def test_attributes_field_exists():
    """Create an entity with attributes and verify they are stored."""
    entity = ExtractedEntity(
        name="Acme Corp",
        entity_type_id=1,
        attributes={"jurisdiction": "US", "registration_number": "12345"},
    )
    assert entity.attributes == {"jurisdiction": "US", "registration_number": "12345"}


def test_attributes_default_empty():
    """Create an entity without specifying attributes; default must be {}."""
    entity = ExtractedEntity(name="Jane Doe", entity_type_id=2)
    assert entity.attributes == {}


def test_attributes_serialization():
    """Verify attributes survive a model_dump / model_validate round-trip."""
    original = ExtractedEntity(
        name="Treasury Dept",
        entity_type_id=3,
        attributes={"country": "UK", "amount": "1000000"},
    )
    dumped = original.model_dump()
    restored = ExtractedEntity.model_validate(dumped)
    assert restored.attributes == original.attributes
    assert restored.name == original.name
    assert restored.entity_type_id == original.entity_type_id


# ---------------------------------------------------------------------------
# Prompt-level tests
# ---------------------------------------------------------------------------

def _minimal_context(**overrides):
    """Return the minimum context dict needed by extraction prompts."""
    base = {
        "entity_types": "1: Person\n2: Organization",
        "previous_episodes": [],
        "episode_content": "Alice works at Acme Corp.",
        "source_description": "Test data",
        "custom_extraction_instructions": "",
    }
    base.update(overrides)
    return base


def test_prompt_mentions_attributes():
    """All three extraction prompts must mention 'attributes'."""
    ctx = _minimal_context()
    for prompt_fn, label in [
        (extract_message, "extract_message"),
        (extract_json, "extract_json"),
        (extract_text, "extract_text"),
    ]:
        messages = prompt_fn(ctx)
        combined = " ".join(m.content for m in messages)
        assert "attributes" in combined.lower(), (
            f"{label} prompt does not mention 'attributes'"
        )
