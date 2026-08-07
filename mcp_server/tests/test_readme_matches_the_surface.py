"""A-D5: the README documented a tool surface that does not exist.

It listed six tools the server never serves (`add_triplet`, `search_nodes`,
`search_memory_facts`, `summarize_saga`, `get_episode_entities`,
`get_entity_edge`) and documented NONE of the ten every Aletheia consumer depends
on. It also described `add_memory` parameters absent from the served input schema.
That is stale upstream text which survived the fork — and it survived because
nothing checked it.

Documentation rot is only preventable mechanically, so this is the check.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from tool_annotations import TOOL_ANNOTATIONS

README = pathlib.Path(__file__).parent.parent / 'README.md'
TEXT = README.read_text()

# The tool tables live between the "Available Tools" heading and the section that
# follows them.
_TOOLS_SECTION = TEXT.split('## Available Tools', 1)[1].split('\n## ', 1)[0]

# Tools the upstream README invented. None has ever been served by this fork.
PHANTOM_TOOLS = (
    'add_triplet',
    'search_nodes',
    'search_memory_facts',
    'summarize_saga',
    'get_episode_entities',
    'get_entity_edge',
)


def _documented_tools() -> set[str]:
    """Tool names in the section's tables, as `| \\`name\\` | ...` cells."""
    return set(re.findall(r'^\|\s*`(\w+)`\s*\|', _TOOLS_SECTION, re.MULTILINE))


@pytest.mark.parametrize('tool_name', sorted(TOOL_ANNOTATIONS))
def test_every_served_tool_is_documented(tool_name):
    assert tool_name in _documented_tools(), f'{tool_name} is served but undocumented'


def test_no_documented_tool_is_unserved():
    """The half of the rot that costs a consumer a wasted call."""
    extra = sorted(_documented_tools() - set(TOOL_ANNOTATIONS))
    assert not extra, f'documented but not served: {extra}'


@pytest.mark.parametrize('phantom', PHANTOM_TOOLS)
def test_the_upstream_phantoms_are_gone(phantom):
    assert phantom not in TEXT, f'{phantom} does not exist on this server'


def test_the_destructive_tools_are_documented_as_destructive():
    section = TEXT.split('### Destructive', 1)[1].split('\n## ', 1)[0]
    for name in ('build_communities', 'delete_entity_edge', 'delete_episode', 'clear_graph'):
        assert f'`{name}`' in section, f'{name} is not in the destructive table'
    assert 'cannot be undone' in section.lower()
    assert 'irreversible' in section.lower()


def test_the_error_contract_deviation_is_documented_as_a_decision():
    """ADR-015 R4 differs from the spec's `isError: true` convention. A reader
    must be able to tell that from the README, or they will "fix" it."""
    section = TEXT.split('## Error contract', 1)[1].split('\n## ', 1)[0]
    assert 'deliberate' in section.lower()
    assert 'ADR-015 R4' in section
    assert 'isError' in section
    for claim in ('always a string', 'absent on success', 'hint'):
        assert claim in section, f'the error contract does not state: {claim}'


def test_the_error_contract_names_the_channel_and_the_portable_check():
    """The flattened models emit an explicit `error: null` in structuredContent on
    success, so key presence is NOT a portable check.

    Saying "absent on success" without naming the channel tells a
    structured-channel client — Claude Desktop, MCP Inspector, the TS SDK's
    structured path — to read every successful call as a failure.
    """
    section = TEXT.split('## Error contract', 1)[1].split('\n## ', 1)[0]
    assert 'text content' in section
    assert 'structuredContent' in section
    assert '"error": null' in section
    assert 'result.get("error")' in section
    assert 'NOT a portable check' in section


def test_the_deprecated_transport_is_labelled():
    """`sse` still works; the spec deprecated HTTP+SSE. Say so where it is chosen."""
    section = TEXT.split('### Available Command-Line Arguments', 1)[1].split('\n### ', 1)[0]
    assert '`sse`' in section
    assert 'deprecated' in section.lower()


def test_the_readme_does_not_invent_add_memory_parameters():
    """It documented six parameters the served input schema does not have."""
    for invented in (
        'reference_time',
        'excluded_entity_types',
        'custom_extraction_instructions',
        'previous_episode_uuids',
        'update_communities',
    ):
        assert invented not in TEXT, f'README documents a nonexistent parameter: {invented}'


def test_the_readme_tells_maintainers_to_bump_the_version():
    """M4: the announced build is only true if someone bumps pyproject at release.

    Two tags shipped without it, so the instruction has to live where a releaser
    reads it, not only in a test failure message.
    """
    section = TEXT.split('## Releasing', 1)[1].split('\n## ', 1)[0]
    assert 'pyproject.toml' in section
    assert 'get_status' in section
    assert 'SemVer' in section or 'semver' in section
