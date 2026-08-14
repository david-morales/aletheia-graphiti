"""P2: the announced `prompts` capability must be backed by a real prompt.

At the 2026-07-28 era the SDK announces `prompts` iff a `prompts/list` handler is
registered, and `MCPServer` registers one unconditionally
(`mcp/server/lowlevel/server.py:583`, `mcp/server/mcpserver/server.py:213`). So
this connector has announced `prompts` — with `prompts.listChanged: true`,
derived from the unconditionally-served `subscriptions/listen` — while serving an
EMPTY prompt list. Declared-and-empty is the ADR-019 R7 defect the P3 comment
block names and defers here.

This module is the guard for the fix: one prompt, `investigate`, whose LIST entry
is static and domain-agnostic and whose MESSAGES are rendered from the live
domain profile at `prompts/get` time. The lazy render is the whole staleness
story — there is no cached text to go stale, so P2 needs none of P3's
fingerprint-and-notify machinery.
"""

from __future__ import annotations

import re

import pytest

import graphiti_mcp_server as srv
import prompt_surface
from domain_profile import DomainProfile, EdgeTypeInfo, EntityTypeInfo
from tool_annotations import TOOL_ANNOTATIONS

# Selected by the CI `contract` job (.github/workflows/mcp-server-tests.yml):
# these guards need no database and no API key, so they gate every change.
pytestmark = pytest.mark.contract


def _profile() -> DomainProfile:
    """A profile whose own vocabulary cannot be mistaken for domain leakage."""
    return DomainProfile(
        group_id='prompt_graph',
        entity_types={
            'Widget': EntityTypeInfo('Widget', 12, 'A widget', ['W-1']),
            'Gadget': EntityTypeInfo('Gadget', 4, 'A gadget', ['G-1']),
        },
        edge_types={'USES': EdgeTypeInfo('USES', 6, 'Widget uses Gadget', 'Widget -> Gadget')},
        time_range=None,
    )


class _Service:
    """The one field the prompt reads off the live service."""

    def __init__(self, profile: DomainProfile | None):
        self.domain_profile = profile


@pytest.fixture
def live_profile(monkeypatch):
    """Install a live service carrying `profile`, the way startup would."""

    def _install(profile: DomainProfile | None) -> _Service:
        service = _Service(profile)
        monkeypatch.setattr(srv, 'graphiti_service', service)
        return service

    return _install


async def _render(topic: str = 'a topic') -> str:
    """The rendered `prompts/get` text, as a client would read it."""
    result = await srv.mcp.get_prompt(prompt_surface.INVESTIGATE_PROMPT_NAME, {'topic': topic})
    return '\n'.join(
        m['content']['text'] for m in result.messages if m['content']['type'] == 'text'
    )


# ---------------------------------------------------------------------------
# The LIST entry: exactly one prompt, static, domain-agnostic
# ---------------------------------------------------------------------------


class TestTheAnnouncedListIsBackedByARealPrompt:
    @pytest.mark.asyncio
    async def test_exactly_one_prompt_is_served(self):
        """Declared-and-empty was the defect; declared-and-two would be scope creep."""
        prompts = await srv.mcp.list_prompts()
        assert [p.name for p in prompts] == ['investigate']

    @pytest.mark.asyncio
    async def test_topic_is_a_required_argument(self):
        (prompt,) = await srv.mcp.list_prompts()
        by_name = {a.name: a for a in prompt.arguments or []}
        assert 'topic' in by_name, f'arguments: {sorted(by_name)}'
        assert by_name['topic'].required is True
        assert by_name['topic'].description, 'an argument with no description is a riddle'

    @pytest.mark.asyncio
    async def test_the_list_entry_describes_itself(self):
        (prompt,) = await srv.mcp.list_prompts()
        assert prompt.description
        assert prompt.title

    @pytest.mark.asyncio
    async def test_the_list_entry_carries_no_profile_data(self, live_profile):
        """The LIST is static: it must not move when the graph does.

        A description rendered from the profile would make `prompts/list` a
        surface that goes stale, and P2 would inherit the whole P3 problem for
        no benefit.
        """
        live_profile(_profile())
        before = [p.model_dump() for p in await srv.mcp.list_prompts()]
        live_profile(
            DomainProfile(
                group_id='totally_different',
                entity_types={'Sprocket': EntityTypeInfo('Sprocket', 99, '', [])},
            )
        )
        assert [p.model_dump() for p in await srv.mcp.list_prompts()] == before


class TestTheCapabilityIsNowHonest:
    """The bit the SDK derives, and what now backs it."""

    def test_modern_announces_prompts_with_list_changed(self):
        caps = srv.mcp._lowlevel_server.get_capabilities(protocol_version='2026-07-28')
        assert caps.prompts is not None
        assert caps.prompts.list_changed is True

    @pytest.mark.asyncio
    async def test_the_announcement_is_no_longer_empty(self):
        """The whole point of P2: the capability now has something behind it."""
        caps = srv.mcp._lowlevel_server.get_capabilities(protocol_version='2026-07-28')
        assert caps.prompts is not None
        assert await srv.mcp.list_prompts(), (
            'prompts announced with an empty list — the ADR-019 R7 defect'
        )


# ---------------------------------------------------------------------------
# The RENDER: live, lazy, and never stale
# ---------------------------------------------------------------------------


class TestTheRenderReadsTheLiveProfile:
    @pytest.mark.asyncio
    async def test_the_census_is_embedded(self, live_profile):
        live_profile(_profile())
        text = await _render()
        assert 'prompt_graph' in text, 'the group_id is not in the rendered prompt'
        assert 'Widget' in text
        assert '12' in text, 'entity counts are not in the rendered prompt'
        assert 'USES' in text

    @pytest.mark.asyncio
    async def test_the_topic_argument_reaches_the_messages(self, live_profile):
        live_profile(_profile())
        assert 'sprocket alignment' in await _render('sprocket alignment')

    @pytest.mark.asyncio
    async def test_a_second_render_reflects_a_changed_profile(self, live_profile):
        """The lazy-render guarantee, which is what removes staleness.

        Render, move the graph under it, render again. A cached template would
        pass the first assertion and fail this one — and would need P3's
        fingerprint-and-notify wiring to ever be correct.
        """
        live_profile(_profile())
        first = await _render()
        assert 'Sprocket' not in first

        live_profile(
            DomainProfile(
                group_id='moved_graph',
                entity_types={'Sprocket': EntityTypeInfo('Sprocket', 77, 'A sprocket', [])},
                edge_types={'FITS': EdgeTypeInfo('FITS', 3, 'fits', '')},
            )
        )
        second = await _render()
        assert 'Sprocket' in second
        assert 'moved_graph' in second
        assert 'Widget' not in second, 'the render served a cached census'


class TestTheNoProfilePathRendersRatherThanCrashes:
    """Episode-only / pre-census / degraded startup: `domain_profile` is None.

    `instructions` degrades to a marked fallback there rather than failing, and
    a prompt that raises at `prompts/get` would be a worse outcome than one that
    admits it has not censused yet.
    """

    @pytest.mark.asyncio
    async def test_no_profile_on_the_service(self, live_profile):
        live_profile(None)
        text = await _render()
        assert text.strip(), 'the no-profile path rendered nothing at all'
        assert prompt_surface.NO_CENSUS_MARKER in text

    @pytest.mark.asyncio
    async def test_no_service_at_all(self, monkeypatch):
        """Before `initialize_server` assigns it, the global is None."""
        monkeypatch.setattr(srv, 'graphiti_service', None)
        text = await _render()
        assert text.strip()
        assert prompt_surface.NO_CENSUS_MARKER in text

    @pytest.mark.asyncio
    async def test_an_empty_but_real_profile_still_names_its_partition(self, live_profile):
        """A censused-but-empty graph is not the same as an uncensused one."""
        live_profile(DomainProfile(group_id='empty_graph'))
        text = await _render()
        assert 'empty_graph' in text


# ---------------------------------------------------------------------------
# The CONTRACT: canonical names only, no domain, no unresolvable references
# ---------------------------------------------------------------------------


class TestTheWorkflowUsesTheRealSurface:
    """A prompt naming a tool this server does not serve is worse than no prompt."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'tool', ['get_schema', 'search', 'explore_entity', 'graph_query']
    )
    async def test_the_workflow_names_its_step(self, live_profile, tool):
        live_profile(_profile())
        assert tool in await _render()

    @pytest.mark.asyncio
    @pytest.mark.parametrize('legacy', ['run_cypher', 'explore_node', 'profile_graph'])
    async def test_no_pre_p1_tool_name_survives(self, live_profile, legacy):
        """P1 renamed these on the fork with no aliases. A prompt is a place a
        stale name can hide, because nothing else validates its text."""
        live_profile(_profile())
        assert legacy not in await _render()

    @pytest.mark.asyncio
    async def test_every_tool_name_it_mentions_is_actually_served(self, live_profile):
        live_profile(_profile())
        text = await _render()
        mentioned = {
            name for name in re.findall(r'\b[a-z_]{4,}\b', text) if name in TOOL_ANNOTATIONS
        }
        assert mentioned, 'the prompt names no tool at all'
        assert mentioned <= set(TOOL_ANNOTATIONS)

    @pytest.mark.asyncio
    async def test_it_defers_the_dialect_to_get_schema(self, live_profile):
        """ADR-019 R6: the dialect is DATA the flavour owns and `get_schema`
        serves. Inlining dialect text here would fork it per-backend in a place
        no flavour test looks."""
        live_profile(_profile())
        assert 'dialect_reference' in await _render()


class TestThePromptIsDomainAgnostic:
    """Hard Rule 2 / ADR-003, at the one surface that renders free prose.

    `test_no_domain_leakage.py` scans every file under `src/` statically, so the
    module is already covered there. This is the WIRE scan: what a client reads
    after the profile has been substituted in.
    """

    @pytest.mark.asyncio
    async def test_the_rendered_prompt_names_no_foreign_domain(self, live_profile):
        from test_no_domain_leakage import _BANNED_RE

        live_profile(_profile())
        hits = sorted({m.group(0) for m in _BANNED_RE.finditer(await _render())})
        assert not hits, f'the investigate prompt serves a foreign-domain term: {hits}'

    @pytest.mark.asyncio
    async def test_the_no_profile_render_names_no_foreign_domain(self, live_profile):
        """The degraded text is unmasked prose — exactly where a leak survives."""
        from test_no_domain_leakage import _BANNED_RE

        live_profile(None)
        hits = sorted({m.group(0) for m in _BANNED_RE.finditer(await _render())})
        assert not hits, f'the no-profile render serves a foreign-domain term: {hits}'

    @pytest.mark.asyncio
    async def test_the_rendered_prompt_cites_no_repo_relative_document(self, live_profile):
        """A-D10: a wire consumer cannot resolve `docs/whatever.md`."""
        from test_no_domain_leakage import _REPO_DOC_RE

        live_profile(_profile())
        hits = sorted({m.group(0) for m in _REPO_DOC_RE.finditer(await _render())})
        assert not hits, f'the investigate prompt cites an unresolvable repo path: {hits}'

    def test_the_static_template_carries_no_profile_vocabulary(self):
        """The template is the thing that must be domain-free BEFORE substitution.

        Rendered against an empty profile there is no domain data to hide behind,
        so what remains is the template itself.
        """
        from test_no_domain_leakage import _BANNED_RE

        template = prompt_surface.build_investigate_prompt('T', None)
        assert not _BANNED_RE.search(template)
