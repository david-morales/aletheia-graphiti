"""P3 — `listChanged` push freshness: what we announce, we emit.

Three measurements sit behind this module, all in
`docs/plans/2026-08-13-mcp-p3-listchanged-design.md` (aletheia):

1. At the 2026-07-28 era the SDK derives `tools.listChanged`,
   `prompts.listChanged`, `resources.listChanged` and `resources.subscribe`
   from ONE condition — whether `subscriptions/listen` is served — and
   `MCPServer` registers that handler unconditionally. So this connector has
   announced all four as `true` since the modern-era migration.
2. Nothing published to the bus. Four declared-and-never-emitted capability
   bits: exactly the ADR-019 R7 defect P2 names.
3. Nothing could have. The startup census completes BEFORE the transport binds,
   and the tool/resource lists were frozen for the process lifetime thereafter,
   while the six `_schema_dirty` sites made every rendered description stale
   without ever re-rendering it.

So the fix is not a flag. It is the missing runtime event: re-census on a
debounce after the graph changes, re-render the surface, and announce it — and
announce it ONLY when the surface actually moved, which is what most of the
assertions below are about.
"""

from __future__ import annotations

import asyncio

import pytest
from mcp.shared.subscriptions import ResourcesListChanged, ToolsListChanged

import graphiti_mcp_server as srv
from domain_profile import DomainProfile, EdgeTypeInfo, EntityTypeInfo
from services.queue_service import QueueService


@pytest.fixture(autouse=True)
def _restore_server_globals():
    """Put the module-global server and refresh state back as we found them.

    Same reasoning as `test_canonical_tool_names.py`: `register_dynamic_tools` /
    `register_fallback_tools` mutate `srv.mcp` in place, and a leaked
    registration silently satisfies another module's surface guard.
    """
    mcp = srv.mcp
    tools = dict(mcp._tool_manager._tools)
    resources = dict(mcp._resource_manager._resources)
    instructions = mcp._lowlevel_server.instructions
    service = srv.graphiti_service
    try:
        yield
    finally:
        mcp._tool_manager._tools.clear()
        mcp._tool_manager._tools.update(tools)
        mcp._resource_manager._resources.clear()
        mcp._resource_manager._resources.update(resources)
        mcp._lowlevel_server.instructions = instructions
        srv.graphiti_service = service
        srv._surface_refresh_marks = 0
        task = srv._surface_refresh_task
        if task is not None and not task.done():
            task.cancel()
        srv._surface_refresh_task = None


class _Bus:
    """Stand-in `SubscriptionBus` that records what was published."""

    def __init__(self):
        self.published = []

    async def publish(self, event):
        self.published.append(event)

    def subscribe(self, listener):  # pragma: no cover - never exercised here
        return lambda: None


class _Service:
    """Minimal `GraphitiService` stand-in for the refresh path."""

    def __init__(self, *, census=None, fails=False):
        self.flavour = None
        self.ontology_client = None
        self.domain_profile = None
        self._schema_dirty = False
        self._census = census
        self._fails = fails
        self.censuses = 0

    async def get_client(self):
        return object()


def _profile(group_id: str = 'p3_graph', *, label: str = 'Widget', count: int = 3):
    return DomainProfile(
        group_id=group_id,
        entity_types={label: EntityTypeInfo(label, count, f'A {label}', [f'{label}-1'])},
        edge_types={'USES': EdgeTypeInfo('USES', 1, 'uses', f'{label} -> {label}')},
        time_range=None,
    )


@pytest.fixture
def wired(monkeypatch):
    """A server wired for refresh: fake service, fake bus, scripted census."""
    bus = _Bus()
    monkeypatch.setattr(srv.mcp, '_subscriptions', bus, raising=False)
    service = _Service()
    monkeypatch.setattr(srv, 'graphiti_service', service)

    class _Cfg:
        class graphiti:
            group_id = 'p3_graph'
            ontology_graph = None

    monkeypatch.setattr(srv, 'config', _Cfg, raising=False)

    profiles = {'next': _profile()}

    async def _census(client, group_id, ontology_client=None, flavour=None):
        service.censuses += 1
        if isinstance(profiles['next'], Exception):
            raise profiles['next']
        return profiles['next']

    monkeypatch.setattr(srv, 'build_domain_profile', _census)
    return bus, service, profiles


class TestTheEmissionIsHonest:
    """ADR-019 R7: what is announced must be what is emitted — and *only* what
    actually changed. A notification for a surface that did not move is the same
    lie in the other direction."""

    @pytest.mark.asyncio
    async def test_a_re_census_that_changes_descriptions_announces_tools(self, wired):
        bus, _service, profiles = wired
        srv.register_dynamic_tools(_profile(label='Widget'))

        profiles['next'] = _profile(label='Gadget')
        assert await srv.refresh_domain_surface('test') is True

        assert ToolsListChanged() in bus.published, (
            'the nine dynamic tools were re-rendered from a different profile, '
            'so their announced descriptions changed and consumers must be told'
        )

    @pytest.mark.asyncio
    async def test_recovering_from_degraded_announces_resources(self, wired):
        """The one case where the resource LIST — not just its bodies — moves.

        `register_fallback_tools` prunes the three profile-rendered resources; a
        successful refresh puts them back, taking `resources/list` from 1 to 4.
        """
        bus, _service, _profiles = wired
        srv.register_fallback_tools(reason='census failed at boot')
        before = set(srv.mcp._resource_manager._resources)

        assert await srv.refresh_domain_surface('test') is True

        after = set(srv.mcp._resource_manager._resources)
        assert after > before, (before, after)
        assert ResourcesListChanged() in bus.published

    @pytest.mark.asyncio
    async def test_a_no_op_re_census_announces_nothing(self, wired):
        """The honesty assertion in the other direction.

        Re-rendering the SAME profile changes no announced entry. Publishing
        anyway would train every consumer to ignore the channel.
        """
        bus, _service, _profiles = wired
        srv.register_dynamic_tools(_profile())
        srv.register_resources(_profile())

        assert await srv.refresh_domain_surface('test') is True

        assert bus.published == [], bus.published

    @pytest.mark.asyncio
    async def test_a_body_only_change_does_not_announce_a_resource_list_change(
        self, wired
    ):
        """`resources/list_changed` is about the LIST, not the bodies.

        A re-census re-renders `domain_summary` & co. with new text, but the
        announced entries (uri, name, description, mimeType) are identical.
        Saying `list_changed` there is P4's `resources/updated` wearing the wrong
        name — and P4 is not this wave.
        """
        bus, _service, profiles = wired
        srv.register_dynamic_tools(_profile(label='Widget'))
        srv.register_resources(_profile(label='Widget'))
        uris_before = set(srv.mcp._resource_manager._resources)

        profiles['next'] = _profile(label='Widget', count=99)
        await srv.refresh_domain_surface('test')

        assert set(srv.mcp._resource_manager._resources) == uris_before
        assert ResourcesListChanged() not in bus.published, bus.published


class TestAFailedRefreshKeepsTheSurface:
    """A transient census failure on a LIVE connector must not degrade an
    announcement that is currently correct. The startup path's failure handler
    (`register_fallback_tools`) is right for startup and wrong here: it would
    turn one timed-out query into a connector that tells every consumer its
    guidance is not derived from this graph."""

    @pytest.mark.asyncio
    async def test_the_served_tools_survive_a_failed_re_census(self, wired):
        bus, _service, profiles = wired
        srv.register_dynamic_tools(_profile(label='Widget'))
        srv.register_resources(_profile(label='Widget'))
        tools_before = {
            n: t.description for n, t in srv.mcp._tool_manager._tools.items()
        }
        resources_before = set(srv.mcp._resource_manager._resources)

        profiles['next'] = RuntimeError('graph unreachable')
        assert await srv.refresh_domain_surface('test') is False

        assert {
            n: t.description for n, t in srv.mcp._tool_manager._tools.items()
        } == tools_before
        assert set(srv.mcp._resource_manager._resources) == resources_before
        assert bus.published == [], 'a failed refresh announced a change'


class TestTheDebounceCoalesces:
    """One census per window, and never a lost mark."""

    @pytest.mark.asyncio
    async def test_a_burst_of_mutations_produces_one_census(self, wired, monkeypatch):
        _bus, service, _profiles = wired
        monkeypatch.setenv('GRAPHITI_SURFACE_REFRESH_DEBOUNCE_SECONDS', '0.05')

        for _ in range(50):
            srv.mark_schema_dirty()

        assert service.censuses == 0, 'the refresh ran before the window closed'
        await asyncio.sleep(0.25)
        assert service.censuses == 1, service.censuses
        assert service._schema_dirty is True

    @pytest.mark.asyncio
    async def test_a_mark_during_the_census_is_not_lost(self, wired, monkeypatch):
        """The window is a COALESCING window, not a quiescence timer: it always
        fires, so staleness is bounded. A mark landing mid-census therefore has
        to earn its own next window rather than being folded into the one that
        already started."""
        _bus, service, _profiles = wired
        monkeypatch.setenv('GRAPHITI_SURFACE_REFRESH_DEBOUNCE_SECONDS', '0.05')

        srv.mark_schema_dirty()
        await asyncio.sleep(0.08)
        assert service.censuses == 1
        srv.mark_schema_dirty()
        await asyncio.sleep(0.15)
        assert service.censuses == 2, service.censuses

    @pytest.mark.asyncio
    async def test_zero_disables_the_scheduler(self, wired, monkeypatch):
        """The escape hatch for a graph too large to re-census on a timer: the
        surface then behaves exactly as it did before this wave."""
        _bus, service, _profiles = wired
        monkeypatch.setenv('GRAPHITI_SURFACE_REFRESH_DEBOUNCE_SECONDS', '0')

        srv.mark_schema_dirty()
        await asyncio.sleep(0.1)

        assert service.censuses == 0
        assert service._schema_dirty is True, (
            'disabling the re-census must not disable the schema cache flag — '
            'get_schema still has to notice'
        )

    @pytest.mark.parametrize('raw', ['nonsense', '-5', 'inf', 'nan', ''])
    def test_an_unusable_window_falls_back_to_the_default(self, monkeypatch, raw):
        monkeypatch.setenv('GRAPHITI_SURFACE_REFRESH_DEBOUNCE_SECONDS', raw)
        assert (
            srv.surface_refresh_debounce_seconds()
            == srv.DEFAULT_SURFACE_REFRESH_DEBOUNCE_SECONDS
        )


class TestEveryMutationSiteGoesThroughTheSeam:
    """Six copies of a one-line invariant is how the seventh mutation site ships
    without it. `mark_schema_dirty()` is the only spelling; a raw assignment is
    a site that will never schedule a re-census."""

    def test_no_raw_schema_dirty_assignment_survives(self):
        import inspect

        # The seam itself is the one permitted assignment; the constructor's
        # `self._schema_dirty` initialiser is not a mutation site.
        allowed = set(inspect.getsource(srv.mark_schema_dirty).splitlines())
        offenders = [
            line.strip()
            for line in inspect.getsource(srv).splitlines()
            if '_schema_dirty = True' in line
            and 'self._schema_dirty' not in line
            and line not in allowed
        ]
        assert offenders == [], offenders

    def test_the_seam_sets_the_flag_and_schedules(self, wired, monkeypatch):
        _bus, service, _profiles = wired
        monkeypatch.setenv('GRAPHITI_SURFACE_REFRESH_DEBOUNCE_SECONDS', '30')
        service._schema_dirty = False

        srv.mark_schema_dirty()

        assert service._schema_dirty is True

    def test_the_seam_survives_having_no_running_loop(self, monkeypatch):
        """`clear_graph` and friends are also reachable from sync test/CLI
        contexts. No loop means nothing to schedule onto — it must not raise."""
        service = _Service()
        monkeypatch.setattr(srv, 'graphiti_service', service)
        srv.mark_schema_dirty()
        assert service._schema_dirty is True


class TestTheQueuedIngestPathAnnounces:
    """H1 — the review's ship-blocker, and the case that matters most.

    `add_memory` without `sync=True` RETURNS as soon as the episode is queued;
    the graph only moves when the worker's LLM extraction lands, which routinely
    takes longer than any freshness window. Marking at ENQUEUE therefore ran the
    single census against the PRE-ingest graph, found nothing changed, published
    nothing, and left the debounce loop with `marks == seen` — so it exited and
    nothing ever marked again. The DEFAULT ingest path never announced at all,
    while bulk/sync/delete/clear_graph did, which is exactly the subset the
    original wire proof exercised.
    """

    @pytest.mark.asyncio
    async def test_an_episode_landing_after_the_window_still_announces(
        self, wired, monkeypatch
    ):
        """The scenario, end to end: enqueue, let the window close with nothing
        to see, and land the episode afterwards."""
        bus, service, profiles = wired
        monkeypatch.setenv('GRAPHITI_SURFACE_REFRESH_DEBOUNCE_SECONDS', '0.05')
        srv.register_dynamic_tools(_profile(label='Widget'))

        queue = QueueService(on_episode_processed=srv._on_episode_processed)
        await queue.initialize(object())

        landed = asyncio.Event()

        async def slow_extraction():
            # Outlives the debounce window, as real extraction does.
            await asyncio.sleep(0.2)
            profiles['next'] = _profile(label='Gadget')  # the graph moved
            landed.set()

        await queue.add_episode_task('g', slow_extraction)
        await landed.wait()
        await asyncio.sleep(0.25)  # one window after the episode landed

        assert service.censuses >= 1, (
            'no census ran after the episode landed — the queued ingest path '
            'never announces (H1)'
        )
        assert ToolsListChanged() in bus.published, bus.published

    @pytest.mark.asyncio
    async def test_the_hook_fires_even_when_the_episode_fails(self):
        """A failed extraction can still have written part of the graph, and the
        worker must survive either way."""
        seen: list[str] = []
        queue = QueueService(on_episode_processed=seen.append)
        await queue.initialize(object())

        async def boom():
            raise RuntimeError('extraction failed')

        await queue.add_episode_task('g', boom)
        for _ in range(100):
            if seen:
                break
            await asyncio.sleep(0.01)
        assert seen == ['g'], seen

    @pytest.mark.asyncio
    async def test_a_raising_hook_does_not_stop_the_queue_worker(self):
        """The hook runs in the worker's `finally`; an escape would be caught by
        the outer handler and end ingestion for that group in silence."""
        calls = 0

        def raises(_group_id):
            nonlocal calls
            calls += 1
            raise RuntimeError('hook is broken')

        queue = QueueService(on_episode_processed=raises)
        await queue.initialize(object())

        done = asyncio.Event()

        async def first():
            pass

        async def second():
            done.set()

        await queue.add_episode_task('g', first)
        await queue.add_episode_task('g', second)
        await asyncio.wait_for(done.wait(), timeout=2)
        assert calls >= 1

    def test_enqueue_no_longer_marks(self):
        """The half of the fix that is a DELETION: marking at enqueue is a claim
        the graph changed when it has not."""
        import inspect

        src = inspect.getsource(srv.add_memory)
        queued = src.split('await queue_service.add_episode(')[1]
        assert 'mark_schema_dirty()' not in queued, (
            'the enqueue path still marks the surface stale before the episode '
            'has landed (H1)'
        )

    def test_the_queue_is_constructed_with_the_hook(self):
        """A hook nobody wires is the same bug one layer out."""
        import inspect

        src = inspect.getsource(srv.initialize_server)
        assert 'QueueService(on_episode_processed=_on_episode_processed)' in src


class TestTheStartupPathDoesNotAnnounce:
    """No client can be listening before the transport binds
    (`initialize_server()` finishes the census at graphiti_mcp_server.py:2783;
    `run_streamable_http_async` is only reached later). A notification nobody can
    receive is the dishonesty this item exists to remove."""

    @pytest.mark.asyncio
    async def test_building_the_surface_at_startup_publishes_nothing(self, wired):
        bus, _service, _profiles = wired
        await srv._build_and_register_domain_surface()
        assert bus.published == [], bus.published


class TestTheCapabilityIsAnnouncedAtTheModernEra:
    """Documents the SDK derivation the design measured, so an SDK change that
    moves it is caught here rather than on a droplet."""

    def test_modern_announces_list_changed_for_tools_and_resources(self):
        caps = srv.mcp._lowlevel_server.get_capabilities(
            protocol_version='2026-07-28'
        )
        assert caps.tools.list_changed is True
        assert caps.resources.list_changed is True

    def test_the_listen_method_is_what_drives_it(self):
        assert (
            'subscriptions/listen' in srv.mcp._lowlevel_server._request_handlers
        ), (
            'MCPServer stopped registering ListenHandler — the announcement '
            'above would silently flip to false and the push channel would '
            'vanish with it'
        )
