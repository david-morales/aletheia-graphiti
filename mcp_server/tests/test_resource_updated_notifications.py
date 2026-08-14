"""P4 — honest `resources.subscribe`: emit `resources/updated` when a body moves.

At the 2026-07-28 era the SDK derives all four freshness bits — `tools`,
`prompts`, `resources.listChanged` and `resources.subscribe` — from ONE
condition, whether `subscriptions/listen` is served, and `MCPServer` wires that
handler unconditionally. After P2 and P3, `resources.subscribe` was the last of
the four announced with nothing behind it: a client could subscribe to
`graphiti://domain_summary` and never hear that a re-census had rewritten it.

The bit cannot be un-announced at this era, so the only honest fix is the
missing runtime event. This module is the guard for it.

WHAT DECIDES AN EVENT is movement in the SERVED BODY, not the fact that a
refresh ran. The server keeps a per-URI fingerprint of what it last SERVED at
each URI; after a refresh settles — rebuilt OR restored — every URI in the
CURRENT list whose body differs from its last-served fingerprint gets one
`ResourceUpdated`, and nothing else does. That is the same level-trigger honesty
P3 established for the list events: a byte-identical re-render announces
nothing, because a channel that fires on no-ops is a channel consumers learn to
ignore.

The map keys LAST-SERVED state, not previous-refresh state, which is what makes
the degraded path come out right: `register_fallback_tools` prunes the three
profile-rendered resources, and when a later refresh restores them their
membership move rides `ResourcesListChanged` while their bodies are compared
against what was last actually served at those URIs.
"""

from __future__ import annotations

import pytest
from mcp.server.subscriptions import (
    InMemorySubscriptionBus,
    ResourcesListChanged,
    ResourceUpdated,
    ToolsListChanged,
)
from mcp.shared.subscriptions import event_matches, event_to_notification
from mcp_types import SubscriptionFilter

import graphiti_mcp_server as srv
from domain_profile import DomainProfile, EdgeTypeInfo, EntityTypeInfo

# Selected by the CI `contract` job (.github/workflows/mcp-server-tests.yml):
# these guards need no database and no API key, so they gate every change. This
# is the R7 guard for the CONTENT half — an announced `resources.subscribe` with
# no publisher behind it is the same defect the prompt surface guards for
# `prompts`, and it is not one to discover on a droplet.
pytestmark = pytest.mark.contract

DOMAIN_SUMMARY = 'graphiti://domain_summary'
ENTITY_CATALOG = 'graphiti://entity_catalog'
RELATIONSHIP_TYPES = 'graphiti://relationship_types'
SCHEMA = 'graphiti://schema'


@pytest.fixture(autouse=True)
def _restore_server_globals():
    """Put the module-global server and refresh state back as we found them.

    Same reasoning as `test_listchanged_notifications.py`: `register_resources`
    and `register_fallback_tools` mutate `srv.mcp` in place, and the last-served
    fingerprint map is process-global by design — a leaked entry would make
    another module's refresh look like a no-op (or the reverse). The map is
    CLEARED for the duration, not merely restored, so each test's baseline is
    the one its own `_boot` lays down.
    """
    mcp = srv.mcp
    tools = dict(mcp._tool_manager._tools)
    resources = dict(mcp._resource_manager._resources)
    instructions = mcp._lowlevel_server.instructions
    service = srv.graphiti_service
    served = dict(srv._last_served_resource_bodies)
    srv._last_served_resource_bodies.clear()
    try:
        yield
    finally:
        mcp._tool_manager._tools.clear()
        mcp._tool_manager._tools.update(tools)
        mcp._resource_manager._resources.clear()
        mcp._resource_manager._resources.update(resources)
        mcp._lowlevel_server.instructions = instructions
        srv.graphiti_service = service
        srv._last_served_resource_bodies.clear()
        srv._last_served_resource_bodies.update(served)
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

    def subscribe(self, listener):  # pragma: no cover - the wire test uses the real bus
        return lambda: None


class _Service:
    """Minimal `GraphitiService` stand-in for the refresh path."""

    def __init__(self):
        self.flavour = None
        self.ontology_client = None
        self.domain_profile = None
        self._schema_dirty = False
        self._schema_cache = None
        self.censuses = 0

    async def get_client(self):
        return object()


def _profile(
    group_id: str = 'p4_graph',
    *,
    label: str = 'Widget',
    count: int = 3,
    samples: tuple[str, ...] = ('Widget-1',),
    edge_count: int = 1,
) -> DomainProfile:
    """A profile whose four knobs each move a DIFFERENT subset of the bodies.

    - `samples` appears only in `render_entity_catalog` — one resource moves.
    - `count` appears in the catalog AND the domain summary — two move.
    - `edge_count` appears in the summary and the relationship list.
    - `group_id` is the heading of all three — everything moves.

    That separation is what lets the parametrised test below assert an EXACT
    set of updated URIs rather than "something changed".
    """
    return DomainProfile(
        group_id=group_id,
        entity_types={label: EntityTypeInfo(label, count, f'A {label}', list(samples))},
        edge_types={'USES': EdgeTypeInfo('USES', edge_count, 'uses', f'{label} -> {label}')},
        time_range=None,
    )


@pytest.fixture
def wired(monkeypatch):
    """A server wired for refresh: fake service, recording bus, scripted census."""
    bus = _Bus()
    monkeypatch.setattr(srv.mcp, '_subscriptions', bus, raising=False)
    service = _Service()
    monkeypatch.setattr(srv, 'graphiti_service', service)

    class _Cfg:
        class graphiti:
            group_id = 'p4_graph'
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


async def _boot(profiles, profile: DomainProfile) -> None:
    """Take the startup path with `profile`, which is what seeds the baseline.

    Deliberately the REAL startup path rather than a hand-populated map: the
    baseline every later comparison rests on is "what this process served at
    startup", and a test that seeded it by hand would not notice startup
    forgetting to record.
    """
    profiles['next'] = profile
    await srv._build_and_register_domain_surface()


def _updated(published) -> set[str]:
    return {e.uri for e in published if isinstance(e, ResourceUpdated)}


class TestOnlyAMovedBodyIsAnnounced:
    """`resources/updated` is a claim that a specific body a client may be
    holding is now stale. Publishing it for a byte-identical re-render is the
    same lie P3 removed from the list channel, one primitive down."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'nxt, expected',
        [
            pytest.param(_profile(), set(), id='none-moved'),
            pytest.param(
                _profile(samples=('Widget-1', 'Widget-2')),
                {ENTITY_CATALOG},
                id='one-moved',
            ),
            pytest.param(
                _profile(count=9),
                {ENTITY_CATALOG, DOMAIN_SUMMARY},
                id='two-moved',
            ),
            pytest.param(
                _profile(group_id='p4_graph_renamed'),
                {ENTITY_CATALOG, DOMAIN_SUMMARY, RELATIONSHIP_TYPES},
                id='all-moved',
            ),
        ],
    )
    async def test_exactly_the_uris_whose_body_moved(self, wired, nxt, expected):
        bus, _service, profiles = wired
        await _boot(profiles, _profile())
        assert bus.published == [], 'startup announced to a client that cannot exist'

        profiles['next'] = nxt
        assert await srv.refresh_domain_surface('test') is True

        assert _updated(bus.published) == expected, bus.published

    @pytest.mark.asyncio
    async def test_a_body_change_is_not_dressed_as_a_list_change(self, wired):
        """The two events answer different questions and P3 pinned the negative
        half of this from the other side. Here it is from P4's: a re-render that
        rewrote three bodies must announce THOSE, and must not claim the
        announced URI set moved when it did not."""
        bus, _service, profiles = wired
        await _boot(profiles, _profile())
        uris_before = set(srv.mcp._resource_manager._resources)

        profiles['next'] = _profile(group_id='p4_graph_renamed')
        await srv.refresh_domain_surface('test')

        assert set(srv.mcp._resource_manager._resources) == uris_before
        assert ResourcesListChanged() not in bus.published, bus.published
        assert _updated(bus.published), 'three bodies moved and nothing was announced'

    @pytest.mark.asyncio
    async def test_a_second_identical_refresh_announces_nothing_more(self, wired):
        """The map must be UPDATED after it publishes, not only read.

        A comparison that keeps comparing against the startup baseline would
        re-announce the same move on every subsequent refresh — an event storm
        proportional to ingest rather than to change.
        """
        bus, _service, profiles = wired
        await _boot(profiles, _profile())

        profiles['next'] = _profile(count=9)
        await srv.refresh_domain_surface('first')
        assert _updated(bus.published) == {ENTITY_CATALOG, DOMAIN_SUMMARY}

        bus.published.clear()
        await srv.refresh_domain_surface('second')
        assert bus.published == [], bus.published


class TestAFailedRefreshAnnouncesNoUpdate:
    """A restored surface is serving exactly what it served before. Telling a
    subscriber to refetch there is a transient graph blip billed to every
    consumer."""

    @pytest.mark.asyncio
    async def test_a_raising_census_publishes_no_updated(self, wired):
        bus, _service, profiles = wired
        await _boot(profiles, _profile())

        profiles['next'] = RuntimeError('graph unreachable')
        assert await srv.refresh_domain_surface('test') is False

        assert bus.published == [], bus.published

    @pytest.mark.asyncio
    async def test_a_raise_during_re_registration_publishes_no_updated(
        self, wired, monkeypatch
    ):
        """The M4 shape from P3, on the content channel: `register_resources`
        REPLACES each resource in turn, so a raise partway can leave some bodies
        rewritten and some not. The snapshot restore puts them all back — and
        the comparison then has to agree that nothing moved."""
        bus, _service, profiles = wired
        await _boot(profiles, _profile())

        real = srv.register_resources

        def half_registers(profile):
            real(_profile(group_id='p4_graph_partial'))
            raise RuntimeError('re-registration blew up')

        monkeypatch.setattr(srv, 'register_resources', half_registers)
        profiles['next'] = _profile(group_id='p4_graph_renamed')
        assert await srv.refresh_domain_surface('test') is False

        assert _updated(bus.published) == set(), (
            'the surface was restored to what it served, so no body moved'
        )


class TestAFailedRefreshSTILLAnnouncesABodyThatMoved:
    """The other half of the `finally`, and the reason it has to be one.

    Restoring the registries restores OBJECTS, not payloads.
    `graphiti://schema` is a lazy `FunctionResource`: the snapshot puts the same
    function back, and that function renders live. So a refresh whose census
    raised can still be serving a schema body different from the one it served
    before — the graph moved, the census could not read it, and the resource
    that reads through `get_schema` reflects the move anyway.

    Deciding emission on the success path instead would lose exactly this case,
    and lose it silently: the mutant passes every other test in this module,
    because the other three resources are eager text that a restore really does
    put back byte-for-byte.
    """

    @pytest.mark.asyncio
    async def test_a_failed_census_with_a_moved_schema_payload_announces_it(
        self, wired, monkeypatch
    ):
        bus, _service, profiles = wired
        payload = {'nodes': [{'label': 'Widget', 'count': 3}]}

        async def _schema():
            return payload

        monkeypatch.setattr(srv, 'get_schema', _schema)
        await _boot(profiles, _profile())

        # The graph moved; the census that would have seen it fails.
        payload = {'nodes': [{'label': 'Widget', 'count': 9}]}
        profiles['next'] = RuntimeError('graph unreachable mid-refresh')

        assert await srv.refresh_domain_surface('test') is False
        assert SCHEMA in _updated(bus.published), (
            'a failed refresh left a MOVED body served and announced nothing — '
            'the content comparison is not in the `finally`'
        )

    @pytest.mark.asyncio
    async def test_the_eager_resources_are_silent_on_that_same_refresh(
        self, wired, monkeypatch
    ):
        """Scoped, not blanket: the three profile-rendered bodies really were
        restored, so announcing them too would be the no-op event the whole
        design forbids."""
        bus, _service, profiles = wired
        payload = {'nodes': [{'label': 'Widget', 'count': 3}]}

        async def _schema():
            return payload

        monkeypatch.setattr(srv, 'get_schema', _schema)
        await _boot(profiles, _profile())

        payload = {'nodes': [{'label': 'Widget', 'count': 9}]}
        profiles['next'] = RuntimeError('graph unreachable mid-refresh')
        await srv.refresh_domain_surface('test')

        assert _updated(bus.published) == {SCHEMA}, bus.published


class TestTheDegradedPathComparesAgainstWhatWasSERVED:
    """`register_fallback_tools` prunes the three profile-rendered resources —
    the one case where the URI set genuinely moves. The prune must not reset the
    content baseline: a URI that returns with the body it had before is not an
    update, and a URI that returns with a different one is."""

    @pytest.mark.asyncio
    async def test_a_restore_with_the_same_bodies_announces_only_the_list(self, wired):
        bus, _service, profiles = wired
        await _boot(profiles, _profile())
        srv.register_fallback_tools(reason='census failed')
        assert set(srv.mcp._resource_manager._resources) == {SCHEMA}

        profiles['next'] = _profile()  # the graph did not move while degraded
        assert await srv.refresh_domain_surface('test') is True

        assert ResourcesListChanged() in bus.published, 'the URI set went 1 -> 4'
        assert _updated(bus.published) == set(), (
            'the restored bodies are the ones last served; nothing to refetch'
        )

    @pytest.mark.asyncio
    async def test_a_restore_with_different_bodies_announces_both(self, wired):
        bus, _service, profiles = wired
        await _boot(profiles, _profile())
        srv.register_fallback_tools(reason='census failed')

        profiles['next'] = _profile(group_id='p4_graph_renamed')
        assert await srv.refresh_domain_surface('test') is True

        assert ResourcesListChanged() in bus.published
        assert _updated(bus.published) == {
            DOMAIN_SUMMARY,
            ENTITY_CATALOG,
            RELATIONSHIP_TYPES,
        }, bus.published

    @pytest.mark.asyncio
    async def test_a_pruned_uri_is_never_announced_as_updated(self, wired, monkeypatch):
        """Nothing is served at a pruned URI, so nothing there can be stale.

        The refresh below fails, so the degraded surface stays: `domain_summary`
        & co. are absent from `resources/list` and must draw no event even
        though the last-served map still remembers them.
        """
        bus, _service, profiles = wired
        await _boot(profiles, _profile())
        srv.register_fallback_tools(reason='census failed')
        bus.published.clear()

        profiles['next'] = RuntimeError('still unreachable')
        assert await srv.refresh_domain_surface('test') is False

        assert _updated(bus.published) == set(), bus.published


class TestTheSchemaResourceIsFingerprintedOnItsRenderedPayload:
    """`graphiti://schema` is a lazy `FunctionResource`: there is no `.text` to
    hash, and the payload it serves is the live `get_schema` one. Fingerprinting
    it means rendering it — once per refresh, which is the cost the design
    accepted."""

    @pytest.mark.asyncio
    async def test_a_changed_schema_payload_announces_the_schema_resource(
        self, wired, monkeypatch
    ):
        bus, _service, profiles = wired
        payload = {'nodes': [{'label': 'Widget', 'count': 3}]}

        async def _schema():
            return payload

        monkeypatch.setattr(srv, 'get_schema', _schema)
        await _boot(profiles, _profile())

        payload = {'nodes': [{'label': 'Widget', 'count': 9}]}
        await srv.refresh_domain_surface('test')

        assert SCHEMA in _updated(bus.published), bus.published

    @pytest.mark.asyncio
    async def test_an_unchanged_schema_payload_announces_nothing(
        self, wired, monkeypatch
    ):
        bus, _service, profiles = wired

        async def _schema():
            return {'nodes': [{'label': 'Widget', 'count': 3}]}

        monkeypatch.setattr(srv, 'get_schema', _schema)
        await _boot(profiles, _profile())

        await srv.refresh_domain_surface('test')

        assert SCHEMA not in _updated(bus.published), bus.published

    @pytest.mark.asyncio
    async def test_a_body_that_cannot_be_rendered_announces_nothing(
        self, wired, monkeypatch
    ):
        """A resource that raises on read was not served either, so its
        last-served entry stands and it draws no event. The alternative —
        treating an unreadable body as "moved" — announces a refetch that would
        fail for the same reason."""
        bus, _service, profiles = wired
        await _boot(profiles, _profile())

        async def _boom():
            raise RuntimeError('schema unavailable')

        monkeypatch.setattr(srv, 'get_schema', _boom)
        await srv.refresh_domain_surface('test')

        assert SCHEMA not in _updated(bus.published), bus.published


class TestTheStartupCensusIsARecordAndNotAnAnnouncement:
    """No client can be listening before the transport binds, and a URI this
    process has never served cannot be stale in anyone's cache."""

    @pytest.mark.asyncio
    async def test_startup_records_every_served_body_and_publishes_nothing(self, wired):
        bus, _service, profiles = wired
        await _boot(profiles, _profile())

        assert bus.published == [], bus.published
        assert set(srv._last_served_resource_bodies) >= {
            DOMAIN_SUMMARY,
            ENTITY_CATALOG,
            RELATIONSHIP_TYPES,
            SCHEMA,
        }

    @pytest.mark.asyncio
    async def test_a_degraded_startup_records_what_it_serves(self, wired, monkeypatch):
        """The fallback path serves one resource. It is still a baseline: the
        recovery refresh has to compare `graphiti://schema` against what the
        degraded surface actually served, not against nothing."""
        bus, _service, profiles = wired
        profiles['next'] = RuntimeError('census failed at boot')

        await srv._build_and_register_domain_surface()

        assert set(srv.mcp._resource_manager._resources) == {SCHEMA}
        assert SCHEMA in srv._last_served_resource_bodies
        assert bus.published == [], bus.published

    @pytest.mark.asyncio
    async def test_a_uri_never_served_before_draws_no_update(self, wired):
        """First appearance is a LIST change, not a content change. Announcing
        `updated` for a body nobody could be holding is the declared-and-
        unbacked defect in the other direction."""
        bus, _service, profiles = wired
        srv._last_served_resource_bodies.clear()
        for uri in (DOMAIN_SUMMARY, ENTITY_CATALOG, RELATIONSHIP_TYPES):
            srv.mcp._resource_manager._resources.pop(uri, None)

        profiles['next'] = _profile()
        await srv.refresh_domain_surface('test')

        assert _updated(bus.published) == set(), bus.published


class TestTheWireAdmissionIsPerURI:
    """The delivery half is the SDK's: `ListenHandler` filters every event
    through `event_matches`, and for `ResourceUpdated` that is membership in the
    stream's honored `resource_subscriptions`. This test stands where the bus
    meets that predicate, so a server that published the wrong URI — or the
    wrong event type — is caught here rather than on a droplet."""

    @pytest.mark.asyncio
    async def test_a_stream_honoring_one_uri_receives_only_that_uri(
        self, wired, monkeypatch
    ):
        _bus, _service, profiles = wired
        bus = InMemorySubscriptionBus()
        monkeypatch.setattr(srv.mcp, '_subscriptions', bus, raising=False)

        honored = SubscriptionFilter(
            resource_subscriptions=[ENTITY_CATALOG], resources_list_changed=None
        )
        honored_uris = frozenset(honored.resource_subscriptions or ())
        delivered: list = []

        def deliver(event):
            # Exactly `ListenHandler.deliver`'s admission test.
            if event_matches(honored, honored_uris, event):
                delivered.append(event)

        bus.subscribe(deliver)

        await _boot(profiles, _profile())
        profiles['next'] = _profile(count=9)  # moves entity_catalog AND domain_summary
        await srv.refresh_domain_surface('test')

        assert delivered == [ResourceUpdated(uri=ENTITY_CATALOG)], delivered

    @pytest.mark.asyncio
    async def test_the_published_event_maps_to_the_updated_notification(
        self, wired
    ):
        """What rides the wire, spelled out: the method a subscriber's client
        demultiplexes on, and the `uri` it refetches."""
        bus, _service, profiles = wired
        await _boot(profiles, _profile())

        profiles['next'] = _profile(samples=('Widget-1', 'Widget-2'))
        await srv.refresh_domain_surface('test')

        (event,) = [e for e in bus.published if isinstance(e, ResourceUpdated)]
        notification = event_to_notification(event, {'x': 1})
        assert notification.method == 'notifications/resources/updated'
        assert notification.params.uri == ENTITY_CATALOG

    @pytest.mark.asyncio
    async def test_a_stream_honoring_only_list_changes_gets_no_updated(
        self, wired, monkeypatch
    ):
        """The mirror: a consumer that asked for list changes and no resource
        subscriptions must never see a body event, however many bodies move."""
        _bus, _service, profiles = wired
        bus = InMemorySubscriptionBus()
        monkeypatch.setattr(srv.mcp, '_subscriptions', bus, raising=False)

        honored = SubscriptionFilter(tools_list_changed=True, resources_list_changed=True)
        delivered: list = []
        bus.subscribe(
            lambda event: delivered.append(event)
            if event_matches(honored, frozenset(), event)
            else None
        )

        await _boot(profiles, _profile())
        profiles['next'] = _profile(group_id='p4_graph_renamed', count=9)
        await srv.refresh_domain_surface('test')

        assert not [e for e in delivered if isinstance(e, ResourceUpdated)], delivered
        assert ToolsListChanged() in delivered, (
            'the tool descriptions were re-rendered from a different profile'
        )


class TestEveryAnnouncedFreshnessBitIsNowBacked:
    """The ADR-019 R7 ledger, closed. All four bits come from the one
    unconditionally-served `subscriptions/listen` handler; after this wave each
    one has a publisher behind it (`prompts.listChanged` is the honest
    exception, recorded in the module: one statically registered prompt, a list
    that cannot move)."""

    def test_the_modern_era_announces_resource_subscriptions(self):
        caps = srv.mcp._lowlevel_server.get_capabilities(protocol_version='2026-07-28')
        assert caps.resources.subscribe is True

    @pytest.mark.asyncio
    async def test_the_publisher_puts_a_resource_updated_on_the_bus(self, monkeypatch):
        """Driven, not grepped.

        This assertion used to read `'ResourceUpdated(' in
        inspect.getsource(...)` — satisfiable by a comment, a docstring or dead
        code, so the message "the announced bit has no publisher" was one the
        test had not earned. Publishing through the seam and reading the bus is
        the claim itself.
        """
        bus = _Bus()
        monkeypatch.setattr(srv.mcp, '_subscriptions', bus, raising=False)

        await srv._publish_surface_change(
            tools=False, resources=False, updated=('graphiti://anything',)
        )

        assert bus.published == [ResourceUpdated(uri='graphiti://anything')], (
            'the announced `resources.subscribe` has no publisher behind it'
        )

    @pytest.mark.asyncio
    async def test_the_publisher_stays_silent_when_nothing_moved(self, monkeypatch):
        """The same seam's other half: no event kind, no publish at all."""
        bus = _Bus()
        monkeypatch.setattr(srv.mcp, '_subscriptions', bus, raising=False)

        await srv._publish_surface_change(tools=False, resources=False, updated=())

        assert bus.published == []
