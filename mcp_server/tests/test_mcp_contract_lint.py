"""The premise the connector's flat envelope rests on — pinned against the real SDK.

WHY THIS MODULE EXISTS

Wave 1 settled ADR-019 R2/A-D7 by giving every tool a concrete typed return, so
the SDK publishes a FLAT `structured_content` and the ADR-015 R4 client rule
`if "error" in result` works for the whole surface. That guarantee is not
something the fork owns outright — it is a *property of the installed MCP SDK*:
`func_metadata` synthesises a `{"result": ...}` envelope for any return
annotation it cannot express as an object schema (primitives such as `-> str`,
and — less obviously — `A | B` unions, which fail its `GenericAlias` test).

So the flat envelope holds because of two facts that must both stay true:

  1. the SDK wraps primitives and unions, and only those;
  2. no served tool is annotated with either.

Fact 2 alone is guarded by `test_typed_output_schemas.py`, which reads published
schemas. Fact 1 is a premise about a dependency, and nothing checked it: an SDK
bump that started wrapping typed models — or stopped wrapping unions — would
change the served envelope for every consumer while every existing test stayed
green. This module drives a REAL in-process client session so the assertions are
against wire shapes rather than an idealisation of them.

MEASURED ACROSS THE 1.x -> 2.x BUMP (wave 6, 2026-08-07). The same five return
shapes were driven through an identical in-process round trip on mcp 1.26.0 and
mcp 2.0.0. Every observable was byte-identical — the envelope keys, the JSON in
the text channel down to its indentation, the error flag:

    return annotation      structured_content              1.26.0   2.0.0
    -> str                 {"result": "plain text"}        wrapped  wrapped
    -> A | B  (union)      {"result": {...}}               wrapped  wrapped
    -> BaseModel           {"message": ..., "rows": ...}   FLAT     FLAT
    -> dict[str, Any]      {"message": ..., "rows": ...}   FLAT     FLAT
    raise (plain Exception) None, is_error=True, text      in-band  in-band

The bump therefore required NO change to any consumer's wrapper-stripping logic.
The `dict` and error rows are new here: the 1.x edition pinned only the first
three, so `dict[str, Any]` (three tools carried it before wave 1) and the ADR-015
R4 in-band error path were premises nothing measured. Both are pinned now.

One 2.x behaviour that is NOT pinned as safe, recorded so the next reader does
not have to rediscover it: raising `MCPError` from a tool no longer comes back
in-band — the client re-raises it. The fork does not raise from tools (every
error path returns a typed payload carrying `error`), which is why the surface is
unaffected; `test_a_typed_error_payload_stays_in_band` pins the path it does use.

What this measures is the *pinned* SDK: CI runs `uv sync`, so the round trip below
exercises the version in `uv.lock`. That is the intent — the guard gates the bump
rather than tracking whatever happens to be installed.

The consumer half of this premise lives in aletheia
(`tests/mcp/test_structured_content_wire.py::TestWirePremise`) and measures
*aletheia's* installed SDK. This one measures the fork's. Two repos, two
dependency pins, two guards — a version skew between them is exactly the failure
neither could see alone.

The second class is bookkeeping, not behaviour: it keeps the ADR-019 guards
attached to the CI job that runs them, because a guard nothing executes is the
same as no guard (2026-08-06 analysis, F3).
"""

from __future__ import annotations

import pathlib
import re
import types
import typing
from typing import Any

import pytest
import yaml
from mcp import Client
from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel

import graphiti_mcp_server as srv
from tool_annotations import TOOL_ANNOTATIONS, annotations_for

pytestmark = pytest.mark.contract

MCP_SERVER = pathlib.Path(__file__).parent.parent
REPO = MCP_SERVER.parent


# ---------------------------------------------------------------------------
# Probe server: the shapes the SDK wraps, and the ones it does not
# ---------------------------------------------------------------------------

class _Payload(BaseModel):
    message: str
    rows: list[str] = []


class _Failure(BaseModel):
    error: str


class _ErrorCapable(BaseModel):
    message: str
    error: str | None = None


_probe = MCPServer('wire-premise-probe')


@_probe.tool()
async def primitive_return() -> str:
    """The shape the SDK wraps: a bare primitive."""
    return 'plain text'


@_probe.tool()
async def union_return() -> _Payload | _Failure:
    """The shape the SDK wraps less obviously: `X | ErrorResponse`. This was the
    fork's pre-wave-1 annotation on eleven tools."""
    return _Payload(message='ok', rows=['r1'])


@_probe.tool()
async def typed_return() -> _Payload:
    """The shape the SDK does NOT wrap — what the whole surface uses today."""
    return _Payload(message='ok', rows=['r1'])


@_probe.tool()
async def dict_return() -> dict[str, Any]:
    """Also NOT wrapped, but for a different reason than a typed model: the SDK can
    express a mapping as an object schema, it just cannot say what is IN it. Three
    tools carried this annotation before wave 1 replaced it (A-D6)."""
    return {'message': 'ok', 'rows': ['r1']}


@_probe.tool()
async def raising_return() -> _Payload:
    """A tool that raises rather than returning. Pins the ADR-015 R4 boundary."""
    raise ValueError('boom in the tool')


@_probe.tool()
async def inband_error_return() -> _ErrorCapable:
    """What the fork's error paths ACTUALLY do: return a typed payload carrying
    `error`. No raise, so no JSON-RPC error — `if "error" in result` sees it."""
    return _ErrorCapable(message='', error='something went wrong')


# `mode='legacy'` is load-bearing, not a leftover. For an in-process server the
# SDK's own connector docstring reads: "legacy mode drives the stream loop via
# InMemoryTransport; any other mode drives the modern per-request path through a
# DirectDispatcher peer pair (no streams, no JSON-RPC framing, no initialize
# handshake)". The default `auto` would therefore skip the framing this module
# claims to measure — the assertions would be against an idealisation, which is
# exactly what the docstring above says they are not. Observables were compared
# across both modes and are identical, so this costs nothing and keeps the claim
# true.
#
# It is also GUARDED, one class below — wave 6 set the constant and left the claim
# resting on a comment (F7, accepted LOW). A constant nobody checks is a comment: the
# next person to write `mode='auto'` here, or an SDK release that changes what the
# modes mean, would silently move every assertion in this module one layer off the
# framing it exists to measure, and nothing would go red.
_WIRE_MODE = 'legacy'


def _dispatcher_name(client: Client) -> str:
    """The dispatcher class actually driving a connected client.

    Reaches through two private attributes on purpose. It is the only place the mode
    becomes OBSERVABLE rather than declared — asserting `_WIRE_MODE == 'legacy'` would
    just read the constant back to itself. If the SDK renames or restructures these,
    this raises AttributeError and the guard fails loudly, which is the correct
    outcome: the premise would need rechecking against the new internals.
    """
    return type(client._session._dispatcher).__name__


async def _call(name: str):
    async with Client(_probe, mode=_WIRE_MODE) as client:
        return await client.call_tool(name, {})


class TestTheProbesReallyRunOverJsonRpcFraming:
    """F7 (wave-6 review, accepted LOW): guard the wire mode instead of asserting it.

    This module's whole claim is that it measures wire shapes rather than an
    idealisation of them. That claim rests on `_WIRE_MODE`, and until now on nothing
    else. These two tests make the difference between the modes observable, so the
    claim fails when it stops being true rather than when someone rereads a comment.
    """

    async def test_the_probe_client_is_driven_by_the_jsonrpc_dispatcher(self):
        async with Client(_probe, mode=_WIRE_MODE) as client:
            name = _dispatcher_name(client)
        assert 'JSONRPC' in name, (
            f'the probe client is driven by {name}, not a JSON-RPC dispatcher — every '
            'assertion in this module is being made one layer above the framing it '
            'claims to measure. Check _WIRE_MODE and the SDK connector docstring.'
        )

    async def test_the_other_mode_still_bypasses_that_framing(self):
        """The other side of the guard, and the reason the first one is not circular.

        If a future SDK collapsed both modes onto the same dispatcher, the test above
        would keep passing while `_WIRE_MODE` stopped meaning anything. This one goes
        red instead, which is the signal to reread the premise — not to delete it.
        """
        async with Client(_probe, mode='auto') as client:
            name = _dispatcher_name(client)
        assert 'JSONRPC' not in name, (
            f"mode='auto' now also uses {name}: the two modes no longer differ, so the "
            'reason this module pins legacy has changed. Recheck the SDK connector '
            'docstring before relaxing anything here.'
        )


class TestTheSdkStillWrapsWhatWeThinkItWraps:
    async def test_a_bare_str_return_is_wrapped_under_result(self):
        result = await _call('primitive_return')
        assert set(result.structured_content) == {'result'}, (
            'the installed MCP SDK no longer wraps primitive returns — the fork chose '
            'typed returns precisely to avoid this envelope, and that reasoning needs '
            'rechecking before the next release'
        )
        assert result.structured_content['result'] == 'plain text'

    async def test_a_union_return_is_wrapped_under_result(self):
        result = await _call('union_return')
        assert set(result.structured_content) == {'result'}, (
            'the installed MCP SDK no longer wraps union returns — A-D7 (two envelope '
            'families) was diagnosed from this behaviour'
        )
        assert isinstance(result.structured_content['result'], dict)

    async def test_a_typed_model_return_is_not_wrapped(self):
        """The control: without this, the two assertions above could pass on an SDK
        that wraps everything, which would break the surface silently."""
        result = await _call('typed_return')
        assert set(result.structured_content) == {'message', 'rows'}

    async def test_a_bare_dict_return_is_not_wrapped(self):
        """The second control, and a distinct code path from the typed model: a
        mapping the SDK can call an object without knowing its fields. A-D6's three
        tools returned this, so an SDK that started wrapping it would change the
        envelope for anything that regressed to a `dict` annotation."""
        result = await _call('dict_return')
        assert set(result.structured_content) == {'message', 'rows'}

    async def test_a_raising_tool_comes_back_in_band(self):
        """ADR-015 R4's outer boundary. A plain exception must NOT reach the client
        as a JSON-RPC error: it is reported in-band with `is_error` set and the
        message in the text channel, which is what keeps a failed tool call a
        readable result rather than a transport fault."""
        result = await _call('raising_return')
        assert result.is_error is True
        assert result.structured_content is None
        assert 'boom in the tool' in result.content[0].text

    async def test_a_typed_error_payload_stays_in_band(self):
        """ADR-015 R4 as the fork actually implements it: not a raise at all, but a
        typed payload with `error` populated. This must stay a NON-error result with
        flat structured content, or `if "error" in result` stops being reachable."""
        result = await _call('inband_error_return')
        assert result.is_error is False
        assert set(result.structured_content) == {'message', 'error'}
        assert result.structured_content['error'] == 'something went wrong'


# ---------------------------------------------------------------------------
# ...and no served tool is annotated with a shape that would be wrapped
# ---------------------------------------------------------------------------

def _return_annotation(tool_name: str):
    fn = getattr(srv, tool_name)
    return typing.get_type_hints(fn).get('return')


@pytest.mark.parametrize('tool_name', sorted(TOOL_ANNOTATIONS))
def test_no_served_tool_returns_a_shape_the_sdk_would_wrap(tool_name):
    """ADR-019 R2 / A-D7 at the annotation level: one envelope family, by
    construction rather than by inspection of the published schema."""
    annotation = _return_annotation(tool_name)
    assert annotation is not None, f'{tool_name} has no return annotation'
    assert annotation is not str, (
        f'{tool_name} returns a bare `str` — the SDK will nest its payload under '
        f'"result" and `if "error" in result` stops working for it (ADR-019 R2)'
    )
    origin = typing.get_origin(annotation)
    assert not isinstance(annotation, types.UnionType) and origin is not typing.Union, (
        f'{tool_name} returns a union ({annotation}) — the SDK nests union payloads '
        f'under "result", which is the second envelope family A-D7 removed'
    )


@pytest.fixture(scope='module')
def served_surface():
    """The whole surface on a throwaway MCPServer — never the module global."""
    m = MCPServer('served-surface-probe')
    for name in sorted(TOOL_ANNOTATIONS):
        m.add_tool(getattr(srv, name), annotations=annotations_for(name))
    return m


async def test_the_served_output_schemas_are_flat_over_a_real_client_session(
    served_surface,
):
    """The same claim as `test_typed_output_schemas.py`, made one layer out: what a
    client actually receives from `tools/list`, not what `MCPServer.list_tools()`
    returns in-process."""
    async with Client(served_surface, mode=_WIRE_MODE) as client:
        listed = (await client.list_tools()).tools

    assert {t.name for t in listed} == set(TOOL_ANNOTATIONS)
    wrapped = [
        t.name for t in listed if set((t.output_schema or {}).get('properties') or {}) == {'result'}
    ]
    assert wrapped == [], f'served with a wrapped envelope: {wrapped}'


# ---------------------------------------------------------------------------
# The guards have to actually run somewhere
# ---------------------------------------------------------------------------

# Each ADR-019 dimension and the module that guards it. The CI job runs the
# `contract` marker, so a new guard joins CI by declaring the marker — there is
# no second list to keep in step.
CONTRACT_GUARDS = {
    'R1 instructions announce the whole surface': 'test_instructions_catalog.py',
    'R2 typed, flat output schemas': 'test_typed_output_schemas.py',
    'R3 truthful tool annotations': 'test_tool_annotations.py',
    'served text names no foreign domain': 'test_no_domain_leakage.py',
    'the SDK envelope premise': 'test_mcp_contract_lint.py',
    # The three dimensions the SDK 2.x migration made load-bearing. Each is a
    # property of the SERVED transport rather than of the tool surface, and each
    # has a silent failure mode: a default that changed under the bump, or an
    # invariant that was only ever a comment.
    'output fields absent from a payload stay nullable': 'test_output_field_nullability.py',
    'the DNS rebinding policy follows FASTMCP_HOST': 'test_transport_security.py',
    'bulk request bodies are not capped at the SDK default': 'test_request_body_limit.py',
}

CI_WORKFLOW = REPO / '.github' / 'workflows' / 'mcp-server-tests.yml'


@pytest.mark.parametrize(('dimension', 'module'), sorted(CONTRACT_GUARDS.items()))
def test_every_contract_dimension_has_a_marked_guard(dimension, module):
    source = (pathlib.Path(__file__).parent / module).read_text(encoding='utf-8')
    assert 'pytestmark = pytest.mark.contract' in source, (
        f'{module} guards "{dimension}" but does not declare the contract marker, '
        f'so the CI job does not select it'
    )


def test_ci_runs_the_contract_marker():
    """The whole point of F3: enforcement that nothing executes is not enforcement."""
    workflow = CI_WORKFLOW.read_text(encoding='utf-8')
    assert '-m contract' in workflow, (
        'no CI job selects the contract guards — they would only ever run by hand'
    )


def _ci_pytest_target() -> str:
    """The positional path the CI job hands pytest (`tests/`)."""
    workflow = CI_WORKFLOW.read_text(encoding='utf-8')
    match = re.search(r'uv run pytest (\S+)', workflow)
    assert match, 'cannot find the pytest invocation in the workflow'
    return match.group(1)


def _resolved_ini() -> pathlib.Path:
    """The pytest.ini the CI invocation actually loads.

    pytest takes the rootdir from the common ancestor of the positional args and
    then walks UP looking for a config file, so `pytest tests/` from mcp_server
    resolves `mcp_server/tests/pytest.ini` — not `mcp_server/pytest.ini`.
    """
    start = (MCP_SERVER / _ci_pytest_target()).resolve()
    for candidate in [start, *start.parents]:
        ini = candidate / 'pytest.ini'
        if ini.is_file():
            return ini
    raise AssertionError('no pytest.ini found above the CI target')


def test_the_contract_marker_is_registered_in_the_ini_ci_resolves():
    """The guard that missed F-1 by reading the wrong file.

    mcp_server carries two pytest.ini files and the CI command loads the deeper
    one, which sets `--strict-markers`. Asserting against a hardcoded path let
    this test pass green while the job it describes failed collection on all five
    guard modules. It now resolves the file the way pytest does.
    """
    ini = _resolved_ini()
    assert 'contract:' in ini.read_text(encoding='utf-8'), (
        f'{ini.relative_to(REPO)} is the config the CI command loads and it does not '
        f'register the `contract` marker — with --strict-markers that is a collection '
        f'error on every guard module'
    )


_TRACKED_INIS = sorted(
    p.relative_to(MCP_SERVER).as_posix()
    for p in MCP_SERVER.rglob('pytest.ini')
    if not {'.venv', 'node_modules', '__pycache__', 'site-packages'} & set(p.parts)
)


@pytest.mark.parametrize('ini', _TRACKED_INIS)
def test_every_pytest_ini_registers_the_marker(ini):
    """Belt to the brace above: whichever ini a future invocation resolves, the
    marker is there. Two config files is the condition that produced F-1."""
    assert 'contract:' in (MCP_SERVER / ini).read_text(encoding='utf-8')


def test_ci_triggers_on_the_branch_this_fork_develops_on():
    """F-2: the workflow was inherited from upstream and fired on `main` only, while
    every fork MR targets `aletheia`. A job wired correctly and triggered on a branch
    nobody pushes is still a job that never runs."""
    on_block = yaml.safe_load(CI_WORKFLOW.read_text(encoding='utf-8'))[True]
    for event in ('push', 'pull_request'):
        assert 'aletheia' in on_block[event]['branches'], (
            f'{event} does not trigger on `aletheia`, the branch this fork develops on'
        )


def test_no_contract_guard_hides_in_the_ci_ignore_list():
    """The job skips four modules that fail at COLLECTION (plus test_fixtures.py, which collects clean at 0 items) for reasons of their own.
    That list is the obvious place to quietly park an inconvenient guard, so pin
    that nothing in it carries the marker."""
    workflow = CI_WORKFLOW.read_text(encoding='utf-8')
    ignored = re.findall(r'--ignore=(tests/\S+\.py)', workflow)
    assert ignored, 'the ignore list vanished — re-check whether it is still needed'
    for rel in ignored:
        path = MCP_SERVER / rel
        if not path.exists():
            continue
        assert 'pytest.mark.contract' not in path.read_text(encoding='utf-8'), (
            f'{rel} is excluded from the CI contract job but declares the marker'
        )
