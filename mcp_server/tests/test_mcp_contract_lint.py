"""The premise the connector's flat envelope rests on — pinned against the real SDK.

WHY THIS MODULE EXISTS

Wave 1 settled ADR-019 R2/A-D7 by giving every tool a concrete typed return, so
FastMCP publishes a FLAT `structuredContent` and the ADR-015 R4 client rule
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

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session
from pydantic import BaseModel

import graphiti_mcp_server as srv
from tool_annotations import TOOL_ANNOTATIONS, annotations_for

pytestmark = pytest.mark.contract

MCP_SERVER = pathlib.Path(__file__).parent.parent
REPO = MCP_SERVER.parent


# ---------------------------------------------------------------------------
# Probe server: the two shapes the SDK wraps, and one it does not
# ---------------------------------------------------------------------------

class _Payload(BaseModel):
    message: str
    rows: list[str] = []


class _Failure(BaseModel):
    error: str


_probe = FastMCP('wire-premise-probe')


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


async def _call(name: str):
    async with create_connected_server_and_client_session(_probe._mcp_server) as session:
        await session.initialize()
        return await session.call_tool(name, {})


class TestTheSdkStillWrapsWhatWeThinkItWraps:
    async def test_a_bare_str_return_is_wrapped_under_result(self):
        result = await _call('primitive_return')
        assert set(result.structuredContent) == {'result'}, (
            'the installed MCP SDK no longer wraps primitive returns — the fork chose '
            'typed returns precisely to avoid this envelope, and that reasoning needs '
            'rechecking before the next release'
        )

    async def test_a_union_return_is_wrapped_under_result(self):
        result = await _call('union_return')
        assert set(result.structuredContent) == {'result'}, (
            'the installed MCP SDK no longer wraps union returns — A-D7 (two envelope '
            'families) was diagnosed from this behaviour'
        )
        assert isinstance(result.structuredContent['result'], dict)

    async def test_a_typed_model_return_is_not_wrapped(self):
        """The control: without this, the two assertions above could pass on an SDK
        that wraps everything, which would break the surface silently."""
        result = await _call('typed_return')
        assert set(result.structuredContent) == {'message', 'rows'}


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
        f'{tool_name} returns a bare `str` — FastMCP will nest its payload under '
        f'"result" and `if "error" in result` stops working for it (ADR-019 R2)'
    )
    origin = typing.get_origin(annotation)
    assert not isinstance(annotation, types.UnionType) and origin is not typing.Union, (
        f'{tool_name} returns a union ({annotation}) — FastMCP nests union payloads '
        f'under "result", which is the second envelope family A-D7 removed'
    )


@pytest.fixture(scope='module')
def served_surface():
    """The whole surface on a throwaway FastMCP — never the module global."""
    m = FastMCP('served-surface-probe')
    for name in sorted(TOOL_ANNOTATIONS):
        m.add_tool(getattr(srv, name), annotations=annotations_for(name))
    return m


async def test_the_served_output_schemas_are_flat_over_a_real_client_session(
    served_surface,
):
    """The same claim as `test_typed_output_schemas.py`, made one layer out: what a
    client actually receives from `tools/list`, not what `FastMCP.list_tools()`
    returns in-process."""
    async with create_connected_server_and_client_session(
        served_surface._mcp_server
    ) as session:
        await session.initialize()
        listed = (await session.list_tools()).tools

    assert {t.name for t in listed} == set(TOOL_ANNOTATIONS)
    wrapped = [
        t.name for t in listed if set((t.outputSchema or {}).get('properties') or {}) == {'result'}
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


def test_the_contract_marker_is_registered():
    """An unregistered marker is a warning, and with `-m` it silently selects nothing."""
    ini = (MCP_SERVER / 'pytest.ini').read_text(encoding='utf-8')
    assert 'contract:' in ini


def test_no_contract_guard_hides_in_the_ci_ignore_list():
    """The job skips five modules that fail at COLLECTION for reasons of their own.
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
