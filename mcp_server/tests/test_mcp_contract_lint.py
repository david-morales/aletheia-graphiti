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

What it measures is the *pinned* SDK: CI runs `uv sync`, so the round trip below
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

import pytest
import yaml
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
