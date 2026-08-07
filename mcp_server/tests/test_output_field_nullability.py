"""Every output field of every served tool must accept `None`. All 18, mechanically.

WHY THIS MODULE EXISTS

The tool return types are `total=False` TypedDicts, so a payload may legitimately
omit any field. The SDK does not preserve that omission: it builds a pydantic model
with every field defaulted to `None` and dumps it WITHOUT `exclude_unset`, so a
field absent from what the tool returned is emitted as `None` in
`structured_content` — and then validated against the published `output_schema`.

If that field's declared type does not admit `None`, the server rejects its own
output. Measured over a real client session:

    RuntimeError: Invalid structured content returned by tool <name>:
    None is not of type 'string'

That is a HARD failure, not an ADR-015 R4 in-band error — the caller gets an
exception where it expected a result. It has happened twice: `get_schema` (`error`)
and `run_cypher` (`truncated`), both fixed by adding `| None`.

`response_types.py` says "DO NOT drop the `| None` — it is load-bearing", and
until now that was the only thing protecting the invariant: a comment. Nothing
enforced it, and the failure is invisible to the capability-level tests, which call
the tool functions directly and never build structured content. Adding one
`error: str` to any of ~11 tools' return types would have shipped green.

Two halves, deliberately redundant:

  * the ROUND TRIP is the real behaviour — an all-unset payload through the same
    `convert_result` + schema validation the lowlevel server performs;
  * the INTROSPECTION says WHY when it breaks, naming the offending field instead
    of leaving a jsonschema message to be decoded.

Both are driven off `TOOL_ANNOTATIONS`, so a new tool joins this guard by being
registered — there is no second list to keep in step.
"""

from __future__ import annotations

import types
import typing

import jsonschema
import pytest
from mcp.server.mcpserver import MCPServer
from typing_extensions import TypedDict

import graphiti_mcp_server as srv
from tool_annotations import TOOL_ANNOTATIONS, annotations_for

pytestmark = pytest.mark.contract


def _return_type(tool_name: str):
    return typing.get_type_hints(getattr(srv, tool_name)).get('return')


def _accepts_none(annotation) -> bool:
    """Does this declared type admit `None`?

    `Any` does (it admits everything). Anything else must be a union with
    `NoneType` in it — `X | None` or `Optional[X]`, which normalise to the same
    thing under `get_type_hints`.
    """
    if annotation is typing.Any:
        return True
    origin = typing.get_origin(annotation)
    if origin is typing.Union or isinstance(annotation, types.UnionType):
        return type(None) in typing.get_args(annotation)
    return annotation is type(None)


@pytest.fixture(scope='module')
def isolated_tools():
    """The whole surface on a throwaway server — never the module global."""
    m = MCPServer('nullability-probe')
    for name in sorted(TOOL_ANNOTATIONS):
        m.add_tool(getattr(srv, name), annotations=annotations_for(name))
    return m._tool_manager._tools


def _minimal_payload(return_type) -> dict:
    """The emptiest payload the type permits: required keys only, nothing optional.

    Required keys are exempt from the nullability rule precisely because they can
    never be absent, so they must be PRESENT here or pydantic rejects the payload
    for an unrelated reason and the test stops measuring nullability.
    """
    dummies = {str: 'probe', int: 0, float: 0.0, bool: False, list: [], dict: {}}
    hints = typing.get_type_hints(return_type)
    payload = {}
    for key in getattr(return_type, '__required_keys__', frozenset()):
        annotation = hints[key]
        origin = typing.get_origin(annotation) or annotation
        payload[key] = dummies.get(origin)
    return payload


@pytest.mark.parametrize('tool_name', sorted(TOOL_ANNOTATIONS))
def test_every_absentable_output_field_accepts_none(tool_name):
    """The invariant `response_types.py` documents, now enforced.

    Scoped to keys that can actually be ABSENT. A `total=True` field is always
    supplied by the tool, so the SDK never injects `None` into it and requiring
    nullability there would be noise — `StatusResponse` is deliberately total and
    always returns all three of its fields.
    """
    return_type = _return_type(tool_name)
    assert return_type is not None, f'{tool_name} has no return annotation'

    hints = typing.get_type_hints(return_type)
    assert hints, f'{tool_name} returns {return_type!r}, which declares no fields'

    optional_keys = getattr(return_type, '__optional_keys__', frozenset())
    offenders = sorted(k for k in optional_keys if not _accepts_none(hints[k]))
    assert not offenders, (
        f'{tool_name} returns {return_type.__name__} whose OPTIONAL field(s) {offenders} '
        f'do not admit None. An optional field may be absent from a returned payload — '
        f'and the SDK emits an absent field as None, then validates it against the '
        f'published output_schema. A non-nullable optional field therefore makes the '
        f'tool raise "Invalid structured content" over the wire on every call that omits '
        f'it. Declare it `X | None`, or make it required (see response_types.py).'
    )


@pytest.mark.parametrize('tool_name', sorted(TOOL_ANNOTATIONS))
def test_the_minimal_payload_survives_output_validation(isolated_tools, tool_name):
    """The same claim as behaviour rather than as types: the emptiest payload a tool
    can return must still validate. This is the exact path that broke `get_schema`
    and `run_cypher`."""
    meta = isolated_tools[tool_name].fn_metadata
    payload = _minimal_payload(_return_type(tool_name))
    structured = meta.convert_result(payload).structured_content
    try:
        jsonschema.validate(structured, meta.output_schema)
    except jsonschema.ValidationError as exc:
        pytest.fail(
            f'{tool_name} cannot validate its own minimal output: {exc.message}\n'
            f'sent={payload}\nstructured_content={structured}\n'
            f'The SDK injected None for an absent field whose declared type rejects it.'
        )


# ---------------------------------------------------------------------------
# ...and the guard above actually bites
# ---------------------------------------------------------------------------

class _Violating(TypedDict, total=False):
    """What a future edit looks like: `error` declared non-nullable."""

    message: str | None
    error: str


class _Compliant(TypedDict, total=False):
    message: str | None
    error: str | None


def test_the_introspection_guard_rejects_a_non_nullable_field():
    hints = typing.get_type_hints(_Violating)
    assert not _accepts_none(hints['error']), 'the control type is not violating'
    assert _accepts_none(typing.get_type_hints(_Compliant)['error'])


def test_the_round_trip_guard_reproduces_the_real_failure():
    """Without this, `test_an_all_unset_payload_survives_output_validation` could
    pass on an SDK that stopped injecting None and would be measuring nothing."""
    m = MCPServer('violating-probe')

    async def violating() -> _Violating:
        return {'message': 'ok'}

    m.add_tool(violating, name='violating')
    meta = m._tool_manager._tools['violating'].fn_metadata

    structured = meta.convert_result({'message': 'ok'}).structured_content
    assert structured == {'message': 'ok', 'error': None}, (
        'the SDK no longer injects None for absent optional fields — the whole '
        'nullability invariant rests on that behaviour, so re-derive it'
    )
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(structured, meta.output_schema)
