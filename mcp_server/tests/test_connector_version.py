"""A-D11: the connector must say which build it is.

`serverInfo.version` on the wire is `1.29.0` — the MCP SDK's version, which
FastMCP reports and which is the same on every connector in the fleet. The
connector's own build appeared nowhere in the surface, so an operator debugging a
bad answer could not tell whether the arm in front of them predated a fix.

Two places, because they answer different questions: `get_status` for "what am I
talking to right now" (a probe an operator or health check already makes), and the
instructions header for "what wrote this guidance" (it travels with the
announcement a consumer captures and caches).
"""

from __future__ import annotations

import importlib.metadata
import re
import subprocess
from pathlib import Path

try:  # 3.11+
    import tomllib
except ModuleNotFoundError:  # 3.10
    import tomli as tomllib

import pytest

from domain_profile import DomainProfile
from flavours.falkordb import FalkorDbFlavour
from tool_descriptions import build_degraded_instructions, build_instructions
from version import CONNECTOR_NAME, CONNECTOR_VERSION, connector_version

PYPROJECT = Path(__file__).parent.parent / 'pyproject.toml'


def test_the_version_matches_the_packaging_metadata():
    """Read independently, with a real TOML parser rather than the impl's regex."""
    declared = tomllib.loads(PYPROJECT.read_text())['project']['version']
    assert declared == CONNECTOR_VERSION, (
        'the announced version must be the packaged one, not a hand-maintained copy'
    )


def _git_tags(*args: str) -> list[str] | None:
    """`mcp-v*` tags matching the given git selector, or None outside a checkout."""
    try:
        out = subprocess.run(
            ['git', 'tag', *args, '--list', 'mcp-v*'],
            cwd=PYPROJECT.parent,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return [t.strip().removeprefix('mcp-v') for t in out.stdout.split('\n') if t.strip()]


def _newest_prior_tag() -> str | None:
    """The newest `mcp-v*` tag reachable from HEAD but NOT pointing at it.

    Tags AT HEAD are excluded on purpose. Cutting `mcp-v1.5.0` on the merge commit
    would otherwise make this test fail on main from the moment of release until
    somebody bumped again — turning a release into a red suite. A tag at HEAD is
    this build being released; the rule is only about releases that came BEFORE it.
    """
    reachable = _git_tags('--merged', 'HEAD')
    if reachable is None:
        return None
    at_head = set(_git_tags('--points-at', 'HEAD') or ())
    prior = [t for t in reachable if t not in at_head]
    if not prior:
        return None
    return max(prior, key=lambda v: tuple(int(p) for p in v.split('.')[:3]))


def test_the_packaged_version_is_ahead_of_every_shipped_tag():
    """THE check the test above cannot make.

    Comparing `CONNECTOR_VERSION` to the pyproject it is read from is tautological
    — it passes no matter how stale that file is. `mcp-v1.3.0` and `mcp-v1.4.0`
    both shipped while pyproject still said `1.2.2`, so the connector would have
    announced a version two releases behind. A confident wrong answer is worse than
    the SDK-version noise it replaces.

    A tag reachable from HEAD is a release this build already contains, so the
    packaged version must be strictly greater than the newest of them. Tags AT
    HEAD are excluded — see `_newest_prior_tag`.
    """
    newest = _newest_prior_tag()
    if newest is None:
        pytest.skip('not a git checkout with prior mcp-v* tags')

    def _parts(v: str) -> tuple[int, ...]:
        return tuple(int(p) for p in v.split('.')[:3])

    assert _parts(CONNECTOR_VERSION) > _parts(newest), (
        f'pyproject says {CONNECTOR_VERSION} but mcp-v{newest} is already reachable '
        f'from HEAD — bump mcp_server/pyproject.toml before shipping'
    )


def test_the_wire_value_agrees_with_installed_metadata_when_there_is_any():
    """Where the project IS installed, the served value must match its metadata.

    Skips where no distribution exists — the dev venv and the Docker image both run
    from source, which is precisely why pyproject is read first.
    """
    try:
        installed = importlib.metadata.version('mcp-server')
    except importlib.metadata.PackageNotFoundError:
        pytest.skip('mcp-server is not installed as a distribution here')
    assert installed == CONNECTOR_VERSION


def test_the_version_looks_like_a_version():
    assert re.fullmatch(r'\d+\.\d+\.\d+([-.+].*)?', CONNECTOR_VERSION), CONNECTOR_VERSION


def test_the_version_is_never_empty_even_without_packaging_metadata(monkeypatch):
    """A connector that cannot read its own metadata must still answer something.

    Returning '' or raising here would make `get_status` fail for a reason that has
    nothing to do with the graph.
    """

    def _missing(_name):
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(importlib.metadata, 'version', _missing)
    monkeypatch.setattr('version._PYPROJECT', Path('/nonexistent/pyproject.toml'))
    assert connector_version() == 'unknown'


def test_pyproject_wins_over_stale_installed_metadata(monkeypatch):
    """pyproject is the source of truth; installed metadata is only the fallback.

    An editable install can carry metadata older than the source beside it. Reading
    metadata first would let that stale snapshot shadow a freshly bumped pyproject
    — announcing a version this build is not, which is the defect M4 is about.
    """
    monkeypatch.setattr(importlib.metadata, 'version', lambda _name: '0.0.1')
    declared = tomllib.loads(PYPROJECT.read_text())['project']['version']
    assert connector_version() == declared


@pytest.mark.parametrize(
    'build',
    [
        lambda: build_instructions(DomainProfile(group_id='g'), FalkorDbFlavour()),
        lambda: build_degraded_instructions(
            group_id='g', flavour=FalkorDbFlavour(), reason='boom', marker='!! D !!'
        ),
    ],
    ids=['healthy', 'degraded'],
)
def test_the_instructions_announce_the_connector_build(build):
    text = build()
    assert CONNECTOR_NAME in text
    assert CONNECTOR_VERSION in text


def test_the_degraded_marker_still_leads_the_announcement():
    """The version line must not displace the degraded marker from first position."""
    text = build_degraded_instructions(
        group_id='g', flavour=None, reason='boom', marker='!! DEGRADED !!'
    )
    assert text.startswith('!! DEGRADED !!')


@pytest.mark.asyncio
async def test_get_status_reports_the_connector_version(monkeypatch):
    import graphiti_mcp_server as srv

    monkeypatch.setattr(srv, 'graphiti_service', None)
    status = await srv.get_status()
    assert status['version'] == CONNECTOR_VERSION, (
        'get_status must answer "what am I talking to" even when the service is down'
    )


@pytest.mark.asyncio
async def test_get_status_still_reports_health(monkeypatch):
    import graphiti_mcp_server as srv

    monkeypatch.setattr(srv, 'graphiti_service', None)
    status = await srv.get_status()
    assert status['status'] == 'error'
    assert 'not initialized' in status['message']
