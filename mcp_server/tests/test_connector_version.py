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

import re
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
    declared = tomllib.loads(PYPROJECT.read_text())['project']['version']
    assert declared == CONNECTOR_VERSION, (
        'the announced version must be the packaged one, not a hand-maintained copy'
    )


def test_the_version_looks_like_a_version():
    assert re.fullmatch(r'\d+\.\d+\.\d+([-.+].*)?', CONNECTOR_VERSION), CONNECTOR_VERSION


def test_the_version_is_never_empty_even_without_packaging_metadata(monkeypatch):
    """A connector that cannot read its own metadata must still answer something.

    Returning '' or raising here would make `get_status` fail for a reason that has
    nothing to do with the graph.
    """
    import importlib.metadata

    def _missing(_name):
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(importlib.metadata, 'version', _missing)
    monkeypatch.setattr('version._PYPROJECT', Path('/nonexistent/pyproject.toml'))
    assert connector_version() == 'unknown'


def test_the_pyproject_fallback_reads_the_real_file(monkeypatch):
    """Docker installs the project without packaging metadata; pyproject is copied
    beside src/, so the fallback is the path that actually runs in the image."""
    import importlib.metadata

    def _missing(_name):
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(importlib.metadata, 'version', _missing)
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
