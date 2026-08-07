"""The connector is domain-agnostic by construction — its TEXT must be too.

A-D3: three docstrings hardcoded aviation examples, and one of them
(`add_memory`'s "Aircraft PH-KZB experienced...") was served verbatim to a
Spanish police graph on both arms. The other two are normally masked by the
profile-driven descriptions — and become the served text the moment graph
introspection fails, which is precisely when an agent is least equipped to spot
that the example is fiction.

A-D10: two descriptions ended with "See docs/extraction-integration.md" — a
repo-relative path that means nothing to a wire consumer, pointing at a file that
does not exist in this repository.

This module is the mechanical guard ADR-019 has no enforcement for (F3): every
leakage finding in the audit survived because nothing scanned the served surface.
"""

from __future__ import annotations

import asyncio
import pathlib
import re

import pytest
from mcp.server.fastmcp import FastMCP

import graphiti_mcp_server as srv
from domain_profile import DomainProfile, EdgeTypeInfo, EntityTypeInfo
from tool_annotations import TOOL_ANNOTATIONS, annotations_for
from tool_descriptions import build_degraded_instructions, build_instructions

# Terms from domains this connector must know nothing about. A connector that
# names one is describing a graph it is not looking at.
BANNED_TERMS = (
    'aircraft',
    'airworthiness',
    'ICAO',
    'EASA',
    'ECCAIRS',
    'PH-KZB',
    'aviation',
    'Boeing',
)

# Leading word boundary only: the leak that shipped was `AirworthinessDirective`,
# a compound identifier with no boundary after the banned stem.
_BANNED_RE = re.compile(
    r'\b(' + '|'.join(re.escape(t) for t in BANNED_TERMS) + r')', re.IGNORECASE
)

# A repo-relative documentation path is unresolvable for a wire consumer.
_REPO_DOC_RE = re.compile(r'\bdocs/[\w./-]+\.md\b')

SRC = pathlib.Path(__file__).parent.parent / 'src'


def _neutral_profile() -> DomainProfile:
    """A profile whose own vocabulary cannot be mistaken for leakage."""
    return DomainProfile(
        group_id='neutral_graph',
        entity_types={
            'Widget': EntityTypeInfo('Widget', 12, 'A widget', ['W-1']),
            'Gadget': EntityTypeInfo('Gadget', 4, 'A gadget', ['G-1']),
        },
        edge_types={'USES': EdgeTypeInfo('USES', 6, 'Widget uses Gadget', 'Widget -> Gadget')},
        time_range=None,
    )


def _served_descriptions() -> dict[str, str]:
    """Every tool description as `tools/list` would serve it, on a neutral graph."""
    m = FastMCP('leakage-probe')
    profile = _neutral_profile()
    from tool_descriptions import (
        build_explore_node_description,
        build_explore_ontology_description,
        build_get_schema_description,
        build_run_cypher_description,
        build_search_description,
        build_search_ontology_description,
    )

    dynamic = {
        'search': build_search_description(profile),
        'explore_node': build_explore_node_description(profile),
        'search_ontology': build_search_ontology_description(profile),
        'explore_ontology': build_explore_ontology_description(profile),
        'get_schema': build_get_schema_description(profile),
        'run_cypher': build_run_cypher_description(profile, None),
    }
    for name in sorted(TOOL_ANNOTATIONS):
        m.add_tool(
            getattr(srv, name),
            description=dynamic.get(name),
            annotations=annotations_for(name),
        )
    return {t.name: t.description or '' for t in asyncio.run(m.list_tools())}


def _degraded_descriptions() -> dict[str, str]:
    """Every tool description in the DEGRADED state — docstrings, unmasked.

    This is the set that matters most: `search_ontology`'s and
    `explore_ontology`'s aviation examples are invisible on a healthy server and
    become the served text exactly when introspection fails.
    """
    m = FastMCP('leakage-probe-degraded')
    for name in sorted(TOOL_ANNOTATIONS):
        m.add_tool(getattr(srv, name), annotations=annotations_for(name))
    return {t.name: t.description or '' for t in asyncio.run(m.list_tools())}


@pytest.fixture(scope='module', params=['healthy', 'degraded'])
def served_descriptions(request):
    return _served_descriptions() if request.param == 'healthy' else _degraded_descriptions()


@pytest.mark.parametrize('path', sorted(SRC.rglob('*.py')), ids=lambda p: p.name)
def test_no_source_file_names_a_foreign_domain(path):
    """The static scan: docstrings are the wire description on every static tool."""
    hits = sorted({m.group(0) for m in _BANNED_RE.finditer(path.read_text())})
    assert not hits, f'{path.relative_to(SRC)} names a foreign domain: {hits}'


@pytest.mark.parametrize('tool_name', sorted(TOOL_ANNOTATIONS))
def test_no_served_description_names_a_foreign_domain(served_descriptions, tool_name):
    """The wire scan: what a client actually reads, on a neutral graph."""
    hits = sorted({m.group(0) for m in _BANNED_RE.finditer(served_descriptions[tool_name])})
    assert not hits, f'{tool_name} serves a foreign-domain term: {hits}'


@pytest.mark.parametrize('tool_name', sorted(TOOL_ANNOTATIONS))
def test_no_served_description_cites_a_repo_relative_document(
    served_descriptions, tool_name
):
    """A-D10: a wire consumer cannot resolve `docs/whatever.md`."""
    hits = sorted({m.group(0) for m in _REPO_DOC_RE.finditer(served_descriptions[tool_name])})
    assert not hits, f'{tool_name} cites an unresolvable repo path: {hits}'


def test_the_healthy_instructions_name_no_foreign_domain():
    text = build_instructions(_neutral_profile())
    assert not _BANNED_RE.search(text)
    assert not _REPO_DOC_RE.search(text)


def test_the_degraded_instructions_name_no_foreign_domain():
    """The degraded path is where the masked docstrings surface — scan it too."""
    text = build_degraded_instructions(
        group_id='neutral_graph', flavour=None, reason='boom', marker='!! DEGRADED !!'
    )
    assert not _BANNED_RE.search(text)
    assert not _REPO_DOC_RE.search(text)


def test_the_guard_actually_catches_the_terms_it_claims_to():
    """A scanner that matches nothing passes everything."""
    assert _BANNED_RE.search('Aircraft PH-KZB experienced a runway excursion')
    assert _BANNED_RE.search('e.g., "AirworthinessDirective"')
    assert _REPO_DOC_RE.search('See docs/extraction-integration.md for the contract')
    # ...and does not fire on ordinary prose that merely contains the letters.
    assert not _BANNED_RE.search('release the increased easement')
