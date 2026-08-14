"""Pytest configuration for MCP server tests.

This file prevents pytest from loading the parent project's conftest.py, and it
carries the two session guards that make the offline suite structurally safe.

ORDER MATTERS HERE. `sys.path` gets `src/` and the provider credentials get
neutralized BEFORE the first project import, because `graphiti_mcp_server` calls
`load_dotenv()` over `mcp_server/.env` at module import and that file carries a
live key on a developer machine. `load_dotenv` does not override an already-set
variable, so stamping first is what makes the neutralization stick.
"""

import sys
from pathlib import Path

import pytest

# Add src to the Python path. Deliberately NOT this directory: `tests/` is a
# package, and putting it on `sys.path` as well would make its modules importable
# under two names — which quietly resurrects `test_async_operations` and
# `test_stress_load` (they do a top-level `from test_fixtures import ...` that
# only resolves that way). Those two are heavy live modules that have failed to
# collect here for as long as this suite has existed; making them collect is a
# separate decision, not a side effect of adding a guard.
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from tests._env_guard import (  # noqa: E402
    LIVE_GATE_ENV,
    live_gate_is_open,
    stamp_dummy_api_keys,
    unsafe_falkordb_uri,
)

# The gate is read ONCE, here, before any test module is imported — several of
# them decide at import time whether to skip, and a gate that moved underneath
# them would make that decision incoherent.
_LIVE_GATE_OPEN = live_gate_is_open()

if not _LIVE_GATE_OPEN:
    stamp_dummy_api_keys()

from config.schema import GraphitiConfig  # noqa: E402


def pytest_configure(config: pytest.Config) -> None:
    """Refuse an offline run that is aimed at a real graph store (BUG-85).

    Raised as a UsageError so the session ends before collection: a warning here
    would be advice, and the failure mode being guarded is precisely the run
    that did not read the advice.
    """
    global _LIVE_GATE_OPEN
    # Re-evaluated with the parsed options in hand: `-m integration` is an
    # explicit opt-in, and it is only visible once pytest has parsed the CLI.
    expression = config.getoption('-m', default='') or ''
    _LIVE_GATE_OPEN = live_gate_is_open({'-m': expression})
    if _LIVE_GATE_OPEN:
        return

    import os

    reason = unsafe_falkordb_uri(os.environ.get('FALKORDB_URI'))
    if reason:
        raise pytest.UsageError(reason)


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Deselect every `integration`-marked test unless the run asked for live.

    DESELECT rather than skip: a deselected item never has its fixtures set up,
    and the fixture is where the damage lives — `test_falkordb_dialect_integration`
    opens a hardcoded `FalkorDB(host='localhost', port=6379)` in a module-scoped
    one, ignoring `FALKORDB_URI` entirely. A skip mark would still leave that
    connection one collection-order accident away.
    """
    if _LIVE_GATE_OPEN:
        return

    kept, deselected = [], []
    for item in items:
        (deselected if item.get_closest_marker('integration') else kept).append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = kept


def pytest_report_header(config: pytest.Config) -> str:
    """Say which arm this run is, in the header nobody has to ask for."""
    if _LIVE_GATE_OPEN:
        return f'live gate: OPEN ({LIVE_GATE_ENV} or -m integration) — live tests selected'
    return f'live gate: closed — integration tests deselected, provider keys stubbed ({LIVE_GATE_ENV}=1 to open)'


@pytest.fixture
def config():
    """Provide a default GraphitiConfig for tests."""
    return GraphitiConfig()
