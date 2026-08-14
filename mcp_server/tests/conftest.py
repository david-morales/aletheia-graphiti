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

_LIVE_GATE_OPEN = False
"""Set for real in `pytest_configure`, once the CLI has been parsed.

NOT decided at import time, and that ordering is the whole point. `-m
integration` — the fork's own live CI invocation — is invisible until pytest
has parsed its arguments, so an import-time gate reads CLOSED for it. When the
stamp below hung off that early answer, a live run with real secrets had every
credential overwritten with a dummy before the first call, and a keyless fork PR
had a dummy planted where the live module's `if not OPENAI_API_KEY: skip` looks.
"""

from config.schema import GraphitiConfig  # noqa: E402


def pytest_configure(config: pytest.Config) -> None:
    """Settle the gate, then arm the two guards it governs.

    Runs after CLI parsing and BEFORE collection, which is the only window that
    works for both halves: `-m` is readable here, and no test module has been
    imported yet — so the credential stamp still lands ahead of
    `graphiti_mcp_server`'s `load_dotenv()`, which is what it has to beat.
    """
    global _LIVE_GATE_OPEN
    expression = config.getoption('-m', default='') or ''
    _LIVE_GATE_OPEN = live_gate_is_open({'-m': expression})

    if _LIVE_GATE_OPEN:
        # An opted-in run gets its environment untouched: it was started to talk
        # to a real backend with real credentials, and stamping over them would
        # break exactly the run that asked for them.
        return

    stamp_dummy_api_keys()

    import os

    # Refuse an offline run aimed at a real graph store (BUG-85). A UsageError
    # ends the session before collection: a warning here would be advice, and
    # the failure mode being guarded is precisely the run that did not read it.
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
