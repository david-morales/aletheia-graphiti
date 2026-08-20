"""Root pytest configuration.

Adds the project root to `sys.path` so `tests.*` imports resolve, and carries the
session guard that keeps this suite off a real graph store by default (BUG-109).

Concretely, with the gate closed the guard covers three things: the `graph_driver`
fixture's provider params (NEO4J, FALKORDB, KUZU, NEPTUNE — emptied at the source,
so there is nothing to dial), the FalkorDB host/port, and the AGE DSN that
`tests/driver/` reads on its own. It is not a claim about anything a test
constructs from a hardcoded endpoint of its own making; `integration`-marked items
are deselected to cover that class, but the guarantee is the params and the
endpoints named above, not a promise about code this file has never seen.

ORDER MATTERS HERE, and it is the reason the guard is wired in this file rather
than in `tests/conftest.py`. `tests/helpers_test.py` decides which providers the
`graph_driver` fixture is parametrized over IN ITS MODULE BODY, from the
`DISABLE_*` env vars, and calls `load_dotenv()` there too. That module is
imported from this one, three lines down. A guard that waits for
`pytest_configure` — or that lives in a conftest pytest loads later — runs after
the driver list has already been built and after `.env` has already been read,
which is to say after the decision it exists to make.
"""

import os
import sys

# This code adds the project root directory to the Python path, allowing imports to work correctly when running tests.
# Without this file, you might encounter ModuleNotFoundError when trying to import modules from your project, especially when running tests.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

import pytest  # noqa: E402

from tests._env_guard import (  # noqa: E402
    LIVE_GATE_ENV,
    disable_live_drivers,
    live_gate_is_open,
    marker_expression_from_argv,
    stamp_dummy_api_keys,
)

# Settled at IMPORT time, deliberately — see the module docstring. `-m` comes off
# `sys.argv` because pytest has not parsed its command line yet.
LIVE_GATE_OPEN = live_gate_is_open({'-m': marker_expression_from_argv(sys.argv)})

if not LIVE_GATE_OPEN:
    # Default-closed. Neither of these consults what the environment already
    # said: a `FALKORDB_HOST` aimed at the reserved store and a real API key are
    # exactly the conditions the guard exists for, not exceptions to it.
    disable_live_drivers()
    stamp_dummy_api_keys()

from tests.helpers_test import graph_driver, mock_embedder  # noqa: E402

# Exclude mcp_server from test collection - it has its own test suite and conftest
collect_ignore_glob = ['mcp_server/*']

__all__ = ['graph_driver', 'mock_embedder']


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Deselect every `integration`-marked test unless the run asked for live.

    DESELECT rather than skip: a deselected item never has its fixtures set up,
    and the fixture is where the damage lives. The empty driver list already
    covers everything that goes through `graph_driver`; this covers the modules
    that build their own driver from `helpers_test.get_driver` instead.
    """
    if LIVE_GATE_OPEN:
        return

    kept, deselected = [], []
    for item in items:
        (deselected if item.get_closest_marker('integration') else kept).append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = kept


def pytest_report_header(config: pytest.Config) -> str:
    """Say which arm this run is, in the header nobody has to ask for."""
    if LIVE_GATE_OPEN:
        return f'live gate: OPEN ({LIVE_GATE_ENV} or -m integration) — live drivers selected'
    return (
        f'live gate: closed — live driver params not generated, integration tests '
        f'deselected, provider keys stubbed ({LIVE_GATE_ENV}=1 to open)'
    )
