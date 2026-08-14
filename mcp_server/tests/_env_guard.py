"""Test-session environment guards. Imported by `conftest.py` before anything else.

Two jobs, both structural rather than procedural — the whole point is that no
run has to remember a flag:

* `live_gate_is_open` / `unsafe_falkordb_uri` — keep the offline suite away from
  a real graph store (BUG-85 / H-F2).
* `stamp_dummy_api_keys` — keep the offline suite keyless no matter what the
  developer's shell or `mcp_server/.env` carries.

Underscore-prefixed so pytest's `python_files = test_*.py` never collects it;
a plain module so the decisions are unit-testable without driving a subprocess.
"""

from __future__ import annotations

import os
from urllib.parse import urlsplit

# The one opt-in for every live/`integration`-marked module in this suite.
LIVE_GATE_ENV = 'MCP_LIVE_TESTS'

# Loopback under every spelling a URI can use. NOT a general "is this local"
# test: on a developer machine the reserved store is ALSO on loopback, which is
# why the port rule below exists alongside this one.
_LOOPBACK_HOSTS = frozenset({'localhost', '127.0.0.1', '::1', ''})

# FalkorDB/Redis's default port. On a developer machine this is a real, populated
# store — here it is the RESERVED production instance the project forbids
# touching — and on CI it is a throwaway container. The difference is not visible
# from the URI, so the gate decides: closed means "you did not ask for a live
# run", and a run that did not ask for one has no business on this port.
_RESERVED_PORT = 6379

# Every provider credential the server or graphiti_core reads. A single present
# key un-skips the fork's live suites, so the offline arm neutralizes all of them
# or none: leaving one behind reproduces the exact hazard.
_PROVIDER_KEY_VARS: tuple[str, ...] = (
    'OPENAI_API_KEY',
    'ANTHROPIC_API_KEY',
    'GOOGLE_API_KEY',
    'GEMINI_API_KEY',
    'GROQ_API_KEY',
    'VOYAGE_API_KEY',
    'AZURE_OPENAI_API_KEY',
    'AWS_ACCESS_KEY_ID',
    'AWS_SECRET_ACCESS_KEY',
    'AWS_SESSION_TOKEN',
)

DUMMY_KEY = 'sk-dummy-offline-suite'


def live_gate_is_open(known_args: dict[str, str] | None = None) -> bool:
    """Has this run explicitly asked for live/integration tests?

    Two ways to say yes, because there are two callers with different habits:

    * the `MCP_LIVE_TESTS` env var — the gate a human sets deliberately;
    * naming the marker in `-m`, which is what the fork's own live CI job does
      (`pytest tests/test_live_falkordb_int.py -m integration`). Selecting a
      marker BY NAME is an explicit request; the accident this guards against is
      the run that never mentions it at all. Honouring it is also what lets this
      guard land without editing a workflow that is outside this change.

    `-m "not integration"` is the opposite request and must NOT open the gate,
    so the expression is inspected rather than merely searched for a substring.
    """
    if os.environ.get(LIVE_GATE_ENV):
        return True
    expression = (known_args or {}).get('-m') or ''
    if not expression:
        return False
    # Deliberately simple: a term is "asked for" when it appears unnegated. The
    # expressions in play are `integration` and `not integration`; anything more
    # elaborate should set the env var rather than rely on parsing.
    tokens = expression.replace('(', ' ').replace(')', ' ').split()
    for index, token in enumerate(tokens):
        if token == 'integration' and (index == 0 or tokens[index - 1] != 'not'):
            return True
    return False


def unsafe_falkordb_uri(uri: str | None) -> str | None:
    """Why this endpoint must not be dialled by a run that is not live, or None.

    Fails CLOSED: anything that cannot be parsed into a loopback host on a
    non-reserved port is refused. An unset/empty `FALKORDB_URI` is fine — the
    offline suite has no endpoint and needs none.
    """
    if not uri:
        return None

    parsed = urlsplit(uri)
    if not parsed.scheme or not parsed.netloc:
        return (
            f'FALKORDB_URI={uri!r} could not be parsed as an endpoint. Refusing to '
            f'run: an endpoint we cannot read is not one we can call harmless. '
            f'Set {LIVE_GATE_ENV}=1 if this really is a live run.'
        )

    try:
        host = (parsed.hostname or '').lower()
        port = parsed.port
    except ValueError:  # malformed port
        host, port = '', None

    if host not in _LOOPBACK_HOSTS:
        return (
            f'FALKORDB_URI={uri!r} points at a non-loopback host. The offline suite '
            f'must not reach a remote graph store. Set {LIVE_GATE_ENV}=1 to run live '
            f'tests deliberately.'
        )
    if port == _RESERVED_PORT:
        return (
            f'FALKORDB_URI={uri!r} points at the reserved default port '
            f'{_RESERVED_PORT}, which on a developer machine is a real populated '
            f'store. The offline suite must not touch it. Set {LIVE_GATE_ENV}=1 to '
            f'run live tests deliberately.'
        )
    return None


def stamp_dummy_api_keys(environ: dict[str, str] | None = None) -> list[str]:
    """Force every provider credential to a dummy. Returns the names changed.

    Called from `conftest.py` BEFORE the first project import, which is what
    makes it work at all: `graphiti_mcp_server` runs `load_dotenv()` over
    `mcp_server/.env` at MODULE IMPORT, and that file carries a live key on a
    developer machine. `load_dotenv` does not override an already-set variable,
    so a value stamped here wins — the offline suite is keyless by construction
    rather than by whoever remembered to sanitize their shell.

    OVERWRITES rather than fills gaps, and that is the point: a real key in the
    developer's own environment is precisely what un-skips the fork's live
    suites and points them at a reserved database. Only called with the live
    gate CLOSED; an opted-in run gets its environment untouched.
    """
    env = os.environ if environ is None else environ
    changed = []
    for name in _PROVIDER_KEY_VARS:
        if env.get(name) != DUMMY_KEY:
            env[name] = DUMMY_KEY
            changed.append(name)
    return changed
