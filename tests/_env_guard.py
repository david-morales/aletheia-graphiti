"""Test-session environment guard for the ROOT suite. Imported by the repo-root
`conftest.py` before anything else.

The sibling suite already has one (`mcp_server/tests/_env_guard.py`); this is the
same mechanism and the same opt-in convention, adapted to what the root suite
actually does. The difference that matters: `mcp_server` reaches a live store
through a `FALKORDB_URI` and an `integration` marker, while the root suite
reaches one through the `graph_driver` FIXTURE PARAMS in `helpers_test.py`,
which are generated from `DISABLE_*` env vars and default the FalkorDB endpoint
to `localhost:6379` — the RESERVED production instance (BUG-109).

So the lever here is the driver list itself. With the gate closed the four
`DISABLE_*` switches are set BEFORE `helpers_test` is imported, the list comes
out empty, and every parametrized test collects as pytest's empty-parameter-set
placeholder: a skip with no fixture setup behind it, which is the only state in
which no connection can be opened. Neutralizing the endpoint as well is
belt-and-braces for any code that reads the host/port directly.

Default-CLOSED is the whole point. The hand-written `env -i ... DISABLE_FALKORDB=1`
recipe this replaces protected only the runs that remembered it.

Underscore-prefixed so pytest's `test_*.py` collection never picks it up; a plain
module so the decisions are unit-testable without driving a subprocess.
"""

from __future__ import annotations

import os

# The one opt-in for live/`integration` runs in this suite. Mirrors
# `MCP_LIVE_TESTS` next door: same `<SUITE>_LIVE_TESTS` shape, same semantics.
LIVE_GATE_ENV = 'GRAPHITI_LIVE_TESTS'

# The switches `tests/helpers_test.py` reads at import to decide which providers
# the `graph_driver` fixture is parametrized over. Setting all four is what makes
# the live params cease to exist rather than merely fail politely.
DRIVER_DISABLE_VARS: tuple[str, ...] = (
    'DISABLE_NEO4J',
    'DISABLE_FALKORDB',
    'DISABLE_KUZU',
    'DISABLE_NEPTUNE',
)

# A closed loopback port. Not load-bearing on its own — the empty driver list is
# — but it means that even a test that builds its own driver from the ambient
# environment gets a connection refused instead of the reserved store.
CLOSED_ENDPOINT_HOST = '127.0.0.1'
CLOSED_ENDPOINT_PORT = '1'

# The AGE suite reaches a SECOND store, and not through the driver list at all:
# `tests/driver/conftest.py` and `tests/driver/test_age_driver.py` each read
# `AGE_TEST_DSN` at module scope, defaulting to a live Postgres+AGE bed on 5433.
# Emptying the driver list does nothing for those, so the endpoint is neutralized
# directly. Same loopback host and closed port as above: the AGE fixtures skip on
# connectivity-class errors, so a refused connection turns them into clean skips
# rather than failures.
CLOSED_AGE_DSN = 'postgresql://age:age@127.0.0.1:1/age_test'

# FalkorDB/Redis's default port. On a developer machine this is a real, populated
# store — here the RESERVED production instance the project forbids touching.
RESERVED_PORT = '6379'

# Every provider credential the root suite or graphiti_core reads. A single
# present key un-skips this fork's live suites, so the offline arm neutralizes
# all of them or none: leaving one behind reproduces the exact hazard.
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

# Values that mean "no" when someone writes them into the gate variable. Without
# this, `GRAPHITI_LIVE_TESTS=0` — the obvious way to say "not this run" — reads
# as truthy and OPENS the gate, turning the one deliberate off-switch into a
# silent on-switch.
_FALSY_GATE_VALUES = frozenset({'', '0', 'false', 'no', 'off'})


def marker_expression_from_argv(argv: list[str]) -> str:
    """The EFFECTIVE `-m` expression, read straight off the command line.

    `mcp_server`'s guard gets this from `config.getoption` in `pytest_configure`.
    This suite cannot wait that long: `helpers_test` builds its driver list in
    its own module body, and that import happens from the repo-root conftest's
    module body — before pytest has parsed anything. By the time
    `pytest_configure` runs, the list is already built.

    Returns the LAST `-m`, not the first, because that is what argparse gives
    pytest: `-m integration -m "not integration"` runs as `not integration`. A
    first-wins reader disagrees with the run it is supposed to describe, and it
    disagrees in the dangerous direction — it would open the gate for a session
    that had just asked for the opposite.

    Handles the three spellings pytest accepts (`-m X`, `-mX`, `-m=X`) and
    deliberately does NOT prefix-match, which would read `--markers` as `-m`
    with a value of `arkers`.
    """
    expression = ''
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == '-m':
            expression = argv[index + 1] if index + 1 < len(argv) else ''
            index += 2
            continue
        if token.startswith('-m=') and len(token) > 3:
            expression = token[3:]
        elif token.startswith('-m') and not token.startswith('--') and len(token) > 2:
            expression = token[2:]
        index += 1
    return expression


def live_gate_is_open(known_args: dict[str, str] | None = None) -> bool:
    """Has this run explicitly asked for live tests?

    Two ways to say yes, because there are two callers with different habits:

    * the `GRAPHITI_LIVE_TESTS` env var — the gate a human (or a CI job that
      talks to a real database) sets deliberately;
    * naming the marker in `-m` and nothing else, which is an explicit request
      by definition. The accident this guards against is the run that never
      mentions it.

    The marker arm is deliberately STRICT and deliberately stupid: the
    expression opens the gate only when it is exactly `integration` after
    stripping whitespace. Nothing else — not `integration or slow`, not
    `not (slow or integration)`.

    That is a real limitation, stated rather than papered over (the sibling
    guard makes the same admission about its own parser). The alternative is
    parsing marker expressions, and the first attempt here did try: it split on
    whitespace after DISCARDING parentheses, so `not (slow or integration)` —
    a negation applied to a group — read as an unnegated `integration` and
    FALSE-OPENED the gate. A guard that fails open on an expression a human
    would call obviously negative is worse than one that fails closed on an
    expression a human would call obviously positive: the closed direction
    costs a skipped test, the open direction reaches the reserved store.

    Anything more elaborate than the exact string should set the env var, which
    is unambiguous by construction.
    """
    value = os.environ.get(LIVE_GATE_ENV)
    if value is not None and value.strip().lower() not in _FALSY_GATE_VALUES:
        return True
    expression = (known_args or {}).get('-m') or ''
    return expression.strip() == 'integration'


def disable_live_drivers(environ: dict[str, str] | None = None) -> list[str]:
    """Make the live driver params impossible to generate. Returns names changed.

    MUST be called before `tests.helpers_test` is imported — it reads these at
    module scope. Also points the FalkorDB endpoint and the AGE DSN at a closed
    port, so that code paths reading an endpoint directly (rather than through
    the fixture) get ECONNREFUSED instead of a real store. The AGE arm is not
    covered by the driver list at all — its tests read `AGE_TEST_DSN` themselves
    — so without this it would stay live while the rest of the suite was closed.
    """
    env = os.environ if environ is None else environ
    changed = []
    for name in DRIVER_DISABLE_VARS:
        if env.get(name) != '1':
            env[name] = '1'
            changed.append(name)
    for name, value in (
        ('FALKORDB_HOST', CLOSED_ENDPOINT_HOST),
        ('FALKORDB_PORT', CLOSED_ENDPOINT_PORT),
        ('AGE_TEST_DSN', CLOSED_AGE_DSN),
    ):
        if env.get(name) != value:
            env[name] = value
            changed.append(name)
    return changed


def stamp_dummy_api_keys(environ: dict[str, str] | None = None) -> list[str]:
    """Force every provider credential to a dummy. Returns the names changed.

    Called from the repo-root `conftest.py` BEFORE the first project import,
    which is what makes it work at all: `tests/helpers_test.py` runs
    `load_dotenv()` at MODULE IMPORT, and that file carries a live key on a
    developer machine. `load_dotenv` does not override an already-set variable,
    so a value stamped here wins — the offline suite is keyless by construction
    rather than by whoever remembered to sanitize their shell.

    OVERWRITES rather than fills gaps, and that is the point: a real key in the
    developer's own environment is precisely what un-skips this fork's live
    suites. Only called with the gate CLOSED; an opted-in run is untouched.
    """
    env = os.environ if environ is None else environ
    changed = []
    for name in _PROVIDER_KEY_VARS:
        if env.get(name) != DUMMY_KEY:
            env[name] = DUMMY_KEY
            changed.append(name)
    return changed
