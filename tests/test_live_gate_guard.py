"""BUG-109: the ROOT suite must be unable to reach a real graph store by default.

`tests/helpers_test.py` parametrizes the `graph_driver` fixture over NEO4J and
FALKORDB unconditionally, and defaults the FalkorDB endpoint to
`localhost:6379` — which on a developer machine is a real, populated store
(here: the RESERVED production FalkorDB). Nothing in this suite deselected or
skipped those params, so a plain `pytest tests/` dialled it. It fired for real
on 2026-08-19: a full-suite run connected to the reserved instance and left a
stray key behind.

The mitigation in use was a hand-written `env -i ... DISABLE_FALKORDB=1 ...`
prefix. That is a habit, not a guard: it protects only the runs that remember
it, and the run that forgets is exactly the dangerous one. This module pins the
STRUCTURAL replacement — the same shape `mcp_server/tests/_env_guard.py` already
uses for the other suite:

* default-CLOSED — with no opt-in, the live driver params are not generated at
  all, no matter what `FALKORDB_HOST`/`FALKORDB_PORT` or which API keys the
  ambient environment happens to carry;
* one opt-in — `GRAPHITI_LIVE_TESTS`, or naming `-m integration`;
* opted in, everything comes back: the guard gates, it does not delete.

The unit tests below cover the DECISION. The subprocess tests cover the WIRING,
which is the half a unit test cannot see: the guard has to actually be installed
in the real conftest, ahead of the `helpers_test` import, for any of it to
matter. Every subprocess arm is `--collect-only`, so this module never opens a
connection to anything even when the guard it is testing is broken.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests._env_guard import (
    DRIVER_DISABLE_VARS,
    DUMMY_KEY,
    LIVE_GATE_ENV,
    live_gate_is_open,
    marker_expression_from_argv,
    stamp_dummy_api_keys,
)

REPO_ROOT = Path(__file__).parent.parent

# A module that takes the `graph_driver` fixture and carries NO `integration`
# marker — deliberately. The marker-deselect arm is the easy half; this module
# proves the harder one, that the live PARAMS themselves are gone.
PARAMETRIZED_MODULE = 'tests/test_add_triplet.py'

# What a live param looks like in a collected node id: `...[GraphProvider.FALKORDB]`.
# With the gate closed the same test collects as `...[NOTSET]`, pytest's shape
# for an empty parameter set (one skipped placeholder, no fixture setup).
LIVE_PARAM_MARKER = 'GraphProvider.'

# The suite's ENTIRE live surface: every module that takes the `graph_driver`
# fixture, plus the one that parametrizes over `helpers_test.drivers` itself.
# Scoped deliberately — several offline modules parametrize over `GraphProvider`
# ENUM VALUES to assert per-dialect query shapes against mock drivers, and those
# ids contain `GraphProvider.` while never opening a connection. A bare scan for
# the marker across `tests/` reads those as live params and is simply wrong.
LIVE_SURFACE_MODULES = (
    'tests/test_add_triplet.py',
    'tests/test_edge_int.py',
    'tests/test_graphiti_int.py',
    'tests/test_graphiti_mock.py',
    'tests/test_node_int.py',
    'tests/test_entity_exclusion_int.py',
)


# ---------------------------------------------------------------------------
# The decision (fast, in-process)
# ---------------------------------------------------------------------------


def test_the_gate_reads_the_environment(monkeypatch):
    monkeypatch.delenv(LIVE_GATE_ENV, raising=False)
    assert live_gate_is_open({}) is False
    monkeypatch.setenv(LIVE_GATE_ENV, '1')
    assert live_gate_is_open({}) is True


@pytest.mark.parametrize('value', ['0', 'false', 'False', 'FALSE', 'no', 'off', 'NO', ''])
def test_an_explicitly_FALSY_gate_value_stays_closed(monkeypatch, value):
    """`GRAPHITI_LIVE_TESTS=0` is how a human says "not this run".

    Reading any non-empty string as an opt-in turns the one deliberate
    off-switch into a silent on-switch (the sibling suite shipped that bug).
    """
    monkeypatch.setenv(LIVE_GATE_ENV, value)
    assert live_gate_is_open({}) is False


@pytest.mark.parametrize('value', ['1', 'true', 'TRUE', 'yes', 'on'])
def test_a_truthy_gate_value_opens(monkeypatch, value):
    monkeypatch.setenv(LIVE_GATE_ENV, value)
    assert live_gate_is_open({}) is True


def test_naming_the_marker_opts_in_without_the_env_gate(monkeypatch):
    monkeypatch.delenv(LIVE_GATE_ENV, raising=False)
    assert live_gate_is_open({'-m': 'integration'}) is True
    assert live_gate_is_open({'-m': 'not integration'}) is False
    assert live_gate_is_open({'-m': 'contract'}) is False


@pytest.mark.parametrize(
    'expression',
    [
        'not (slow or integration)',
        'not(slow or integration)',
        '(not integration)',
        'not integration',
        'integration or slow',
        'slow and integration',
        'contract',
        '',
        '   ',
    ],
)
def test_only_the_bare_marker_opens_the_gate(monkeypatch, expression):
    """The gate opens on EXACTLY `integration` and nothing else.

    `not (slow or integration)` is the case that made this strict. The previous
    parser stripped parentheses before splitting on whitespace, which turned a
    negation applied to a GROUP into a bare, unnegated `integration` token —
    so the most emphatic way of saying "not live" opened the live gate.

    The compound POSITIVE forms (`integration or slow`) are here to pin the
    trade deliberately: they stay closed. Failing closed on a positive costs a
    skipped test; failing open on a negative reaches the reserved store.
    """
    monkeypatch.delenv(LIVE_GATE_ENV, raising=False)
    assert live_gate_is_open({'-m': expression}) is False, expression


@pytest.mark.parametrize('expression', ['integration', ' integration', 'integration '])
def test_the_bare_marker_opens_regardless_of_surrounding_whitespace(monkeypatch, expression):
    monkeypatch.delenv(LIVE_GATE_ENV, raising=False)
    assert live_gate_is_open({'-m': expression}) is True, expression


@pytest.mark.parametrize(
    'argv,opens',
    [
        (['pytest', '-m', 'integration'], True),
        (['pytest', '-m', 'not (slow or integration)'], False),
        # argparse keeps the LAST occurrence; the gate must agree with the run.
        (['pytest', '-m', 'integration', '-m', 'not integration'], False),
        (['pytest', '-m', 'not integration', '-m', 'integration'], True),
    ],
)
def test_the_gate_agrees_with_the_run_argparse_will_actually_do(monkeypatch, argv, opens):
    """End to end over argv: the two halves (parse, then decide) must compose.

    A first-wins reader disagreed with pytest itself — `-m integration
    -m "not integration"` RUNS as `not integration`, but the gate read the
    first token and opened. The dangerous direction, for a session that had
    just asked for the opposite.
    """
    monkeypatch.delenv(LIVE_GATE_ENV, raising=False)
    expression = marker_expression_from_argv(argv)
    assert live_gate_is_open({'-m': expression}) is opens, (argv, expression)


@pytest.mark.parametrize(
    'argv,expected',
    [
        (['pytest', 'tests/'], ''),
        (['pytest', '-m', 'integration'], 'integration'),
        (['pytest', '-m', 'not integration'], 'not integration'),
        (['pytest', '-mintegration'], 'integration'),
        (['pytest', '-m=integration'], 'integration'),
        (['pytest', '--markers'], ''),
        (['pytest', '-m'], ''),
        # LAST wins, matching argparse — across all three spellings.
        (['pytest', '-m', 'integration', '-m', 'not integration'], 'not integration'),
        (['pytest', '-m', 'not integration', '-m', 'integration'], 'integration'),
        (['pytest', '-mnot integration', '-m=integration'], 'integration'),
        (['pytest', '-m=integration', '-m', 'contract'], 'contract'),
        # A trailing bare `-m` has no value; it must not resurrect an earlier one.
        (['pytest', '-m', 'integration', '-m'], ''),
    ],
)
def test_the_marker_expression_is_read_off_argv(argv, expected):
    """The root conftest has to decide the gate at IMPORT time, before pytest
    has parsed its CLI — `helpers_test` builds the driver list in its own module
    body, and that import happens from the conftest's module body. So `-m` is
    read off `sys.argv` here rather than from `config.getoption`.

    `--markers` is in the list because a prefix match on `-m` would swallow it.
    Repeated `-m` resolves to the LAST occurrence, which is what argparse hands
    pytest — reading the first disagrees with the run being described.
    """
    assert marker_expression_from_argv(argv) == expected


def test_the_gate_governs_the_driver_disable_vars():
    """The four `DISABLE_*` switches `helpers_test` reads are the lever the
    guard pulls: with the gate closed it sets all of them, so the driver list
    comes out EMPTY and there is no live param to run."""
    assert set(DRIVER_DISABLE_VARS) == {
        'DISABLE_NEO4J',
        'DISABLE_FALKORDB',
        'DISABLE_KUZU',
        'DISABLE_NEPTUNE',
    }


def test_the_stamp_overwrites_a_real_looking_key():
    """OVERWRITE, not fill-gaps: a real key in the developer's shell is
    precisely what un-skips this fork's live suites."""
    env = {'OPENAI_API_KEY': 'sk-proj-a-real-looking-key'}
    changed = stamp_dummy_api_keys(env)
    assert 'OPENAI_API_KEY' in changed
    assert env['OPENAI_API_KEY'] == DUMMY_KEY


def test_this_sessions_own_key_is_a_dummy():
    """Driven by a subprocess arm below; proves the stamp reached this process."""
    assert os.environ.get('OPENAI_API_KEY') == DUMMY_KEY


def test_this_sessions_driver_list_is_empty():
    """Driven by a subprocess arm below; proves the disable reached `helpers_test`
    before it built its list. Passes trivially under the sanitized recipe too —
    the arm that MEASURES is the child run under a poisoned environment."""
    from tests.helpers_test import drivers

    assert drivers == [], drivers


# ---------------------------------------------------------------------------
# The wiring (subprocess — the real conftest, the real hook)
# ---------------------------------------------------------------------------


def _run_pytest(*args, env_overrides=None):
    """Collect-only pytest in a child process, under a DELIBERATELY POISONED env.

    This is the environment the guard exists for: the reserved endpoint spelled
    out, a real-looking key present, and none of the `DISABLE_*` flags the
    hand-written recipe relied on. Nothing here is ever executed — every caller
    passes `--collect-only` — so a guard that fails still cannot dial.
    """
    env = {
        **os.environ,
        'FALKORDB_HOST': 'localhost',
        'FALKORDB_PORT': '6379',
        'OPENAI_API_KEY': 'sk-proj-a-real-looking-key',
    }
    for name in DRIVER_DISABLE_VARS:
        env.pop(name, None)
    env.pop(LIVE_GATE_ENV, None)
    env.update(env_overrides or {})
    return subprocess.run(
        [sys.executable, '-m', 'pytest', *args, '-p', 'no:cacheprovider', '--color=no'],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )


def test_live_params_are_not_generated_without_the_opt_in():
    """THE named check for BUG-109.

    A poisoned environment pointing straight at the reserved store, no opt-in,
    no `--ignore`, no `env -i` prefix — and not one live driver param is
    generated. Before the guard this collected 67 FALKORDB and 54 NEO4J params
    across the suite, every one of which would have dialled on execution.
    """
    result = _run_pytest(PARAMETRIZED_MODULE, '--collect-only', '-q')
    assert result.returncode == 0, result.stdout + result.stderr
    assert LIVE_PARAM_MARKER not in result.stdout, result.stdout
    # The tests still exist — they collect as pytest's empty-parameter-set
    # placeholder, which is a skip with no fixture setup behind it.
    assert 'NOTSET' in result.stdout, result.stdout


def test_the_driver_list_itself_comes_out_empty():
    """The structural claim, at its single source.

    Every live param in this suite is generated from ONE list —
    `helpers_test.drivers`. Asserting it is empty under a poisoned environment
    covers the whole surface at once and cannot be fooled by a module that
    merely mentions a provider enum. Driven as a subprocess so it measures the
    real conftest doing its real import-ordering job.
    """
    result = _run_pytest(
        'tests/test_live_gate_guard.py::test_this_sessions_driver_list_is_empty', '-q'
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '1 passed' in result.stdout, result.stdout


def test_no_live_surface_module_generates_a_live_param():
    """The same claim read off real collection, across every module that can
    actually reach a store."""
    result = _run_pytest(
        *LIVE_SURFACE_MODULES, '--collect-only', '-q', '--continue-on-collection-errors'
    )
    offenders = [line for line in result.stdout.splitlines() if LIVE_PARAM_MARKER in line]
    assert not offenders, offenders[:10]


def test_integration_marked_items_are_deselected_without_the_opt_in():
    """The marker arm, mirroring the sibling suite: an `integration` module
    named directly still collects nothing. Exit 5 is pytest's "no tests
    collected", the shape a full deselect takes."""
    result = _run_pytest('tests/test_entity_exclusion_int.py', '-q', '--collect-only')
    assert 'deselected' in result.stdout or result.returncode == 5, result.stdout


def test_the_opt_in_brings_the_live_params_back():
    """The guard gates, it does not delete. Collect-only and a closed port even
    here: proving the params are generated needs no connection."""
    result = _run_pytest(
        PARAMETRIZED_MODULE,
        '--collect-only',
        '-q',
        env_overrides={LIVE_GATE_ENV: '1', 'FALKORDB_PORT': '1'},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'GraphProvider.FALKORDB' in result.stdout, result.stdout
    assert 'GraphProvider.NEO4J' in result.stdout, result.stdout


def test_a_closed_gate_stamps_over_a_real_key_in_the_child():
    """The credential half of the guard, end to end through the real conftest."""
    result = _run_pytest(
        f'{PARAMETRIZED_MODULE.replace("test_add_triplet.py", "test_live_gate_guard.py")}'
        '::test_this_sessions_own_key_is_a_dummy',
        '-q',
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '1 passed' in result.stdout, result.stdout
