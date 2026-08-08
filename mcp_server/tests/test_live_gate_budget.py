"""The live gate's timeout budget, checked mechanically rather than in a comment.

`pytest.ini` sets `timeout = 300` and pytest-timeout is a DEV dependency, so the
ceiling is live under CI's `uv sync` and absent from a bare local interpreter. That
asymmetry is the classic "passes on a laptop, flakes in CI" shape, and it hides a real
one here: the e2e test's own waits add up to more than 300s, so the ini ceiling could
fire before they were exhausted and report a timeout instead of "the episode never
processed" — a CI red pointing at the wrong thing.

These assertions run everywhere, because they read the two files rather than importing
the live module (which self-skips without a key and a database — so a guard living
inside it would never run in the suite that needs it).
"""

from __future__ import annotations

import ast
import configparser
import pathlib

import pytest

TESTS_DIR = pathlib.Path(__file__).parent
LIVE_MODULE = TESTS_DIR / 'test_live_falkordb_int.py'
PYTEST_INI = TESTS_DIR / 'pytest.ini'

E2E_TEST = 'test_end_to_end_add_search_context_delete_clear'


def _ini() -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    parser.read(PYTEST_INI)
    return parser


def _live_tree() -> ast.Module:
    return ast.parse(LIVE_MODULE.read_text())


def _module_constants() -> dict[str, float]:
    """Every module-level numeric constant of the live gate, by name."""
    values: dict[str, float] = {}
    for node in _live_tree().body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = node.targets[0]
        if not (isinstance(target, ast.Name) and isinstance(node.value, ast.Constant)):
            continue
        value = node.value.value
        # `bool` is an `int` subclass; a flag is not a duration.
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values[target.id] = float(value)
    return values


def test_the_timeout_marker_is_registered_so_strict_markers_cannot_break_collection():
    """`--strict-markers` is on and pytest-timeout is optional at runtime.

    Without the ini registration, any interpreter lacking the plugin fails to COLLECT
    every module carrying `@pytest.mark.timeout` — it does not merely ignore the
    marker. That would take out far more than the live gate.
    """
    markers = _ini()['pytest']['markers']
    assert 'timeout:' in markers, (
        'the `timeout` marker is not registered in pytest.ini; with --strict-markers '
        'and no pytest-timeout installed, collection fails rather than degrades'
    )


def test_the_e2e_test_carries_an_explicit_timeout():
    for node in ast.walk(_live_tree()):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == E2E_TEST:
            names = {
                d.func.attr
                for d in node.decorator_list
                if isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute)
            }
            assert 'timeout' in names, (
                f'{E2E_TEST} has no explicit @pytest.mark.timeout, so it inherits the '
                f'300s ini ceiling — which is below its own worst-case waits'
            )
            return
    pytest.fail(f'{E2E_TEST} not found in {LIVE_MODULE.name}')


def test_the_e2e_budget_really_does_exceed_the_ini_ceiling():
    """The arithmetic itself, so the comment beside it cannot quietly go stale.

    Recomputed from the live module's OWN constants. If someone shortens the waits far
    enough that the budget fits under the ini ceiling, this fails and says the marker
    is now redundant — which is a fine outcome, and a deliberate prompt to delete it
    rather than leave an unexplained override behind.
    """
    consts = _module_constants()
    required = {
        '_SERVER_BOOT_SECONDS',
        '_ASSUMED_CALL_SECONDS',
        '_EPISODE_WAIT_SECONDS',
        '_SEARCH_ATTEMPTS',
        '_SEARCH_POLL',
        '_TAIL_CALLS',
    }
    missing = required - consts.keys()
    assert not missing, f'the live gate no longer declares {sorted(missing)}'

    worst_case = (
        consts['_SERVER_BOOT_SECONDS']
        + consts['_ASSUMED_CALL_SECONDS']
        + consts['_EPISODE_WAIT_SECONDS']
        + 2 * consts['_SEARCH_ATTEMPTS'] * (consts['_SEARCH_POLL'] + consts['_ASSUMED_CALL_SECONDS'])
        + consts['_TAIL_CALLS'] * consts['_ASSUMED_CALL_SECONDS']
    )
    ini_ceiling = float(_ini()['pytest']['timeout'])

    assert worst_case > ini_ceiling, (
        f"the e2e worst case is now {worst_case:.0f}s, under the {ini_ceiling:.0f}s ini "
        f'ceiling — @pytest.mark.timeout on {E2E_TEST} is no longer load-bearing and '
        f'should be removed along with the budget comment'
    )
