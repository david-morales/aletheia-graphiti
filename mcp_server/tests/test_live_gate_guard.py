"""BUG-85 / H-F2: the offline suite must be unable to reach a real graph store.

`test_falkordb_dialect_integration.py` opens `FalkorDB(host='localhost', port=6379)`
in a module-scoped fixture. The host and port are HARDCODED — `FALKORDB_URI` is
not consulted at all — and while the module carried an `integration` marker,
nothing in this suite ever deselected it. So a plain `pytest tests` collected it,
ran it, and dialled 6379, which on a developer machine is a real, populated
store (here: the RESERVED production FalkorDB). The whole sanitize recipe for
this suite ended in `--ignore=tests/test_falkordb_dialect_integration.py`, which
is a habit, not a guard: it protects only the runs that remember it.

Two guards, both in `tests/conftest.py` so they cannot be forgotten per-run:

1. `integration`-marked items are DESELECTED unless the run opts in — by env
   gate, or by naming the marker itself in `-m` (which is what CI's live job
   does, and what makes this safe to land without touching the workflow).
2. With the gate off, a `FALKORDB_URI` pointing at a non-loopback host or at the
   reserved 6379 is a session error rather than a quiet invitation.

The unit tests below cover the decision; the subprocess tests cover the WIRING,
which is the half a unit test cannot see — the hook has to actually be installed
in the real conftest for any of it to matter.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests._env_guard import LIVE_GATE_ENV, live_gate_is_open, unsafe_falkordb_uri

TESTS_DIR = Path(__file__).parent
DIALECT_MODULE = TESTS_DIR / 'test_falkordb_dialect_integration.py'


# ---------------------------------------------------------------------------
# The decision (fast, in-process)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'uri',
    [
        '',
        'redis://127.0.0.1:9',
        'redis://localhost:6399',
        'redis://[::1]:16379',
    ],
)
def test_a_throwaway_loopback_endpoint_is_allowed(uri):
    """The sanitize recipe's own `redis://127.0.0.1:9` must keep working."""
    assert unsafe_falkordb_uri(uri) is None, uri


@pytest.mark.parametrize(
    'uri,why',
    [
        ('redis://localhost:6379', 'the reserved default port'),
        ('redis://127.0.0.1:6379', 'the reserved default port'),
        ('redis://falkordb.example.com:6399', 'a non-loopback host'),
        ('redis://10.0.0.4:6399', 'a non-loopback host'),
    ],
)
def test_a_real_looking_endpoint_is_refused(uri, why):
    reason = unsafe_falkordb_uri(uri)
    assert reason is not None, (uri, why)
    assert uri in reason and LIVE_GATE_ENV in reason, reason


def test_an_unparseable_uri_is_refused_rather_than_ignored():
    """Fail closed: a URI we cannot read is not a URI we can call harmless."""
    assert unsafe_falkordb_uri('not a uri at all') is not None


def test_the_gate_reads_the_environment(monkeypatch):
    monkeypatch.delenv(LIVE_GATE_ENV, raising=False)
    assert live_gate_is_open({}) is False
    monkeypatch.setenv(LIVE_GATE_ENV, '1')
    assert live_gate_is_open({}) is True


def test_naming_the_marker_opts_in_without_the_env_gate(monkeypatch):
    """CI's live job selects with `-m integration` and sets no gate of ours.

    Asking for the marker BY NAME is an explicit opt-in; the accident this
    guards against is the run that never mentions it.
    """
    monkeypatch.delenv(LIVE_GATE_ENV, raising=False)
    assert live_gate_is_open({'-m': 'integration'}) is True
    assert live_gate_is_open({'-m': 'not integration'}) is False
    assert live_gate_is_open({'-m': 'contract'}) is False


# ---------------------------------------------------------------------------
# The wiring (subprocess — the real conftest, the real hook)
# ---------------------------------------------------------------------------


def _run_pytest(*args, env_overrides=None):
    """Run pytest in a child process from the mcp_server directory.

    Sanitized to this suite's own recipe, then overridden per case: dummy
    provider keys and a dead loopback endpoint, so a guard that fails to fire
    still cannot reach anything.
    """
    env = {
        **os.environ,
        'OPENAI_API_KEY': 'sk-dummy',
        'ANTHROPIC_API_KEY': 'sk-dummy',
        'FALKORDB_URI': 'redis://127.0.0.1:9',
        'DISABLE_NEO4J': '1',
        'DISABLE_FALKORDB': '1',
    }
    env.pop(LIVE_GATE_ENV, None)
    env.update(env_overrides or {})
    return subprocess.run(
        [sys.executable, '-m', 'pytest', *args, '-p', 'no:cacheprovider', '--color=no'],
        cwd=TESTS_DIR.parent,
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )


def test_the_dialect_module_collects_nothing_without_the_gate():
    """THE named check: no `--ignore` flag, no opt-in, no network.

    Exit 5 is pytest's "no tests collected", which is the shape a full deselect
    takes when it is the only module named.
    """
    result = _run_pytest(str(DIALECT_MODULE.relative_to(TESTS_DIR.parent)), '-q')
    assert result.returncode == 5, result.stdout + result.stderr
    assert 'deselected' in result.stdout, result.stdout
    # Nothing ran, so no fixture opened a connection to anything.
    assert ' passed' not in result.stdout, result.stdout
    assert ' failed' not in result.stdout, result.stdout


def test_the_gate_selects_the_dialect_module_back_in():
    """Opt in and the module is collectable again — the guard gates, not deletes."""
    result = _run_pytest(
        str(DIALECT_MODULE.relative_to(TESTS_DIR.parent)),
        '--collect-only',
        '-q',
        env_overrides={LIVE_GATE_ENV: '1'},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'test_falkordb_dialect_integration.py' in result.stdout, result.stdout
    assert 'tests collected' in result.stdout, result.stdout
    assert 'deselected' not in result.stdout, result.stdout


def test_the_offline_suite_refuses_a_run_aimed_at_the_reserved_port():
    """Gate off + a real-looking endpoint = a loud session error, not a quiet run."""
    result = _run_pytest(
        'tests/test_flavours.py',
        '--collect-only',
        '-q',
        env_overrides={'FALKORDB_URI': 'redis://localhost:6379'},
    )
    assert result.returncode != 0, result.stdout
    combined = result.stdout + result.stderr
    assert 'redis://localhost:6379' in combined, combined
    assert LIVE_GATE_ENV in combined, combined


def test_the_gate_permits_the_endpoint_it_was_opened_for():
    """With the gate ON the operator has said what they are pointing at.

    CI's live job runs exactly this shape (`FALKORDB_URI: redis://localhost:6379`
    against a throwaway container), so the refusal must not fire there.
    """
    result = _run_pytest(
        'tests/test_flavours.py',
        '--collect-only',
        '-q',
        env_overrides={'FALKORDB_URI': 'redis://localhost:6379', LIVE_GATE_ENV: '1'},
    )
    assert result.returncode == 0, result.stdout + result.stderr


# ---------------------------------------------------------------------------
# The offline suite is keyless BY CONSTRUCTION (task 4)
# ---------------------------------------------------------------------------


def test_the_bootstrap_stamps_every_provider_key():
    from tests._env_guard import _PROVIDER_KEY_VARS, DUMMY_KEY, stamp_dummy_api_keys

    env = {'OPENAI_API_KEY': 'sk-proj-a-real-looking-key', 'UNRELATED': 'kept'}
    stamp_dummy_api_keys(env)

    assert env['UNRELATED'] == 'kept'
    for name in _PROVIDER_KEY_VARS:
        assert env[name] == DUMMY_KEY, name


def test_this_sessions_own_key_is_a_dummy():
    """The bootstrap ran for THIS process — so assert on it, not on a copy.

    With the gate closed there is no arrangement of shell and `.env` under which
    a live key should still be readable here.
    """
    from tests._env_guard import DUMMY_KEY

    assert os.environ.get('OPENAI_API_KEY') == DUMMY_KEY


def test_a_dotenv_carrying_a_real_key_cannot_win_after_the_stamp(tmp_path):
    """The mechanism, exercised in the order the server module actually uses.

    `graphiti_mcp_server` calls `load_dotenv(mcp_server/.env)` at MODULE IMPORT,
    and on a developer machine that file carries a live key. `load_dotenv` does
    not override an already-set variable — so a value stamped BEFORE the first
    project import wins, and the offline suite is keyless no matter what is in
    the file or in the developer's shell.

    Both arms run in a child process against a TEMPORARY `.env`; the real one is
    never read, and this process's own environment is never mutated.
    """
    dotenv = tmp_path / '.env'
    dotenv.write_text('OPENAI_API_KEY=sk-proj-a-real-looking-key\n')

    script = (
        'import os, sys, json\n'
        f'sys.path.insert(0, {str(TESTS_DIR.parent)!r})\n'
        'from dotenv import load_dotenv\n'
        'from tests._env_guard import stamp_dummy_api_keys, DUMMY_KEY\n'
        'if os.environ.pop("ARM") == "stamped":\n'
        '    stamp_dummy_api_keys()\n'
        f'load_dotenv({str(dotenv)!r})\n'
        'print(json.dumps({"key": os.environ.get("OPENAI_API_KEY"), "dummy": DUMMY_KEY}))\n'
    )

    def _arm(name):
        env = {k: v for k, v in os.environ.items() if k != 'OPENAI_API_KEY'}
        env['ARM'] = name
        out = subprocess.run(
            [sys.executable, '-c', script],
            capture_output=True, text=True, timeout=60, env=env, check=True,
        )
        return __import__('json').loads(out.stdout.strip().splitlines()[-1])

    stamped = _arm('stamped')
    assert stamped['key'] == stamped['dummy'], stamped

    # The control arm: without the stamp the file's real key lands in the
    # environment. If this ever stops being true the test above proves nothing.
    unstamped = _arm('bare')
    assert unstamped['key'] == 'sk-proj-a-real-looking-key', unstamped
