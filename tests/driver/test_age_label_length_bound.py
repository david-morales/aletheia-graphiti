"""BUG-47: an AGE label of 64+ characters hard-fails the whole save.

Since `aletheia-v0.9.3` the driver materialises labels EXPLICITLY (BUG-38), and
`_ensure_label`/`_label_exists` bind the label through a `::name` cast.
PostgreSQL truncates an over-long name only when it is a literal in the SQL text;
a *parameter* bound to `name` at 64+ characters is rejected outright:

    NameTooLongError: identifier too long  (42622)

so the save raises mid-ingest and nothing is written — where the pre-BUG-38
implicit-DDL path silently truncated to 63 and carried on (with a silent
COLLISION hazard: two names sharing their first 63 characters became one label).
Measured boundary on the live bed: 62 OK · 63 OK · 64 raises.

The deterministic ontology lane cannot reach it (the longest local-name in either
policia TTL is 23 characters, `INVOLUCRA_DOCUMENTACION`); the exposure is
LLM-produced narrative relationship names, which `_node_label`/`_edge_label`
admitted with no length cap at all.

Fix under test: ONE shared bound, `_bounded_label`, applied by both label
choosers. A label longer than PostgreSQL's 63-character limit is shortened
DETERMINISTICALLY — kept prefix plus a `_<8 hex>` digest of the WHOLE original —
and the shortening is logged at WARNING with both forms. That keeps a long ingest
alive where a hard rejection would lose it, without reintroducing the truncation
collision: the digest covers the part that truncation would have thrown away, so
two names that differ only past character 63 still get two labels.

Determinism has to hold ACROSS processes and driver instances, not merely within
one: two writers on the same graph that shortened the same name differently would
write two labels for one relationship type. That is why the digest is
`hashlib.sha256` and not `hash()` — `hash()` of a str is salted per process by
PYTHONHASHSEED, so a `hash()`-based implementation passes an in-process test and
fails in production. `test_shortening_is_identical_in_a_separate_process` is the
one that discriminates between the two.

NOT PROVEN HERE: the real 63/64 PostgreSQL boundary. These helpers are pure
string functions and are tested as such, offline; that a 63-character label is
accepted by `create_vlabel`/`create_elabel` and a 64-character one is not is a
property of PostgreSQL's NAMEDATALEN, measured on the :5433 AGE bed and recorded
in the ledger. There is no AGE bed in this lane, so the live regression at the
boundary is deferred to the AGE-bed decision / Lane C.
"""

import logging
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from graphiti_core.driver.graph_operations.age_graph_operations import (
    _IDENT_RE,
    _MAX_LABEL_LEN,
    _bounded_label,
    _edge_label,
    _node_label,
    _warn_label_shortened,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _forget_warned_labels():
    """The warn-once dedupe is process-scoped, so without this a label another
    test already warned about would silently stop warning here — order-dependent
    tests that pass alone and fail in a suite."""
    _warn_label_shortened.cache_clear()
    yield
    _warn_label_shortened.cache_clear()


# --------------------------------------------------------------- the boundary


def test_postgres_name_limit_is_63():
    """The bound is PostgreSQL's, not a taste: NAMEDATALEN 64 minus the NUL."""
    assert _MAX_LABEL_LEN == 63


@pytest.mark.parametrize('length', [1, 10, 62, 63])
def test_labels_at_or_under_the_limit_are_returned_untouched(length):
    """62 OK · 63 OK — the measured accept side of the boundary. A label that
    PostgreSQL can store must reach the graph EXACTLY as written, or every
    already-ingested label would be rewritten by this change."""
    label = 'L' + 'a' * (length - 1)
    assert len(label) == length
    assert _bounded_label(label) == label


@pytest.mark.parametrize('length', [64, 65, 120, 400])
def test_labels_over_the_limit_are_shortened_to_the_limit(length):
    """64 raises on the live bed — so 64 is the first length that must shorten."""
    label = 'L' + 'a' * (length - 1)
    bounded = _bounded_label(label)
    assert len(bounded) == _MAX_LABEL_LEN
    assert bounded != label


def test_a_shortened_label_is_still_a_safe_identifier():
    """The shortened form goes straight into a Cypher pattern and a `::name`
    cast, so it has to satisfy the same identifier shape the choosers enforce."""
    bounded = _bounded_label('Relacion_' + 'X' * 200)
    assert _IDENT_RE.match(bounded), bounded


def test_the_shortened_label_keeps_a_readable_prefix_of_the_original():
    """Shortening is only defensible if the result is still traceable to what it
    came from; a bare digest would make every long label unreadable."""
    original = 'ESTABLECE_RELACION_DE_COLABORACION_OPERATIVA_CONTINUADA_CON_LA_ORGANIZACION'
    bounded = _bounded_label(original)
    assert original.startswith(bounded[: _MAX_LABEL_LEN - 9])


# ------------------------------------------------------------- no collisions


def test_two_names_differing_only_past_the_limit_do_not_collide():
    """The historical hazard of silent truncation, in its exact shape. Plain
    truncation maps both of these onto one label and merges two relationship
    types into one; the digest covers the whole name, so they stay two."""
    shared = 'A' * 80
    assert _bounded_label(shared + 'X') != _bounded_label(shared + 'Y')


def test_a_long_name_does_not_collide_with_its_own_truncation():
    """A graph can hold both `X*63` and `X*200`. Shortening the second onto the
    first would silently fuse them."""
    short = 'X' * _MAX_LABEL_LEN
    assert _bounded_label('X' * 200) != _bounded_label(short) == short


def test_many_long_names_sharing_a_prefix_stay_distinct():
    names = ['P' * 90 + f'_{i}' for i in range(200)]
    bounded = {_bounded_label(n) for n in names}
    assert len(bounded) == len(names)


# -------------------------------------------------------------- determinism


def test_shortening_is_stable_across_repeated_calls():
    label = 'Q' * 130
    assert len({_bounded_label(label) for _ in range(50)}) == 1


def test_shortening_carries_no_per_process_state():
    """Interleaving different names must not change any of their results — a
    counter- or cache-based scheme would fail this."""
    a, b = 'M' * 100 + '_a', 'M' * 100 + '_b'
    first_a = _bounded_label(a)
    _bounded_label(b)
    _bounded_label(b)
    assert _bounded_label(a) == first_a


@pytest.mark.parametrize('hash_seed', ['0', '12345'])
def test_shortening_is_identical_in_a_separate_process(hash_seed):
    """The test that rules out `hash()`.

    Python salts `hash()` of a str per process (PYTHONHASHSEED), so a
    `hash()`-based shortening is stable inside one process and different in the
    next — two writers on the same graph would then create two labels for one
    relationship type. Run the same input under two different seeds in a fresh
    interpreter and require the same answer as this process.

    Offline by construction: a stdlib-only probe, no network, no store.
    """
    label = 'Z' * 200
    probe = (
        'import sys; sys.path.insert(0, sys.argv[1]); '
        'from graphiti_core.driver.graph_operations.age_graph_operations '
        'import _bounded_label; print(_bounded_label(sys.argv[2]))'
    )
    result = subprocess.run(
        [sys.executable, '-c', probe, str(REPO_ROOT), label],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(REPO_ROOT),
        # Minimal environment: no API keys reach the child, and nothing but the
        # hash seed varies between the two runs.
        env={'PYTHONHASHSEED': hash_seed, 'PATH': os.environ.get('PATH', '')},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == _bounded_label(label)


# ------------------------------------------------------- both choosers apply it


def test_node_label_shortens_an_over_long_leaf_instead_of_failing():
    """The leaf is the semantically correct pick, so an over-long leaf is
    shortened — NOT skipped in favour of a shorter ancestor, which would silently
    mis-type the vertex."""
    leaf = 'Organizacion' + 'Criminal' * 12
    assert len(leaf) > _MAX_LABEL_LEN
    assert _node_label(['Entity', 'Organizacion', leaf]) == _bounded_label(leaf)


def test_edge_label_shortens_an_over_long_relationship_name():
    """LLM-produced narrative predicates are the whole exposure. An
    identifier-safe one that is merely long keeps its typed label, shortened —
    falling back to RELATES_TO would throw the type away."""
    name = 'ESTABLECE_' + 'CONTACTO_' * 12
    assert _IDENT_RE.match(name) and len(name) > _MAX_LABEL_LEN
    assert _edge_label(name) == _bounded_label(name)


def test_edge_label_still_falls_back_for_a_name_that_is_not_an_identifier():
    """The length bound must not weaken the identifier check that precedes it."""
    assert _edge_label('actúa junto a, en varias ocasiones, ' + 'x' * 90) == 'RELATES_TO'


def test_node_label_still_falls_back_to_entity():
    assert _node_label(['Entity']) == 'Entity'
    assert _node_label([]) == 'Entity'


@pytest.mark.parametrize('length', [62, 63])
def test_choosers_leave_labels_at_the_accept_side_of_the_boundary_alone(length):
    label = 'L' + 'a' * (length - 1)
    assert _node_label(['Entity', label]) == label
    assert _edge_label(label) == label


# ------------------------------------------------------------------- warning


def test_shortening_logs_both_the_original_and_the_result(caplog):
    """A label that is not the one the caller asked for must never be silent —
    the WARNING is what makes a shortened label traceable in an ingest log."""
    original = 'V' * 100
    with caplog.at_level(logging.WARNING, logger='graphiti_core.driver.graph_operations'):
        bounded = _bounded_label(original)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, caplog.records
    message = warnings[0].getMessage()
    assert original in message
    assert bounded in message


def test_a_label_within_the_limit_logs_nothing(caplog):
    with caplog.at_level(logging.WARNING, logger='graphiti_core.driver.graph_operations'):
        _bounded_label('W' * _MAX_LABEL_LEN)
    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


def test_the_same_long_label_warns_once_however_many_times_it_is_seen(caplog):
    """`_bounded_label` is called far more often than there are labels — the bulk
    path runs `_node_label` three times per node — so a warning per CALL floods
    the log and destroys the traceability it exists for. Measured before the
    dedupe: 30 lines for a 10-node bulk of one label, ~15,000 on a 5,000-entity
    ingest."""
    label = 'F' * 120
    with caplog.at_level(logging.WARNING, logger='graphiti_core.driver.graph_operations'):
        results = {_bounded_label(label) for _ in range(200)}

    assert len(results) == 1  # still the same answer every time
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


def test_a_different_long_label_still_warns(caplog):
    """Deduping must be per NAME. Silencing the second distinct label would hide
    exactly the case an operator needs to see."""
    with caplog.at_level(logging.WARNING, logger='graphiti_core.driver.graph_operations'):
        _bounded_label('G' * 120)
        _bounded_label('G' * 120)
        _bounded_label('H' * 120)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2
    assert 'G' * 120 in warnings[0].getMessage()
    assert 'H' * 120 in warnings[1].getMessage()


def test_the_dedupe_is_bounded_and_cannot_grow_without_limit():
    """A plain seen-set would grow forever in a long-lived server fed an endless
    stream of novel LLM-produced names. The LRU caps it: the lifetime is 'until
    evicted', so an evicted label warns again — a re-notification, never a flood
    and never unbounded memory."""
    for i in range(3000):
        _bounded_label(f'{"I" * 100}_{i}')

    info = _warn_label_shortened.cache_info()
    assert info.maxsize is not None
    assert info.currsize <= info.maxsize


@pytest.mark.asyncio
async def test_a_bulk_save_of_one_long_label_warns_once_not_once_per_call(caplog):
    """The measured shape, through the real bulk writer rather than the helper."""
    from datetime import datetime, timezone

    from graphiti_core.driver.graph_operations.age_graph_operations import AGEGraphOperations
    from graphiti_core.nodes import EntityNode

    long_class = 'Organizacion' + 'Criminal' * 12
    rows = [
        {
            'uuid': f'bulk-largo-{i}',
            'name': f'B{i}',
            'group_id': 'bug47',
            'summary': '',
            'labels': ['Entity', long_class],
            'created_at': datetime.now(timezone.utc),
            'name_embedding': None,
        }
        for i in range(10)
    ]

    driver = _CapturingDriver()
    with caplog.at_level(logging.WARNING, logger='graphiti_core.driver.graph_operations'):
        await AGEGraphOperations().node_save_bulk(EntityNode, driver, None, rows)

    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


# ------------------------------------------- the identifier gate is `\Z`-anchored


@pytest.mark.parametrize('suffix', ['\n', '\n\n'])
def test_a_name_with_a_trailing_newline_is_not_an_identifier(suffix):
    """Python's `$` also matches just BEFORE a trailing newline, so a `$`-anchored
    gate admitted `'BROADER\\n'`. One character, but it splits the two things that
    must never disagree: the DECLARED label keeps the newline (it goes through a
    `::name` parameter) while the MERGEd one ends at the whitespace in the Cypher
    text. Declared `BROADER\\n`, merged `BROADER` — the write is back on implicit
    DDL for a label nobody declared."""
    assert not _IDENT_RE.match('BROADER' + suffix)
    assert _edge_label('BROADER' + suffix) == 'RELATES_TO'
    assert _node_label(['Entity', 'Persona' + suffix]) == 'Entity'


def test_the_plain_name_without_the_newline_is_still_accepted():
    """The control: closing the hole must not reject the legitimate name."""
    assert _IDENT_RE.match('BROADER')
    assert _edge_label('BROADER') == 'BROADER'
    assert _node_label(['Entity', 'Persona']) == 'Persona'


@pytest.mark.asyncio
async def test_a_trailing_newline_name_never_reaches_the_declaration():
    """The end-to-end consequence: whatever the writer declares is what it MERGEs
    on, and neither of them carries a newline."""
    from datetime import datetime, timezone

    from graphiti_core.driver.graph_operations.age_graph_operations import AGEGraphOperations
    from graphiti_core.edges import EntityEdge
    from graphiti_core.nodes import EntityNode

    driver = _CapturingDriver()
    ops = AGEGraphOperations()
    node = EntityNode(
        name='N',
        group_id='bug47',
        labels=['Entity', 'Persona\n'],
        created_at=datetime.now(timezone.utc),
        summary='',
        attributes={},
    )
    node.name_embedding = None
    edge = EntityEdge(
        source_node_uuid=node.uuid,
        target_node_uuid=node.uuid,
        name='BROADER\n',
        group_id='bug47',
        fact='f',
        created_at=datetime.now(timezone.utc),
    )
    edge.fact_embedding = None

    await ops.node_save(node, driver)
    await ops.edge_save(edge, driver)

    assert driver.ensured == [('v', 'Entity'), ('e', 'RELATES_TO')]
    for _kind, label in driver.ensured:
        assert '\n' not in label
    assert _labels_named_in_merges(driver) - set(driver.ensured) == set()


# ------------------------------------------- the write declares what it MERGEs

# The bound is worthless if the label a writer MERGEs on and the label it
# declares to `ensure_*_label` can differ: the declared one would be created and
# the MERGE would then take AGE's implicit-DDL path on the other (BUG-38). Both
# come from the same chooser call, and this drives the real writers to prove it.


class _CapturingDriver:
    def __init__(self) -> None:
        self.queries: list[str] = []
        self.ensured: list[tuple[str, str]] = []
        self._node_tbl = 'age_nodes'
        self._edge_tbl = 'age_edges'

    async def execute_query(self, cypher: str, **_kwargs):
        self.queries.append(cypher)
        return [{'uuid': 'merged'}], ['uuid'], None

    async def execute_sql(self, _sql: str, *_args):
        return []

    async def ensure_vertex_label(self, label: str) -> None:
        self.ensured.append(('v', label))

    async def ensure_edge_label(self, label: str) -> None:
        self.ensured.append(('e', label))


def _labels_named_in_merges(driver: _CapturingDriver) -> set[tuple[str, str]]:
    named: set[tuple[str, str]] = set()
    for query in driver.queries:
        for merge in re.findall(r'MERGE\s+(.+?)(?:\s+SET\b|$)', query):
            named |= {('v', m) for m in re.findall(r'\(\s*\w*\s*:(\w+)', merge)}
            named |= {('e', m) for m in re.findall(r'\[\s*\w*\s*:(\w+)', merge)}
    return named


@pytest.mark.asyncio
async def test_an_over_long_label_is_declared_and_merged_under_the_same_name():
    from datetime import datetime, timezone

    from graphiti_core.driver.graph_operations.age_graph_operations import AGEGraphOperations
    from graphiti_core.edges import EntityEdge
    from graphiti_core.nodes import EntityNode

    long_class = 'Organizacion' + 'Criminal' * 12
    long_predicate = 'ESTABLECE_' + 'CONTACTO_' * 12

    driver = _CapturingDriver()
    ops = AGEGraphOperations()
    node = EntityNode(
        name='N',
        group_id='bug47',
        labels=['Entity', long_class],
        created_at=datetime.now(timezone.utc),
        summary='',
        attributes={},
    )
    node.name_embedding = None
    edge = EntityEdge(
        source_node_uuid=node.uuid,
        target_node_uuid=node.uuid,
        name=long_predicate,
        group_id='bug47',
        fact='f',
        created_at=datetime.now(timezone.utc),
    )
    edge.fact_embedding = None

    await ops.node_save(node, driver)
    await ops.edge_save(edge, driver)

    named = _labels_named_in_merges(driver)
    assert named, 'no MERGE emitted to check'
    assert named - set(driver.ensured) == set()
    assert ('v', _bounded_label(long_class)) in named
    assert ('e', _bounded_label(long_predicate)) in named
    for _kind, label in named:
        assert len(label) <= _MAX_LABEL_LEN, label
