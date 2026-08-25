"""BUG-108: interface delegation must dispatch on CAPABILITY, not on `NotImplementedError`.

The old idiom was::

    if driver.search_interface:
        try:
            return await driver.search_interface.edge_bfs_search(...)
        except NotImplementedError:
            pass
    # ... provider-generic Cypher ...

which reads ANY `NotImplementedError` as "this interface does not implement the
method" — including one raised from DEEPER inside a working implementation. The
interface's query has already run; the exception is swallowed; the generic
Cypher is then re-issued against the same driver. Measured on the AGE flavour
(ledger BUG-108): `AGE SQL issued: 1 / generic Cypher ALSO issued: 1 / result: [[]]`.
A genuine implementation bug is converted into a misleading downstream failure.

The fix dispatches on whether the interface OVERRIDES the method
(`graphiti_core.driver.interface_dispatch.implements`), so an error raised by an
override — of any type, at any depth — propagates to the caller.

Three behaviours are pinned at every kind of site:

1. an override that fails deep inside  -> the error PROPAGATES, generic never issued
2. an interface that does not override -> falls back to generic, exactly as before
3. an override that works              -> its result is returned, generic never issued

plus a fork-wide AST guard that the old idiom cannot come back.
"""

from __future__ import annotations

import ast
import pathlib
import re
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

import graphiti_core
from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.graph_operations.graph_operations import GraphOperationsInterface
from graphiti_core.driver.interface_dispatch import implements
from graphiti_core.driver.search_interface.search_interface import SearchInterface
from graphiti_core.edges import EpisodicEdge
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import edge_bfs_search

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _driver(*, search_interface=None, graph_operations_interface=None):
    """A driver whose only real behaviour is recording generic-Cypher calls."""
    driver = MagicMock()
    driver.provider = GraphProvider.FALKORDB
    driver.search_interface = search_interface
    driver.graph_operations_interface = graph_operations_interface
    driver.execute_query = AsyncMock(return_value=([], None, None))
    return driver


class _DeepFailure(SearchInterface):
    """Overrides the method; the implementation runs, then fails deep inside.

    This is the trap: the SQL has already been issued when the error surfaces.
    """

    ran: int = 0

    async def edge_bfs_search(self, driver, *args, **kwargs):
        self.ran += 1
        return await self._hydrate()

    async def _hydrate(self):
        # Deep inside a WORKING implementation — e.g. a helper that has not been
        # written yet, or a library that raises NotImplementedError on some input.
        raise NotImplementedError('hydration helper is missing')


class _NoOverride(SearchInterface):
    """A real interface that genuinely does not implement edge_bfs_search."""


class _Working(SearchInterface):
    ran: int = 0

    async def edge_bfs_search(self, driver, *args, **kwargs):
        self.ran += 1
        return ['the-interface-result']


class _OpsDeepFailure(GraphOperationsInterface):
    ran: int = 0

    async def episodic_edge_save(self, edge, driver):
        self.ran += 1
        raise NotImplementedError('save helper is missing')


class _OpsNoOverride(GraphOperationsInterface):
    pass


class _OpsWorking(GraphOperationsInterface):
    ran: int = 0

    async def episodic_edge_save(self, edge, driver):
        self.ran += 1
        return 'the-interface-result'


def _episodic_edge():
    return EpisodicEdge(
        uuid='e1',
        source_node_uuid='s1',
        target_node_uuid='t1',
        group_id='g',
        created_at=datetime.now(timezone.utc),
    )


# --------------------------------------------------------------------------
# the dispatch primitive
# --------------------------------------------------------------------------


def test_implements_is_false_for_no_interface():
    assert implements(None, SearchInterface.edge_bfs_search) is False


def test_implements_is_false_when_the_method_is_not_overridden():
    assert implements(_NoOverride(), SearchInterface.edge_bfs_search) is False


def test_implements_is_false_for_the_base_class_itself():
    assert implements(SearchInterface(), SearchInterface.edge_bfs_search) is False


def test_implements_is_true_when_the_method_is_overridden():
    assert implements(_Working(), SearchInterface.edge_bfs_search) is True


def test_implements_is_true_for_an_override_that_raises_not_implemented_error():
    """The whole point: an override is a capability regardless of what it raises."""
    assert implements(_DeepFailure(), SearchInterface.edge_bfs_search) is True


def test_implements_is_true_through_an_intermediate_subclass():
    """A subclass that inherits an OVERRIDE still has the capability."""

    class Grandchild(_Working):
        pass

    assert implements(Grandchild(), SearchInterface.edge_bfs_search) is True


def test_implements_accepts_a_duck_typed_interface():
    """An adapter that supplies the method without inheriting the base still has
    the capability — the attribute may live on the instance, as it does on a Mock."""
    duck = MagicMock()
    duck.edge_bfs_search = AsyncMock()

    assert implements(duck, SearchInterface.edge_bfs_search) is True


def test_implements_follows_an_instance_attribute_that_shadows_an_inherited_stub():
    """Adversarial review F2 — the shape that makes the class lookup wrong.

    A subclass that does NOT override inherits the base stub, so resolving on
    `type(interface)` finds the stub and reports "not implemented". But an
    instance attribute SHADOWS the inherited one, so the delegation call one
    line later runs the real callable — and the generic leg runs too. That is
    BUG-108's double execution reached from the other side, so the answer has
    to follow what the call site resolves.
    """

    class Sub(SearchInterface):
        pass

    async def real_impl(*args, **kwargs):
        return ['real result']

    iface = Sub()
    object.__setattr__(iface, 'edge_bfs_search', real_impl)

    assert Sub.edge_bfs_search is SearchInterface.edge_bfs_search, (
        'precondition: the CLASS still resolves to the inherited stub'
    )
    assert iface.edge_bfs_search is real_impl, (
        'precondition: the call site reaches the instance attribute'
    )
    assert implements(iface, SearchInterface.edge_bfs_search) is True


@pytest.mark.asyncio
async def test_a_shadowing_instance_attribute_is_delegated_to_and_generic_is_not_run():
    """The same shape driven through a real call site, end to end."""

    class Sub(SearchInterface):
        pass

    async def real_impl(*args, **kwargs):
        return ['real result']

    iface = Sub()
    object.__setattr__(iface, 'edge_bfs_search', real_impl)
    driver = _driver(search_interface=iface)

    result = await edge_bfs_search(driver, ['origin'], 2, SearchFilters())

    assert result == ['real result']
    assert driver.execute_query.await_count == 0, (
        'the generic Cypher must not run alongside a real implementation'
    )


def test_implements_is_false_for_an_object_without_the_method():
    class Empty:
        pass

    assert implements(Empty(), SearchInterface.edge_bfs_search) is False


def test_implements_discriminates_per_method():
    iface = _Working()
    assert implements(iface, SearchInterface.edge_bfs_search) is True
    assert implements(iface, SearchInterface.node_bfs_search) is False


# --------------------------------------------------------------------------
# search_interface site: edge_bfs_search
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deep_failure_propagates_and_generic_is_not_reissued():
    """RED before the fix: the error was swallowed and generic Cypher re-issued."""
    iface = _DeepFailure()
    driver = _driver(search_interface=iface)

    with pytest.raises(NotImplementedError, match='hydration helper is missing'):
        await edge_bfs_search(driver, ['origin'], 2, SearchFilters())

    assert iface.ran == 1, 'the interface implementation must have run'
    assert driver.execute_query.await_count == 0, (
        'the generic Cypher must NOT be re-issued after an implementation failed'
    )


@pytest.mark.asyncio
async def test_absent_override_still_falls_back_to_generic():
    """Unchanged behaviour: a genuinely unimplemented method uses the generic leg."""
    driver = _driver(search_interface=_NoOverride())

    await edge_bfs_search(driver, ['origin'], 2, SearchFilters())

    assert driver.execute_query.await_count >= 1, 'the generic Cypher must be issued'


@pytest.mark.asyncio
async def test_no_interface_at_all_falls_back_to_generic():
    driver = _driver(search_interface=None)

    await edge_bfs_search(driver, ['origin'], 2, SearchFilters())

    assert driver.execute_query.await_count >= 1


@pytest.mark.asyncio
async def test_working_override_is_returned_and_generic_never_issued():
    iface = _Working()
    driver = _driver(search_interface=iface)

    result = await edge_bfs_search(driver, ['origin'], 2, SearchFilters())

    assert result == ['the-interface-result']
    assert iface.ran == 1
    assert driver.execute_query.await_count == 0


# --------------------------------------------------------------------------
# graph_operations_interface site: EpisodicEdge.save
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ops_deep_failure_propagates_and_generic_is_not_reissued():
    iface = _OpsDeepFailure()
    driver = _driver(graph_operations_interface=iface)

    with pytest.raises(NotImplementedError, match='save helper is missing'):
        await _episodic_edge().save(driver)

    assert iface.ran == 1
    assert driver.execute_query.await_count == 0, (
        'a half-run save must not be silently re-issued through the generic leg'
    )


@pytest.mark.asyncio
async def test_ops_absent_override_still_falls_back_to_generic():
    driver = _driver(graph_operations_interface=_OpsNoOverride())

    await _episodic_edge().save(driver)

    assert driver.execute_query.await_count >= 1


@pytest.mark.asyncio
async def test_ops_working_override_is_returned_and_generic_never_issued():
    iface = _OpsWorking()
    driver = _driver(graph_operations_interface=iface)

    result = await _episodic_edge().save(driver)

    assert result == 'the-interface-result'
    assert iface.ran == 1
    assert driver.execute_query.await_count == 0


# --------------------------------------------------------------------------
# fork-wide guard: the old idiom must not come back
# --------------------------------------------------------------------------

# Derived from the IMPORTED package, never from `__file__` arithmetic: a path
# built by counting `.parents[...]` can silently point at nothing, and an AST
# scan over nothing reports zero offenders — a guard that passes by scanning
# no files is worse than no guard. `test_the_guard_scans_the_real_package`
# below pins that this path really is the package under test.
_CORE = pathlib.Path(graphiti_core.__file__).resolve().parent
_DELEGATION = re.compile(r'\w+\.\w*(?:search|graph_operations)_interface\.\w+')


def _idiom_sites() -> list[str]:
    """Every `try: ... await <driver>.<x>_interface.<m>(...) ... except NotImplementedError`."""
    offenders = []
    for path in sorted(_CORE.rglob('*.py')):
        source = path.read_text()
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Try):
                continue
            catches = False
            for handler in node.handlers:
                exc = handler.type
                names = []
                if isinstance(exc, ast.Name):
                    names = [exc.id]
                elif isinstance(exc, ast.Tuple):
                    names = [e.id for e in exc.elts if isinstance(e, ast.Name)]
                if 'NotImplementedError' in names:
                    catches = True
            if not catches:
                continue
            segment = ast.get_source_segment(source, node) or ''
            if _DELEGATION.search(segment):
                offenders.append(f'{path.relative_to(_CORE.parent)}:{node.lineno}')
    return offenders


def test_no_interface_delegation_catches_not_implemented_error():
    """BUG-108: dispatch on capability, never on the exception — at EVERY site.

    A partial migration makes the idiom inconsistent, which the ledger calls
    worse than the trap itself. This guard is what makes 'all sites' checkable.
    """
    offenders = _idiom_sites()
    assert offenders == [], (
        'interface delegation must be guarded by `implements(...)`, not by '
        'catching NotImplementedError:\n  ' + '\n  '.join(offenders)
    )


def test_the_guard_scans_the_real_package():
    """A scan over an empty file set reports zero offenders and passes vacuously."""
    files = list(_CORE.rglob('*.py'))
    assert _CORE.name == 'graphiti_core'
    assert (_CORE / 'search' / 'search_utils.py').is_file()
    assert len(files) > 50, f'only {len(files)} files scanned under {_CORE}'


def test_the_guard_counts_delegation_sites():
    """The package really does contain the delegation sites this guard is about,
    so a zero-offender result means they were migrated — not that they vanished."""
    total = 0
    for path in _CORE.rglob('*.py'):
        total += len(_DELEGATION.findall(path.read_text()))
    assert total > 50, f'expected the fork-wide delegation surface, found {total}'


def test_the_guard_can_see_the_idiom():
    """The guard above is only meaningful if it detects the shape it forbids."""
    sample = (
        'async def f(driver):\n'
        '    if driver.search_interface:\n'
        '        try:\n'
        '            return await driver.search_interface.edge_bfs_search(driver)\n'
        '        except NotImplementedError:\n'
        '            pass\n'
    )
    tree = ast.parse(sample)
    tries = [n for n in ast.walk(tree) if isinstance(n, ast.Try)]
    assert len(tries) == 1
    segment = ast.get_source_segment(sample, tries[0])
    assert _DELEGATION.search(segment or '') is not None
