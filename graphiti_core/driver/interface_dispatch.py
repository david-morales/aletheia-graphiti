"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import inspect
from typing import Any, TypeVar

from typing_extensions import TypeIs

__all__ = ['implements']

TInterface = TypeVar('TInterface')


def implements(interface: TInterface | None, base_method: Any) -> TypeIs[TInterface]:
    """Does `interface` actually implement `base_method`'s capability?

    `SearchInterface` and `GraphOperationsInterface` declare their whole surface
    as bare `raise NotImplementedError` stubs, and every caller in this package
    delegates to the driver's interface when it can and falls back to
    provider-generic Cypher when it cannot. That question — *can it?* — is a
    question about the CLASS, not about what a call happens to raise.

    The idiom this replaces asked it with `except NotImplementedError: pass`,
    which reads ANY `NotImplementedError` as "not implemented" — including one
    raised from deep inside a working implementation, after its query has
    already run. The generic query was then re-issued against the same driver,
    turning an implementation bug into a misleading downstream failure (BUG-108).

    Dispatching on capability instead means an override owns its own errors: it
    is called only when it exists, and anything it raises — of any type, at any
    depth — reaches the caller.

    Usage — pass the method UNBOUND, off the base class, so a typo is an
    `AttributeError` at import time rather than a silently-never-true check::

        if implements(driver.search_interface, SearchInterface.edge_bfs_search):
            return await driver.search_interface.edge_bfs_search(...)

        # ... provider-generic fallback ...

    The lookup deliberately goes through the INSTANCE — `getattr(interface, …)`,
    exactly what the call site one line below will do — rather than through
    `type(interface)`. Resolving on the class instead looks equivalent and is
    not: an instance attribute shadows an inherited one, so a subclass that
    inherits the base stub but carries a real callable on the instance would be
    reported as "not implemented" while the delegation call ran that callable.
    The generic leg would then run *as well as* the real implementation — the
    same double-execution BUG-108 is about, arrived at from the other side.

    Because attribute access on an instance builds a fresh bound-method object
    every time, identity is compared against `__func__` for genuine bound
    methods. That is what lets a non-overriding subclass still answer False: its
    bound method unwraps to the base declaration itself. Anything that is not a
    bound method — a duck-typed adapter, a plain function assigned to the
    instance, a `MagicMock` attribute — is compared as-is, so only the base
    class's own stub means "not implemented".

    Args:
        interface: The interface instance to test, or None when the driver
            supplies none.
        base_method: The unbound method as declared on the base interface class
            (e.g. `SearchInterface.edge_bfs_search`).

    Declared as a `TypeIs` so it also does the narrowing the truthiness test it
    replaces used to do: the driver attributes are `... | None`, and inside the
    guarded block a type checker must see the non-optional interface.

    Returns:
        True if `interface` resolves that method to something other than the
        base declaration — i.e. it, or one of its bases, overrides it.
    """
    if interface is None:
        return False

    # Resolve it the way the delegation call will resolve it.
    resolved = getattr(interface, base_method.__name__, None)
    if resolved is None:
        return False

    func = resolved.__func__ if inspect.ismethod(resolved) else resolved
    return func is not base_method
