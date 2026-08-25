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

    `GraphDriver` declares these attributes as `SearchInterface | None` and
    `GraphOperationsInterface | None`, so the class lookup is the real contract.
    An object that supplies the method WITHOUT inheriting the base declaration —
    a duck-typed adapter, or a test double such as `MagicMock`, whose attributes
    live on the instance rather than the class — still has the capability, and
    is treated as such: only the base class's own stub means "not implemented".

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

    name = base_method.__name__
    override = getattr(type(interface), name, None)
    if override is None:
        override = getattr(interface, name, None)
    return override is not None and override is not base_method
