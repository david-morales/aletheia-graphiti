import asyncio
import pytest
from graphiti_core.utils.entity_lock_manager import EntityLockManager


@pytest.mark.asyncio
async def test_get_lock_returns_same_lock_for_same_entity():
    mgr = EntityLockManager()
    lock_a = mgr.get_lock("Location", "Madrid")
    lock_b = mgr.get_lock("Location", "Madrid")
    assert lock_a is lock_b


@pytest.mark.asyncio
async def test_get_lock_returns_different_locks_for_different_entities():
    mgr = EntityLockManager()
    lock_a = mgr.get_lock("Location", "Madrid")
    lock_b = mgr.get_lock("Location", "Barcelona")
    assert lock_a is not lock_b


@pytest.mark.asyncio
async def test_normalizes_entity_names():
    mgr = EntityLockManager()
    lock_a = mgr.get_lock("Location", "  Madrid  ")
    lock_b = mgr.get_lock("Location", "madrid")
    assert lock_a is lock_b


@pytest.mark.asyncio
async def test_register_and_get_resolved():
    from graphiti_core.nodes import EntityNode
    mgr = EntityLockManager()
    node = EntityNode(name="Madrid", group_id="g", labels=["Location"])
    mgr.register_resolved("Location", "Madrid", node)
    assert mgr.get_resolved("Location", "Madrid") is node
    assert mgr.get_resolved("Location", "Barcelona") is None


@pytest.mark.asyncio
async def test_concurrent_access_serializes_same_entity():
    """Two coroutines trying to resolve the same entity are serialized."""
    mgr = EntityLockManager()
    order = []

    first_acquired = asyncio.Event()

    async def first_worker():
        async with mgr.get_lock("Location", "Madrid"):
            first_acquired.set()
            order.append("A_start")
            await asyncio.sleep(0.05)
            order.append("A_end")

    async def second_worker():
        await first_acquired.wait()
        async with mgr.get_lock("Location", "Madrid"):
            order.append("B_start")
            order.append("B_end")

    await asyncio.gather(first_worker(), second_worker())
    assert order == ["A_start", "A_end", "B_start", "B_end"]


@pytest.mark.asyncio
async def test_concurrent_access_parallelizes_different_entities():
    """Two coroutines resolving different entities run concurrently."""
    mgr = EntityLockManager()

    a_inside = asyncio.Event()
    b_inside = asyncio.Event()
    both_observed = {"a_saw_b": False, "b_saw_a": False}

    async def worker_a():
        async with mgr.get_lock("Location", "Madrid"):
            a_inside.set()
            await b_inside.wait()
            both_observed["a_saw_b"] = True

    async def worker_b():
        async with mgr.get_lock("Location", "Barcelona"):
            b_inside.set()
            await a_inside.wait()
            both_observed["b_saw_a"] = True

    await asyncio.wait_for(asyncio.gather(worker_a(), worker_b()), timeout=2.0)
    assert both_observed["a_saw_b"] is True
    assert both_observed["b_saw_a"] is True


@pytest.mark.asyncio
async def test_clear_resets_state():
    from graphiti_core.nodes import EntityNode
    mgr = EntityLockManager()
    node = EntityNode(name="Madrid", group_id="g", labels=["Location"])
    mgr.register_resolved("Location", "Madrid", node)
    lock_before = mgr.get_lock("Location", "Madrid")

    mgr.clear()

    assert mgr.get_resolved("Location", "Madrid") is None
    lock_after = mgr.get_lock("Location", "Madrid")
    assert lock_after is not lock_before


@pytest.mark.asyncio
async def test_normalize_key_consistent_with_get_lock():
    """normalize_key produces the same key that get_lock uses internally."""
    mgr = EntityLockManager()
    key = mgr.normalize_key("Location", "  Madrid  ")
    _ = mgr.get_lock("Location", "madrid")
    assert key in mgr._locks
