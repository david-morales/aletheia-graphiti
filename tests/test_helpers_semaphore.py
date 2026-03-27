"""Tests for the global shared semaphore in helpers.py.

These tests verify that:
1. All semaphore_gather() calls without an explicit max_coroutines use a single
   shared global semaphore, so concurrent batches cannot exceed SEMAPHORE_LIMIT
   total concurrent coroutines.
2. Passing max_coroutines explicitly uses a LOCAL semaphore (backward compat).
3. reset_global_semaphore() replaces the module-level semaphore with a fresh one.
"""

import asyncio

import pytest

import graphiti_core.helpers as helpers_mod
from graphiti_core.helpers import reset_global_semaphore, semaphore_gather


@pytest.fixture(autouse=True)
def _reset_global_semaphore_after_test():
    """Reset global semaphore to None after each test to prevent state leakage."""
    yield
    helpers_mod._global_semaphore = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _counting_coroutine(
    active: list[int],
    peak: list[int],
    delay: float = 0.01,
) -> None:
    """Increments active counter while running, records peak concurrency."""
    active[0] += 1
    if active[0] > peak[0]:
        peak[0] = active[0]
    await asyncio.sleep(delay)
    active[0] -= 1


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSharedSemaphoreLimitsTotalConcurrency:
    """Global semaphore keeps all concurrent batches within the limit."""

    @pytest.mark.asyncio
    async def test_shared_semaphore_limits_total_concurrency(self):
        """Two concurrent semaphore_gather calls share one semaphore.

        Global limit=3.  Each gather has 5 coroutines (10 total).
        Peak concurrency must not exceed 3.
        """
        global_limit = 3
        reset_global_semaphore(global_limit)

        active: list[int] = [0]
        peak: list[int] = [0]

        async def make_coroutine():
            return _counting_coroutine(active, peak, delay=0.02)

        # Build 5 coroutines for each of the two concurrent batches.
        batch_a = [await make_coroutine() for _ in range(5)]
        batch_b = [await make_coroutine() for _ in range(5)]

        # Launch both batches concurrently — they share the global semaphore.
        await asyncio.gather(
            semaphore_gather(*batch_a),
            semaphore_gather(*batch_b),
        )

        assert peak[0] <= global_limit, (
            f'Peak concurrency {peak[0]} exceeded global limit {global_limit}'
        )

    @pytest.mark.asyncio
    async def test_shared_semaphore_is_same_object_across_calls(self):
        """Two default semaphore_gather calls should use the exact same semaphore."""
        reset_global_semaphore()

        sem_ids: list[int] = []

        async def capture_semaphore_id():
            sem = helpers_mod._get_global_semaphore()
            sem_ids.append(id(sem))
            await asyncio.sleep(0)

        await semaphore_gather(capture_semaphore_id(), capture_semaphore_id())

        # Both coroutines should have seen the same semaphore object.
        assert len(sem_ids) == 2
        assert sem_ids[0] == sem_ids[1], 'Different semaphore instances were used'


class TestMaxCoroutinesOverrideUsesLocalSemaphore:
    """Explicit max_coroutines creates a local semaphore, not the global one."""

    @pytest.mark.asyncio
    async def test_max_coroutines_override_uses_local_semaphore(self):
        """Explicit max_coroutines=1 limits that call to 1, ignoring global limit.

        Global limit=2; explicit limit=1.  Peak for the explicit call must be 1.
        """
        global_limit = 2
        reset_global_semaphore(global_limit)

        active: list[int] = [0]
        peak: list[int] = [0]

        coroutines = [_counting_coroutine(active, peak, delay=0.02) for _ in range(4)]

        # Pass max_coroutines=1 explicitly — should enforce a limit of 1.
        await semaphore_gather(*coroutines, max_coroutines=1)

        assert peak[0] <= 1, (
            f'Peak concurrency {peak[0]} exceeded local limit 1'
        )

    @pytest.mark.asyncio
    async def test_explicit_max_coroutines_does_not_affect_global_semaphore(self):
        """A call with explicit max_coroutines should not replace the global semaphore."""
        reset_global_semaphore()

        sem_before = helpers_mod._get_global_semaphore()

        async def noop():
            await asyncio.sleep(0)

        # Run with explicit limit — should not touch the global semaphore.
        await semaphore_gather(noop(), noop(), max_coroutines=1)

        sem_after = helpers_mod._get_global_semaphore()

        assert sem_before is sem_after, (
            'Global semaphore was replaced by an explicit-max_coroutines call'
        )


class TestResetGlobalSemaphore:
    """reset_global_semaphore() replaces the module-level semaphore."""

    @pytest.mark.asyncio
    async def test_reset_global_semaphore_creates_new_semaphore(self):
        """reset_global_semaphore() replaces the semaphore with a new object."""
        reset_global_semaphore()
        sem_before = helpers_mod._get_global_semaphore()

        reset_global_semaphore()
        sem_after = helpers_mod._get_global_semaphore()

        assert sem_before is not sem_after

    @pytest.mark.asyncio
    async def test_reset_global_semaphore_with_specified_limit(self):
        """reset_global_semaphore(limit) creates a semaphore with the given limit."""
        new_limit = 7
        reset_global_semaphore(new_limit)

        sem = helpers_mod._get_global_semaphore()

        # asyncio.Semaphore exposes _value (internal counter = initial limit when idle).
        assert sem._value == new_limit, (
            f'Expected semaphore limit {new_limit}, got {sem._value}'
        )

    @pytest.mark.asyncio
    async def test_reset_global_semaphore_without_limit_uses_default(self):
        """reset_global_semaphore() without args uses SEMAPHORE_LIMIT."""
        reset_global_semaphore()

        sem = helpers_mod._get_global_semaphore()

        assert sem._value == helpers_mod.SEMAPHORE_LIMIT

    @pytest.mark.asyncio
    async def test_reset_global_semaphore_new_semaphore_enforces_new_limit(self):
        """After reset with limit=2, concurrent calls are bounded to 2."""
        reset_global_semaphore(limit=2)

        active: list[int] = [0]
        peak: list[int] = [0]

        coroutines = [_counting_coroutine(active, peak, delay=0.02) for _ in range(6)]

        await semaphore_gather(*coroutines)

        assert peak[0] <= 2, f'Peak {peak[0]} exceeded post-reset limit of 2'
