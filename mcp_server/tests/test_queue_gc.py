"""Tests for the queue worker task's lifecycle: GC prevention (#1176) and the
per-group_id sequential guarantee (BUG-134).

Two properties of the same object, and the second one is why the first one was
not enough. `QueueService` keeps strong references to the worker `asyncio.Task`
objects so the garbage collector cannot cancel them mid-execution — but a
reference only protects the task it still points at, and a second worker
spawned for the same group_id both overwrote that reference AND broke the
sequential processing the queue exists to provide.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
from unittest.mock import AsyncMock

import pytest

from services import queue_service as queue_service_module
from services.queue_service import QueueService

WORKER_COROUTINE = '_process_episode_queue'


def _live_worker_tasks() -> list[asyncio.Task]:
    """Every pending worker task on this loop, found by coroutine rather than
    by the service's own bookkeeping.

    `_worker_tasks` cannot answer "how many workers are running": the BUG-134
    defect was a second worker *overwriting* that dict entry, so the dict reads
    1 while two workers race. The loop is the only honest witness.
    """
    return [
        task
        for task in asyncio.all_tasks()
        if getattr(task.get_coro(), '__qualname__', '').endswith(WORKER_COROUTINE)
    ]


async def _cancel_live_workers() -> None:
    """Stop every worker this test started, including ones the service lost
    track of — an abandoned worker would otherwise leak into the next test."""
    tasks = _live_worker_tasks()
    for task in tasks:
        task.cancel()
    for task in tasks:
        try:
            await task
        except asyncio.CancelledError:
            pass
    await asyncio.sleep(0)


async def _wait_for(predicate, timeout: float = 2.0) -> None:
    """Poll until `predicate()` holds, so a failure reports the state rather
    than hanging the suite."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)


class TestWorkerTasksDict:
    """Verify the _worker_tasks dict is initialised on construction."""

    def test_worker_tasks_dict_exists(self):
        svc = QueueService()
        assert hasattr(svc, '_worker_tasks')
        assert isinstance(svc._worker_tasks, dict)
        assert len(svc._worker_tasks) == 0


class TestTaskStoredAfterAdd:
    """Adding an episode task must store the worker Task reference."""

    @pytest.mark.asyncio
    async def test_task_stored_after_add(self):
        svc = QueueService()
        group_id = 'test-group'

        # A simple async callable that the worker will invoke
        process_func = AsyncMock()

        await svc.add_episode_task(group_id, process_func)

        # The worker task should now be stored under the group_id key
        assert group_id in svc._worker_tasks
        task = svc._worker_tasks[group_id]
        assert isinstance(task, asyncio.Task)
        assert not task.done()

        # Clean up: cancel the long-lived worker so it doesn't block
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


class TestTaskCleanedUpAfterDone:
    """The done callback must remove the reference when the worker finishes."""

    @pytest.mark.asyncio
    async def test_task_cleaned_up_after_done(self):
        svc = QueueService()
        group_id = 'cleanup-group'

        process_func = AsyncMock()

        await svc.add_episode_task(group_id, process_func)

        assert group_id in svc._worker_tasks
        task = svc._worker_tasks[group_id]

        # Cancel the worker to make it finish
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Allow the event loop to run the done callback
        await asyncio.sleep(0)

        # The done callback should have removed the entry
        assert group_id not in svc._worker_tasks


def _tracking_process(order: list[str], tag: str):
    """A process_func that records when it starts and when it ends, with a real
    suspension in between — exactly where LLM extraction lives, and the only
    window in which a second worker can interleave."""

    async def process() -> None:
        order.append(f'start:{tag}')
        await asyncio.sleep(0.02)
        order.append(f'end:{tag}')

    return process


class TestOneWorkerPerGroupId:
    """BUG-134 — the busy flag was set INSIDE the worker coroutine but read in
    `add_episode_task`, so two enqueues landing before the first worker got its
    first step both read False and both spawned a worker.

    `asyncio.create_task` only *schedules*; the enqueue that created the task
    returns before the worker runs a single line. On the default ingest path
    (`group_id` omitted → all traffic shares one) this put concurrent LLM
    extraction on one partition, which is the single thing the queue exists to
    prevent.
    """

    @pytest.mark.asyncio
    async def test_the_flag_is_claimed_before_the_worker_gets_its_first_step(self):
        """The mechanism, in one assertion: the claim must be visible the moment
        the enqueue returns, not one loop iteration later."""
        svc = QueueService()
        try:
            await svc.add_episode_task('claim-group', AsyncMock())

            assert svc.is_worker_running('claim-group') is True, (
                'the busy flag is still False after the enqueue returned — it is '
                'being set inside the worker coroutine, so the next enqueue will '
                'spawn a second worker (BUG-134)'
            )
        finally:
            await _cancel_live_workers()

    @pytest.mark.asyncio
    async def test_two_immediate_enqueues_spawn_exactly_one_worker(self):
        """The measured reproduction: two enqueues with no await between them."""
        svc = QueueService()
        group_id = 'race-group'
        order: list[str] = []

        try:
            await svc.add_episode_task(group_id, _tracking_process(order, 'A'))
            await svc.add_episode_task(group_id, _tracking_process(order, 'B'))

            workers = _live_worker_tasks()
            assert len(workers) == 1, (
                f'{len(workers)} workers running for one group_id — the second '
                f'enqueue spawned its own (BUG-134)'
            )

            await _wait_for(lambda: len(order) == 4)
            assert order == ['start:A', 'end:A', 'start:B', 'end:B'], (
                f'processing was not sequential on one group_id: {order}'
            )
        finally:
            await _cancel_live_workers()

    @pytest.mark.asyncio
    async def test_the_first_workers_reference_survives_a_second_enqueue(self):
        """The GC half. The second worker overwrote `_worker_tasks[group_id]`,
        dropping the first task's only strong reference — the exact protection
        this module was written for (#1176)."""
        svc = QueueService()
        group_id = 'reference-group'
        order: list[str] = []

        try:
            await svc.add_episode_task(group_id, _tracking_process(order, 'A'))
            first = svc._worker_tasks[group_id]

            await svc.add_episode_task(group_id, _tracking_process(order, 'B'))

            assert svc._worker_tasks[group_id] is first, (
                'the second enqueue replaced the first worker task reference, '
                'leaving the running worker unprotected from GC (BUG-134)'
            )
        finally:
            await _cancel_live_workers()

    @pytest.mark.asyncio
    async def test_distinct_group_ids_still_get_their_own_worker(self):
        """The no-regression half: the guarantee is per-group_id, not global.
        Separate partitions must still run concurrently."""
        svc = QueueService()
        group_ids = ('g1', 'g2', 'g3')
        order: list[str] = []

        try:
            for group_id in group_ids:
                await svc.add_episode_task(group_id, _tracking_process(order, group_id))

            assert len(_live_worker_tasks()) == len(group_ids)
            assert {svc._worker_tasks[g] for g in group_ids} == set(_live_worker_tasks())
            for group_id in group_ids:
                assert svc.is_worker_running(group_id) is True

            await _wait_for(lambda: len(order) == 2 * len(group_ids))
            assert sorted(order) == sorted(
                [f'{prefix}:{g}' for g in group_ids for prefix in ('start', 'end')]
            ), order
        finally:
            await _cancel_live_workers()


class TestTheClaimIsReleasedOnEveryPath:
    """A latched flag is worse than the race it prevents: no worker will ever be
    spawned for that group_id again, so its queue fills and nothing drains it —
    a permanently dead partition. Every claim therefore needs a release that
    cannot be skipped."""

    @pytest.mark.asyncio
    async def test_a_create_task_failure_does_not_latch_the_claim(self, monkeypatch):
        """`create_task` raises on a loop that is shutting down. The claim is
        already in place by then, so the failure path has to undo it."""
        svc = QueueService()
        group_id = 'no-loop-group'

        def exploding_create_task(*_args, **_kwargs):
            raise RuntimeError('no running event loop')

        monkeypatch.setattr(asyncio, 'create_task', exploding_create_task)

        with pytest.raises(RuntimeError):
            await svc.add_episode_task(group_id, AsyncMock())

        assert svc.is_worker_running(group_id) is False, (
            'the busy flag stayed latched after create_task failed — this '
            'group_id can never start a worker again'
        )
        assert group_id not in svc._worker_tasks

    @pytest.mark.asyncio
    async def test_a_worker_cancelled_before_its_first_step_releases_the_claim(self):
        """Measured asyncio semantics: a task cancelled before it runs a single
        step never executes its coroutine body, so a release living in the
        worker's own `finally` never fires. The done callback does fire, which
        is why the release lives there."""
        svc = QueueService()
        group_id = 'stillborn-group'

        await svc.add_episode_task(group_id, AsyncMock())
        task = svc._worker_tasks[group_id]
        task.cancel()  # no await since create_task: the body never ran
        try:
            await task
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(0)

        assert svc.is_worker_running(group_id) is False, (
            'a worker that never got a first step left the claim latched, '
            'killing this group_id permanently'
        )
        assert group_id not in svc._worker_tasks

        # And the partition is genuinely revivable, not just flag-clean.
        order: list[str] = []
        try:
            await svc.add_episode_task(group_id, _tracking_process(order, 'after'))
            await _wait_for(lambda: order == ['start:after', 'end:after'])
            assert order == ['start:after', 'end:after'], order
        finally:
            await _cancel_live_workers()


class TestTheBusyFlagHasOneWriterPerTransition:
    """The defect was not a wrong value, it was two owners: the flag was set in
    the worker and read in the enqueue. Locking the writer set keeps a future
    edit from reintroducing the split.
    """

    def test_only_the_claim_and_the_release_write_the_flag(self):
        """AST, not substring matching: find every assignment whose target is a
        `self._queue_workers[...]` subscript and name the function it sits in."""
        tree = ast.parse(inspect.getsource(queue_service_module))
        writers: set[str] = set()

        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            for inner in ast.walk(node):
                targets: list[ast.expr] = []
                if isinstance(inner, ast.Assign):
                    targets = list(inner.targets)
                elif isinstance(inner, ast.AugAssign | ast.AnnAssign):
                    targets = [inner.target]
                for target in targets:
                    if (
                        isinstance(target, ast.Subscript)
                        and isinstance(target.value, ast.Attribute)
                        and target.value.attr == '_queue_workers'
                    ):
                        writers.add(node.name)

        assert writers == {'add_episode_task', '_release_worker'}, (
            f'the busy flag is written from {sorted(writers)}; it must be claimed '
            f'in add_episode_task and released in _release_worker, nowhere else'
        )
