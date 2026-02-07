"""Tests for queue worker task GC prevention (#1176).

Verifies that QueueService stores strong references to asyncio.Task objects
created for queue workers, preventing the garbage collector from cancelling
them mid-execution.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from services.queue_service import QueueService


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
