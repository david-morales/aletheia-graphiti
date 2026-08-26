"""Queue service for managing episode processing."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


class QueueService:
    """Service for managing sequential episode processing queues by group_id."""

    def __init__(self, on_episode_processed: Callable[[str], None] | None = None):
        """Initialize the queue service.

        Args:
            on_episode_processed: Called with the group_id after each queued
                episode finishes processing, successfully or not. This is the
                only moment the graph has actually changed — enqueueing changes
                nothing, and LLM extraction routinely takes longer than any
                freshness window a caller might set. A hook that raises is
                logged and ignored: it must not stop the queue worker.
        """
        # Dictionary to store queues for each group_id
        self._episode_queues: dict[str, asyncio.Queue] = {}
        # Whether a worker is CLAIMED for each group_id. Claimed, not "running":
        # the claim is taken synchronously in `add_episode_task` before the
        # worker task exists, and released only by `_release_worker`. Anything
        # that writes it from a third place reopens BUG-134.
        self._queue_workers: dict[str, bool] = {}
        # Strong references to worker tasks to prevent garbage collection
        self._worker_tasks: dict[str, asyncio.Task] = {}
        # Store the graphiti client after initialization
        self._graphiti_client: Any = None
        self._on_episode_processed = on_episode_processed

    async def add_episode_task(
        self, group_id: str, process_func: Callable[[], Awaitable[None]]
    ) -> int:
        """Add an episode processing task to the queue.

        Args:
            group_id: The group ID for the episode
            process_func: The async function to process the episode

        Returns:
            The position in the queue
        """
        # Initialize queue for this group_id if it doesn't exist
        if group_id not in self._episode_queues:
            self._episode_queues[group_id] = asyncio.Queue()

        # Add the episode processing function to the queue
        await self._episode_queues[group_id].put(process_func)

        # Start a worker for this queue if one isn't already claimed.
        #
        # The claim is taken HERE and synchronously. `asyncio.create_task` only
        # schedules the coroutine, so a worker that sets the flag itself has not
        # set it yet when this call returns — two enqueues arriving before the
        # first worker's first step would both read False and both spawn a
        # worker, interleaving extraction on one group_id (BUG-134).
        #
        # Single-threaded asyncio makes check-then-claim atomic if and only if
        # no await intervenes between them, which is why the claim sits directly
        # under the check with nothing awaitable in between. Do not insert one.
        if not self._queue_workers.get(group_id, False):
            self._queue_workers[group_id] = True
            try:
                task = asyncio.create_task(self._process_episode_queue(group_id))
            except BaseException:
                # `create_task` fails on a loop that is closing. The claim is
                # already in place, and a claim with no task to release it is a
                # permanently dead partition — worse than the race.
                self._queue_workers[group_id] = False
                raise
            self._worker_tasks[group_id] = task
            task.add_done_callback(lambda t: self._release_worker(group_id, t))

        return self._episode_queues[group_id].qsize()

    def _release_worker(self, group_id: str, task: asyncio.Task) -> None:
        """Release the worker claim for `group_id`, and drop its strong reference.

        Runs as the worker task's done callback, which is the ONLY site that
        pairs with every claim. A task cancelled before its first step never
        executes its coroutine body — so a release living in the worker's own
        `finally` silently never fires and the claim latches forever — but its
        done callback still runs. Every other terminal state (return, exception,
        cancellation mid-flight) reaches both, so the callback is a superset.

        Releasing the claim and dropping the reference in the same synchronous
        step is what keeps them consistent: the next claim can only be taken
        after this returns, so it cannot have its reference popped by a
        predecessor's callback.
        """
        # Both writers move the reference and the claim together, so the slot
        # settles the ownership question on its own: if it does not hold THIS
        # task, then either a successor owns the claim — releasing it would put
        # a second worker on one group_id, which is BUG-134 itself, not a
        # cheaper failure than it — or the slot is empty and the claim is
        # already released. Neither is ours to release.
        if self._worker_tasks.get(group_id) is not task:
            return
        del self._worker_tasks[group_id]
        self._queue_workers[group_id] = False

    async def _process_episode_queue(self, group_id: str) -> None:
        """Process episodes for a specific group_id sequentially.

        This function runs as a long-lived task that processes episodes
        from the queue one at a time.

        It does NOT touch the busy flag. `add_episode_task` claimed it before
        this task existed, and `_release_worker` releases it when this task
        reaches any terminal state; a set here would be the second writer that
        BUG-134 was made of. There is also no normal exit: the loop below parks
        in `await queue.get()` when the queue drains, so the only ways out are
        cancellation and an unexpected exception.
        """
        logger.info(f'Starting episode queue worker for group_id: {group_id}')

        try:
            while True:
                # Get the next episode processing function from the queue
                # This will wait if the queue is empty
                process_func = await self._episode_queues[group_id].get()

                try:
                    # Process the episode
                    await process_func()
                except Exception as e:
                    logger.error(
                        f'Error processing queued episode for group_id {group_id}: {str(e)}'
                    )
                finally:
                    # Mark the task as done regardless of success/failure
                    self._episode_queues[group_id].task_done()
                    self._notify_episode_processed(group_id)
        except asyncio.CancelledError:
            logger.info(f'Episode queue worker for group_id {group_id} was cancelled')
        except Exception as e:
            logger.error(f'Unexpected error in queue worker for group_id {group_id}: {str(e)}')
        finally:
            logger.info(f'Stopped episode queue worker for group_id: {group_id}')

    def _notify_episode_processed(self, group_id: str) -> None:
        """Run the completion hook, isolating the worker from it.

        Called from the worker's `finally`, so an exception escaping here would
        propagate out of the inner try, be caught by the outer handler, and stop
        the queue worker for this group entirely — one bad hook would silently
        end ingestion. Hence the swallow.
        """
        if self._on_episode_processed is None:
            return
        try:
            self._on_episode_processed(group_id)
        except Exception:
            logger.exception(
                'Episode-processed hook raised for group_id %s; continuing', group_id
            )

    def get_queue_size(self, group_id: str) -> int:
        """Get the current queue size for a group_id."""
        if group_id not in self._episode_queues:
            return 0
        return self._episode_queues[group_id].qsize()

    def is_worker_running(self, group_id: str) -> bool:
        """Whether a worker is CLAIMED for a group_id.

        Claimed is not the same as running, and the difference is the fix for
        BUG-134: this reads True from the moment `add_episode_task` claims the
        slot, which is before the worker task has run a single step, and it
        stays True for one loop hop after a worker dies, until its done callback
        releases the claim. Callers wanting "is anything being processed" want
        `get_queue_size`; what this answers is "would an enqueue start a new
        worker", which is the question the queue's own sequencing turns on.
        """
        return self._queue_workers.get(group_id, False)

    async def initialize(self, graphiti_client: Any) -> None:
        """Initialize the queue service with a graphiti client.

        Args:
            graphiti_client: The graphiti client instance to use for processing episodes
        """
        self._graphiti_client = graphiti_client
        logger.info('Queue service initialized with graphiti client')

    async def add_episode(
        self,
        group_id: str,
        name: str,
        content: str,
        source_description: str,
        episode_type: Any,
        entity_types: Any,
        uuid: str | None,
        reference_time: datetime | None = None,
        edge_types: Any = None,
        edge_type_map: Any = None,
        excluded_entity_types: list[str] | None = None,
        previous_episode_uuids: list[str] | None = None,
        custom_extraction_instructions: str | None = None,
        update_communities: bool = False,
        saga: str | None = None,
        saga_previous_episode_uuid: str | None = None,
    ) -> int:
        """Add an episode for processing.

        Args:
            group_id: The group ID for the episode
            name: Name of the episode
            content: Episode content
            source_description: Description of the episode source
            episode_type: Type of the episode
            entity_types: Entity types for extraction
            uuid: Episode UUID
            reference_time: Event occurrence time for the episode. Defaults to
                the current UTC time when not provided (bi-temporal model).
            edge_types: Optional mapping of edge (fact) type name to Pydantic model
            edge_type_map: Optional mapping of (source, target) entity type pairs to
                allowed edge type names
            excluded_entity_types: Optional list of entity type names to exclude
                from extraction
            previous_episode_uuids: Optional explicit list of prior episode UUIDs to
                use as context (overrides automatic retrieval)
            custom_extraction_instructions: Optional extra natural-language
                instructions for the extraction LLM
            update_communities: Whether to incrementally update communities after
                ingestion
            saga: Optional saga name/id to attach this episode to
            saga_previous_episode_uuid: Optional UUID of the prior episode in the saga

        Returns:
            The position in the queue
        """
        if self._graphiti_client is None:
            raise RuntimeError('Queue service not initialized. Call initialize() first.')

        async def process_episode():
            """Process the episode using the graphiti client."""
            try:
                logger.info(f'Processing episode {uuid} for group {group_id}')

                # Process the episode using the graphiti client
                await self._graphiti_client.add_episode(
                    name=name,
                    episode_body=content,
                    source_description=source_description,
                    source=episode_type,
                    group_id=group_id,
                    reference_time=reference_time or datetime.now(timezone.utc),
                    entity_types=entity_types,
                    edge_types=edge_types,
                    edge_type_map=edge_type_map,
                    excluded_entity_types=excluded_entity_types,
                    previous_episode_uuids=previous_episode_uuids,
                    custom_extraction_instructions=custom_extraction_instructions,
                    update_communities=update_communities,
                    saga=saga,
                    saga_previous_episode_uuid=saga_previous_episode_uuid,
                    uuid=uuid,
                )

                logger.info(f'Successfully processed episode {uuid} for group {group_id}')

            except Exception as e:
                logger.error(f'Failed to process episode {uuid} for group {group_id}: {str(e)}')
                raise

        # Use the existing add_episode_task method to queue the processing
        return await self.add_episode_task(group_id, process_episode)
