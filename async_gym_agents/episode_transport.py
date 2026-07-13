import multiprocessing
import queue
import time
from typing import Any, Optional, TypeAlias

from async_gym_agents.constants import QUEUE_PUT_RETRY_TIMEOUT_SECONDS
from async_gym_agents.data_classes import (
    EpisodePacket,
    EpisodeSendResult,
    EpisodeTransportStats,
)

GenericQueue: TypeAlias = Any
GenericSemaphore: TypeAlias = Any
GenericSharedCounter: TypeAlias = Any
GenericStopEvent: TypeAlias = Any


class EpisodeSender:
    def __init__(
        self,
        worker_index: int,
        episode_queue: GenericQueue,
        ready_queue: GenericQueue,
        capacity: GenericSemaphore,
        pending_episodes: GenericSharedCounter,
        max_pending_episodes: GenericSharedCounter,
        pending_bytes: GenericSharedCounter,
        max_pending_bytes: GenericSharedCounter,
        sent_episodes: GenericSharedCounter,
        sent_bytes: GenericSharedCounter,
        use_mp: bool,
    ) -> None:
        self.worker_index = worker_index
        self._episode_queue = episode_queue
        self._ready_queue = ready_queue
        self._capacity = capacity
        self._pending_episodes = pending_episodes
        self._max_pending_episodes = max_pending_episodes
        self._pending_bytes = pending_bytes
        self._max_pending_bytes = max_pending_bytes
        self._sent_episodes = sent_episodes
        self._sent_bytes = sent_bytes
        self._use_mp = use_mp

    def send(
        self,
        packet: EpisodePacket,
        stop: GenericStopEvent,
        timeout: Optional[float],
    ) -> EpisodeSendResult:
        """Send one complete episode while respecting global and worker bounds."""
        if packet.worker_index != self.worker_index:
            raise ValueError("Episode packet was sent through the wrong worker channel")

        start_ns = time.perf_counter_ns()
        deadline = None if timeout is None else time.monotonic() + timeout
        waiting_ns = 0
        if not self._capacity.acquire(block=False):
            waiting_start_ns = time.perf_counter_ns()
            capacity_acquired = self._acquire_capacity(stop, deadline)
            waiting_ns += time.perf_counter_ns() - waiting_start_ns
            if not capacity_acquired:
                return self._build_result(False, waiting_ns, start_ns)

        try:
            try:
                self._episode_queue.put_nowait(packet)
            except queue.Full:
                waiting_start_ns = time.perf_counter_ns()
                packet_sent = self._put_packet(packet, stop, deadline)
                waiting_ns += time.perf_counter_ns() - waiting_start_ns
                if not packet_sent:
                    self._capacity.release()
                    return self._build_result(False, waiting_ns, start_ns)
        except BaseException:
            self._capacity.release()
            raise

        self._record_send(len(packet.payload))
        self._ready_queue.put(self.worker_index)
        return self._build_result(True, waiting_ns, start_ns)

    def close(self) -> None:
        """Close this process's queue handles after its worker exits."""
        if not self._use_mp:
            return
        for target_queue in (self._episode_queue, self._ready_queue):
            target_queue.close()
            target_queue.cancel_join_thread()

    def _acquire_capacity(
        self,
        stop: GenericStopEvent,
        deadline: Optional[float],
    ) -> bool:
        while not stop.is_set():
            retry_timeout = self._calculate_retry_timeout(deadline)
            if retry_timeout is None:
                return False
            if self._capacity.acquire(timeout=retry_timeout):
                return True
        return False

    def _put_packet(
        self,
        packet: EpisodePacket,
        stop: GenericStopEvent,
        deadline: Optional[float],
    ) -> bool:
        while not stop.is_set():
            retry_timeout = self._calculate_retry_timeout(deadline)
            if retry_timeout is None:
                return False
            try:
                self._episode_queue.put(
                    packet,
                    timeout=retry_timeout,
                )
                return True
            except queue.Full:
                continue
        return False

    @staticmethod
    def _calculate_retry_timeout(deadline: Optional[float]) -> Optional[float]:
        if deadline is None:
            return QUEUE_PUT_RETRY_TIMEOUT_SECONDS
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        return min(QUEUE_PUT_RETRY_TIMEOUT_SECONDS, remaining)

    def _record_send(self, payload_size: int) -> None:
        with self._pending_episodes.get_lock():
            self._pending_episodes.value += 1
            pending_episodes = self._pending_episodes.value
        with self._max_pending_episodes.get_lock():
            self._max_pending_episodes.value = max(
                self._max_pending_episodes.value,
                pending_episodes,
            )
        with self._pending_bytes.get_lock():
            self._pending_bytes.value += payload_size
            pending_bytes = self._pending_bytes.value
        with self._max_pending_bytes.get_lock():
            self._max_pending_bytes.value = max(
                self._max_pending_bytes.value,
                pending_bytes,
            )
        with self._sent_episodes.get_lock():
            self._sent_episodes.value += 1
        with self._sent_bytes.get_lock():
            self._sent_bytes.value += payload_size

    @staticmethod
    def _build_result(
        sent: bool,
        waiting_ns: int,
        start_ns: int,
    ) -> EpisodeSendResult:
        total_ns = time.perf_counter_ns() - start_ns
        return EpisodeSendResult(
            sent=sent,
            waiting_ns=waiting_ns,
            transport_ns=max(0, total_ns - waiting_ns),
        )


class EpisodeTransport:
    def __init__(
        self,
        worker_count: int,
        max_pending_episodes: int,
        use_mp: bool,
        mp_ctx: Optional[multiprocessing.context.BaseContext] = None,
    ) -> None:
        if worker_count <= 0:
            raise ValueError("Episode transport requires at least one worker")
        if max_pending_episodes <= 0:
            raise ValueError("Episode transport capacity must be positive")

        self.worker_count = worker_count
        self.max_pending_episodes = max_pending_episodes
        self.use_mp = use_mp
        self._mp_ctx = mp_ctx or multiprocessing.get_context()
        self._ready_queue = self._create_queue()
        self._episode_queues = [
            self._create_queue(maxsize=1) for _ in range(worker_count)
        ]
        self._capacity = self._mp_ctx.BoundedSemaphore(max_pending_episodes)
        self._pending_episodes = self._mp_ctx.Value("q", 0)
        self._max_pending_episodes = self._mp_ctx.Value("q", 0)
        self._pending_bytes = self._mp_ctx.Value("q", 0)
        self._max_pending_bytes = self._mp_ctx.Value("q", 0)
        self._sent_episodes = self._mp_ctx.Value("q", 0)
        self._sent_bytes = self._mp_ctx.Value("q", 0)
        self._senders = [
            EpisodeSender(
                worker_index=worker_index,
                episode_queue=self._episode_queues[worker_index],
                ready_queue=self._ready_queue,
                capacity=self._capacity,
                pending_episodes=self._pending_episodes,
                max_pending_episodes=self._max_pending_episodes,
                pending_bytes=self._pending_bytes,
                max_pending_bytes=self._max_pending_bytes,
                sent_episodes=self._sent_episodes,
                sent_bytes=self._sent_bytes,
                use_mp=use_mp,
            )
            for worker_index in range(worker_count)
        ]
        self._ready_workers = set()
        self._next_worker_index = 0
        self._received_episodes = 0
        self._shutdown = False

    def get_sender(self, worker_index: int) -> EpisodeSender:
        """Return the single-producer endpoint for one worker."""
        return self._senders[worker_index]

    def receive(self, timeout: Optional[float] = None) -> EpisodePacket:
        """Receive one complete episode, selecting ready workers round-robin."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            self._collect_ready_workers(deadline)
            worker_index = self._select_ready_worker()
            remaining = self._calculate_remaining_timeout(deadline)
            try:
                packet = self._episode_queues[worker_index].get(timeout=remaining)
            except queue.Empty:
                self._ready_workers.add(worker_index)
                continue

            self._capacity.release()
            self._record_receive(len(packet.payload))
            self._next_worker_index = (worker_index + 1) % self.worker_count
            return packet

    def get_stats(self) -> EpisodeTransportStats:
        """Return current and peak dynamic transport memory usage."""
        return EpisodeTransportStats(
            pending_episodes=self._pending_episodes.value,
            max_pending_episodes=self._max_pending_episodes.value,
            pending_bytes=self._pending_bytes.value,
            max_pending_bytes=self._max_pending_bytes.value,
            sent_episodes=self._sent_episodes.value,
            sent_bytes=self._sent_bytes.value,
            received_episodes=self._received_episodes,
        )

    def shutdown(self) -> None:
        """Close all parent-side queue handles once workers have stopped."""
        if self._shutdown:
            return
        self._shutdown = True
        if not self.use_mp:
            return
        for target_queue in [*self._episode_queues, self._ready_queue]:
            target_queue.close()
            target_queue.cancel_join_thread()

    def _create_queue(self, maxsize: int = 0) -> GenericQueue:
        if self.use_mp:
            return self._mp_ctx.Queue(maxsize=maxsize)
        return queue.Queue(maxsize=maxsize)

    def _collect_ready_workers(self, deadline: Optional[float]) -> None:
        if not self._ready_workers:
            timeout = self._calculate_remaining_timeout(deadline)
            self._ready_workers.add(self._ready_queue.get(timeout=timeout))

        while True:
            try:
                self._ready_workers.add(self._ready_queue.get_nowait())
            except queue.Empty:
                return

    def _select_ready_worker(self) -> int:
        for offset in range(self.worker_count):
            worker_index = (self._next_worker_index + offset) % self.worker_count
            if worker_index in self._ready_workers:
                self._ready_workers.remove(worker_index)
                return worker_index
        raise RuntimeError("Ready notification did not identify a worker")

    def _calculate_remaining_timeout(
        self, deadline: Optional[float]
    ) -> Optional[float]:
        if deadline is None:
            return None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise queue.Empty
        return remaining

    def _record_receive(self, payload_size: int) -> None:
        with self._pending_episodes.get_lock():
            self._pending_episodes.value -= 1
        with self._pending_bytes.get_lock():
            self._pending_bytes.value -= payload_size
        self._received_episodes += 1
