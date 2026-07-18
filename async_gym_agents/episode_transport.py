import multiprocessing
import queue
import struct
import time
from multiprocessing.connection import Connection, wait
from typing import Any, Callable, Dict, Optional, Set, Tuple, TypeAlias

from async_gym_agents.constants import (
    EPISODE_KIND_OFF_POLICY_CODE,
    EPISODE_KIND_ON_POLICY_CODE,
    EPISODE_PACKET_HEADER,
    PER_WORKER_PENDING_EPISODE_CAPACITY,
    SHARED_COUNTER_TYPE_CODE,
    TRANSPORT_ACQUIRE_RETRY_TIMEOUT_SECONDS,
)
from async_gym_agents.data_classes import (
    EpisodePacket,
    EpisodeSendResult,
    EpisodeTransportStats,
)
from async_gym_agents.enums import EpisodeKind

GenericSemaphore: TypeAlias = Any
GenericSharedCounter: TypeAlias = Any
GenericStopEvent: TypeAlias = Any


def _encode_episode_packet_header(packet: EpisodePacket, enqueue_ns: int) -> bytes:
    """Encode fixed-size packet metadata without copying the episode payload."""
    if packet.episode_kind is EpisodeKind.ON_POLICY:
        episode_kind_code = EPISODE_KIND_ON_POLICY_CODE
    elif packet.episode_kind is EpisodeKind.OFF_POLICY:
        episode_kind_code = EPISODE_KIND_OFF_POLICY_CODE
    else:
        raise ValueError(f"Unsupported episode kind: {packet.episode_kind!r}")

    has_policy_version = packet.policy_version is not None
    policy_version = 0 if packet.policy_version is None else packet.policy_version
    return EPISODE_PACKET_HEADER.pack(
        has_policy_version,
        policy_version,
        episode_kind_code,
        packet.transition_count,
        enqueue_ns,
    )


def _decode_pipe_packet(
    worker_index: int,
    header: bytes,
    payload: bytes,
) -> EpisodePacket:
    """Reconstruct one packet from its fixed-size header and raw payload frame."""
    try:
        (
            has_policy_version,
            policy_version,
            episode_kind_code,
            transition_count,
            enqueue_ns,
        ) = EPISODE_PACKET_HEADER.unpack(header)
    except struct.error as error:
        raise ValueError("Received an invalid episode packet header") from error

    if has_policy_version not in (0, 1):
        raise ValueError("Episode packet header has an invalid policy marker")

    if episode_kind_code == EPISODE_KIND_ON_POLICY_CODE:
        episode_kind = EpisodeKind.ON_POLICY
    elif episode_kind_code == EPISODE_KIND_OFF_POLICY_CODE:
        episode_kind = EpisodeKind.OFF_POLICY
    else:
        raise ValueError(
            f"Episode packet header has unknown kind code {episode_kind_code}"
        )

    return EpisodePacket(
        worker_index=worker_index,
        policy_version=policy_version if has_policy_version else None,
        episode_kind=episode_kind,
        transition_count=transition_count,
        payload=payload,
        transport_enqueue_ns=enqueue_ns,
    )


class EpisodeSender:
    def __init__(
        self,
        worker_index: int,
        send_connection: Connection,
        worker_capacity: GenericSemaphore,
        capacity: GenericSemaphore,
        pending_episodes: GenericSharedCounter,
        max_pending_episodes: GenericSharedCounter,
        pending_bytes: GenericSharedCounter,
        max_pending_bytes: GenericSharedCounter,
        sent_episodes: GenericSharedCounter,
        sent_bytes: GenericSharedCounter,
        clock: Callable[[], int],
    ) -> None:
        self.worker_index = worker_index
        self._send_connection = send_connection
        self._worker_capacity = worker_capacity
        self._capacity = capacity
        self._pending_episodes = pending_episodes
        self._max_pending_episodes = max_pending_episodes
        self._pending_bytes = pending_bytes
        self._max_pending_bytes = max_pending_bytes
        self._sent_episodes = sent_episodes
        self._sent_bytes = sent_bytes
        self._clock = clock
        self._closed = False

    def send(
        self,
        packet: EpisodePacket,
        stop: GenericStopEvent,
        timeout: Optional[float],
    ) -> EpisodeSendResult:
        """Stream one complete episode while respecting global backpressure."""
        if packet.worker_index != self.worker_index:
            raise ValueError("Episode packet was sent through the wrong worker channel")

        start_ns = self._clock()
        if stop.is_set():
            return self._build_result(False, 0, start_ns)

        deadline = None if timeout is None else time.monotonic() + timeout
        waiting_ns = 0
        worker_capacity_acquired, worker_waiting_ns = self._acquire_slot(
            self._worker_capacity,
            stop,
            deadline,
        )
        waiting_ns += worker_waiting_ns
        if not worker_capacity_acquired:
            return self._build_result(False, waiting_ns, start_ns)

        capacity_acquired, capacity_waiting_ns = self._acquire_slot(
            self._capacity,
            stop,
            deadline,
        )
        waiting_ns += capacity_waiting_ns
        if not capacity_acquired:
            self._worker_capacity.release()
            return self._build_result(False, waiting_ns, start_ns)

        enqueue_ns = self._clock()
        payload_size = len(packet.payload)
        try:
            header = _encode_episode_packet_header(packet, enqueue_ns)
        except BaseException:
            self._capacity.release()
            self._worker_capacity.release()
            raise
        self._record_send_started(payload_size)
        try:
            self._send_connection.send_bytes(header)
            self._send_connection.send_bytes(packet.payload)
        except BaseException as error:
            self._rollback_send(payload_size)
            self._capacity.release()
            self._worker_capacity.release()
            if stop.is_set() and isinstance(error, (EOFError, OSError)):
                return self._build_result(False, waiting_ns, start_ns)
            raise

        return self._build_result(True, waiting_ns, start_ns)

    def close(self) -> None:
        """Close this worker's sending endpoint."""
        if self._closed:
            return
        self._closed = True
        self._send_connection.close()

    def _acquire_slot(
        self,
        target: GenericSemaphore,
        stop: GenericStopEvent,
        deadline: Optional[float],
    ) -> Tuple[bool, int]:
        if target.acquire(block=False):
            return True, 0

        waiting_start_ns = self._clock()
        while not stop.is_set():
            retry_timeout = self._calculate_retry_timeout(deadline)
            if retry_timeout is None:
                return False, self._clock() - waiting_start_ns
            if target.acquire(timeout=retry_timeout):
                return True, self._clock() - waiting_start_ns
        return False, self._clock() - waiting_start_ns

    @staticmethod
    def _calculate_retry_timeout(deadline: Optional[float]) -> Optional[float]:
        if deadline is None:
            return TRANSPORT_ACQUIRE_RETRY_TIMEOUT_SECONDS
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        return min(TRANSPORT_ACQUIRE_RETRY_TIMEOUT_SECONDS, remaining)

    def _record_send_started(self, payload_size: int) -> None:
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

    def _rollback_send(self, payload_size: int) -> None:
        with self._pending_episodes.get_lock():
            self._pending_episodes.value -= 1
        with self._pending_bytes.get_lock():
            self._pending_bytes.value -= payload_size
        with self._sent_episodes.get_lock():
            self._sent_episodes.value -= 1
        with self._sent_bytes.get_lock():
            self._sent_bytes.value -= payload_size

    def _build_result(
        self,
        sent: bool,
        waiting_ns: int,
        start_ns: int,
    ) -> EpisodeSendResult:
        total_ns = self._clock() - start_ns
        return EpisodeSendResult(
            sent=sent,
            waiting_ns=max(0, waiting_ns),
            transport_ns=max(0, total_ns - waiting_ns),
        )


class EpisodeTransport:
    def __init__(
        self,
        worker_count: int,
        max_pending_episodes: int,
        use_mp: bool,
        mp_ctx: Optional[multiprocessing.context.BaseContext] = None,
        clock: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if worker_count <= 0:
            raise ValueError("Episode transport requires at least one worker")
        if max_pending_episodes <= 0:
            raise ValueError("Episode transport capacity must be positive")

        self.worker_count = worker_count
        self.max_pending_episodes = max_pending_episodes
        self.use_mp = use_mp
        self._mp_ctx = mp_ctx or multiprocessing.get_context()
        self._clock = clock
        connections = [self._mp_ctx.Pipe(duplex=False) for _ in range(worker_count)]
        self._receive_connections = [pair[0] for pair in connections]
        send_connections = [pair[1] for pair in connections]
        self._worker_by_connection = {
            connection: worker_index
            for worker_index, connection in enumerate(self._receive_connections)
        }
        self._worker_capacities = [
            self._mp_ctx.BoundedSemaphore(PER_WORKER_PENDING_EPISODE_CAPACITY)
            for _ in range(worker_count)
        ]
        self._capacity = self._mp_ctx.BoundedSemaphore(max_pending_episodes)
        self._pending_episodes = self._mp_ctx.Value(SHARED_COUNTER_TYPE_CODE, 0)
        self._max_pending_episodes = self._mp_ctx.Value(SHARED_COUNTER_TYPE_CODE, 0)
        self._pending_bytes = self._mp_ctx.Value(SHARED_COUNTER_TYPE_CODE, 0)
        self._max_pending_bytes = self._mp_ctx.Value(SHARED_COUNTER_TYPE_CODE, 0)
        self._sent_episodes = self._mp_ctx.Value(SHARED_COUNTER_TYPE_CODE, 0)
        self._sent_bytes = self._mp_ctx.Value(SHARED_COUNTER_TYPE_CODE, 0)
        self._senders = [
            EpisodeSender(
                worker_index=worker_index,
                send_connection=send_connections[worker_index],
                worker_capacity=self._worker_capacities[worker_index],
                capacity=self._capacity,
                pending_episodes=self._pending_episodes,
                max_pending_episodes=self._max_pending_episodes,
                pending_bytes=self._pending_bytes,
                max_pending_bytes=self._max_pending_bytes,
                sent_episodes=self._sent_episodes,
                sent_bytes=self._sent_bytes,
                clock=clock,
            )
            for worker_index in range(worker_count)
        ]
        self._ready_workers: Set[int] = set()
        self._pending_headers: Dict[int, bytes] = {}
        self._next_worker_index = 0
        self._received_episodes = 0
        self._received_bytes = 0
        self._receive_attempts = 0
        self._receive_timeouts = 0
        self._receive_timeouts_with_pending = 0
        self._readiness_wait_ns = 0
        self._readiness_wait_count = 0
        self._readiness_timeout_ns = 0
        self._readiness_timeouts = 0
        self._payload_receive_ns = 0
        self._payload_receive_count = 0
        self._payload_receive_timeout_ns = 0
        self._payload_receive_timeouts = 0
        self._pipe_latency_ns = 0
        self._pipe_latency_count = 0
        self._interrupted = False
        self._shutdown = False

    def get_sender(self, worker_index: int) -> EpisodeSender:
        """Return the single-producer endpoint for one worker."""
        return self._senders[worker_index]

    def close_parent_senders(self) -> None:
        """Close parent copies after multiprocessing workers inherit their endpoints."""
        if not self.use_mp:
            return
        for sender in self._senders:
            sender.close()

    def receive(self, timeout: Optional[float] = None) -> EpisodePacket:
        """Receive one complete episode from readable worker pipes fairly."""
        self._receive_attempts += 1
        deadline = None if timeout is None else time.monotonic() + timeout
        try:
            self._collect_ready_workers(deadline)
            worker_index = self._select_ready_worker()
            packet = self._receive_packet(worker_index, deadline)
            self._record_receive(packet)
            self._capacity.release()
            self._worker_capacities[worker_index].release()
            self._next_worker_index = (worker_index + 1) % self.worker_count
            return packet
        except queue.Empty:
            self._receive_timeouts += 1
            if self._pending_episodes.value > 0:
                self._receive_timeouts_with_pending += 1
            raise

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
            received_bytes=self._received_bytes,
            receive_attempts=self._receive_attempts,
            receive_timeouts=self._receive_timeouts,
            receive_timeouts_with_pending=self._receive_timeouts_with_pending,
            readiness_wait_ns=self._readiness_wait_ns,
            readiness_wait_count=self._readiness_wait_count,
            readiness_timeout_ns=self._readiness_timeout_ns,
            readiness_timeouts=self._readiness_timeouts,
            payload_receive_ns=self._payload_receive_ns,
            payload_receive_count=self._payload_receive_count,
            payload_receive_timeout_ns=self._payload_receive_timeout_ns,
            payload_receive_timeouts=self._payload_receive_timeouts,
            pipe_latency_ns=self._pipe_latency_ns,
            pipe_latency_count=self._pipe_latency_count,
        )

    def interrupt(self) -> None:
        """Close receiving endpoints so blocked worker sends can stop promptly."""
        if self._interrupted:
            return
        self._interrupted = True
        for connection in self._receive_connections:
            connection.close()

    def shutdown(self) -> None:
        """Close all parent-side pipe handles."""
        if self._shutdown:
            return
        self._shutdown = True
        self.interrupt()
        for sender in self._senders:
            sender.close()

    def _collect_ready_workers(self, deadline: Optional[float]) -> None:
        if self._ready_workers:
            return

        timeout = self._calculate_remaining_timeout(deadline)
        readiness_start_ns = self._clock()
        ready_connections = wait(self._receive_connections, timeout=timeout)
        readiness_ns = self._clock() - readiness_start_ns
        self._record_readiness_wait(
            readiness_ns,
            timed_out=not ready_connections,
        )
        if not ready_connections:
            raise queue.Empty

        self._ready_workers.update(
            self._worker_by_connection[connection] for connection in ready_connections
        )

    def _select_ready_worker(self) -> int:
        for offset in range(self.worker_count):
            worker_index = (self._next_worker_index + offset) % self.worker_count
            if worker_index in self._ready_workers:
                self._ready_workers.remove(worker_index)
                return worker_index
        raise RuntimeError("Readable pipe did not identify a worker")

    def _receive_packet(
        self,
        worker_index: int,
        deadline: Optional[float],
    ) -> EpisodePacket:
        connection = self._receive_connections[worker_index]
        receive_start_ns = self._clock()
        header = self._pending_headers.pop(worker_index, None)
        if header is None:
            header = connection.recv_bytes()

        try:
            remaining = self._calculate_remaining_timeout(deadline)
        except queue.Empty:
            self._pending_headers[worker_index] = header
            self._record_payload_receive(
                self._clock() - receive_start_ns,
                timed_out=True,
            )
            raise

        if remaining is not None and not connection.poll(remaining):
            self._pending_headers[worker_index] = header
            self._record_payload_receive(
                self._clock() - receive_start_ns,
                timed_out=True,
            )
            raise queue.Empty

        payload = connection.recv_bytes()
        self._record_payload_receive(
            self._clock() - receive_start_ns,
            timed_out=False,
        )
        return _decode_pipe_packet(worker_index, header, payload)

    def _calculate_remaining_timeout(
        self,
        deadline: Optional[float],
    ) -> Optional[float]:
        if deadline is None:
            return None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise queue.Empty
        return remaining

    def _record_readiness_wait(self, duration_ns: int, timed_out: bool) -> None:
        self._readiness_wait_ns += max(0, duration_ns)
        self._readiness_wait_count += 1
        if timed_out:
            self._readiness_timeout_ns += max(0, duration_ns)
            self._readiness_timeouts += 1

    def _record_payload_receive(self, duration_ns: int, timed_out: bool) -> None:
        self._payload_receive_ns += max(0, duration_ns)
        self._payload_receive_count += 1
        if timed_out:
            self._payload_receive_timeout_ns += max(0, duration_ns)
            self._payload_receive_timeouts += 1

    def _record_receive(self, packet: EpisodePacket) -> None:
        payload_size = len(packet.payload)
        with self._pending_episodes.get_lock():
            self._pending_episodes.value -= 1
        with self._pending_bytes.get_lock():
            self._pending_bytes.value -= payload_size
        self._received_episodes += 1
        self._received_bytes += payload_size
        if packet.transport_enqueue_ns is not None:
            self._pipe_latency_ns += max(
                0,
                self._clock() - packet.transport_enqueue_ns,
            )
            self._pipe_latency_count += 1
