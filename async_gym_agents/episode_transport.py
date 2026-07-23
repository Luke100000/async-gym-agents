import multiprocessing
import queue
import struct
import threading
import time
from multiprocessing.connection import Connection, wait
from typing import Any, Callable, Dict, Optional, Set, Tuple, TypeAlias

from async_gym_agents import constants
from async_gym_agents.data_classes import (
    EpisodePacket,
    EpisodeReservation,
    EpisodeSendResult,
    EpisodeSubmissionResult,
    EpisodeTransportStats,
)
from async_gym_agents.enums import EpisodeKind

GenericSemaphore: TypeAlias = Any
GenericSharedCounter: TypeAlias = Any
GenericStopEvent: TypeAlias = Any


class WorkerTransportClosedError(RuntimeError):
    """Report which worker-side episode channel closed unexpectedly."""

    def __init__(self, worker_index: int) -> None:
        self.worker_index = worker_index
        super().__init__(
            constants.WORKER_FAILURE_MESSAGE.format(
                worker_index=worker_index,
                reason=constants.WORKER_TRANSPORT_CLOSED_REASON,
            )
        )


def _encode_episode_packet_header(packet: EpisodePacket) -> bytes:
    """Encode fixed-size packet metadata without copying the episode payload."""
    if packet.episode_kind is EpisodeKind.ON_POLICY:
        episode_kind_code = constants.EPISODE_KIND_ON_POLICY_CODE
    elif packet.episode_kind is EpisodeKind.OFF_POLICY:
        episode_kind_code = constants.EPISODE_KIND_OFF_POLICY_CODE
    else:
        raise ValueError(f"Unsupported episode kind: {packet.episode_kind!r}")

    has_policy_version = packet.policy_version is not None
    policy_version = 0 if packet.policy_version is None else packet.policy_version
    return constants.EPISODE_PACKET_HEADER.pack(
        has_policy_version,
        policy_version,
        episode_kind_code,
        packet.transition_count,
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
        ) = constants.EPISODE_PACKET_HEADER.unpack(header)
    except struct.error as error:
        raise ValueError("Received an invalid episode packet header") from error

    if has_policy_version not in (0, 1):
        raise ValueError("Episode packet header has an invalid policy marker")

    if episode_kind_code == constants.EPISODE_KIND_ON_POLICY_CODE:
        episode_kind = EpisodeKind.ON_POLICY
    elif episode_kind_code == constants.EPISODE_KIND_OFF_POLICY_CODE:
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
    )


class EpisodeSender:
    def __init__(
        self,
        worker_index: int,
        send_connection: Connection,
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
        submission = self.reserve(packet, stop, timeout)
        if not submission:
            return EpisodeSendResult(
                sent=False,
                waiting_ns=submission.waiting_ns,
                transport_ns=0,
            )
        if submission.reservation is None:
            raise RuntimeError("Successful episode submission has no reservation")
        return self.send_reserved(submission.reservation, stop)

    def reserve(
        self,
        packet: EpisodePacket,
        stop: GenericStopEvent,
        timeout: Optional[float],
    ) -> EpisodeSubmissionResult:
        """Reserve global capacity before an asynchronous feeder accepts a packet."""
        if packet.worker_index != self.worker_index:
            raise ValueError("Episode packet was sent through the wrong worker channel")

        if stop.is_set():
            return EpisodeSubmissionResult(submitted=False, waiting_ns=0)

        deadline = None if timeout is None else time.monotonic() + timeout
        capacity_acquired, waiting_ns = self._acquire_slot(
            self._capacity,
            stop,
            deadline,
        )
        if not capacity_acquired:
            return EpisodeSubmissionResult(
                submitted=False,
                waiting_ns=waiting_ns,
            )

        enqueue_ns = self._clock()
        self._record_reservation(len(packet.payload))
        return EpisodeSubmissionResult(
            submitted=True,
            waiting_ns=waiting_ns,
            reservation=EpisodeReservation(
                packet=packet,
                enqueue_ns=enqueue_ns,
                waiting_ns=waiting_ns,
            ),
        )

    def send_reserved(
        self,
        reservation: EpisodeReservation,
        stop: GenericStopEvent,
    ) -> EpisodeSendResult:
        """Write an already bounded episode reservation to this worker's pipe."""
        packet = reservation.packet
        if packet.worker_index != self.worker_index:
            raise ValueError("Episode reservation belongs to another worker channel")
        if stop.is_set():
            return self.cancel(reservation)

        try:
            header = _encode_episode_packet_header(packet)
            self._send_connection.send_bytes(header)
            self._send_connection.send_bytes(packet.payload)
        except BaseException as error:
            result = self.cancel(reservation)
            if stop.is_set() and isinstance(error, (EOFError, OSError)):
                return result
            raise

        return self._build_send_result(True, reservation)

    def cancel(self, reservation: EpisodeReservation) -> EpisodeSendResult:
        """Release one accepted episode that cannot be delivered."""
        self._rollback_reservation(len(reservation.packet.payload))
        self._capacity.release()
        return self._build_send_result(False, reservation)

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
            return constants.TRANSPORT_ACQUIRE_RETRY_TIMEOUT_SECONDS
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        return min(constants.TRANSPORT_ACQUIRE_RETRY_TIMEOUT_SECONDS, remaining)

    def _record_reservation(self, payload_size: int) -> None:
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

    def _rollback_reservation(self, payload_size: int) -> None:
        with self._pending_episodes.get_lock():
            self._pending_episodes.value -= 1
        with self._pending_bytes.get_lock():
            self._pending_bytes.value -= payload_size
        with self._sent_episodes.get_lock():
            self._sent_episodes.value -= 1
        with self._sent_bytes.get_lock():
            self._sent_bytes.value -= payload_size

    def _build_send_result(
        self,
        sent: bool,
        reservation: EpisodeReservation,
    ) -> EpisodeSendResult:
        return EpisodeSendResult(
            sent=sent,
            waiting_ns=max(0, reservation.waiting_ns),
            transport_ns=max(0, self._clock() - reservation.enqueue_ns),
        )


class EpisodeFeeder:
    def __init__(
        self,
        sender: EpisodeSender,
        stop: GenericStopEvent,
        on_send_complete: Optional[Callable[[EpisodeSendResult], None]] = None,
    ) -> None:
        self._sender = sender
        self._stop = stop
        self._on_send_complete = on_send_complete
        self._reservations = queue.Queue()
        self._error_lock = threading.Lock()
        self._error: Optional[BaseException] = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name=f"episode-feeder-{sender.worker_index}",
            daemon=True,
        )
        self._thread.start()

    def submit(
        self,
        packet: EpisodePacket,
        timeout: Optional[float],
    ) -> EpisodeSubmissionResult:
        """Accept a globally bounded packet without waiting for pipe delivery."""
        if self._closed:
            raise RuntimeError("Cannot submit an episode to a closed feeder")
        self.raise_if_failed()
        submission = self._sender.reserve(packet, self._stop, timeout)
        if submission:
            if submission.reservation is None:
                raise RuntimeError("Successful episode submission has no reservation")
            self._reservations.put_nowait(submission.reservation)
        return submission

    def shutdown(self) -> None:
        """Drain or cancel accepted episodes and stop the feeder thread."""
        if self._closed:
            return
        self._closed = True
        self._reservations.put_nowait(None)
        self._thread.join()
        self.raise_if_failed()

    def raise_if_failed(self) -> None:
        """Raise a worker-visible error when asynchronous delivery failed."""
        with self._error_lock:
            error = self._error
        if error is not None:
            raise RuntimeError("Episode feeder failed") from error

    def _run(self) -> None:
        while True:
            reservation = self._reservations.get()
            if reservation is None:
                return
            try:
                result = self._sender.send_reserved(reservation, self._stop)
                if self._on_send_complete is not None:
                    self._on_send_complete(result)
            except BaseException as error:
                with self._error_lock:
                    if self._error is None:
                        self._error = error
                self._stop.set()


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
        connections = [self._mp_ctx.Pipe(duplex=False) for _ in range(worker_count)]
        self._receive_connections = [pair[0] for pair in connections]
        send_connections = [pair[1] for pair in connections]
        self._worker_by_connection = {
            connection: worker_index
            for worker_index, connection in enumerate(self._receive_connections)
        }
        self._capacity = self._mp_ctx.BoundedSemaphore(max_pending_episodes)
        counter_type = constants.SHARED_COUNTER_TYPE_CODE
        self._pending_episodes = self._mp_ctx.Value(counter_type, 0)
        self._max_pending_episodes = self._mp_ctx.Value(counter_type, 0)
        self._pending_bytes = self._mp_ctx.Value(counter_type, 0)
        self._max_pending_bytes = self._mp_ctx.Value(counter_type, 0)
        self._sent_episodes = self._mp_ctx.Value(counter_type, 0)
        self._sent_bytes = self._mp_ctx.Value(counter_type, 0)
        self._senders = [
            EpisodeSender(
                worker_index=worker_index,
                send_connection=send_connections[worker_index],
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
        deadline = None if timeout is None else time.monotonic() + timeout
        self._collect_ready_workers(deadline)
        worker_index = self._select_ready_worker()
        packet = self._receive_packet(worker_index, deadline)
        self._record_receive(packet)
        self._capacity.release()
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
            received_bytes=self._received_bytes,
        )

    def has_pending_episodes(self) -> bool:
        """Return whether at least one complete episode is currently queued."""
        return self._pending_episodes.value > 0

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
        ready_connections = wait(self._receive_connections, timeout=timeout)
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
        header = self._pending_headers.pop(worker_index, None)
        if header is None:
            header = self._receive_worker_bytes(connection, worker_index)

        try:
            remaining = self._calculate_remaining_timeout(deadline)
        except queue.Empty:
            self._pending_headers[worker_index] = header
            raise

        if remaining is not None and not connection.poll(remaining):
            self._pending_headers[worker_index] = header
            raise queue.Empty

        payload = self._receive_worker_bytes(connection, worker_index)
        return _decode_pipe_packet(worker_index, header, payload)

    @staticmethod
    def _receive_worker_bytes(
        connection: Connection,
        worker_index: int,
    ) -> bytes:
        try:
            return connection.recv_bytes()
        except (EOFError, OSError) as error:
            raise WorkerTransportClosedError(worker_index) from error

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

    def _record_receive(self, packet: EpisodePacket) -> None:
        payload_size = len(packet.payload)
        with self._pending_episodes.get_lock():
            self._pending_episodes.value -= 1
        with self._pending_bytes.get_lock():
            self._pending_bytes.value -= payload_size
        self._received_episodes += 1
        self._received_bytes += payload_size
