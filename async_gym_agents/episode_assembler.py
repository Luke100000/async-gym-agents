import queue
import threading
from typing import Optional

from async_gym_agents.constants import (
    ASSEMBLER_RECEIVE_TIMEOUT_SECONDS,
    EPISODE_ASSEMBLER_THREAD_NAME,
    PROFILE_PHASE_ASSEMBLER_TRANSPORT,
    PROFILE_PHASE_ASSEMBLER_WAITING,
    PROFILE_PHASE_EPISODE_DESERIALIZATION,
)
from async_gym_agents.data_classes import AssembledEpisode, EpisodeAssembly
from async_gym_agents.enums import EpisodeKind
from async_gym_agents.episode_codec import decode_episode_packet
from async_gym_agents.episode_transport import EpisodeTransport
from async_gym_agents.profiler import RuntimeProfiler


class AsyncEpisodeAssembler:
    def __init__(
        self,
        transport: EpisodeTransport,
        target_transition_count: int,
        expected_episode_kind: EpisodeKind,
        profiler: RuntimeProfiler,
    ) -> None:
        if target_transition_count <= 0:
            raise ValueError("Episode assembly target must be positive")

        self._transport = transport
        self._target_transition_count = target_transition_count
        self._expected_episode_kind = expected_episode_kind
        self._profiler = profiler
        self._fill_requested = threading.Event()
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._state_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._assembly: Optional[EpisodeAssembly] = None
        self._error: Optional[BaseException] = None
        self._active = False
        self._filled_transition_count = 0

    def start(self) -> None:
        """Start filling the first background episode buffer."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name=EPISODE_ASSEMBLER_THREAD_NAME,
            daemon=True,
        )
        self._fill_requested.set()
        self._thread.start()

    def acquire(self, timeout: Optional[float] = None) -> EpisodeAssembly:
        """Wait for and acquire the next complete-episode buffer."""
        if self._thread is None:
            raise RuntimeError("Episode assembler has not been started")
        if not self._ready.wait(timeout):
            raise TimeoutError("Timed out waiting for assembled episodes")

        with self._state_lock:
            if self._error is not None:
                raise RuntimeError("Episode assembler failed") from self._error
            if self._assembly is None:
                raise RuntimeError("Episode assembler stopped before filling a buffer")
            if self._active:
                raise RuntimeError("Episode assembly is already active")
            assembly = self._assembly
            self._assembly = None
            self._active = True
            self._ready.clear()
            return assembly

    def release(self) -> None:
        """Release the active buffer and begin filling its replacement."""
        with self._state_lock:
            if not self._active:
                return
            self._active = False
            self._filled_transition_count = 0
        self._fill_requested.set()

    def shutdown(self) -> None:
        """Stop background assembly and unblock all waits."""
        self._stop.set()
        self._fill_requested.set()
        self._ready.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join()

    @property
    def filled_transition_count(self) -> int:
        """Return how many transitions currently occupy buffer B."""
        with self._state_lock:
            return self._filled_transition_count

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                self._fill_requested.wait()
                self._fill_requested.clear()
                if self._stop.is_set():
                    return
                assembly = self._assemble_episodes()
                if assembly is None:
                    return
                with self._state_lock:
                    self._assembly = assembly
                self._ready.set()
        except BaseException as error:
            with self._state_lock:
                self._error = error
            self._ready.set()

    def _assemble_episodes(self) -> Optional[EpisodeAssembly]:
        episodes = []
        transition_count = 0
        payload_bytes = 0
        while (
            transition_count < self._target_transition_count and not self._stop.is_set()
        ):
            transport_stats = self._transport.get_stats()
            phase = (
                PROFILE_PHASE_ASSEMBLER_WAITING
                if transport_stats.pending_episodes == 0
                else PROFILE_PHASE_ASSEMBLER_TRANSPORT
            )
            try:
                with self._profiler.track(phase):
                    packet = self._transport.receive(ASSEMBLER_RECEIVE_TIMEOUT_SECONDS)
            except queue.Empty:
                continue

            if packet.episode_kind is not self._expected_episode_kind:
                raise ValueError("Assembler received an incompatible episode kind")
            with self._profiler.track(PROFILE_PHASE_EPISODE_DESERIALIZATION):
                batch = decode_episode_packet(packet)
            episodes.append(AssembledEpisode(packet=packet, batch=batch))
            transition_count += packet.transition_count
            payload_bytes += len(packet.payload)
            with self._state_lock:
                self._filled_transition_count = transition_count

        if self._stop.is_set():
            return None
        return EpisodeAssembly(
            episodes=episodes,
            transition_count=transition_count,
            payload_bytes=payload_bytes,
        )
