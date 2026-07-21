import queue
import threading
from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

from async_gym_agents import constants
from async_gym_agents.data_classes import AssembledEpisode, EpisodeAssemblerStats
from async_gym_agents.enums import EpisodeKind
from async_gym_agents.episode_codec import decode_episode_packet
from async_gym_agents.episode_transport import EpisodeTransport
from async_gym_agents.profiler import RuntimeProfiler

PreparedAssembly = TypeVar("PreparedAssembly")


class AsyncEpisodeAssembler(ABC, Generic[PreparedAssembly]):
    """Prepare one bounded episode assembly while the trainer uses another."""

    def __init__(
        self,
        transport: EpisodeTransport,
        episode_kind: EpisodeKind,
        target_transition_count: int,
        profiler: RuntimeProfiler,
        thread_name: str,
    ) -> None:
        if target_transition_count <= 0:
            raise ValueError("Episode assembly target must be positive")

        self._transport = transport
        self._episode_kind = episode_kind
        self._target_transition_count = target_transition_count
        self._profiler = profiler
        self._thread_name = thread_name
        self._fill_requested = threading.Event()
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._state_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._prepared_assembly: Optional[PreparedAssembly] = None
        self._error: Optional[BaseException] = None
        self._filled_transition_count = 0
        self._completed_assemblies = 0
        self._last_transition_count = 0
        self._max_transition_count = 0
        self._last_payload_bytes = 0
        self._max_payload_bytes = 0

    def start(self) -> None:
        """Start preparing the first trainer-ready assembly."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name=self._thread_name,
            daemon=True,
        )
        self._fill_requested.set()
        self._thread.start()

    def acquire(self, timeout: Optional[float] = None) -> PreparedAssembly:
        """Acquire the prepared assembly and immediately request its replacement."""
        if self._thread is None:
            raise RuntimeError("Episode assembler has not been started")
        if not self._ready.wait(timeout):
            raise TimeoutError("Timed out waiting for a prepared episode assembly")

        with self._state_lock:
            if self._error is not None:
                raise RuntimeError("Episode assembler failed") from self._error
            if self._prepared_assembly is None:
                raise RuntimeError(
                    "Episode assembler stopped before preparing an assembly"
                )
            prepared_assembly = self._prepared_assembly
            self._prepared_assembly = None
            self._filled_transition_count = 0
            self._ready.clear()

        self._fill_requested.set()
        return prepared_assembly

    def shutdown(self) -> None:
        """Stop background preparation and unblock all waits."""
        self._stop.set()
        self._fill_requested.set()
        self._ready.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join()

    def get_stats(self) -> EpisodeAssemblerStats:
        """Return current fill progress and completed assembly peaks."""
        with self._state_lock:
            return EpisodeAssemblerStats(
                target_transition_count=self._target_transition_count,
                filling_transition_count=self._filled_transition_count,
                completed_assemblies=self._completed_assemblies,
                last_transition_count=self._last_transition_count,
                max_transition_count=self._max_transition_count,
                last_payload_bytes=self._last_payload_bytes,
                max_payload_bytes=self._max_payload_bytes,
            )

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                self._fill_requested.wait()
                self._fill_requested.clear()
                if self._stop.is_set():
                    return
                prepared_assembly = self._prepare_assembly()
                if prepared_assembly is None:
                    return
                with self._state_lock:
                    self._prepared_assembly = prepared_assembly
                self._ready.set()
        except BaseException as error:
            with self._state_lock:
                self._error = error
            self._ready.set()

    def _prepare_assembly(self) -> Optional[PreparedAssembly]:
        episodes = []
        transition_count = 0
        payload_bytes = 0
        while (
            transition_count < self._target_transition_count and not self._stop.is_set()
        ):
            episode = self._receive_episode()
            if episode is None:
                continue
            episodes.append(episode)
            transition_count += episode.batch.transition_count
            payload_bytes += episode.payload_bytes
            with self._state_lock:
                self._filled_transition_count = transition_count

        if self._stop.is_set():
            return None

        prepared_assembly = self._build_assembly(episodes, transition_count)
        with self._state_lock:
            self._completed_assemblies += 1
            self._last_transition_count = transition_count
            self._max_transition_count = max(
                self._max_transition_count,
                transition_count,
            )
            self._last_payload_bytes = payload_bytes
            self._max_payload_bytes = max(
                self._max_payload_bytes,
                payload_bytes,
            )
        return prepared_assembly

    def _receive_episode(self) -> Optional[AssembledEpisode]:
        transport_stats = self._transport.get_stats()
        phase = (
            "assembler_waiting"
            if transport_stats.pending_episodes == 0
            else "assembler_transport"
        )
        try:
            with self._profiler.track(phase):
                packet = self._transport.receive(
                    constants.ASSEMBLER_RECEIVE_TIMEOUT_SECONDS
                )
        except queue.Empty:
            return None

        if packet.episode_kind is not self._episode_kind:
            raise ValueError(
                f"Assembler expected {self._episode_kind.value} episodes, "
                f"received {packet.episode_kind.value}"
            )
        with self._profiler.track("episode_deserialization"):
            batch = decode_episode_packet(packet)
        return AssembledEpisode(
            policy_version=packet.policy_version,
            payload_bytes=len(packet.payload),
            batch=batch,
        )

    @abstractmethod
    def _build_assembly(
        self,
        episodes: list[AssembledEpisode],
        transition_count: int,
    ) -> PreparedAssembly:
        """Build one policy-specific trainer input from complete episodes."""
