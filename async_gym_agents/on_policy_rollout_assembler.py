import copy
import queue
import threading
from typing import Any, Optional

import numpy as np
import torch
from stable_baselines3.common.buffers import RolloutBuffer

from async_gym_agents import constants
from async_gym_agents.data_classes import (
    AssembledEpisode,
    EpisodeAssemblerStats,
    PreparedOnPolicyRollout,
)
from async_gym_agents.enums import EpisodeKind
from async_gym_agents.episode_codec import decode_episode_packet
from async_gym_agents.episode_transport import EpisodeTransport
from async_gym_agents.profiler import RuntimeProfiler


class AsyncOnPolicyRolloutAssembler:
    """Build the inactive on-policy rollout buffer while the active buffer trains."""

    def __init__(
        self,
        transport: EpisodeTransport,
        target_transition_count: int,
        profiler: RuntimeProfiler,
        rollout_buffer_template: RolloutBuffer,
    ) -> None:
        if target_transition_count <= 0:
            raise ValueError("Episode assembly target must be positive")

        self._transport = transport
        self._target_transition_count = target_transition_count
        self._profiler = profiler
        self._rollout_buffer_template = copy.copy(rollout_buffer_template)
        self._fill_requested = threading.Event()
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._state_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._prepared_rollout: Optional[PreparedOnPolicyRollout] = None
        self._error: Optional[BaseException] = None
        self._filled_transition_count = 0
        self._completed_assemblies = 0
        self._last_transition_count = 0
        self._max_transition_count = 0
        self._last_payload_bytes = 0
        self._max_payload_bytes = 0

    def start(self) -> None:
        """Start preparing the first trainer-ready rollout buffer."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="on-policy-rollout-assembler",
            daemon=True,
        )
        self._fill_requested.set()
        self._thread.start()

    def acquire(self, timeout: Optional[float] = None) -> PreparedOnPolicyRollout:
        """Swap in the prepared buffer and immediately start its replacement."""
        if self._thread is None:
            raise RuntimeError("Rollout assembler has not been started")
        if not self._ready.wait(timeout):
            raise TimeoutError("Timed out waiting for a prepared rollout buffer")

        with self._state_lock:
            if self._error is not None:
                raise RuntimeError("Rollout assembler failed") from self._error
            if self._prepared_rollout is None:
                raise RuntimeError(
                    "Rollout assembler stopped before preparing a buffer"
                )
            prepared_rollout = self._prepared_rollout
            self._prepared_rollout = None
            self._filled_transition_count = 0
            self._ready.clear()

        self._fill_requested.set()
        return prepared_rollout

    def shutdown(self) -> None:
        """Stop background preparation and unblock all waits."""
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

    def get_stats(self) -> EpisodeAssemblerStats:
        """Return current fill progress and completed buffer peaks."""
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
                prepared_rollout = self._prepare_rollout()
                if prepared_rollout is None:
                    return
                with self._state_lock:
                    self._prepared_rollout = prepared_rollout
                self._ready.set()
        except BaseException as error:
            with self._state_lock:
                self._error = error
            self._ready.set()

    def _prepare_rollout(self) -> Optional[PreparedOnPolicyRollout]:
        episodes = []
        transition_count = 0
        payload_bytes = 0
        while (
            transition_count < self._target_transition_count and not self._stop.is_set()
        ):
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
                continue

            if packet.episode_kind is not EpisodeKind.ON_POLICY:
                raise ValueError("Assembler received a non-on-policy episode")
            with self._profiler.track("episode_deserialization"):
                batch = decode_episode_packet(packet)
            episodes.append(AssembledEpisode(packet=packet, batch=batch))
            transition_count += packet.transition_count
            payload_bytes += len(packet.payload)
            with self._state_lock:
                self._filled_transition_count = transition_count

        if self._stop.is_set():
            return None

        with self._profiler.track("rollout_buffer_building"):
            rollout_buffer = self._build_rollout_buffer(episodes, transition_count)

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
        return PreparedOnPolicyRollout(
            rollout_buffer=rollout_buffer,
            episodes=episodes,
            transition_count=transition_count,
        )

    def _build_rollout_buffer(
        self,
        episodes: list[AssembledEpisode],
        transition_count: int,
    ) -> RolloutBuffer:
        rollout_buffer = copy.copy(self._rollout_buffer_template)
        rollout_buffer.buffer_size = transition_count
        rollout_buffer.n_envs = 1
        rollout_buffer.reset()

        observations = self._concatenate_episode_field(
            episodes,
            "last_obs",
        )
        self._copy_observations(rollout_buffer.observations, observations)
        self._copy_array(
            rollout_buffer.actions,
            self._concatenate_episode_field(
                episodes,
                "actions",
            ),
        )
        self._copy_array(
            rollout_buffer.rewards,
            self._concatenate_episode_field(
                episodes,
                "rewards",
            ),
        )
        self._copy_array(
            rollout_buffer.episode_starts,
            self._concatenate_episode_field(
                episodes,
                "last_dones",
            ),
        )
        self._copy_array(
            rollout_buffer.values,
            self._concatenate_episode_field(
                episodes,
                "values",
            ),
        )
        self._copy_array(
            rollout_buffer.log_probs,
            self._concatenate_episode_field(
                episodes,
                "log_probs",
            ),
        )

        dones = self._concatenate_episode_field(
            episodes,
            "dones",
        )
        final_dones = np.asarray(dones[-1:]).reshape(rollout_buffer.n_envs)
        rollout_buffer.compute_returns_and_advantage(
            last_values=torch.zeros(rollout_buffer.n_envs),
            dones=final_dones,
        )
        rollout_buffer.pos = transition_count
        rollout_buffer.full = True
        return rollout_buffer

    def _concatenate_episode_field(
        self,
        episodes: list[AssembledEpisode],
        field_name: str,
    ) -> Any:
        return self._concatenate_values(
            [episode.batch.fields[field_name] for episode in episodes]
        )

    def _concatenate_values(self, values: list[Any]) -> Any:
        first_value = values[0]
        if isinstance(first_value, np.ndarray):
            return np.concatenate(values, axis=0)
        if isinstance(first_value, dict):
            return {
                key: self._concatenate_values([value[key] for value in values])
                for key in first_value
            }
        raise TypeError(
            f"Cannot build a rollout buffer from {type(first_value)!r} observations"
        )

    def _copy_observations(self, destination: Any, source: Any) -> None:
        if isinstance(destination, dict):
            for key, destination_value in destination.items():
                self._copy_observations(destination_value, source[key])
            return
        self._copy_array(destination, source)

    @staticmethod
    def _copy_array(destination: np.ndarray, source: Any) -> None:
        np.copyto(destination, np.asarray(source).reshape(destination.shape))
