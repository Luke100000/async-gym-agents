import copy
from typing import Any

import numpy as np
import torch
from stable_baselines3.common.buffers import RolloutBuffer

from async_gym_agents.data_classes import (
    AssembledEpisode,
    PreparedOnPolicyRollout,
)
from async_gym_agents.enums import EpisodeKind
from async_gym_agents.episode_assembler import AsyncEpisodeAssembler
from async_gym_agents.episode_transport import EpisodeTransport
from async_gym_agents.profiler import RuntimeProfiler


class AsyncOnPolicyRolloutAssembler(AsyncEpisodeAssembler[PreparedOnPolicyRollout]):
    """Build the inactive on-policy rollout buffer while the active buffer trains."""

    def __init__(
        self,
        transport: EpisodeTransport,
        target_transition_count: int,
        profiler: RuntimeProfiler,
        rollout_buffer_template: RolloutBuffer,
    ) -> None:
        super().__init__(
            transport=transport,
            episode_kind=EpisodeKind.ON_POLICY,
            target_transition_count=target_transition_count,
            profiler=profiler,
            thread_name="on-policy-rollout-assembler",
        )
        self._rollout_buffer_template = copy.copy(rollout_buffer_template)

    def _build_assembly(
        self,
        episodes: list[AssembledEpisode],
        transition_count: int,
    ) -> PreparedOnPolicyRollout:
        with self._profiler.track("rollout_buffer_building"):
            rollout_buffer = self._build_rollout_buffer(episodes, transition_count)
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
