import copy
from typing import Any

import numpy as np
from stable_baselines3.common.buffers import RolloutBuffer

from async_gym_agents import constants
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

        self._fill_episode_field(
            episodes,
            "last_obs",
            rollout_buffer.observations,
        )
        self._fill_episode_field(
            episodes,
            "actions",
            rollout_buffer.actions,
        )
        self._fill_episode_field(
            episodes,
            constants.ON_POLICY_TRAINING_REWARDS_FIELD,
            rollout_buffer.rewards,
        )
        self._fill_episode_field(
            episodes,
            "last_dones",
            rollout_buffer.episode_starts,
        )
        self._fill_episode_field(
            episodes,
            "values",
            rollout_buffer.values,
        )
        self._fill_episode_field(
            episodes,
            "log_probs",
            rollout_buffer.log_probs,
        )

        # Rewards here are the raw environment rewards; the trainer applies the
        # TimeLimit-truncation bootstrap with its own live policy and computes
        # returns/advantages afterwards, since only it can access that policy
        # without racing its concurrent training thread.
        return rollout_buffer

    def _fill_episode_field(
        self,
        episodes: list[AssembledEpisode],
        field_name: str,
        destination: Any,
    ) -> None:
        self.fill_concatenated_values(
            destination,
            [episode.batch.fields[field_name] for episode in episodes],
        )

    @classmethod
    def fill_concatenated_values(
        cls,
        destination: Any,
        values: list[Any],
    ) -> None:
        """Concatenate episode chunks directly into a rollout-buffer destination."""
        if isinstance(destination, np.ndarray):
            first_value = values[0]
            if not isinstance(first_value, np.ndarray):
                raise TypeError(
                    f"Cannot fill an array from {type(first_value)!r} values"
                )
            concatenated_shape = (
                sum(value.shape[0] for value in values),
                *first_value.shape[1:],
            )
            destination_view = destination.reshape(concatenated_shape)
            np.concatenate(values, axis=0, out=destination_view)
            return
        if isinstance(destination, dict):
            for key, destination_value in destination.items():
                cls.fill_concatenated_values(
                    destination_value,
                    [value[key] for value in values],
                )
            return
        raise TypeError(
            f"Cannot fill rollout destination of type {type(destination)!r}"
        )
