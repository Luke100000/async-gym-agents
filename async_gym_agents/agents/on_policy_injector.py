from typing import Generator, Optional, Type

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

from async_gym_agents import constants
from async_gym_agents.agents.injector import AsyncAgentInjector, InjectorWorkerBase
from async_gym_agents.callback_batching import CallbackBatchDispatcher
from async_gym_agents.data_classes import (
    EpisodeCallbackContext,
    OnPolicyTransition,
)
from async_gym_agents.enums import EpisodeKind
from async_gym_agents.episode_codec import (
    get_episode_infos,
    get_episode_reset_infos,
    slice_episode_field,
)
from async_gym_agents.on_policy_rollout_assembler import (
    AsyncOnPolicyRolloutAssembler,
)
from async_gym_agents.utils import copy_obs, single_slice


def bootstrap_truncated_rewards(
    policy: BasePolicy,
    gamma: float,
    rewards: np.ndarray,
    dones: np.ndarray,
    infos: list[dict],
) -> None:
    """Bootstrap time-limit rewards with the rollout policy's terminal value."""
    for index, done in enumerate(dones):
        terminal_observation = infos[index].get("terminal_observation")
        if (
            not done
            or terminal_observation is None
            or not infos[index].get("TimeLimit.truncated", False)
        ):
            continue

        terminal_obs = policy.obs_to_tensor(terminal_observation)[0]
        with torch.inference_mode():
            terminal_value = policy.predict_values(terminal_obs)[0]
        rewards[index] += gamma * terminal_value.item()


class OnPolicyAlgorithmInjector(AsyncAgentInjector, OnPolicyAlgorithm):
    def __init__(
        self,
        *args,
        max_episodes_in_buffer: int = 8,
        use_mp: bool = False,
        worker_start_interval_seconds: float = 0.0,
        skip_truncated: bool = False,
        queue_put_timeout: Optional[float] = None,
        worker_join_timeout: float = 120.0,
        profiler_sync_interval: float = 1.0,
        mp_threads: int = 1,
        **kwargs,
    ):
        super().__init__(
            max_episodes_in_buffer=max_episodes_in_buffer,
            use_mp=use_mp,
            worker_start_interval_seconds=worker_start_interval_seconds,
            skip_truncated=skip_truncated,
            queue_put_timeout=queue_put_timeout,
            worker_join_timeout=worker_join_timeout,
            profiler_sync_interval=profiler_sync_interval,
            mp_threads=mp_threads,
        )
        super(AsyncAgentInjector, self).__init__(*args, **kwargs)

    # This implementation mirrors SB3's rollout lifecycle and must track upgrades.
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment.
        :param callback: Callback that will be called at each step.
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts.
        :param n_rollout_steps: Number of experiences to collect per environment.
        :return: True if the function returned with at least `n_rollout_steps`.
            Collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"

        self.policy.set_training_mode(False)
        self.pre_collect_preparation(self.policy)
        self._initialize_rollout_assembler(n_rollout_steps)
        with self._profiler_main.track("assembler_acquire"):
            prepared_rollout = self._acquire_prepared_assembly()
        self.rollout_buffer = prepared_rollout.rollout_buffer
        rollout_buffer = prepared_rollout.rollout_buffer
        for assembled_episode in prepared_rollout.episodes:
            self._record_policy_lag(
                assembled_episode.policy_version,
                assembled_episode.batch.transition_count,
            )

        n_steps = 0
        if self.use_sde:
            self.policy.reset_noise(1)

        callback.on_rollout_start()
        callback_dispatcher = CallbackBatchDispatcher(
            callback,
            EpisodeKind.ON_POLICY,
        )

        new_obs = None
        dones = None
        buffer_index = 0
        with self._profiler_main.track("transition_processing"):
            for assembled_episode in prepared_rollout.episodes:
                batch = assembled_episode.batch
                episode_start_timestep = self.num_timesteps
                if callback_dispatcher.needs_step_callbacks:
                    for transition_index in range(batch.transition_count):
                        if (
                            self.use_sde
                            and self.sde_sample_freq > 0
                            and n_steps % self.sde_sample_freq == 0
                        ):
                            self.policy.reset_noise(1)

                        new_obs = slice_episode_field(
                            batch,
                            "new_obs",
                            transition_index,
                        )
                        self._last_obs = slice_episode_field(
                            batch,
                            "last_obs",
                            transition_index,
                        )
                        actions = rollout_buffer.actions[buffer_index]
                        training_rewards = rollout_buffer.rewards[buffer_index]
                        environment_rewards = slice_episode_field(
                            batch,
                            constants.ON_POLICY_ENVIRONMENT_REWARDS_FIELD,
                            transition_index,
                        )
                        rewards = environment_rewards
                        self._last_episode_starts = rollout_buffer.episode_starts[
                            buffer_index
                        ]
                        values = torch.from_numpy(rollout_buffer.values[buffer_index])
                        log_probs = torch.from_numpy(
                            rollout_buffer.log_probs[buffer_index]
                        )
                        dones = slice_episode_field(
                            batch,
                            "dones",
                            transition_index,
                        )
                        infos = get_episode_infos(batch, transition_index)
                        reset_infos = get_episode_reset_infos(
                            batch,
                            transition_index,
                        )

                        self.num_timesteps += 1
                        if not callback_dispatcher.process_step(locals()):
                            return False

                        self._update_info_buffer(infos, dones)
                        n_steps += 1
                        buffer_index += 1
                else:
                    if self.use_sde and self.sde_sample_freq > 0:
                        first_reset_offset = (-n_steps) % self.sde_sample_freq
                        for reset_offset in range(
                            first_reset_offset,
                            batch.transition_count,
                            self.sde_sample_freq,
                        ):
                            self.num_timesteps = episode_start_timestep + reset_offset
                            self.policy.reset_noise(1)
                        self.num_timesteps = episode_start_timestep

                    for transition_index, infos in batch.infos.items():
                        dones = slice_episode_field(
                            batch,
                            "dones",
                            transition_index,
                        )
                        self._update_info_buffer(infos, dones)

                    transition_index = batch.transition_count - 1
                    final_buffer_index = buffer_index + transition_index
                    new_obs = slice_episode_field(
                        batch,
                        "new_obs",
                        transition_index,
                    )
                    self._last_obs = slice_episode_field(
                        batch,
                        "last_obs",
                        transition_index,
                    )
                    actions = rollout_buffer.actions[final_buffer_index]
                    training_rewards = rollout_buffer.rewards[final_buffer_index]
                    environment_rewards = slice_episode_field(
                        batch,
                        constants.ON_POLICY_ENVIRONMENT_REWARDS_FIELD,
                        transition_index,
                    )
                    rewards = environment_rewards
                    self._last_episode_starts = rollout_buffer.episode_starts[
                        final_buffer_index
                    ]
                    values = torch.from_numpy(rollout_buffer.values[final_buffer_index])
                    log_probs = torch.from_numpy(
                        rollout_buffer.log_probs[final_buffer_index]
                    )
                    dones = slice_episode_field(
                        batch,
                        "dones",
                        transition_index,
                    )
                    infos = get_episode_infos(batch, transition_index)
                    reset_infos = get_episode_reset_infos(
                        batch,
                        transition_index,
                    )
                    self.num_timesteps += batch.transition_count
                    n_steps += batch.transition_count
                    buffer_index += batch.transition_count

                callback_context = EpisodeCallbackContext(
                    batch=batch,
                    start_timestep=episode_start_timestep,
                    end_timestep=self.num_timesteps,
                )
                if not callback_dispatcher.process_episode(callback_context):
                    return False

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def _initialize_rollout_assembler(self, target_transition_count: int) -> None:
        if self._episode_assembler is not None:
            return
        self._episode_assembler = AsyncOnPolicyRolloutAssembler(
            transport=self._episode_transport,
            target_transition_count=target_transition_count,
            profiler=self._profiler_main,
            rollout_buffer_template=self.rollout_buffer,
        )
        self._episode_assembler.start()

    def get_worker_class(self) -> Type[InjectorWorkerBase]:
        return InjectorWorker

    def get_worker_kwargs(self):
        return dict(
            **super().get_worker_kwargs(),
            action_space=self.action_space,
            gamma=self.gamma,
        )


class InjectorWorker(InjectorWorkerBase):
    def __init__(
        self,
        action_space: gym.Space,
        gamma: float,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.action_space = action_space
        self.gamma = gamma

    def generate(self) -> Generator[list[OnPolicyTransition], None, None]:
        """
        Continuously plays the game and returns episodes of Transitions
        """
        with self._profiler.track("resetting"):
            last_obs = self.env.reset()
        last_dones = np.ones((self.env.num_envs,), dtype=bool)

        episodes = {}

        while True:
            with self._profiler.track("inference"):
                with torch.inference_mode():
                    # Convert to pytorch tensor or to TensorDict
                    obs_tensor = obs_as_tensor(last_obs, self.policy.device)
                    actions, values, log_probs = self.policy(obs_tensor)

            actions = actions.cpu().numpy()
            values = values.cpu().numpy()
            log_probs = log_probs.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out-of-bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(
                        actions, self.action_space.low, self.action_space.high
                    )

            with self._profiler.track("stepping"):
                new_obs, rewards, dones, infos = self.env.step(clipped_actions)

            environment_rewards = rewards.copy()
            training_rewards = environment_rewards.copy()
            bootstrap_truncated_rewards(
                self.policy,
                self.gamma,
                training_rewards,
                dones,
                infos,
            )

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Store transition
            with self._profiler.track("transition_building"):
                for idx in range(len(dones)):
                    if idx not in episodes:
                        episodes[idx] = []
                    episodes[idx].append(
                        OnPolicyTransition(
                            actions=single_slice(actions, idx),
                            values=single_slice(values, idx),
                            log_probs=single_slice(log_probs, idx),
                            last_obs=copy_obs(single_slice(last_obs, idx)),
                            new_obs=copy_obs(single_slice(new_obs, idx)),
                            environment_rewards=single_slice(
                                environment_rewards,
                                idx,
                            ),
                            training_rewards=single_slice(training_rewards, idx),
                            dones=single_slice(dones, idx),
                            last_dones=single_slice(last_dones, idx),
                            infos=single_slice(infos, idx),
                            reset_infos=single_slice(self.env.reset_infos, idx),
                        )
                    )
            self._flush_profiler()
            last_obs = new_obs
            last_dones = dones

            # Start a new episode
            for idx, done in enumerate(dones):
                if done:
                    yield episodes.pop(idx)

                    self.copy_policy_from_store()
