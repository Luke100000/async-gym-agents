from typing import Generator, Type

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

from async_gym_agents.agents.injector import AsyncAgentInjector, InjectorWorkerBase
from async_gym_agents.data_classes import OnPolicyTransition as Transition
from async_gym_agents.enums import EpisodeKind
from async_gym_agents.episode_assembler import AsyncEpisodeAssembler
from async_gym_agents.episode_codec import (
    get_episode_infos,
    get_episode_reset_infos,
    slice_episode_field,
)
from async_gym_agents.utils import copy_obs, single_slice


class OnPolicyAlgorithmInjector(AsyncAgentInjector, OnPolicyAlgorithm):
    def __init__(
        self,
        *args,
        max_episodes_in_buffer: int = 8,
        use_mp: bool = False,
        worker_start_interval_seconds: float = 0.0,
        skip_truncated: bool = False,
        queue_put_timeout: float = 60.0,
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
        self._episode_assembler = None

    # must be updated from SB3 (!)
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
        self._initialize_episode_assembler(n_rollout_steps)
        assembly = self._episode_assembler.acquire()
        for assembled_episode in assembly.episodes:
            self._record_policy_lag(
                assembled_episode.packet.policy_version,
                assembled_episode.packet.transition_count,
            )

        n_steps = 0
        rollout_buffer.n_envs = 1
        rollout_buffer.buffer_size = assembly.transition_count
        rollout_buffer.reset()

        if self.use_sde:
            self.policy.reset_noise(1)

        callback.on_rollout_start()

        new_obs = None
        dones = None
        try:
            for assembled_episode in assembly.episodes:
                batch = assembled_episode.batch
                for transition_index in range(batch.transition_count):
                    if (
                        self.use_sde
                        and self.sde_sample_freq > 0
                        and n_steps % self.sde_sample_freq == 0
                    ):
                        self.policy.reset_noise(1)

                    with self._profiler_main.track("processing"):
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
                        actions = slice_episode_field(
                            batch,
                            "actions",
                            transition_index,
                        )
                        rewards = slice_episode_field(
                            batch,
                            "rewards",
                            transition_index,
                        )
                        self._last_episode_starts = slice_episode_field(
                            batch,
                            "last_dones",
                            transition_index,
                        )
                        values = torch.from_numpy(
                            slice_episode_field(
                                batch,
                                "values",
                                transition_index,
                            )
                        )
                        log_probs = torch.from_numpy(
                            slice_episode_field(
                                batch,
                                "log_probs",
                                transition_index,
                            )
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
                        callback.update_locals(locals())
                        if not callback.on_step():
                            return False

                        self._update_info_buffer(infos, dones)
                        n_steps += 1
                        for idx, done in enumerate(dones):
                            if (
                                done
                                and infos[idx].get("terminal_observation") is not None
                                and infos[idx].get("TimeLimit.truncated", False)
                            ):
                                terminal_obs = self.policy.obs_to_tensor(
                                    infos[idx]["terminal_observation"]
                                )[0]
                                with torch.inference_mode():
                                    terminal_value = self.policy.predict_values(
                                        terminal_obs
                                    )[0]
                                rewards[idx] += self.gamma * terminal_value.item()

                        rollout_buffer.add(
                            self._last_obs,
                            actions,
                            rewards,
                            self._last_episode_starts,
                            values,
                            log_probs,
                        )
        finally:
            self._episode_assembler.release()

        with self._profiler_main.track("processing"):
            values = torch.zeros(1, device=self.device)
            rollout_buffer.compute_returns_and_advantage(
                last_values=values, dones=dones
            )

            callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def _initialize_episode_assembler(self, target_transition_count: int) -> None:
        if self._episode_assembler is not None:
            return
        self._episode_assembler = AsyncEpisodeAssembler(
            transport=self._episode_transport,
            target_transition_count=target_transition_count,
            expected_episode_kind=EpisodeKind.ON_POLICY,
            profiler=self._profiler_main,
        )
        self._episode_assembler.start()

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["_episode_assembler"]

    def shutdown(self):
        if self._episode_assembler is not None:
            self._episode_assembler.shutdown()
            self._episode_assembler = None
        return super().shutdown()

    def get_worker_class(self) -> Type[InjectorWorkerBase]:
        return InjectorWorker

    def get_worker_kwargs(self):
        return dict(
            **super().get_worker_kwargs(),
            action_space=self.action_space,
        )


class InjectorWorker(InjectorWorkerBase):
    def __init__(
        self,
        action_space: gym.Space,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.action_space = action_space

    def generate(self) -> Generator[list[Transition], None, None]:
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

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Store transition
            with self._profiler.track("transition_building"):
                for idx in range(len(dones)):
                    if idx not in episodes:
                        episodes[idx] = []
                    episodes[idx].append(
                        Transition(
                            single_slice(actions, idx),
                            single_slice(values, idx),
                            single_slice(log_probs, idx),
                            copy_obs(single_slice(last_obs, idx)),
                            copy_obs(single_slice(new_obs, idx)),
                            single_slice(rewards, idx),
                            single_slice(dones, idx),
                            single_slice(last_dones, idx),
                            single_slice(infos, idx),
                            single_slice(self.env.reset_infos, idx),
                        )
                    )
            self._flush_profiler()
            last_obs = new_obs
            last_dones = dones

            # Start a new episode
            for idx, done in enumerate(dones):
                if done:
                    yield episodes[idx]
                    del episodes[idx]

                    self.copy_policy_from_queue()
