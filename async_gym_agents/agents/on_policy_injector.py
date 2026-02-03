import io
import logging
import multiprocessing
import queue
from copy import deepcopy
from dataclasses import dataclass
from multiprocessing.managers import Namespace
from typing import Dict, Generator, Optional

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.utils import get_device, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs

from async_gym_agents.agents.injector import (
    AsyncAgentInjector,
    AsyncAgentInjectorBase,
    AsyncAgentInjectorMP,
    IAsyncAgentInjector,
    InjectorWorkerBase,
)
from async_gym_agents.envs.multi_env import IndexableMultiEnv
from async_gym_agents.types import EnvFactory, EnvFactoryList
from async_gym_agents.utils import single_slice


@dataclass
class Transition:
    actions: np.ndarray
    values: torch.Tensor
    log_probs: torch.Tensor
    last_obs: VecEnvObs
    new_obs: VecEnvObs
    rewards: np.ndarray
    dones: np.ndarray
    last_dones: np.ndarray
    infos: list[Dict]


class OnPolicyAlgorithmInjectorBase(AsyncAgentInjectorBase, OnPolicyAlgorithm):
    def __init__(self, *args, **kwargs):
        super().__init__(self)
        super(AsyncAgentInjectorBase, self).__init__(*args, **kwargs)

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
        assert (
            self.n_envs == 1
        ), "Do not pass a VecEnv > 1, use IndexableMultiEnv or thr gym.Env interface instead!"

        if not self.initialized:
            self.init_collect_process()

        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)
        self.pre_collect_preparation(self.policy)

        n_steps = 0
        rollout_buffer.reset()

        # Sample new weights for the state-dependent exploration
        if self.use_sde:
            self.policy.reset_noise(1)

        callback.on_rollout_start()

        new_obs = None
        dones = None
        while n_steps < n_rollout_steps:
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and n_steps % self.sde_sample_freq == 0
            ):
                # Sample a new noise matrix
                self.policy.reset_noise(1)

            # Fetch transitions from workers
            transition: Transition = self.fetch_transition()

            # Make locals available for callbacks
            new_obs = transition.new_obs
            self._last_obs = transition.last_obs
            actions = transition.actions
            rewards = transition.rewards
            self._last_episode_starts = transition.last_dones
            values = transition.values
            log_probs = transition.log_probs
            dones = transition.dones
            infos = transition.infos

            self.num_timesteps += 1

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            # Handle timeout by bootstrapping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(
                        infos[idx]["terminal_observation"]
                    )[0]
                    with torch.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
            )

        with torch.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True


class EpisodeGenerator:
    def __init__(self, action_space: gym.Space, device: torch.device) -> None:
        self.action_space = action_space
        self.device = device

    def update_policy(self, policy: BasePolicy) -> BasePolicy:
        return policy

    def generate(
        self, policy: BasePolicy, env: IndexableMultiEnv, index: int
    ) -> Generator[list[Transition], None, None]:
        """
        Continuously plays the game and returns episodes of Transitions
        """
        last_obs = env.reset(index=index)
        last_dones = None

        episodes = {}

        while True:
            with torch.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(last_obs, self.device)
                actions, values, log_probs = policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out-of-bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(
                        actions, self.action_space.low, self.action_space.high
                    )

            new_obs, rewards, dones, infos = env.step(clipped_actions, index=index)

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Store transition
            for idx in range(len(dones)):
                if idx not in episodes:
                    episodes[idx] = []
                episodes[idx].append(
                    Transition(
                        single_slice(actions, idx),
                        single_slice(values, idx),
                        single_slice(log_probs, idx),
                        deepcopy(single_slice(last_obs, idx)),
                        deepcopy(single_slice(new_obs, idx)),
                        single_slice(rewards, idx),
                        single_slice(dones, idx),
                        single_slice(last_dones, idx)
                        if last_dones is not None
                        else [False],
                        single_slice(infos, idx),
                    )
                )
            last_obs = new_obs
            last_dones = dones

            # Start a new episode
            for idx, done in enumerate(dones):
                if done:
                    yield episodes[idx]
                    del episodes[idx]

                    policy = self.update_policy(policy)


class OnPolicyAlgorithmInjector(AsyncAgentInjector, OnPolicyAlgorithmInjectorBase):
    def __init__(self, *args, max_episodes_in_buffer: int = 8, **kwargs) -> None:
        super().__init__(max_episodes_in_buffer)
        super(AsyncAgentInjector, self).__init__(*args, **kwargs)

    def train(self, *args, **kwargs) -> None:
        # update self.training_policy
        with self.training_policy_lock:
            super().train()
        self.training_policy_version += 1

    def _episode_generator(self, index: int) -> Generator[list, None, None]:
        """
        Continuously plays the game and returns episodes of Transitions
        """
        episode_generator = EpisodeGenerator(self.action_space, self.device)
        generator = episode_generator.generate(
            self.policy, self.get_indexable_env(), index
        )
        while self.running:
            yield next(generator)
            self.sync_training_policy_to_rollout_policy_weights_only(index)


class InjectorWorker(InjectorWorkerBase, EpisodeGenerator):
    def __init__(
        self,
        env_func: EnvFactory,
        trajectory: multiprocessing.Queue,
        state: Namespace,
        stop: multiprocessing.Event,
        **kwargs,
    ):
        super().__init__(env_func, trajectory, state, stop)
        EpisodeGenerator.__init__(self, **kwargs)

        self._version = None
        self._policy = None

        self._logger = logging.getLogger("Worker")

    def episode_generator(self, env: IndexableMultiEnv, index: int):
        generator = self.generate(self._copy_policy_from_state(), env, index)

        while self.running:
            episode = next(generator)

            try:
                self._trajectory.put_nowait(episode)
            except queue.Full:
                self._logger.info("Dropped episode due to buffer full")

        self._logger.info("generator cycle is completed")

    def _copy_policy_from_state(self):
        state = self._state
        policy_bytes = state.policy

        if self._version != state.version:
            data = io.BytesIO(policy_bytes)
            # load state
            policy = torch.load(data, weights_only=False, map_location="cpu")
            # turn off the train mode
            policy.set_training_mode(False)

            self._policy = policy
            self._version = state.version

        return self._policy

    def update_policy(self, policy: BasePolicy) -> BasePolicy:
        return self._copy_policy_from_state()


class OnPolicyAlgorithmInjectorMP(AsyncAgentInjectorMP, OnPolicyAlgorithmInjectorBase):
    def __init__(
        self,
        *args,
        max_episodes_in_buffer: int = 8,
        envs: Optional[EnvFactoryList] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            envs=envs,
            worker_class=InjectorWorker,
            max_episodes_in_buffer=max_episodes_in_buffer,
        )
        super(IAsyncAgentInjector, self).__init__(*args, **kwargs)

        # hardcoded override (!)
        self.device = get_device("cpu")

    def get_worker_kwargs(self):
        return dict(action_space=self.action_space, device=self.device)
