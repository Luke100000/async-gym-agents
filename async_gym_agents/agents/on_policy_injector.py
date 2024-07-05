import threading
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

from async_gym_agents.agents.injector import AsyncAgentInjector


@dataclass
class Transition:
    actions: list[any]
    values: list[any]
    log_probs: list[any]
    last_obs: list[any]
    new_obs: list[any]
    rewards: list[float]
    dones: list[bool]
    last_dones: list[bool]
    infos: list[Any]
    index: int


class OnPolicyAlgorithmInjector(AsyncAgentInjector, OnPolicyAlgorithm):
    def __init__(self, *args, max_steps_in_buffer: int = 8, **kwargs) -> None:
        super().__init__(max_steps_in_buffer)
        super(AsyncAgentInjector, self).__init__(*args, **kwargs)

        self.policy_lock = threading.Lock()
        self.policy_lock.acquire()

    def _episode_generator(self, index: int) -> list[Transition]:
        """
        Continuously plays the game and returns episodes of Transitions
        """
        env = self.get_indexable_env()
        last_obs = env.reset(index=index)
        last_dones = np.ones((1,), dtype=bool)

        episode = []

        while self.running:
            with self.policy_lock:
                with th.no_grad():
                    # Convert to pytorch tensor or to TensorDict
                    obs_tensor = obs_as_tensor(last_obs, self.device)
                    actions, values, log_probs = self.policy(obs_tensor)
                actions = actions.cpu().numpy()

                # Rescale and perform action
                clipped_actions = actions

                if isinstance(self.action_space, spaces.Box):
                    if self.policy.squash_output:
                        # Unscale the actions to match env bounds
                        # if they were previously squashed (scaled in [-1, 1])
                        clipped_actions = self.policy.unscale_action(clipped_actions)
                    else:
                        # Otherwise, clip the actions to avoid out of bound error
                        # as we are sampling from an unbounded Gaussian distribution
                        clipped_actions = np.clip(
                            actions, self.action_space.low, self.action_space.high
                        )

            new_obs, rewards, dones, infos = env.step(clipped_actions, index=index)

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Store transition
            episode.append(
                Transition(
                    actions,
                    values,
                    log_probs,
                    deepcopy(last_obs),
                    deepcopy(new_obs),
                    rewards,
                    dones,
                    last_dones,
                    infos,
                    index,
                )
            )
            last_obs = new_obs
            last_dones = dones

            # Start new episode
            if any(dones):
                yield episode
                episode = []

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

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"

        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()

        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(1)

        if not self.initialized:
            self._initialize_threads()
            self.initialized = True

        callback.on_rollout_start()

        transition = None
        while n_steps < n_rollout_steps:
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and n_steps % self.sde_sample_freq == 0
            ):
                # Sample a new noise matrix
                self.policy.reset_noise(1)

            # Fetch transitions from workers
            self.policy_lock.release()
            transition: Transition = self.fetch_transition()
            self.policy_lock.acquire()

            self.num_timesteps += 1

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(transition.infos, transition.dones)
            n_steps += 1

            # Handle timeout by bootstrapping with value function
            # see GitHub issue #633
            for idx, done in enumerate(transition.dones):
                if (
                    done
                    and transition.infos[idx].get("terminal_observation") is not None
                    and transition.infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(
                        transition.infos[idx]["terminal_observation"]
                    )[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    transition.rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                transition.last_obs,
                transition.actions,
                transition.rewards,
                transition.last_dones,
                transition.values,
                transition.log_probs,
            )

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(
                obs_as_tensor(transition.new_obs, self.device)
            )

        rollout_buffer.compute_returns_and_advantage(
            last_values=values, dones=transition.dones
        )

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def shutdown(self):
        self.policy_lock.release()
        super().shutdown()
        self.policy_lock.acquire()
