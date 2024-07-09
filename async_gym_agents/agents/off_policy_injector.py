from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import RolloutReturn, TrainFreq
from stable_baselines3.common.utils import should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv

from async_gym_agents.agents.injector import AsyncAgentInjector
from async_gym_agents.envs.multi_env import IndexableMultiEnv


@dataclass
class Transition:
    buffer_actions: list[any]
    last_obs: list[any]
    new_obs: list[any]
    rewards: list[float]
    dones: list[bool]
    infos: list[Any]


class OffPolicyAlgorithmInjector(AsyncAgentInjector, OffPolicyAlgorithm):
    def __init__(self, *args, max_steps_in_buffer: int = 8, **kwargs) -> None:
        super().__init__(max_steps_in_buffer)
        super(AsyncAgentInjector, self).__init__(*args, **kwargs)

    def _excluded_save_params(self) -> List[str]:
        return [
            *super()._excluded_save_params(),
            *super(AsyncAgentInjector, self)._excluded_save_params(),
        ]

    def _store_transition(*args):
        raise NotImplementedError()

    def _custom_store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        last_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """
        Nearly identical to super but stateless (last_obs now passed)
        """
        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            raise NotImplementedError()

        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(new_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    # Replace next obs for the correct envs
                    for key in new_obs.keys():
                        new_obs[key][i] = next_obs_[key]
                else:
                    new_obs[i] = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        new_obs[i] = self._vec_normalize_env.unnormalize_obs(
                            new_obs[i, :]
                        )

        replay_buffer.add(
            last_obs,
            new_obs,
            buffer_action,
            reward,
            dones,
            infos,
        )

    def _sample_action(*args):
        raise NotImplementedError()

    def _custom_sample_action(
        self,
        learning_starts: int,
        obs,
        action_noise: Optional[ActionNoise] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Very similar but uses passed observation as input
        """
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (
            self.use_sde and self.use_sde_at_warmup
        ):
            # Warmup phase
            unscaled_action = np.array([self.action_space.sample()])
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            unscaled_action, _ = self.predict(obs, deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action

    def _episode_generator(self, index: int) -> list[Transition]:
        """
        Continuously plays the game and returns episodes of Transitions
        """
        env = self.get_indexable_env()
        last_obs = env.reset(index=index)

        episode = []

        while self.running:
            # Select action randomly or according to policy
            actions, buffer_actions = self._custom_sample_action(
                self.learning_starts,
                last_obs,
                self.action_noise,
            )

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions, index=index)

            # Store transition
            episode.append(
                Transition(
                    buffer_actions,
                    deepcopy(last_obs),
                    deepcopy(new_obs),
                    rewards,
                    dones,
                    infos,
                )
            )
            last_obs = new_obs

            # Start new episode
            if any(dones):
                yield episode
                episode = []

    def collect_rollouts(
        self,
        env: IndexableMultiEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        self.learning_starts = learning_starts
        self.action_noise = action_noise

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if self.use_sde:
            self.actor.reset_noise(1)

        if not self.initialized:
            self._initialize_threads()
            self.initialized = True

        callback.on_rollout_start()
        continue_training = True
        while should_collect_more_steps(
            train_freq, num_collected_steps, num_collected_episodes
        ):
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and num_collected_steps % self.sde_sample_freq == 0
            ):
                # Sample a new noise matrix
                self.actor.reset_noise(1)

            # Fetch transition (Also the only significant change to super)
            transition: Transition = self.fetch_transition()

            # Make locals available for callbacks
            buffer_actions = transition.buffer_actions
            self._last_obs = transition.last_obs
            new_obs = transition.new_obs
            rewards = transition.rewards
            dones = transition.dones
            infos = transition.infos

            # Update stats
            self.num_timesteps += 1
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())

            # Only stop training if return value is False, not when it is None.
            if not callback.on_step():
                return RolloutReturn(
                    num_collected_steps,
                    num_collected_episodes,
                    continue_training=False,
                )

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._custom_store_transition(
                replay_buffer,
                buffer_actions,
                self._last_obs,
                new_obs,
                rewards,
                dones,
                infos,
            )

            self._update_current_progress_remaining(
                self.num_timesteps, self._total_timesteps
            )

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        action_noise.reset()

                    # Log training infos
                    if (
                        log_interval is not None
                        and self._episode_num % log_interval == 0
                    ):
                        self._dump_logs()

        callback.on_rollout_end()

        return RolloutReturn(
            num_collected_steps,
            num_collected_episodes,
            continue_training,
        )
