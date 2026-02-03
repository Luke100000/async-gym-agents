import io
import logging
import multiprocessing
import queue
from copy import deepcopy
from dataclasses import dataclass
from multiprocessing.managers import Namespace
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import RolloutReturn, TrainFreq
from stable_baselines3.common.utils import get_device, should_collect_more_steps
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
from async_gym_agents.types import EnvFactoryList
from async_gym_agents.utils import single_slice


@dataclass
class Transition:
    buffer_actions: np.ndarray
    last_obs: VecEnvObs
    new_obs: VecEnvObs
    rewards: np.ndarray
    dones: np.ndarray
    infos: list[Dict]


class OffPolicyAlgorithmInjectorBase(AsyncAgentInjectorBase, OffPolicyAlgorithm):
    def __init__(self, *args, **kwargs):
        super().__init__(self)
        super(AsyncAgentInjectorBase, self).__init__(*args, **kwargs)

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
            Required for deterministic policy (e.g., TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every `log_interval` episode
        :return:
        """

        if not self.initialized:
            self.init_collect_process()

        assert (
            self.n_envs == 1
        ), "Do not pass a VecEnv > 1, use IndexableMultiEnv or thr gym.Env interface instead!"

        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)
        self.pre_collect_preparation(self.policy)

        num_collected_steps, num_collected_episodes = 0, 0

        self.learning_starts = learning_starts
        self.action_noise = action_noise

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if self.use_sde:
            self.actor.reset_noise(1)

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

            # Only stop training if the return value is False, not when it is None.
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


class EpisodeGenerator:
    def __init__(
        self,
        learning_starts: int,
        num_timesteps: int,
        use_sde: bool,
        use_sde_at_warmup: bool,
        action_space: gym.Space,
        action_noise: Optional[ActionNoise],
    ) -> None:
        self.learning_starts = learning_starts
        self.num_timesteps = num_timesteps
        self.use_sde = use_sde
        self.use_sde_at_warmup = use_sde_at_warmup
        self.action_space = action_space
        self.action_noise = action_noise

    def sample_action(
        self,
        policy: BasePolicy,
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
            unscaled_action = np.array([self.action_space.sample() for _ in obs])
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            unscaled_action, _ = policy.predict(obs, deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, spaces.Box):
            scaled_action = policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action

    def update_policy(self, policy: BasePolicy) -> BasePolicy:
        return policy

    def generate(
        self, policy: BasePolicy, env: IndexableMultiEnv, index: int
    ) -> Generator[list[Transition], None, None]:
        """
        Continuously plays the game and returns episodes of Transitions
        """
        last_obs = env.reset(index=index)

        episodes = {}

        while True:
            # Select action randomly or according to policy
            actions, buffer_actions = self.sample_action(
                policy,
                self.learning_starts,
                last_obs,
                self.action_noise,
            )

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions, index=index)

            # Store transition
            for idx in range(len(dones)):
                if idx not in episodes:
                    episodes[idx] = []
                episodes[idx].append(
                    Transition(
                        single_slice(buffer_actions, idx),
                        deepcopy(single_slice(last_obs, idx)),
                        deepcopy(single_slice(new_obs, idx)),
                        single_slice(rewards, idx),
                        single_slice(dones, idx),
                        single_slice(infos, idx),
                    )
                )
            last_obs = new_obs

            # Start a new episode
            for idx, done in enumerate(dones):
                if done:
                    yield episodes[idx]
                    del episodes[idx]

                    policy = self.update_policy(policy)


class OffPolicyAlgorithmInjector(AsyncAgentInjector, OffPolicyAlgorithmInjectorBase):
    def __init__(self, *args, max_episodes_in_buffer: int = 8, **kwargs) -> None:
        super().__init__(max_episodes_in_buffer)
        super(AsyncAgentInjectorBase, self).__init__(*args, **kwargs)

    def train(self, *args, **kwargs) -> None:
        with self.training_policy_lock:
            super().train(*args, **kwargs)
        self.training_policy_version += 1

    def _excluded_save_params(self) -> List[str]:
        return [
            *super()._excluded_save_params(),
            *super(AsyncAgentInjector, self)._excluded_save_params(),
        ]

    def _sample_action(*args):
        raise NotImplementedError()

    def _episode_generator(self, index: int) -> Generator[list, None, None]:
        """
        Continuously plays the game and returns episodes of Transitions
        """
        episode_generator = EpisodeGenerator(
            self.learning_starts,
            self.num_timesteps,
            self.use_sde,
            self.use_sde_at_warmup,
            self.action_space,
            self.action_noise,
        )
        generator = episode_generator.generate(
            self.policy, self.get_indexable_env(), index
        )
        while self.running:
            yield next(generator)
            self.sync_training_policy_to_rollout_policy_weights_only(index)


class InjectorWorker(InjectorWorkerBase, EpisodeGenerator):
    def __init__(
        self,
        env_func,
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


class OffPolicyAlgorithmInjectorMP(
    AsyncAgentInjectorMP, OffPolicyAlgorithmInjectorBase
):
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
        return dict(
            learning_starts=self.learning_starts,
            num_timesteps=self.num_timesteps,
            use_sde=self.use_sde,
            use_sde_at_warmup=self.use_sde_at_warmup,
            action_space=self.action_space,
            action_noise=self.action_noise,
        )
