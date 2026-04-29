from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple, Type

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import RolloutReturn, TrainFreq
from stable_baselines3.common.utils import should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs

from async_gym_agents.agents.injector import AsyncAgentInjector, InjectorWorkerBase
from async_gym_agents.utils import copy_obs, single_slice


@dataclass
class Transition:
    buffer_actions: np.ndarray
    last_obs: VecEnvObs
    new_obs: VecEnvObs
    rewards: np.ndarray
    dones: np.ndarray
    infos: list[Dict]
    reset_infos: list[Dict]


class OffPolicyAlgorithmInjector(AsyncAgentInjector, OffPolicyAlgorithm):
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

    def _store_transition(*args):
        raise NotImplementedError()

    def _custom_store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        last_obs: VecEnvObs,
        new_obs: VecEnvObs,
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

        assert replay_buffer.n_envs == 1
        replay_buffer.add(
            last_obs,
            new_obs,
            buffer_action,
            reward,
            dones,
            infos,
        )

    # must be updated from SB3 (!)
    def collect_rollouts(
        self,
        env: VecEnv,
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

        if replay_buffer.n_envs != 1:
            replay_buffer.n_envs = 1
            replay_buffer.reset()

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

            with self._profiler_main.track("processing"):
                # Make locals available for callbacks
                buffer_actions = transition.buffer_actions
                self._last_obs = transition.last_obs
                new_obs = transition.new_obs
                rewards = transition.rewards
                dones = transition.dones
                infos = transition.infos
                reset_infos = transition.reset_infos

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

    def get_worker_class(self) -> Type[InjectorWorkerBase]:
        return InjectorWorker

    def get_worker_kwargs(self):
        return dict(
            **super().get_worker_kwargs(),
            learning_starts=self.learning_starts,
            num_timesteps=self.num_timesteps,
            use_sde=self.use_sde,
            use_sde_at_warmup=self.use_sde_at_warmup,
            action_space=self.action_space,
            action_noise=self.action_noise,
        )


class InjectorWorker(InjectorWorkerBase):
    def __init__(
        self,
        learning_starts: int,
        num_timesteps: int,
        use_sde: bool,
        use_sde_at_warmup: bool,
        action_space: gym.Space,
        action_noise: Optional[ActionNoise],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

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

    def generate(self) -> Generator[list[Transition], None, None]:
        """
        Continuously plays the game and returns episodes of Transitions
        """
        with self._profiler.track("resetting"):
            last_obs = self.env.reset()

        episodes = {}

        while True:
            # Select action randomly or according to policy
            with self._profiler.track("inference"):
                actions, buffer_actions = self.sample_action(
                    self.policy,
                    self.learning_starts,
                    last_obs,
                    self.action_noise,
                )

            # Rescale and perform action
            with self._profiler.track("stepping"):
                new_obs, rewards, dones, infos = self.env.step(actions)

            # Store transition
            with self._profiler.track("transition_building"):
                for idx in range(len(dones)):
                    if idx not in episodes:
                        episodes[idx] = []
                    episodes[idx].append(
                        Transition(
                            single_slice(buffer_actions, idx),
                            copy_obs(single_slice(last_obs, idx)),
                            copy_obs(single_slice(new_obs, idx)),
                            single_slice(rewards, idx),
                            single_slice(dones, idx),
                            single_slice(infos, idx),
                            single_slice(self.env.reset_infos, idx),
                        )
                    )
            self._flush_profiler()
            last_obs = new_obs

            # Start a new episode
            for idx, done in enumerate(dones):
                if done:
                    yield episodes[idx]
                    del episodes[idx]

                    self.copy_policy_from_queue()
