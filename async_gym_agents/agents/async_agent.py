import threading
from copy import deepcopy
from dataclasses import dataclass
from threading import Thread
from typing import Optional, Union, Dict, List, Any, Tuple

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import (
    RolloutReturn,
    TrainFreq,
    TrainFrequencyUnit,
)
from stable_baselines3.common.vec_env import VecEnv

from async_gym_agents.envs.multi_env import IndexableMultiEnv


@dataclass
class SharedState:
    """
    State shared between worker, reset per collection step

    """

    remaining_steps: int = 0
    sleeping_threads: int = 0

    num_collected_steps: int = 0
    num_collected_episodes: int = 0

    callback: BaseCallback = None
    replay_buffer: ReplayBuffer = None
    action_noise: Optional[ActionNoise] = None
    learning_starts: int = 0
    log_interval: Optional[int] = None


class OffPolicyAlgorithmInjector(OffPolicyAlgorithm):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Synchronization points
        self.thread_condition = threading.Condition()
        self.main_condition = threading.Condition()

        # Shared progress between workers
        self.shared_state = SharedState()

        # Initialize lazy to support providing envs later on
        self.initialized = False

    def get_indexable_env(self) -> IndexableMultiEnv:
        assert isinstance(
            self.env, IndexableMultiEnv
        ), "You must pass a IndexableMultiEnv"
        return self.env

    def initialize_threads(self):
        threads = []
        for index in range(self.get_indexable_env().real_n_envs):
            thread = Thread(
                target=self._collector_loop,
                args=(index,),
            )
            threads.append(thread)
            threads[index].start()

    def _excluded_save_params(self) -> List[str]:
        return [
            *super()._excluded_save_params(),
            "thread_condition",
            "main_condition",
            "shared_state",
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

        # Avoid modification by reference
        last_obs = deepcopy(last_obs)
        next_obs = deepcopy(new_obs)

        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    # Replace next obs for the correct envs
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(
                            next_obs[i, :]
                        )

        replay_buffer.add(
            last_obs,
            next_obs,
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

    def _collector_loop(
        self,
        index: int,
    ):
        env = self.get_indexable_env()
        last_obs = env.reset(index=index)

        while True:
            # Select action randomly or according to policy
            actions, buffer_actions = self._custom_sample_action(
                self.shared_state.learning_starts,
                last_obs,
                self.shared_state.action_noise,
            )

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions, index=index)

            with self.thread_condition:
                # Wait for next loop
                while self.shared_state.remaining_steps <= 0:
                    with self.main_condition:
                        self.main_condition.notify()
                    self.shared_state.sleeping_threads += 1
                    self.thread_condition.wait()
                self.shared_state.sleeping_threads -= 1
                self.shared_state.remaining_steps -= 1

                self.num_timesteps += 1
                self.shared_state.num_collected_steps += 1

                # Give access to local variables
                self.shared_state.callback.update_locals(locals())

                # Only stop training if return value is False, not when it is None.
                if not self.shared_state.callback.on_step():
                    return RolloutReturn(
                        self.shared_state.num_collected_steps,
                        self.shared_state.num_collected_episodes,
                        continue_training=False,
                    )

                # Retrieve reward and episode length if using Monitor wrapper
                self._update_info_buffer(infos, dones)

                # Store data in replay buffer (normalized action and unnormalized observation)
                self._custom_store_transition(
                    self.shared_state.replay_buffer,
                    buffer_actions,
                    last_obs,
                    new_obs,
                    rewards,
                    dones,
                    infos,
                )
                last_obs = new_obs

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
                        self.shared_state.num_collected_episodes += 1
                        self._episode_num += 1

                        if self.shared_state.action_noise is not None:
                            kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                            # noinspection PyArgumentList
                            self.shared_state.action_noise.reset(**kwargs)

                        # Log training infos
                        if (
                            self.shared_state.log_interval is not None
                            and self._episode_num % self.shared_state.log_interval == 0
                        ):
                            self._dump_logs()

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

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."
        assert (
            train_freq.unit == TrainFrequencyUnit.STEP
        ), "You must use only one env when doing episodic training."

        if self.use_sde:
            raise NotImplementedError("Not supported yet.")

        callback.on_rollout_start()

        ####################

        if not self.initialized:
            self.initialize_threads()
            self.initialized = True

        assert train_freq.unit == TrainFrequencyUnit.STEP
        self.shared_state.remaining_steps = train_freq.frequency

        self.shared_state.num_collected_steps = 0
        self.shared_state.num_collected_episodes = 0

        self.shared_state.callback = callback
        self.shared_state.replay_buffer = replay_buffer
        self.shared_state.action_noise = action_noise
        self.shared_state.learning_starts = learning_starts
        self.shared_state.log_interval = log_interval

        # Release threads
        with self.thread_condition:
            self.thread_condition.notify_all()

        # Wait until first thread finishes
        with self.main_condition:
            self.main_condition.wait()

        ####################

        callback.on_rollout_end()

        return RolloutReturn(
            self.shared_state.num_collected_steps,
            self.shared_state.num_collected_episodes,
            True,
        )


def get_injected_agent(clazz: OffPolicyAlgorithm):
    # TODO also support on policy

    class AsyncAgent(OffPolicyAlgorithmInjector, clazz):
        pass

    return AsyncAgent
