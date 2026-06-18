from typing import Optional

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import RolloutReturn, TrainFreq

from async_gym_agents.agents.off_policy_injector import OffPolicyAlgorithmInjector


class FastOffPolicyAlgorithmInjector(OffPolicyAlgorithmInjector):
    def __init__(
        self,
        *args,
        full_speed_collect_steps: int = 32,
        full_speed_train_steps: int = 1,
        full_speed_max_train_bursts: int = 8,
        full_speed_min_replay_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.full_speed_collect_steps = max(1, full_speed_collect_steps)
        self.full_speed_train_steps = max(1, full_speed_train_steps)
        self.full_speed_max_train_bursts = max(1, full_speed_max_train_bursts)
        self.full_speed_min_replay_size = full_speed_min_replay_size

    def _collect_full_speed_rollout(
        self,
        callback: BaseCallback,
        log_interval: Optional[int],
        total_timesteps: int,
        block_for_data: bool,
    ) -> RolloutReturn:
        assert self.replay_buffer is not None

        num_collected_steps, num_collected_episodes = 0, 0
        continue_training = True

        while (
            num_collected_steps < self.full_speed_collect_steps
            and self.num_timesteps < total_timesteps
            and continue_training
        ):
            # Block only when replay cannot keep the trainer busy anymore.
            transition = (
                self.fetch_transition()
                if block_for_data and num_collected_steps == 0
                else self.try_fetch_transition()
            )
            if transition is None:
                break

            continue_training, episodes = self._process_worker_transition(
                self.replay_buffer,
                callback,
                transition,
                self.action_noise,
                log_interval,
            )
            num_collected_steps += 1
            num_collected_episodes += episodes

        return RolloutReturn(
            num_collected_steps,
            num_collected_episodes,
            continue_training,
        )

    def _can_full_speed_train(self, min_replay_size: int) -> bool:
        assert self.replay_buffer is not None
        return (
            self.replay_buffer.size() >= min_replay_size
            and self.num_timesteps > self.learning_starts
        )

    def learn(
        self,
        total_timesteps: int,
        callback=None,
        log_interval: int = 4,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None, (
            "You must set the environment before calling learn()"
        )
        assert isinstance(self.train_freq, TrainFreq)
        assert self.replay_buffer is not None

        if self.replay_buffer.n_envs != 1:
            self.replay_buffer.n_envs = 1
            self.replay_buffer.reset()

        self.policy.set_training_mode(False)
        self.pre_collect_preparation(self.policy)
        callback.on_rollout_start()

        idle_train_bursts = 0
        continue_training = True
        min_replay_size = self.full_speed_min_replay_size or max(
            self.batch_size,
            self.learning_starts,
        )

        while self.num_timesteps < total_timesteps and continue_training:
            can_train = self._can_full_speed_train(min_replay_size)

            # Prefer replay training over waiting, but periodically block for
            # fresh collector data to avoid spinning forever on stale samples.
            rollout = self._collect_full_speed_rollout(
                callback,
                log_interval,
                total_timesteps,
                block_for_data=(
                    not can_train
                    or idle_train_bursts >= self.full_speed_max_train_bursts
                ),
            )
            continue_training = rollout.continue_training
            if not continue_training:
                break

            can_train = self._can_full_speed_train(min_replay_size)
            if can_train:
                self.train(
                    batch_size=self.batch_size,
                    gradient_steps=self.full_speed_train_steps,
                )
                # Workers collect on CPU copies; push each trained policy version.
                self.pre_collect_preparation(self.policy)
                idle_train_bursts = (
                    0 if rollout.episode_timesteps else idle_train_bursts + 1
                )
            else:
                idle_train_bursts = 0

        if continue_training:
            callback.on_rollout_end()
        callback.on_training_end()

        return self
