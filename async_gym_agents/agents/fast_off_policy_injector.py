from typing import Optional

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import RolloutReturn, TrainFreq

from async_gym_agents.agents.off_policy_injector import OffPolicyAlgorithmInjector


class FastOffPolicyAlgorithmInjector(OffPolicyAlgorithmInjector):
    """
    Off-policy trainer that drains all available worker transitions, trains, and
    pushes the new policy each round. Batch size is the only runtime-tuned knob,
    adjusted to hit a target worker-sync freshness.
    """

    def __init__(
        self,
        *args,
        full_speed_train_steps: int = 1,
        full_speed_target_freshness: float = 1.0,
        full_speed_min_batch_size: Optional[int] = None,
        full_speed_max_batch_size: Optional[int] = None,
        full_speed_min_replay_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.full_speed_train_steps = max(1, full_speed_train_steps)
        # Target worker-syncs per update, per worker.
        self.full_speed_target_freshness = max(0.0, full_speed_target_freshness)
        self.full_speed_min_batch_size = full_speed_min_batch_size
        self.full_speed_max_batch_size = full_speed_max_batch_size
        self.full_speed_min_replay_size = full_speed_min_replay_size

        self._full_speed_batch = float(self.batch_size)

    def _drain_available_transitions(
        self,
        callback: BaseCallback,
        log_interval: Optional[int],
        total_timesteps: int,
        block_for_first: bool,
    ) -> RolloutReturn:
        """
        Drain all available worker transitions into the replay buffer, blocking
        for the first only when ``block_for_first`` is set.
        """
        assert self.replay_buffer is not None

        num_collected_steps, num_collected_episodes = 0, 0
        continue_training = True

        while self.num_timesteps < total_timesteps and continue_training:
            transition = (
                self.fetch_transition()
                if block_for_first and num_collected_steps == 0
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

    def _adjust_batch_size(
        self, synced: int, target: float, min_batch: int, max_batch: int
    ) -> None:
        # Grow when workers under-consumed (synced < target), shrink otherwise.
        factor = (target + 1.0) / (synced + 1.0)
        self._full_speed_batch = min(
            float(max_batch),
            max(float(min_batch), self._full_speed_batch * factor),
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

        continue_training = True
        min_replay_size = self.full_speed_min_replay_size or max(
            self.batch_size,
            self.learning_starts,
        )

        self._full_speed_batch = float(self.batch_size)
        min_batch = self.full_speed_min_batch_size or max(1, self.batch_size // 16)
        max_batch = self.full_speed_max_batch_size or self.batch_size * 16
        target = self.full_speed_target_freshness * len(self._update_queues)

        while self.num_timesteps < total_timesteps and continue_training:
            # Only wait for data when we cannot train yet.
            can_train = self._can_full_speed_train(min_replay_size)
            rollout = self._drain_available_transitions(
                callback,
                log_interval,
                total_timesteps,
                block_for_first=not can_train,
            )
            continue_training = rollout.continue_training
            if not continue_training:
                break

            if self._can_full_speed_train(min_replay_size):
                batch_size = int(round(self._full_speed_batch))
                self.train(
                    batch_size=batch_size,
                    gradient_steps=self.full_speed_train_steps,
                )

                # Adapt to the previous policy's consumption, then push the new one.
                synced = self.take_policy_sync_count()
                self._adjust_batch_size(synced, target, min_batch, max_batch)
                self.pre_collect_preparation(self.policy)

        if continue_training:
            callback.on_rollout_end()
        callback.on_training_end()

        return self
