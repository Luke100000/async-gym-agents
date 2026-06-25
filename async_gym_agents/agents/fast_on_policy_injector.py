from typing import Optional

import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor

from async_gym_agents.agents.on_policy_injector import OnPolicyAlgorithmInjector


class FastOnPolicyAlgorithmInjector(OnPolicyAlgorithmInjector):
    """
    On-policy trainer that fetches a rollout (up to n_steps), trains, and pushes
    the new policy each cycle. n_epochs is EMA-tuned toward producing n_steps per
    cycle: fewer produced than n_steps -> more epochs, more -> fewer.
    """

    def __init__(
        self,
        *args,
        full_speed_min_rollout: int = 16,
        full_speed_smoothing: float = 0.3,
        full_speed_min_epochs: int = 1,
        full_speed_max_epochs: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.full_speed_min_rollout = max(1, full_speed_min_rollout)
        # EMA weight on the newest production count (1.0 = no smoothing).
        self.full_speed_smoothing = min(1.0, max(0.0, full_speed_smoothing))
        self.full_speed_min_epochs = max(1, full_speed_min_epochs)
        self.full_speed_max_epochs = full_speed_max_epochs or self.n_epochs * 100

        self._fs_epochs = float(self.n_epochs)
        self._produced_ema: Optional[float] = None

    def _adjust_epochs(self, produced: int) -> None:
        # Under target -> more epochs, over -> fewer.
        if self._produced_ema is None:
            self._produced_ema = float(produced)
        else:
            a = self.full_speed_smoothing
            self._produced_ema = a * produced + (1.0 - a) * self._produced_ema

        factor = (self.n_steps + 1.0) / (self._produced_ema + 1.0)
        self._fs_epochs = min(
            float(self.full_speed_max_epochs),
            max(float(self.full_speed_min_epochs), self._fs_epochs * factor),
        )
        self.n_epochs = max(1, round(self._fs_epochs))
        print(self.n_epochs, produced, factor)

    def _collect_full_speed_rollout(self, callback: BaseCallback) -> tuple[bool, int]:
        assert self._last_obs is not None, "No previous observation was provided"

        self.policy.set_training_mode(False)
        self.pre_collect_preparation(self.policy)
        callback.on_rollout_start()

        # Block to the floor, then drain up to n_steps (cap bounds train time).
        floor = min(self.full_speed_min_rollout, self.n_steps)
        transitions = []
        while len(transitions) < self.n_steps:
            if len(transitions) < floor:
                transitions.append(self.fetch_transition())
            else:
                transition = self.try_fetch_transition()
                if transition is None:
                    break
                transitions.append(transition)

        rollout_buffer = self.rollout_buffer_class(
            len(transitions),
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=1,
            **self.rollout_buffer_kwargs,
        )

        new_obs = None
        dones = None
        for transition in transitions:
            continue_training, new_obs, dones = self._process_worker_transition(
                rollout_buffer, callback, transition
            )
            if not continue_training:
                return False, len(transitions)

        with self._profiler_main.track("processing"):
            with torch.inference_mode():
                values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))
            rollout_buffer.compute_returns_and_advantage(
                last_values=values, dones=dones
            )
            callback.update_locals(locals())

        self.rollout_buffer = rollout_buffer
        callback.on_rollout_end()
        return True, len(transitions)

    def learn(
        self,
        total_timesteps: int,
        callback=None,
        log_interval: int = 1,
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

        self._fs_epochs = float(self.n_epochs)
        self._produced_ema = None

        iteration = 0
        while self.num_timesteps < total_timesteps:
            continue_training, collected = self._collect_full_speed_rollout(callback)
            if not continue_training:
                break

            produced = self.take_produced_samples()
            self._adjust_epochs(produced)

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            if log_interval is not None and iteration % log_interval == 0:
                self.logger.record("rollout/collected", collected)
                self.logger.record("rollout/produced", produced)
                self.logger.record("train/n_epochs", self.n_epochs)
                self._dump_logs(iteration)

            self.train()

        callback.on_training_end()
        return self
