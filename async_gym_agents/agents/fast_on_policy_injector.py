import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor, safe_mean

from async_gym_agents.agents.on_policy_injector import OnPolicyAlgorithmInjector


class FastOnPolicyAlgorithmInjector(OnPolicyAlgorithmInjector):
    """
    On-policy trainer that fetches a fresh rollout, trains, and pushes the new
    policy each cycle. The rollout size floats with worker throughput: it blocks
    until a small floor of samples is collected (a sanity check against training
    on almost no data), then drains whatever else is queued up to ``n_steps``.
    No replay buffer, hence no batch/freshness controller.
    """

    def __init__(self, *args, full_speed_min_rollout: int = 16, **kwargs):
        super().__init__(*args, **kwargs)
        self.full_speed_min_rollout = max(1, full_speed_min_rollout)

    def _collect_full_speed_rollout(self, callback: BaseCallback) -> bool:
        assert self._last_obs is not None, "No previous observation was provided"

        self.policy.set_training_mode(False)
        self.pre_collect_preparation(self.policy)
        callback.on_rollout_start()

        # Block until the floor is met, then drain greedily up to n_steps.
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
                return False

        with self._profiler_main.track("processing"):
            with torch.inference_mode():
                values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))
            rollout_buffer.compute_returns_and_advantage(
                last_values=values, dones=dones
            )
            callback.update_locals(locals())

        self.rollout_buffer = rollout_buffer
        callback.on_rollout_end()
        return True

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

        iteration = 0
        while self.num_timesteps < total_timesteps:
            if not self._collect_full_speed_rollout(callback):
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                self.logger.record("time/iterations", iteration)
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record(
                        "rollout/ep_rew_mean",
                        safe_mean([ep["r"] for ep in self.ep_info_buffer]),
                    )
                    self.logger.record(
                        "rollout/ep_len_mean",
                        safe_mean([ep["l"] for ep in self.ep_info_buffer]),
                    )
                self.logger.record("time/total_timesteps", self.num_timesteps)
                self.logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()
        return self
