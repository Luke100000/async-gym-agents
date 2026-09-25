import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from async_gym_agents.agents.async_agent import get_injected_agent
from async_gym_agents.envs.multi_env import IndexableMultiEnv

N_STEPS = 16
N_ROLLOUTS = 6


class AlwaysTruncatedEnv(gym.Env):
    """Every episode is one zero-reward step ending in a time-limit truncation.

    Training rewards are therefore exactly ``gamma * V(terminal_observation)``.
    """

    observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, True, {}


class RecordTrainingRewards(BaseCallback):
    """Optionally sets gamma at rollout start; records rewards used for training."""

    def __init__(self, gamma=None):
        super().__init__()
        self.gamma = gamma
        self.max_abs_rewards = []

    def _on_rollout_start(self) -> None:
        if self.gamma is not None:
            self.model.gamma = self.gamma

    def _on_rollout_end(self) -> None:
        rollout_buffer = self.model.rollout_buffer
        rewards = rollout_buffer.rewards[: rollout_buffer.pos]
        self.max_abs_rewards.append(float(np.abs(rewards).max()))

    def _on_step(self) -> bool:
        return True


def train(use_mp, gamma_at_rollout_start):
    env = IndexableMultiEnv([AlwaysTruncatedEnv for _ in range(2)])
    agent = get_injected_agent(PPO)(
        "MlpPolicy",
        env,
        n_steps=N_STEPS,
        batch_size=N_STEPS,
        n_epochs=1,
        gamma=0.99,
        device="cpu",
        use_mp=use_mp,
    )
    callback = RecordTrainingRewards(gamma_at_rollout_start)
    try:
        agent.learn(total_timesteps=N_STEPS * N_ROLLOUTS, callback=callback)
    finally:
        agent.shutdown()
    return callback.max_abs_rewards


@pytest.mark.parametrize("use_mp", [False, True])
def test_workers_bootstrap_with_gamma_changed_during_training(use_mp):
    """Episodes collected after the trainer sets gamma=0 carry no bootstrap."""
    max_abs_rewards = train(use_mp, gamma_at_rollout_start=0.0)

    assert len(max_abs_rewards) == N_ROLLOUTS
    # The first rollout is collected before any change, and the second is
    # assembled in the background while the first trains (the same lag the
    # policy has). Every later rollout must be collected with gamma=0.
    assert max_abs_rewards[2:] == [0.0] * (N_ROLLOUTS - 2)


@pytest.mark.parametrize("use_mp", [False, True])
def test_workers_bootstrap_with_initial_gamma_without_changes(use_mp):
    """Control: with the initial gamma, truncated rows are bootstrapped."""
    max_abs_rewards = train(use_mp, gamma_at_rollout_start=None)

    assert all(max_abs_reward > 0.0 for max_abs_reward in max_abs_rewards)
