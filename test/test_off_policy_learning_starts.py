"""Off-policy workers' switch from random warm-up actions to policy actions after learning_starts.

Strict xfails document unresolved findings; they assert the desired behavior.
"""

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import DQNPolicy

from async_gym_agents.agents.async_agent import get_injected_agent
from async_gym_agents.envs.multi_env import IndexableMultiEnv

TOTAL_TIMESTEPS = 64


class OneStepEnv(gym.Env):
    observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(2, np.float32), {}

    def step(self, action):
        return np.zeros(2, np.float32), 0.0, True, False, {}


def count_worker_policy_actions(monkeypatch, learning_starts):
    """Train DQN and count worker actions chosen by the policy (the learner never calls policy.predict)."""
    policy_actions = []
    original_predict = DQNPolicy.predict

    def record_predict(self, *args, **kwargs):
        policy_actions.append(1)
        return original_predict(self, *args, **kwargs)

    monkeypatch.setattr(DQNPolicy, "predict", record_predict)
    agent = get_injected_agent(DQN)(
        "MlpPolicy",
        IndexableMultiEnv([OneStepEnv]),
        learning_starts=learning_starts,
        train_freq=4,
        buffer_size=1000,
        batch_size=8,
        device="cpu",
        use_mp=False,
        worker_join_timeout=5.0,
    )
    try:
        agent.learn(total_timesteps=TOTAL_TIMESTEPS)
    finally:
        agent.shutdown()
    return len(policy_actions)


def test_workers_use_policy_without_warm_up(monkeypatch):
    assert count_worker_policy_actions(monkeypatch, learning_starts=0) >= TOTAL_TIMESTEPS


@pytest.mark.xfail(
    strict=True,
    reason="off_policy_injector.py:323,345 copy num_timesteps once at worker creation; :362 never sees it grow",
)
def test_workers_switch_to_policy_after_learning_starts(monkeypatch):
    assert count_worker_policy_actions(monkeypatch, learning_starts=16) >= TOTAL_TIMESTEPS - 16
