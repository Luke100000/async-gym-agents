"""Batched callback dispatch when a plain step callback stops training on an episode's final transition.

Strict xfails document unresolved findings; they assert the desired behavior.
"""

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from async_gym_agents.agents.async_agent import get_injected_agent
from async_gym_agents.envs.multi_env import IndexableMultiEnv

EPISODE_LENGTH = 2


class TwoStepEnv(gym.Env):
    observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.zeros(2, np.float32), {}

    def step(self, action):
        self.steps += 1
        return np.zeros(2, np.float32), 1.0, self.steps >= EPISODE_LENGTH, False, {}


class StopOnFirstDone(BaseCallback):
    """Plain (per-step) callback halting training on the first episode's final transition."""

    def _on_step(self) -> bool:
        return not np.any(self.locals["dones"])


class RecordEpisodes(BaseCallback):
    """Episode-batchable callback recording the episodes dispatched to it."""

    def __init__(self):
        super().__init__()
        self.episode_lengths = []

    def advance_callback(self, transition_count, num_timesteps):
        self.n_calls += transition_count
        self.num_timesteps = num_timesteps

    def process_episode(self, context):
        self.episode_lengths.append(context.batch.transition_count)
        return True

    def _on_step(self) -> bool:
        return True


def train_until_stopped(algorithm):
    kwargs = (
        dict(n_steps=8, batch_size=8, n_epochs=1)
        if algorithm is PPO
        else dict(learning_starts=0, train_freq=1, buffer_size=100)
    )
    agent = get_injected_agent(algorithm)(
        "MlpPolicy",
        IndexableMultiEnv([TwoStepEnv]),
        device="cpu",
        use_mp=False,
        worker_join_timeout=5.0,
        **kwargs,
    )
    recorder = RecordEpisodes()
    try:
        agent.learn(total_timesteps=16, callback=CallbackList([StopOnFirstDone(), recorder]))
    finally:
        agent.shutdown()
    return agent, recorder


@pytest.mark.parametrize(
    "algorithm",
    [
        pytest.param(
            PPO,
            marks=pytest.mark.xfail(
                strict=True,
                reason="on_policy_injector.py:182-183 returns on the step veto before process_episode at :194",
            ),
        ),
        pytest.param(
            DQN,
            marks=pytest.mark.xfail(
                strict=True,
                reason="off_policy_injector.py:196-201 returns on the step veto before process_episode at :211",
            ),
        ),
    ],
)
def test_batched_callbacks_receive_episode_whose_last_step_stopped_training(algorithm):
    agent, recorder = train_until_stopped(algorithm)

    assert agent.num_timesteps == EPISODE_LENGTH
    assert recorder.episode_lengths == [EPISODE_LENGTH]
