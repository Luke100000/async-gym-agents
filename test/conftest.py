from functools import partial

import gymnasium as gym
import pytest
from stable_baselines3.common.monitor import Monitor

from async_gym_agents.envs.buggy_lunar_lander import BuggyLunarLander
from async_gym_agents.envs.multi_env import IndexableMultiEnv

PROCESSES = 8


@pytest.fixture
def taxi_multi_env():
    """Fixture for creating a multi-environment with discrete action space environment."""
    return IndexableMultiEnv([lambda: gym.make("Taxi-v3") for _ in range(PROCESSES)])


@pytest.fixture
def lunar_lander_multi_env():
    """Fixture for creating a multi-environment with continuous action space environment."""
    return IndexableMultiEnv(
        [lambda: gym.make("LunarLanderContinuous-v3") for _ in range(PROCESSES)]
    )


@pytest.fixture
def taxi_env():
    """Fixture for creating a discrete action space environment."""
    return partial(gym.make, "Taxi-v3")


@pytest.fixture
def lunar_lander_env():
    """Fixture for creating a continuous action space environment."""
    return partial(gym.make, "LunarLanderContinuous-v3")


def get_buggy_env(buggy: bool) -> gym.Env:
    """Environment factory for an env returning truncated episodes."""
    return Monitor(
        BuggyLunarLander(
            crash_probability=0.01 if buggy else 0,
            time_limit=1000,
        )
    )
