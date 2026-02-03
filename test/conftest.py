from typing import List

import gymnasium as gym
import pytest
from stable_baselines3.common.monitor import Monitor

from async_gym_agents.envs.buggy_lunar_lander import BuggyLunarLander
from async_gym_agents.envs.multi_env import IndexableMultiEnv

PROCESSES = 8


@pytest.fixture
def pendulum_multi_env():
    """Fixture for creating a multi-environment with Pendulum-v1."""
    return IndexableMultiEnv(
        [lambda: gym.make("Pendulum-v1") for _ in range(PROCESSES)]
    )


@pytest.fixture
def lunar_lander_multi_env():
    """Fixture for creating a multi-environment with LunarLanderContinuous-v3."""
    return IndexableMultiEnv(
        [lambda: gym.make("LunarLanderContinuous-v3") for _ in range(PROCESSES)]
    )


@pytest.fixture
def taxi_env():
    """Fixture for creating a discrete action space environment."""
    return gym.make("Taxi-v3")


@pytest.fixture
def lunar_lander_env():
    """Fixture for creating a continuous action space environment."""
    return gym.make("LunarLanderContinuous-v3")


def env_func_on() -> gym.Env:
    """Environment factory function for on-policy tests."""
    return gym.make("Taxi-v3")


def env_func_off() -> List[gym.Env]:
    """Environment factory function for off-policy tests."""
    return [gym.make("LunarLanderContinuous-v3") for _ in range(PROCESSES)]


def get_buggy_env(buggy: bool) -> gym.Env:
    """Environment factory for an env returning truncated episodes."""
    return Monitor(
        BuggyLunarLander(
            crash_probability=0.01 if buggy else 0,
            time_limit=1000,
        )
    )
