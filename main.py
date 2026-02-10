from functools import partial
from test.test_agents import run_model_test

import gymnasium as gym
from stable_baselines3 import PPO, SAC

from async_gym_agents.agents.async_agent import get_injected_agent
from async_gym_agents.envs.multi_env import IndexableMultiEnv
from async_gym_agents.envs.threaded_env import ThreadedVecEnv

PROCESSES = 4


def taxi_env():
    return gym.make("Taxi-v3")


def lunar_lander_env():
    return gym.make("LunarLanderContinuous-v3")


def make_venv(env):
    return ThreadedVecEnv([env for _ in range(PROCESSES)])


def multi_env(venv):
    return [venv for _ in range(PROCESSES)]


def test_on_policy():
    """Test on-policy agent with multiprocessing."""
    env = IndexableMultiEnv(multi_env(partial(make_venv, taxi_env)))
    model = get_injected_agent(PPO)("MlpPolicy", env)
    run_model_test(model)


def test_on_policy_mp():
    """Test on-policy agent with multiprocessing."""
    model = get_injected_agent(PPO)(
        "MlpPolicy",
        taxi_env(),
        envs=multi_env(partial(make_venv, taxi_env)),
        use_mp=True,
    )
    run_model_test(model)


def test_off_policy():
    """Test on-policy agent with multiprocessing."""
    env = IndexableMultiEnv(multi_env(partial(make_venv, lunar_lander_env)))
    model = get_injected_agent(SAC)("MlpPolicy", env)
    run_model_test(model)


def test_off_policy_mp():
    """Test on-policy agent with multiprocessing."""
    model = get_injected_agent(SAC)(
        "MlpPolicy",
        lunar_lander_env(),
        envs=multi_env(partial(make_venv, lunar_lander_env)),
        use_mp=True,
    )
    run_model_test(model)


if __name__ == "__main__":
    test_off_policy_mp()
