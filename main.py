from functools import partial
from test.test_agents import run_model_test

import gymnasium as gym
from stable_baselines3 import PPO, SAC

from async_gym_agents.agents.async_agent import get_injected_agent
from async_gym_agents.envs.multi_env import IndexableMultiEnv
from async_gym_agents.envs.threaded_env import ThreadedVecEnv
from async_gym_agents.types import Env

N_ENV = 2
PROCESSES = 1


def taxi_env():
    return gym.make("Taxi-v3")


def lunar_lander_env():
    return gym.make("LunarLanderContinuous-v3")


def make_venv(env) -> Env:
    return ThreadedVecEnv([env for _ in range(N_ENV)])


def multi_env(venv) -> IndexableMultiEnv:
    return IndexableMultiEnv([venv for _ in range(PROCESSES)], venv())


def test_on_policy_vanilla():
    """Test on-policy agent with default SB3."""
    env = multi_env(partial(make_venv, taxi_env))
    model = PPO("MlpPolicy", env)
    run_model_test(model)


def test_on_policy():
    """Test on-policy agent with multiprocessing."""
    env = multi_env(partial(make_venv, taxi_env))
    model = get_injected_agent(PPO)("MlpPolicy", env)
    run_model_test(model)


def test_on_policy_mp():
    """Test on-policy agent with multiprocessing."""
    env = multi_env(partial(make_venv, taxi_env))
    model = get_injected_agent(PPO)("MlpPolicy", env, use_mp=True)
    run_model_test(model)


def test_off_policy_vanilla():
    """Test on-policy agent with default SB3."""
    env = multi_env(partial(make_venv, lunar_lander_env))
    model = SAC("MlpPolicy", env)
    run_model_test(model)


def test_off_policy():
    """Test on-policy agent with multiprocessing."""
    env = multi_env(partial(make_venv, lunar_lander_env))
    model = get_injected_agent(SAC)("MlpPolicy", env)
    run_model_test(model)


def test_off_policy_mp():
    """Test on-policy agent with multiprocessing."""
    env = multi_env(partial(make_venv, lunar_lander_env))
    model = get_injected_agent(SAC)("MlpPolicy", env, use_mp=True)
    run_model_test(model)


if __name__ == "__main__":
    # test_on_policy_vanilla()
    # test_off_policy_vanilla()
    test_on_policy()
    test_off_policy()
    test_on_policy_mp()
    test_off_policy_mp()
