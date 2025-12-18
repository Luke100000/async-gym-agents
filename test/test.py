import logging
import os
from typing import List, Union

import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm

from async_gym_agents.agents.async_agent import get_injected_agent
from async_gym_agents.agents.injector import AsyncAgentInjectorBase
from async_gym_agents.envs.multi_env import IndexableMultiEnv

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


processes = 8


def run_test(model: Union[BaseAlgorithm, AsyncAgentInjectorBase]):
    # Train the model
    model.learn(total_timesteps=100)

    # Test continual learning
    model.learn(total_timesteps=10)
    model.shutdown()

    # Test saving and loading
    model.save("test.zip")
    model.load("test.zip")
    os.remove("test.zip")
    model.learn(total_timesteps=10)
    model.shutdown()


def test_on_policy():
    # Create env with 8 parallel envs
    env = IndexableMultiEnv([lambda: gym.make("Pendulum-v1") for _ in range(processes)])

    # Create the model, injected with async capabilities
    model = get_injected_agent(PPO)("MlpPolicy", env)

    run_test(model)


def env_func_on() -> gym.Env:
    return gym.make("Taxi-v3")


def test_on_policy_mp():
    # Create env
    env = gym.make("Taxi-v3")

    # Create the model, injected with async capabilities
    model = get_injected_agent(PPO, use_mp=True)(
        "MlpPolicy", env, envs=[env_func_on for _ in range(processes)]
    )

    run_test(model)


def test_off_policy():
    # Create env with 8 parallel envs
    env = IndexableMultiEnv(
        [lambda: gym.make("LunarLanderContinuous-v3") for _ in range(processes)]
    )

    # Create the model, injected with async capabilities
    model = get_injected_agent(SAC)("MlpPolicy", env)

    run_test(model)


def env_func_off() -> List[gym.Env]:
    return [gym.make("LunarLanderContinuous-v3") for _ in range(processes)]


def test_off_policy_mp():
    # Create env
    env = gym.make("LunarLanderContinuous-v3")

    # Create the model, injected with async capabilities
    model = get_injected_agent(SAC, use_mp=True)("MlpPolicy", env, envs=[env_func_off])

    run_test(model)


if __name__ == "__main__":
    test_on_policy()
    test_on_policy_mp()
    test_off_policy()
    test_off_policy_mp()
    print("Done")
