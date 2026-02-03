import logging
import os
import time
from typing import Union

from conftest import PROCESSES, env_func_off, env_func_on
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm

from async_gym_agents.agents.async_agent import get_injected_agent
from async_gym_agents.agents.injector import AsyncAgentInjectorBase

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

TRAIN_TIMESTEPS = 1000


def run_model_test(model: Union[BaseAlgorithm, AsyncAgentInjectorBase]):
    """Helper function to test a model's basic functionality."""
    # Train the model
    t = time.time()
    model.learn(total_timesteps=TRAIN_TIMESTEPS, progress_bar=True)
    training_time = time.time() - t

    # Verify training completed
    assert training_time > 0, "Training should take some time"

    # Test continual learning
    model.learn(total_timesteps=10)

    # Test saving and loading
    model_path = "test_model.zip"
    model.save(model_path)
    assert os.path.exists(model_path), "Model file should exist after saving"

    model.load(model_path)
    os.remove(model_path)

    # Final learning step
    model.learn(total_timesteps=10)

    # Cleanup
    if hasattr(model, "shutdown"):
        model.shutdown()


def test_on_policy(pendulum_multi_env):
    """Test on-policy agent."""
    model = get_injected_agent(PPO)("MlpPolicy", pendulum_multi_env)
    run_model_test(model)


def test_on_policy_mp(taxi_env):
    """Test on-policy agent with multiprocessing."""
    model = get_injected_agent(PPO, use_mp=True)(
        "MlpPolicy", taxi_env, envs=[env_func_on for _ in range(PROCESSES)]
    )
    run_model_test(model)


def test_off_policy(lunar_lander_multi_env):
    """Test off-policy agent."""
    model = get_injected_agent(SAC)(
        "MlpPolicy", lunar_lander_multi_env, batch_size=1024
    )
    run_model_test(model)


def test_off_policy_mp(lunar_lander_env):
    """Test off-policy agent with multiprocessing."""
    model = get_injected_agent(SAC, use_mp=True)(
        "MlpPolicy", lunar_lander_env, envs=[env_func_off]
    )
    run_model_test(model)
