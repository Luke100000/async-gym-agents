import logging
import os
import time
from test.conftest import PROCESSES
from typing import Union

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy

from async_gym_agents.agents.async_agent import get_injected_agent
from async_gym_agents.agents.injector import AsyncAgentInjector
from async_gym_agents.envs.multi_env import IndexableMultiEnv

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

TRAIN_TIMESTEPS = 10000
EVAL_TIMESTEPS = 100


def run_model_test(model: Union[BaseAlgorithm, AsyncAgentInjector]):
    """Helper function to test a model's basic functionality."""
    # Train the model
    t = time.time()
    model.learn(total_timesteps=TRAIN_TIMESTEPS, progress_bar=True)
    training_time = time.time() - t

    # Verify training completed
    assert training_time > 0, "Training should take some time"

    # Test continual learning
    model.learn(total_timesteps=100)

    # Test saving and loading
    model_path = "test_model.zip"
    model.save(model_path)
    assert os.path.exists(model_path), "Model file should exist after saving"

    model.load(model_path)
    os.remove(model_path)

    # Learn even after loading
    model.learn(total_timesteps=10)

    # Evaluate
    env = model.env
    if isinstance(env, IndexableMultiEnv):
        env.num_envs = env.real_num_envs
    mean_reward, std_reward = evaluate_policy(
        model, model.env, n_eval_episodes=EVAL_TIMESTEPS
    )
    print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

    # Cleanup
    if hasattr(model, "shutdown"):
        model.shutdown()

        print("Buffer utilization: ", model.buffer_utilization)
        print("Buffer emptiness: ", model.buffer_emptyness)
        print("Discarded episodes: ", model.discarded_episodes_fraction)


def test_on_policy(taxi_multi_env):
    """Test on-policy agent."""
    model = get_injected_agent(PPO)("MlpPolicy", taxi_multi_env)
    run_model_test(model)


def test_on_policy_mp(taxi_env):
    """Test on-policy agent with multiprocessing."""
    model = get_injected_agent(PPO)(
        "MlpPolicy", taxi_env, envs=[taxi_env for _ in range(PROCESSES)], use_mp=True
    )
    run_model_test(model)


def test_off_policy(lunar_lander_multi_env):
    """Test off-policy agent."""
    model = get_injected_agent(SAC)("MlpPolicy", lunar_lander_multi_env)
    run_model_test(model)


def test_off_policy_mp(lunar_lander_env):
    """Test off-policy agent with multiprocessing."""
    model = get_injected_agent(SAC)(
        "MlpPolicy",
        lunar_lander_env,
        envs=[lunar_lander_env for _ in range(PROCESSES)],
        use_mp=True,
    )
    run_model_test(model)
