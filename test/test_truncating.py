from test.conftest import get_buggy_env
from typing import Type

from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from async_gym_agents.agents.async_agent import get_injected_agent
from async_gym_agents.envs.multi_env import IndexableMultiEnv


def test_truncating(threads: int = 8, agent: Type[BaseAlgorithm] = PPO):
    """Test that the agent is able to truncate episodes."""
    env = IndexableMultiEnv([lambda: get_buggy_env(True) for _ in range(threads)])

    model = get_injected_agent(agent)("MlpPolicy", env, learning_rate=3e-4)

    eval_env = get_buggy_env(False)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=None,
        log_path=None,
        n_eval_episodes=100,
        eval_freq=100000,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=1_000, progress_bar=True, callback=eval_callback)

    model.shutdown()

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=1000)
    print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
