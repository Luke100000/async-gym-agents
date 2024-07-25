import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from async_gym_agents.agents.async_agent import get_injected_agent
from async_gym_agents.envs.buggy_lunar_lander import BuggyLunarLander
from async_gym_agents.envs.multi_env import IndexableMultiEnv
from async_gym_agents.envs.truncation_counter_wrapper import TruncationCounterWrapper


def get_env(buggy: bool):
    return Monitor(
        TruncationCounterWrapper(
            BuggyLunarLander(
                crash_probability=0.001 if buggy else 0,
                time_limit=1000,
            )
        )
    )


def evaluate(
    threads: int = 8,
    agent: BaseAlgorithm = PPO,
):
    env = IndexableMultiEnv([lambda: get_env(True) for _ in range(threads)])

    agent = get_injected_agent(agent)

    model = agent("MlpPolicy", env)

    eval_env = get_env(False)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=None,
        log_path=None,
        n_eval_episodes=100,
        eval_freq=100000,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=1_000_000, progress_bar=True, callback=eval_callback)

    model.shutdown()

    factors = [e.envs[0].get_truncation_factor() for e in env.envs]
    print(f"Truncated episodes: {np.mean(factors) * 100}%")

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=1000)
    print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")


if __name__ == "__main__":
    evaluate()
