import time
from enum import Enum
from functools import partial
from typing import List, Type

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from async_gym_agents.agents.async_agent import get_injected_agent
from async_gym_agents.envs.multi_env import IndexableMultiEnv
from async_gym_agents.envs.slow_cartpole import SlowCartPoleEnv
from async_gym_agents.envs.threaded_env import ThreadedVecEnv


class Mode(Enum):
    ASYNC = (0,)
    PARALLEL = (1,)
    SEQUENTIAL = (2,)


def get_env(slow: bool) -> gym.Env:
    return Monitor(SlowCartPoleEnv(min_sleep=0, max_sleep=0.1 if slow else 0))


def get_envs(threads) -> List[gym.Env]:
    return [get_env(True) for _ in range(threads)]


def evaluate(
    mode: Mode = Mode.ASYNC,
    use_mp: bool = False,
    threads: int = 8,
    agent: Type[BaseAlgorithm] = PPO,
):
    env = (ThreadedVecEnv if mode == Mode.PARALLEL else IndexableMultiEnv)(
        [lambda: get_env(True) for _ in range(threads * threads)]
    )

    injected_agent = (
        get_injected_agent(agent, use_mp=use_mp) if mode == Mode.ASYNC else agent
    )

    model = injected_agent(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        **(
            {"envs": [partial(get_envs, threads=threads) for _ in range(threads)]}
            if mode == Mode.ASYNC and use_mp
            else {}
        ),
    )

    model.learn(total_timesteps=1000)

    if mode == Mode.ASYNC:
        model.shutdown()
        print(f"Buffer utilization: {model.buffer_utilization}")
        print(f"Buffer emptiness: {model.buffer_emptyness}")
        print(f"Discarded episodes: {model.discarded_episodes_fraction}")

    eval_env = get_env(False)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)
    print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")


def benchmark(mode: Mode, use_mp: bool = False):
    t = time.time()
    evaluate(mode, use_mp)
    print(f"{mode.name.capitalize()}, MP: {use_mp}: {time.time() - t:.1f}s\n")


if __name__ == "__main__":
    benchmark(Mode.ASYNC, True)
    benchmark(Mode.ASYNC, False)
    benchmark(Mode.PARALLEL)
    benchmark(Mode.SEQUENTIAL)
