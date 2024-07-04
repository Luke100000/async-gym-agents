import time
from enum import Enum

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


def get_env(slow: bool):
    return Monitor(SlowCartPoleEnv(min_sleep=0, max_sleep=0.0 if slow else 0))


def evaluate(
    mode: Mode = Mode.ASYNC,
    threads: int = 8,
    agent: BaseAlgorithm = PPO,
):
    env = (ThreadedVecEnv if mode == Mode.PARALLEL else IndexableMultiEnv)(
        [lambda: get_env(True) for _ in range(threads)]
    )

    agent = get_injected_agent(agent) if mode == Mode.ASYNC else agent

    model = agent("MlpPolicy", env, learning_rate=3e-4)

    model.learn(total_timesteps=10_000)

    if mode == Mode.ASYNC:
        model.shutdown()
        print(f"Buffer utilization: {model.buffer_utilization}")
        print(f"Buffer emptiness: {model.buffer_emptyness}")

    eval_env = get_env(False)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=1000)
    print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")


def benchmark(mode: Mode):
    t = time.time()
    evaluate(mode)
    print(f"{mode.name.capitalize()}: {time.time() - t:.1f}s")


if __name__ == "__main__":
    benchmark(Mode.ASYNC)
    benchmark(Mode.PARALLEL)
    benchmark(Mode.SEQUENTIAL)
