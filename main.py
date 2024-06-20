import time
from enum import Enum

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from torchinfo import summary

from async_gym_agents.agents.async_agent import get_injected_agent
from async_gym_agents.envs.multi_env import IndexableMultiEnv
from async_gym_agents.envs.slow_cartpole import SlowCartPoleEnv
from async_gym_agents.envs.sync_multi_env import SyncIndexableMultiEnv
from async_gym_agents.envs.threaded_env import ThreadedVecEnv


class Mode(Enum):
    ASYNC = (0,)
    PARALLEL = (1,)
    SEQUENTIAL = (2,)


def get_env():
    return SlowCartPoleEnv()


def main(
    mode: Mode = Mode.ASYNC,
    threads: int = 8,
    sync: bool = True,
    agent: OffPolicyAlgorithm = DQN,
):
    env = (
        ThreadedVecEnv
        if mode == Mode.PARALLEL
        else (SyncIndexableMultiEnv if sync else IndexableMultiEnv)
    )([lambda: get_env() for _ in range(threads)])

    model = (get_injected_agent(agent) if mode == Mode.ASYNC else agent)(
        "MlpPolicy",
        env,
        verbose=1,
        # policy_kwargs={"net_arch": dict(pi=[64, 32], qf=[64, 32])},
    )

    summary(model.policy)

    eval_callback = EvalCallback(
        env, eval_freq=1000, deterministic=True, render=False, n_eval_episodes=10
    )

    model.learn(total_timesteps=1000, log_interval=10, callback=eval_callback)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
    print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")


def benchmark(mode: Mode):
    t = time.time()
    main(mode)
    print(f"{mode.name.capitalize()}: {time.time() - t:.1f}s")


if __name__ == "__main__":
    # benchmark(Mode.SEQUENTIAL)
    # benchmark(Mode.PARALLEL)
    benchmark(Mode.ASYNC)
