import time
from enum import Enum

from stable_baselines3 import DQN
from torchinfo import summary

from async_gym_agents.agents.async_dqn import AsyncDQN
from async_gym_agents.envs.multi_env import IndexableMultiEnv
from async_gym_agents.envs.slow_cartpole import SlowCartPoleEnv
from async_gym_agents.envs.sync_multi_env import SyncIndexableMultiEnv
from async_gym_agents.envs.threaded_env import ThreadedVecEnv


class Mode(Enum):
    ASYNC = (0,)
    PARALLEL = (1,)
    SEQUENTIAL = (2,)


def main(mode: Mode = Mode.ASYNC, threads: int = 8, sync: bool = True):
    env = (
        ThreadedVecEnv
        if mode == Mode.PARALLEL
        else (SyncIndexableMultiEnv if sync else IndexableMultiEnv)
    )([lambda: SlowCartPoleEnv() for _ in range(threads)])

    model = (AsyncDQN if mode == Mode.ASYNC else DQN)(
        "MlpPolicy",
        env,
        verbose=1,
        exploration_fraction=0.5,
        learning_rate=0.001,
    )
    summary(model.policy)
    model.learn(total_timesteps=1000, log_interval=10)
    # model.save("dqn_cartpole")


def benchmark(mode: Mode):
    t = time.time()
    main(mode)
    print(f"{mode.name.capitalize()}: {time.time() - t:.1f}s")


if __name__ == "__main__":
    # benchmark(Mode.SEQUENTIAL) # Parallel: 14.5s
    # benchmark(Mode.PARALLEL) # Parallel: 12.5s
    benchmark(Mode.ASYNC)  # Async: 7.7s
