from functools import partial
from typing import Any, Tuple, TypeVar

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs

from async_gym_agents.types import Env

T = TypeVar("T")


def single_slice(x: T, idx: int) -> T | Tuple[Any, ...]:
    return x[idx : idx + 1]


def identity(x):
    return x


def make_venv(e: Env) -> VecEnv:
    if isinstance(e, VecEnv):
        return e
    if isinstance(e, gym.Env):
        return DummyVecEnv([partial(identity, e)])
    if isinstance(e, list) and isinstance(e[0], gym.Env):
        return DummyVecEnv([partial(identity, env) for env in e])
    raise ValueError(f"Cannot make VecEnv from {e}")


def copy_obs(obs: VecEnvObs) -> VecEnvObs:
    if isinstance(obs, np.ndarray):
        return obs.copy()
    if isinstance(obs, dict):
        return {k: v.copy() for k, v in obs.items()}
    if isinstance(obs, tuple):
        return tuple(x.copy() for x in obs)
    raise TypeError(type(obs))
