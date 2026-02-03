from typing import Callable, List, TypeVar, Union

import gymnasium as gym
from stable_baselines3.common.vec_env import VecEnv

Transition = TypeVar("Transition")

Env = Union[gym.Env, VecEnv]
EnvFactory = Callable[[], Union[Env, List[Env]]]
EnvFactoryList = List[EnvFactory]
