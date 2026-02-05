from typing import Callable, List, TypeVar, Union

import gymnasium as gym
from stable_baselines3.common.vec_env import VecEnv

Transition = TypeVar("Transition")

Env = Union[VecEnv, List[gym.Env], gym.Env]
EnvFactory = Callable[[], Union[Env]]
EnvFactoryList = List[EnvFactory]
