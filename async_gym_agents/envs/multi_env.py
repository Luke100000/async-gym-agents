from typing import Any, List, Type, Union

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)

from async_gym_agents.types import EnvFactoryList
from async_gym_agents.utils import make_venv


class RandomEnv(gym.Env):
    def __init__(self, obs_space, action_space):
        self.observation_space = obs_space
        self.action_space = action_space

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self.observation_space.sample(), {}

    def step(self, action):
        obs = self.observation_space.sample()
        reward = 0.0
        terminated = True
        truncated = True
        info = {}
        return obs, reward, terminated, truncated, info


class IndexableMultiEnv(VecEnv):
    """
    A container for multiple VecEnvs / gym.Envs.
    Should not be used outside async-agents since it will return random sampled episodes.
    """

    def __init__(self, env_fns: Union[EnvFactoryList], stub_env: gym.Env | None = None):
        # Make sure they are all VecEnvs
        self.env_fns = env_fns

        stub_env = make_venv(stub_env or env_fns[0]())

        # This is when using the IndexableMultiEnv directly and only serves to not raise exceptions
        self.env = make_venv(
            [
                RandomEnv(stub_env.observation_space, stub_env.action_space)
                for _ in range(stub_env.num_envs)
            ]
        )

        super().__init__(
            stub_env.num_envs, stub_env.observation_space, stub_env.action_space
        )

    def __len__(self):
        return len(self.env_fns)

    def step_async(self, actions: np.ndarray, index: int = 0) -> None:
        self.env.step_async(actions)

    def step_wait(self, index: int = 0) -> VecEnvStepReturn:
        return self.env.step_wait()

    def reset(self, index: int = 0, **kwargs) -> VecEnvObs:
        return self.env.reset()

    def close(self) -> None:
        self.env.close()

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        return self.env.get_attr(attr_name, indices)

    def set_attr(
        self, attr_name: str, value: Any, indices: VecEnvIndices = None
    ) -> None:
        self.env.set_attr(attr_name, value, indices)

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: VecEnvIndices = None,
        **method_kwargs,
    ) -> List[Any]:
        return self.env.env_method(
            method_name, *method_args, indices=indices, **method_kwargs
        )

    def env_is_wrapped(
        self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None
    ) -> List[bool]:
        return self.env.env_is_wrapped(wrapper_class, indices)

    @staticmethod
    def _get_index(indices: VecEnvIndices) -> int:
        """
        Convert a flexibly typed reference to environment indices to an implied list of indices.

        :param indices: Refers to indices of envs.
        :return: The implied list of indices.
        """
        if indices is None:
            return 0
        elif isinstance(indices, int):
            return indices
        raise ValueError(
            f"IndexableMultiEnv only supports a scalar index, not {indices}."
        )
