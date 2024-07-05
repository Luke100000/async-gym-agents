from collections import defaultdict
from typing import List, Optional, Sequence, Any, Type, Callable

import numpy as np
from gymnasium import Env, Wrapper
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnvStepReturn,
    VecEnvObs,
    VecEnvIndices,
)


class IndexableMultiEnv(VecEnv):
    """
    Same as multi env but sync
    """

    def __init__(self, env_fns: List[Callable[[], Env]]):
        self.real_n_envs = len(env_fns)

        self.envs = [DummyVecEnv([e]) for e in env_fns]
        self.additional = defaultdict(dict)

        super().__init__(1, self.envs[0].observation_space, self.envs[0].action_space)

    def step(self, actions: np.ndarray, index: int = 0) -> VecEnvStepReturn:
        self.step_async(actions, index=index)
        return self.step_wait(index=index)

    def step_async(self, actions: np.ndarray, index: int = 0) -> None:
        self.envs[index].step_async(actions)

    def step_wait(self, index: int = 0) -> VecEnvStepReturn:
        # todo here fetch stuff like mask
        return self.envs[index].step_wait()

    def reset(self, index: int = 0) -> VecEnvObs:
        return self.envs[index].reset()

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def get_images(self) -> Sequence[Optional[np.ndarray]]:
        raise NotImplementedError

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        return self.envs[self._get_index(indices)].get_attr(attr_name)

    def set_attr(
        self, attr_name: str, value: Any, indices: VecEnvIndices = None
    ) -> None:
        self.envs[self._get_index(indices)].set_attr(attr_name, value)

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: VecEnvIndices = None,
        **method_kwargs,
    ) -> List[Any]:
        return self.envs[self._get_index(indices)].env_method(
            *method_args, *method_kwargs
        )

    def env_is_wrapped(
        self, wrapper_class: Type[Wrapper], indices: VecEnvIndices = None
    ) -> List[bool]:
        return self.envs[self._get_index(indices)].env_is_wrapped(wrapper_class)

    def _get_index(self, indices: VecEnvIndices) -> int:
        """
        Convert a flexibly-typed reference to environment indices to an implied list of indices.

        :param indices: refers to indices of envs.
        :return: the implied list of indices.
        """
        if indices is None:
            return 0
        elif isinstance(indices, int):
            return indices
        raise ValueError(
            f"IndexableMultiEnv only supports a scalar index, not {indices}."
        )
