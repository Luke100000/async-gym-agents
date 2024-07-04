from collections import defaultdict
from typing import List, Optional, Sequence, Any, Type, Callable

import numpy as np
from gymnasium import Env, Wrapper
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn, VecEnvObs


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

    def get_attr(self, attr_name: str, index: int = 0) -> List[Any]:
        return self.envs[index].get_attr(attr_name)

    def set_attr(self, attr_name: str, value: Any, index: int = 0) -> None:
        self.envs[index].set_attr(attr_name, value)

    def env_method(
        self, method_name: str, *method_args, index: int = 0, **method_kwargs
    ) -> List[Any]:
        # todo conflicts with supers indices
        return self.envs[index].env_method(*method_args, *method_kwargs)

    def env_is_wrapped(
        self, wrapper_class: Type[Wrapper], index: int = 0
    ) -> List[bool]:
        return self.envs[index].env_is_wrapped(wrapper_class)
