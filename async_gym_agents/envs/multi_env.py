import threading
from queue import Queue
from typing import List, Optional, Sequence, Any, Type, Callable

import numpy as np
from gymnasium import Env, Wrapper
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn, VecEnvObs

# noinspection PyProtectedMember
from stable_baselines3.common.vec_env.subproc_vec_env import _flatten_obs

from async_gym_agents.envs.threaded_env import _worker


class IndexableMultiEnv(VecEnv):
    """
    A threaded environment maintains a list of independent environments.
    It only allows access per index, and thus behaves as if it has a n_env of 1.
    When used in classic agents, only index 0 is used.
    We use threads since some operations (e.g., sockets) are thread-bound.
    All access happen via a queue to avoid any race conditions.

    # TODO should threads be used here at all?
    Since its indented
    to be used in a threaded agent anyways the asynchronicity is not used anyways

    # TODO allow for batches when used for classic agents (similar to process env work)
    """

    def __init__(self, env_fns: List[Callable[[], Env]]):
        self.waiting = False  # TODO should be a list as well
        self.closed = False

        # From the outside this env behaves like it has only one env
        # Only the agent could then make use of the other ones
        self.real_n_envs = len(env_fns)

        self.task_queues = [Queue() for _ in range(self.real_n_envs)]
        self.result_queues = [Queue() for _ in range(self.real_n_envs)]

        self.threads = []
        for task_queue, result_queue, env in zip(
            self.task_queues, self.result_queues, env_fns
        ):
            args = (task_queue, result_queue, env)
            thread = threading.Thread(target=_worker, args=args, daemon=True)
            thread.start()
            self.threads.append(thread)

        self.task_queues[0].put(("get_spaces", None))
        observation_space, action_space = self.result_queues[0].get()

        super().__init__(1, observation_space, action_space)

    def step(self, actions: np.ndarray, index: int = 0) -> VecEnvStepReturn:
        """
        Step the environments with the given action

        :param actions: the action
        :param index: the env index
        :return: observation, reward, done, information
        """
        self.step_async(actions, index=index)
        return self.step_wait(index=index)

    def step_async(self, actions: np.ndarray, index: int = 0) -> None:
        self.task_queues[index].put(
            ("step", actions[0])
        )  # TODO it appears I'm using gym environments here,
        # for the sake of using the full power of sb3, work with sb3 gyms.
        # Yes, that would effectively create a batched * threaded env.
        self.waiting = True

    def step_wait(self, index: int = 0) -> VecEnvStepReturn:
        result = self.result_queues[index].get()
        obs, rewards, dones, infos, self.reset_infos = result
        self.waiting = False
        return (
            _flatten_obs([obs], self.observation_space),
            np.stack([rewards]),
            np.stack([dones]),
            [infos],
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        index: int = 0,
    ) -> VecEnvObs:
        self.task_queues[index].put(
            # ("reset", (self._seeds[index], self._options[index]))
            (
                "reset",
                (self._seeds[0], self._options[0]),
            )  # TODO not properly seeded
        )
        obs, reset_infos = self.result_queues[index].get()
        # Seeds and options are only used once
        self._reset_seeds()
        self._reset_options()
        return _flatten_obs([obs], self.observation_space)  # , self.reset_infos

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for queue in self.result_queues:
                queue.get()
        for queue in self.task_queues:
            queue.put(("close", None))
        for thread in self.threads:
            thread.join()
        self.closed = True

    def get_images(self) -> Sequence[Optional[np.ndarray]]:
        raise NotImplementedError

    def get_attr(self, attr_name: str, index: int = 0) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        self.task_queues[index].put(("get_attr", attr_name))
        return [self.result_queues[index].get()]

    def set_attr(self, attr_name: str, value: Any, index: int = 0) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        self.task_queues[index].put(("set_attr", (attr_name, value)))
        self.result_queues[index].get()

    def env_method(
        self,
        method_name: str,
        *method_args,
        index: int = 0,
        **method_kwargs,
    ) -> List[Any]:
        """Call instance methods of vectorized environments."""
        self.task_queues[index].put(
            ("env_method", (method_name, method_args, method_kwargs))
        )
        return [self.result_queues[index].get()]

    def env_is_wrapped(
        self, wrapper_class: Type[Wrapper], index: int = 0
    ) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        self.task_queues[index].put(("is_wrapped", wrapper_class))
        return [self.result_queues[index].get()]
