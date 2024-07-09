import threading
from queue import Queue
from typing import Any, Callable, Dict, List, Optional, Sequence, Type

import numpy as np
from gymnasium import Env, Wrapper
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)

# noinspection PyProtectedMember
from stable_baselines3.common.vec_env.patch_gym import _patch_env

# noinspection PyProtectedMember
from stable_baselines3.common.vec_env.subproc_vec_env import _flatten_obs


def _worker(
    task_queue: Queue,
    result_queue: Queue,
    env: Callable,
) -> None:
    from stable_baselines3.common.env_util import is_wrapped

    env: Env = _patch_env(env())

    reset_info: Optional[Dict[str, Any]] = {}
    while True:
        try:
            cmd, data = task_queue.get()
            if cmd == "step":
                observation, reward, terminated, truncated, info = env.step(data)
                done = terminated or truncated
                info["TimeLimit.truncated"] = truncated and not terminated
                if done:
                    # save final observation where user can get it, then reset
                    info["terminal_observation"] = observation
                    observation, reset_info = env.reset()
                result_queue.put((observation, reward, done, info, reset_info))
            elif cmd == "reset":
                maybe_options = {"options": data[1]} if data[1] else {}
                observation, reset_info = env.reset(seed=data[0], **maybe_options)
                result_queue.put((observation, reset_info))
            elif cmd == "render":
                result_queue.put(env.render())
            elif cmd == "close":
                env.close()
                result_queue.join()
                break
            elif cmd == "get_spaces":
                result_queue.put((env.observation_space, env.action_space))
            elif cmd == "env_method":
                method = getattr(env, data[0])
                result_queue.put(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                result_queue.put(getattr(env, data))
            elif cmd == "set_attr":
                result_queue.put(setattr(env, data[0], data[1]))
            elif cmd == "is_wrapped":
                result_queue.put(is_wrapped(env, data))
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break


class ThreadedVecEnv(VecEnv):
    def __init__(self, envs: List[Callable]):
        self.waiting = False
        self.closed = False
        n_envs = len(envs)

        self.task_queues = [Queue() for _ in range(n_envs)]
        self.result_queues = [Queue() for _ in range(n_envs)]

        self.threads = []
        for task_queue, result_queue, env in zip(
            self.task_queues, self.result_queues, envs
        ):
            args = (task_queue, result_queue, env)
            thread = threading.Thread(target=_worker, args=args, daemon=True)
            thread.start()
            self.threads.append(thread)

        self.task_queues[0].put(("get_spaces", None))
        observation_space, action_space = self.result_queues[0].get()

        super().__init__(len(envs), observation_space, action_space)

    def step_async(self, actions: np.ndarray) -> None:
        for queue, action in zip(self.task_queues, actions):
            queue.put(("step", action))
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        results = [queue.get() for queue in self.result_queues]
        self.waiting = False
        obs, rewards, dones, infos, self.reset_infos = zip(*results)  # type: ignore[assignment]
        return (
            _flatten_obs(obs, self.observation_space),
            np.stack(rewards),
            np.stack(dones),
            infos,
        )  # type: ignore[return-value]

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> VecEnvObs:
        for env_idx, queue in enumerate(self.task_queues):
            queue.put(("reset", (self._seeds[env_idx], self._options[env_idx])))
        results = [queue.get() for queue in self.result_queues]
        obs, self.reset_infos = zip(*results)  # type: ignore[assignment]
        # Seeds and options are only used once
        self._reset_seeds()
        self._reset_options()
        return _flatten_obs(obs, self.observation_space)  # , self.reset_infos

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

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        for queue in self._get_target_queues(self.task_queues, indices):
            queue.put(("get_attr", attr_name))
        return [
            queue.get()
            for queue in self._get_target_queues(self.result_queues, indices)
        ]

    def set_attr(
        self, attr_name: str, value: Any, indices: VecEnvIndices = None
    ) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        for queue in self._get_target_queues(self.task_queues, indices):
            queue.put(("set_attr", (attr_name, value)))
        for queue in self._get_target_queues(self.result_queues, indices):
            queue.get()

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: VecEnvIndices = None,
        **method_kwargs,
    ) -> List[Any]:
        """Call instance methods of vectorized environments."""
        for queue in self._get_target_queues(self.task_queues, indices):
            queue.put(("env_method", (method_name, method_args, method_kwargs)))
        return [
            queue.get()
            for queue in self._get_target_queues(self.result_queues, indices)
        ]

    def env_is_wrapped(
        self, wrapper_class: Type[Wrapper], indices: VecEnvIndices = None
    ) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        for queue in self._get_target_queues(self.task_queues, indices):
            queue.put(("is_wrapped", wrapper_class))
        return [
            queue.get()
            for queue in self._get_target_queues(self.result_queues, indices)
        ]

    def _get_target_queues(
        self, queues: list[Queue], indices: VecEnvIndices
    ) -> List[Any]:
        """
        Get the connection object needed to communicate with the wanted
        envs that are in subprocesses.

        :param indices: Refers to indices of envs.
        :return: Connection object to communicate between processes.
        """
        indices = self._get_indices(indices)
        return [queues[i] for i in indices]
