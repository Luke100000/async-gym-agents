import io
import logging
import multiprocessing
import queue
import threading
from multiprocessing.managers import Namespace
from queue import Queue
from typing import Dict, List, TypeVar, Callable, Type, Any, Optional

import gymnasium as gym
import torch as th
from stable_baselines3.common.base_class import BasePolicy

from async_gym_agents.envs.multi_env import IndexableMultiEnv

logger = logging.getLogger("async_gym_agents")


Transition = TypeVar("Transition")


class IAsyncAgentInjector:
    initialized: bool

    def init_collect_process(self):
        raise NotImplementedError

    def fetch_transition(self) -> Transition:
        raise NotImplementedError

    def fetch_transitions(self) -> List[Transition]:
        raise NotImplementedError

    def shutdown(self):
        raise NotImplementedError

    def _excluded_save_params(self):
        raise NotImplementedError


class AsyncAgentInjectorBase(IAsyncAgentInjector):
    def __init__(self, *args, **kwargs):
        self.initialized = False

    def pre_collect_preparation(self, policy: BasePolicy):
        raise NotImplementedError

    def init_collect_process(self):
        raise NotImplementedError

    def fetch_transition(self) -> Transition:
        raise NotImplementedError

    def fetch_transitions(self) -> List[Transition]:
        raise NotImplementedError

    def shutdown(self):
        raise NotImplementedError

    def _excluded_save_params(self) -> List[str]:
        return [
            "initialized",
        ]


class AsyncAgentInjector(AsyncAgentInjectorBase):
    def __init__(
        self,
        max_episodes_in_buffer: int,
        skip_truncated: bool = False,
        timeout: float = 1.0,
    ):
        AsyncAgentInjectorBase.__init__(self)

        self._buffer_utilization = 0.0
        self._buffer_emptiness = 0.0
        self._buffer_stat_count = 0

        self.running = True
        self.initialized = False
        self.threads = []
        self.thread_lookup: Dict[str, int] = {}

        self.total_episodes = 0
        self.discarded_episodes = 0
        self.skip_truncated = skip_truncated
        self.timeout = timeout

        # The larger the queue, the less wait times, but the more outdated the policies training data are
        self.queue = Queue(max_episodes_in_buffer)
        self.transition_queue = Queue()

        # The policy itself is rarely thread-safe
        self.training_policy_lock = threading.Lock()
        self.training_policy: BasePolicy = (
            getattr(self, "policy") if hasattr(self, "policy") else None
        )
        self.training_policy_version: int = 0

        self.rollout_policies: Dict[int, BasePolicy] = {}
        self.rollout_policy_versions: Dict[int, int] = {}

    @property
    def policy(self):
        thread_name = threading.current_thread().name
        index = self.thread_lookup.get(thread_name, None)
        if index is not None:
            return self.rollout_policies[index]
        return self.training_policy

    @policy.setter
    def policy(self, value):
        self.training_policy = value

    def sync_training_policy_to_rollout_policy_complete(self, index: int):
        if (
            index not in self.rollout_policy_versions
            or self.rollout_policy_versions[index] < self.training_policy_version
        ):
            with self.training_policy_lock:
                buffer = io.BytesIO()
                th.save(self.training_policy, buffer)
                buffer.seek(0)
                self.rollout_policies[index] = th.load(buffer, weights_only=False)
                self.rollout_policy_versions[index] = self.training_policy_version

    def sync_training_policy_to_rollout_policy_weights_only(self, index: int):
        if (
            index not in self.rollout_policy_versions
            or self.rollout_policy_versions[index] < self.training_policy_version
        ):
            with self.training_policy_lock:
                self.rollout_policies[index].load_state_dict(
                    self.training_policy.state_dict()
                )
                self.rollout_policy_versions[index] = self.training_policy_version

    def _excluded_save_params(self) -> List[str]:
        return [
            "threads",
            "queue",
            "transition_queue",
            "training_policy_lock",
            "training_policy",
            "rollout_policies",
            "running",
            "initialized",
        ]

    # noinspection PyUnresolvedReferences
    def get_indexable_env(self) -> IndexableMultiEnv:
        """
        Asserts whether a correct environment is supplied
        """
        assert isinstance(
            self.env, IndexableMultiEnv
        ), "You must pass a IndexableMultiEnv"
        return self.env

    def pre_collect_preparation(self, policy: BasePolicy):
        pass

    def init_collect_process(self):
        self.running = True

        self.threads = []
        for index in range(self.get_indexable_env().real_n_envs):
            thread = threading.Thread(
                name=f"CollectorThread{index}",
                target=self._collector_loop,
                args=(index,),
            )
            self.sync_training_policy_to_rollout_policy_complete(index)
            self.thread_lookup[thread.name] = index
            self.threads.append(thread)
            self.threads[index].start()

        self.initialized = True

    def fetch_transition(self):
        while self.transition_queue.empty():
            self._buffer_utilization += self.queue.qsize()
            self._buffer_emptiness += 1 if self.queue.empty() else 0
            self._buffer_stat_count += 1
            for t in self.queue.get():
                self.transition_queue.put(t)
        return self.transition_queue.get()

    def fetch_transitions(self) -> List[Transition]:
        raise NotImplementedError

    @property
    def buffer_utilization(self) -> float:
        return (
            0
            if self._buffer_stat_count == 0
            else self._buffer_utilization / self._buffer_stat_count
        )

    @property
    def buffer_emptyness(self) -> float:
        return (
            0
            if self._buffer_stat_count == 0
            else self._buffer_emptiness / self._buffer_stat_count
        )

    @property
    def discarded_episodes_fraction(self) -> float:
        return (
            0
            if self.total_episodes == 0
            else self.discarded_episodes / self.total_episodes
        )

    def _episode_generator(self, index: int):
        raise NotImplementedError()

    def _collector_loop(self, index: int):
        """
        Batch-inserts transitions whenever an episode is done.
        """
        for episode in self._episode_generator(index):
            # Keeps track of truncated episodes and optionally removes them
            self.total_episodes += 1
            if episode[-1].infos[0]["TimeLimit.truncated"] and self.skip_truncated:
                self.discarded_episodes += 1
                logger.info("Dropped episode due to truncation")
                continue

            # Feeds the episodes into the queue
            try:
                self.queue.put(episode, block=True, timeout=self.timeout)
            except queue.Full:
                try:
                    self.queue.get(block=False)
                    self.queue.put(episode, block=False)
                except queue.Full:
                    pass
                self.discarded_episodes += 1
                logger.info("Dropped episode due to buffer full")

    def shutdown(self):
        """
        Shuts down the workers.
        Shutting down is required to fully release environments.
        Subsequent calls to e.g., train will restart the workers.
        """
        self.running = False
        for thread in self.threads:
            thread.join()
        self.initialized = False


class InjectorWorkerBase:
    def __init__(
        self,
        env_func: Callable[[], List[gym.Env]],
        trajectory: multiprocessing.Queue,
        state: Namespace,
        **kwargs,
    ):
        self._env_func = env_func
        self._trajectory = trajectory
        self._state = state

    def run(self):
        raise NotImplementedError()


class AsyncAgentInjectorMP(AsyncAgentInjectorBase):
    def __init__(
        self,
        envs: List[Callable[[], List[gym.Env]]],
        worker_class: InjectorWorkerBase,
        max_steps_in_buffer: int = 10000
    ):
        AsyncAgentInjectorBase.__init__(self)

        self._worker_class = worker_class

        self._envs = envs
        # shared memory
        self._trajectory = multiprocessing.Queue(maxsize=max_steps_in_buffer)
        # shared object (!)
        self._manager = multiprocessing.Manager()
        self._state = self._manager.Namespace()
        self._version = 0

        self._workers_inited = False
        self._proc: List[multiprocessing.Process] = []

        self._transitions: Optional[List[Transition]] = None

    def _episode_generator(self, index: int):
        raise NotImplementedError()

    @staticmethod
    def _run_worker(
        worker_class: Type[InjectorWorkerBase],
        env_func,
        trajectory: multiprocessing.Queue,
        state: Namespace,
        worker_kwargs: Dict[str, Any],
    ):
        worker = worker_class(
            env_func=env_func,
            trajectory=trajectory,
            state=state,
            **worker_kwargs
        )
        worker.run()

    def get_worker_kwargs(self) -> Dict[str, Any]:
        raise NotImplementedError()

    def init_collect_process(self):
        if self._workers_inited:
            return

        for env_func in self._envs:
            proc = multiprocessing.Process(
                target=AsyncAgentInjectorMP._run_worker,
                kwargs=dict(
                    worker_class=self._worker_class,
                    env_func=env_func,
                    trajectory=self._trajectory,
                    state=self._state,
                    worker_kwargs=self.get_worker_kwargs(),
                )
            )
            proc.start()

            self._proc.append(proc)

        self._workers_inited = True

    def _excluded_save_params(self) -> List[str]:
        return [
            *super()._excluded_save_params(),
            *super(AsyncAgentInjectorBase, self)._excluded_save_params(),
            "_trajectory",
            "_manager",
            "_state",
            "_proc",
            "_envs",
        ]

    def fetch_transitions(self) -> List[Transition]:
        return self._trajectory.get()

    def fetch_transition(self) -> Transition:
        if self._transitions is None or len(self._transitions) == 0:
            self._transitions = self.fetch_transitions()

        return self._transitions.pop(0)

    def _sync_policy(self, policy):
        weights_buf = io.BytesIO()
        th.save(policy.state_dict(), weights_buf)
        weights_bytes = weights_buf.getvalue()
        policy_buf = io.BytesIO()
        th.save(policy, policy_buf)
        policy_bytes = policy_buf.getvalue()

        self._state.version = self._version
        self._state.weights = weights_bytes
        self._state.policy = policy_bytes
        self._version += 1

    def pre_collect_preparation(self, policy: BasePolicy):
        self._sync_policy(policy)

    def shutdown(self):
        for proc in self._proc:
            proc.kill()
