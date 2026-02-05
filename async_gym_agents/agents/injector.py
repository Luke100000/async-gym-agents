import io
import logging
import multiprocessing
import queue
import threading
from multiprocessing.managers import Namespace
from queue import Queue
from typing import Any, Dict, List, Optional, Type

import torch as th
from stable_baselines3.common.base_class import BasePolicy
from stable_baselines3.common.vec_env import VecEnv

from async_gym_agents.envs.multi_env import IndexableMultiEnv
from async_gym_agents.types import EnvFactory, EnvFactoryList, Transition

logger = logging.getLogger("async_gym_agents")


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

    def _excluded_save_params(self) -> List[str]:
        # Implicitly inherited from BaseAlgorithm
        # noinspection PyProtectedMember,PyUnresolvedReferences
        return super()._excluded_save_params()


class AsyncAgentInjectorBase(IAsyncAgentInjector):
    def __init__(self, *args, envs: Optional[EnvFactoryList] = None, **kwargs):
        self.initialized = False
        self._envs = envs

    def pre_collect_preparation(self, policy: BasePolicy):
        raise NotImplementedError

    def init_collect_process(self):
        raise NotImplementedError

    def fetch_transition(self) -> Transition:
        """Fetch one single batched transition which is guaranteed to be in order and not interleaved with other episodes."""
        raise NotImplementedError

    def fetch_transitions(self) -> List[Transition]:
        raise NotImplementedError

    def shutdown(self):
        raise NotImplementedError

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + [
            "initialized",
        ]

    @property
    def buffer_utilization(self) -> float:
        return -1

    @property
    def buffer_emptyness(self) -> float:
        return -1

    @property
    def discarded_episodes_fraction(self) -> float:
        return -1


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
        return super()._excluded_save_params() + [
            "threads",
            "queue",
            "transition_queue",
            "training_policy_lock",
            "training_policy",
            "rollout_policies",
            "running",
            "initialized",
            "envs",
            "_envs",
        ]

    # noinspection PyUnresolvedReferences
    def get_indexable_env(self) -> IndexableMultiEnv:
        """
        Asserts whether a correct environment is supplied
        """
        assert isinstance(
            self.env, IndexableMultiEnv
        ), "Neither a IndexableMultiEnv nor an env constructor was supplied."
        return self.env

    def pre_collect_preparation(self, policy: BasePolicy):
        pass

    def init_collect_process(self):
        self.running = True

        self.threads = []

        if self._envs is not None:
            # If an environment constructor is supplied, use that one
            env = IndexableMultiEnv(self._envs[0]())
        else:
            # Otherwise an IndexableMultiEnv is required, which can treat environments separately
            env = self.get_indexable_env()

        for index in range(len(env)):
            self._start_thread(index, env.get_env(index))

        self.initialized = True

    def _start_thread(self, index: int, env: VecEnv):
        thread = threading.Thread(
            name=f"collector-thread-{index}",
            target=self._collector_loop,
            args=(index, env),
            daemon=True,
        )
        self.sync_training_policy_to_rollout_policy_complete(index)
        self.thread_lookup[thread.name] = index
        self.threads.append(thread)
        self.threads[index].start()

    def fetch_transition(self):
        while self.transition_queue.empty():
            self._buffer_utilization += self.queue.qsize()
            self._buffer_emptiness += 1 if self.queue.empty() else 0
            self._buffer_stat_count += 1
            for t in self.queue.get():
                self.transition_queue.put(t)
        return self.transition_queue.get()

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

    def _episode_generator(self, index: int, env: VecEnv):
        raise NotImplementedError()

    def _collector_loop(self, index: int, env: VecEnv):
        """
        Batch-inserts transitions whenever an episode is done.
        """
        for episode in self._episode_generator(index, env):
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
        env_func: EnvFactory,
        trajectory: multiprocessing.Queue,
        state: Namespace,
        stop: multiprocessing.Event,
        **kwargs,
    ):
        self._env_func = env_func
        self._trajectory = trajectory
        self._state = state
        self._stop = stop

        self.running = True

    def episode_generator(self, index: int, env: VecEnv):
        raise NotImplementedError()

    def run(self):
        threads = []

        env = IndexableMultiEnv(self._env_func())

        for index in range(len(env)):
            thread_name = f"collector-thread-{index}"
            thread = threading.Thread(
                name=thread_name,
                target=self.episode_generator,
                args=(index, env.get_env(index)),
                daemon=True,
            )
            thread.start()
            threads.append(thread)

        # wait stop outside the worker
        self._stop.wait()

        # stop threads
        self.running = False
        for thread in threads:
            thread.join()

        # stop env
        env.close()


class AsyncAgentInjectorMP(AsyncAgentInjectorBase):
    def __init__(
        self,
        envs: Optional[EnvFactoryList],
        worker_class: InjectorWorkerBase,
        max_episodes_in_buffer: int = 8,
    ):
        AsyncAgentInjectorBase.__init__(self, envs=envs)

        self._worker_class = worker_class

        # shared memory
        self.max_episodes_in_buffer = max_episodes_in_buffer
        self._trajectory: multiprocessing.Queue | None = None

        # shared object (!)
        self._manager: multiprocessing.Manager = None
        self._state: Namespace | None = None
        self._version = 0

        self._stop = multiprocessing.Event()

        self._workers_inited = False
        self._proc: List[multiprocessing.Process] = []

        self._transitions: Optional[List[Transition]] = None

        # 1 minute wait for a new message in the queue
        self._queue_get_timeout = 60.0
        # 2-minute wait before try to kill the process
        self._proc_join_timeout = 120.0

    def _episode_generator(self, index: int):
        raise NotImplementedError()

    @staticmethod
    def _run_worker(
        worker_class: Type[InjectorWorkerBase],
        env_func: EnvFactory,
        trajectory: multiprocessing.Queue,
        state: Namespace,
        stop: multiprocessing.Event,
        worker_kwargs: Dict[str, Any],
    ):
        worker = worker_class(
            env_func=env_func,
            trajectory=trajectory,
            state=state,
            stop=stop,
            **worker_kwargs,
        )
        worker.run()

        # Close the queue
        trajectory.close()
        trajectory.cancel_join_thread()

    def get_worker_kwargs(self) -> Dict[str, Any]:
        raise NotImplementedError()

    def init_collect_process(self):
        if self._workers_inited:
            return

        # shared memory
        if self._trajectory is None:
            self._trajectory = multiprocessing.Queue(
                maxsize=self.max_episodes_in_buffer
            )

        # shared object (!)
        if self._manager is None:
            self._manager = multiprocessing.Manager()
            self._state = self._manager.Namespace()

        if self._stop is None:
            self._stop = multiprocessing.Event()

        if self._envs is None:
            raise ValueError(
                "Multi-processed injectors must have the envs constructor set."
            )

        for env_func in self._envs:
            proc = multiprocessing.Process(
                target=AsyncAgentInjectorMP._run_worker,
                kwargs=dict(
                    worker_class=self._worker_class,
                    env_func=env_func,
                    trajectory=self._trajectory,
                    state=self._state,
                    stop=self._stop,
                    worker_kwargs=self.get_worker_kwargs(),
                ),
            )
            proc.start()

            self._proc.append(proc)

        self._workers_inited = True

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + [
            "_trajectory",
            "_manager",
            "_state",
            "_stop",
            "_workers_inited",
            "_proc",
            "_envs",
        ]

    def fetch_transitions(self) -> List[Transition]:
        try:
            return self._trajectory.get(timeout=self._queue_get_timeout)
        except queue.Empty:
            return []

    def fetch_transition(self) -> Transition:
        # fetch_transitions returns [] in case of timeout is reached
        while self._transitions is None or len(self._transitions) == 0:
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
        logger.info("send stop event to all processes")
        self._stop.set()
        self._stop = None

        for proc in self._proc:
            if not proc.is_alive():
                continue

            proc.join(timeout=self._proc_join_timeout)

            try:
                proc.kill()
            except PermissionError:
                logger.warning("cannot kill process due to permission error")

        # close the queue
        if self._trajectory is not None:
            self._trajectory.close()
            self._trajectory.cancel_join_thread()
            self._trajectory = None

        # release a shared object: manager
        if self._manager is not None:
            self._manager.shutdown()
            self._manager = None
            self._state = None

        self.initialized = False
        self._workers_inited = False

        logger.info("stop manager")
