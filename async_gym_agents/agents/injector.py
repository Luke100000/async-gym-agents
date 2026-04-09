import io
import logging
import multiprocessing
import queue
import threading
import time
from multiprocessing.context import Process as MPProcess
from multiprocessing.managers import Namespace
from multiprocessing.queues import Queue as MPQueue
from multiprocessing.synchronize import Event as MPEvent
from multiprocessing.synchronize import SemLock
from types import SimpleNamespace
from typing import Any, Dict, Generator, List, Optional, Type, TypeAlias

import torch
import torch as th
from stable_baselines3.common.base_class import BasePolicy

from async_gym_agents.envs.multi_env import IndexableMultiEnv
from async_gym_agents.types import EnvFactory, Transition
from async_gym_agents.utils import make_venv


class MockLock:
    def __init__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


GenericState: TypeAlias = Namespace | SimpleNamespace
GenericStateLock: TypeAlias = SemLock | MockLock
GenericEvent: TypeAlias = MPEvent | threading.Event
GenericQueue: TypeAlias = MPQueue | queue.Queue
GenericWorker: TypeAlias = MPProcess | threading.Thread


def _get_torch_thread_settings() -> Dict[str, int]:
    return {
        "num_threads": torch.get_num_threads(),
        "num_interop_threads": torch.get_num_interop_threads(),
    }


def _apply_torch_thread_settings(thread_settings: Dict[str, int]) -> None:
    try:
        torch.set_num_threads(thread_settings["num_threads"])
        torch.set_num_interop_threads(thread_settings["num_interop_threads"])
    except RuntimeError:
        pass


class AsyncAgentInjector:
    def __init__(
        self,
        *args,
        max_episodes_in_buffer: int = 8,
        use_mp: bool = False,
        skip_truncated: bool = False,
        queue_put_timeout: float = 60.0,
        worker_join_timeout: float = 120.0,
        mp_method: Optional[str] = "spawn",
        **kwargs,
    ):
        """
        :param max_episodes_in_buffer: Max episodes in the buffer before blocking
        :param use_mp: Use processes instead of threads
        :param skip_truncated: Skip episodes with truncated signal
        :param queue_put_timeout: Timeout when putting an episode before dropping
        :param worker_join_timeout: Shutdown time before killing the process
        :param mp_method: Method to create processes. None for OS default. See https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
        """
        self.max_episodes_in_buffer = max_episodes_in_buffer
        self.use_mp = use_mp

        self.mp_ctx = multiprocessing.get_context(mp_method)

        self._skip_truncated = skip_truncated
        self._queue_put_timeout = queue_put_timeout
        self._worker_join_timeout = worker_join_timeout

        # shared memory
        self._episode_queue: GenericQueue | None = None
        self._transitions: Optional[List[Transition]] = []

        # shared object (!)
        self._manager: Optional[multiprocessing.Manager] = None
        self._state: GenericState | None = None
        self._state_lock: GenericStateLock | None = (
            self.mp_ctx.Lock() if use_mp else MockLock()
        )
        self._version = 0

        self._stop: GenericEvent | None = None

        self._initialized = False
        self._initialized_workers = False
        self._workers: List[GenericWorker] = []

        # Metrics
        self._buffer_utilization = 0.0
        self._buffer_emptiness = 0.0
        self._buffer_stat_count = 0

        self._logger = logging.getLogger("async_gym_agents")

    @staticmethod
    def _run_worker(
        worker_class: Type["InjectorWorkerBase"],
        env_func: EnvFactory,
        episode_queue: GenericQueue,
        state: GenericState,
        state_lock: GenericStateLock,
        stop: GenericEvent,
        worker_kwargs: Dict[str, Any],
        policy_class: BasePolicy,
        policy_data: Dict[str, Any],
        torch_thread_settings: Dict[str, int],
        use_mp: bool = False,
    ):
        if use_mp:
            _apply_torch_thread_settings(torch_thread_settings)

        worker = worker_class(
            env_func=env_func,
            episode_queue=episode_queue,
            state=state,
            state_lock=state_lock,
            stop=stop,
            policy_class=policy_class,
            policy_data=policy_data,
            **worker_kwargs,
        )
        worker.run()

        # Only close the queue in a child process; closing it in a thread
        # would close the shared queue for all workers.
        if use_mp:
            episode_queue.close()
            episode_queue.cancel_join_thread()

    def get_worker_class(self) -> Type["InjectorWorkerBase"]:
        raise NotImplementedError()

    def get_worker_kwargs(self) -> Dict[str, Any]:
        return dict(
            skip_truncated=self._skip_truncated,
            queue_put_timeout=self._queue_put_timeout,
        )

    # noinspection PyUnresolvedReferences
    def get_indexable_env(self) -> IndexableMultiEnv:
        """
        Asserts whether a correct environment is supplied
        """
        assert isinstance(self.env, IndexableMultiEnv), (
            "You must pass a IndexableMultiEnv"
        )
        return self.env

    def pre_collect_preparation(self, policy: BasePolicy):
        self._init_collect_state()

        # weights -> bytes
        weights_buf = io.BytesIO()
        th.save(policy.state_dict(), weights_buf)
        weights_bytes = weights_buf.getvalue()

        with self._state_lock:
            self._version += 1
            self._state.version = self._version
            self._state.weights = weights_bytes

            self._logger.debug(f"update policy to the version: {self._version}")

        self._init_collect_processes(policy)

    def _init_collect_state(self):
        if self._initialized:
            return

        # Environment queue
        self._episode_queue = (
            self.mp_ctx.Queue(maxsize=self.max_episodes_in_buffer)
            if self.use_mp
            else queue.Queue(maxsize=self.max_episodes_in_buffer)
        )

        # Shared state for policy and metrics
        self._manager = self.mp_ctx.Manager() if self.use_mp else None
        self._state = self._manager.Namespace() if self.use_mp else SimpleNamespace()

        self._state.total_episodes = 0
        self._state.discarded_episodes = 0

        # Stop signal
        self._stop = self.mp_ctx.Event() if self.use_mp else threading.Event()

        self._initialized = True

    def _init_collect_processes(self, policy: BasePolicy):
        if self._initialized_workers:
            return

        # Start workers
        self._workers = []

        torch_thread_settings = _get_torch_thread_settings()

        policy_class = type(policy)
        # noinspection PyProtectedMember
        policy_data = policy._get_constructor_parameters()

        for env_func in self.get_indexable_env().env_fns:
            worker = (self.mp_ctx.Process if self.use_mp else threading.Thread)(
                target=AsyncAgentInjector._run_worker,
                kwargs=dict(
                    worker_class=self.get_worker_class(),
                    env_func=env_func,
                    episode_queue=self._episode_queue,
                    state=self._state,
                    state_lock=self._state_lock,
                    stop=self._stop,
                    worker_kwargs=self.get_worker_kwargs(),
                    policy_class=policy_class,
                    policy_data=policy_data,
                    torch_thread_settings=torch_thread_settings,
                    use_mp=self.use_mp,
                ),
            )
            worker.start()

            self._workers.append(worker)

        self._initialized_workers = True

    def _excluded_save_params(self) -> List[str]:
        # noinspection PyUnresolvedReferences
        return super()._excluded_save_params() + [
            "_envs",
            "_episode_queue",
            "_transitions",
            "_manager",
            "_state",
            "_state_lock",
            "_version",
            "_stop",
            "_initialized",
            "_initialized_workers",
            "_workers",
            "_logger",
        ]

    def _fetch_transitions(self) -> List[Transition]:
        return self._episode_queue.get()

    def fetch_transition(self) -> Transition:
        """
        Each episode is returned as a sequence of transitions, in order, complete,
        and not interleaved with episodes from other workers.
        """
        while len(self._transitions) == 0:
            self._buffer_utilization += self._episode_queue.qsize()
            self._buffer_emptiness += 1 if self._episode_queue.empty() else 0
            self._buffer_stat_count += 1

            self._transitions = self._fetch_transitions()

        return self._transitions.pop(0)

    def shutdown(self):
        self._logger.info("Send stop event to all processes")
        self._stop.set()
        self._stop = None

        for worker in self._workers:
            if not worker.is_alive():
                continue

            worker.join(timeout=self._worker_join_timeout)

            if self.use_mp:
                try:
                    worker.kill()
                except PermissionError:
                    self._logger.warning("cannot kill process due to permission error")

        # close the queue (multiprocessing.Queue needs explicit cleanup)
        if self._episode_queue is not None:
            if self.use_mp:
                self._episode_queue.close()
                self._episode_queue.cancel_join_thread()
            self._episode_queue = None

        # release a shared object: manager
        if self._manager is not None:
            self._state = SimpleNamespace(
                total_episodes=self._state.total_episodes,
                discarded_episodes=self._state.discarded_episodes,
            )
            self._manager.shutdown()
            self._manager = None

        self._initialized = False
        self._initialized_workers = False

        self._logger.info("Stopped manager")

    @property
    def buffer_utilization(self) -> float:
        """
        The average size of the buffer in episodes.
        """
        return (
            0
            if self._buffer_stat_count == 0
            else self._buffer_utilization / self._buffer_stat_count
        )

    @property
    def buffer_emptyness(self) -> float:
        """
        The fraction of the time the buffer was empty.
        """
        return (
            0
            if self._buffer_stat_count == 0
            else self._buffer_emptiness / self._buffer_stat_count
        )

    @property
    def discarded_episodes_fraction(self) -> float:
        """
        The fraction of episodes dropped, either due to full buffer or truncation.
        """
        return (
            0
            if self._state is None or self._state.total_episodes == 0
            else self._state.discarded_episodes / self._state.total_episodes
        )


class InjectorWorkerBase:
    def __init__(
        self,
        env_func: EnvFactory,
        episode_queue: GenericQueue,
        state: GenericState,
        state_lock: GenericStateLock,
        stop: GenericEvent,
        skip_truncated: bool,
        queue_put_timeout: float,
        policy_class: BasePolicy,
        policy_data: Dict[str, Any],
        **kwargs,
    ):
        self.env = make_venv(env_func())

        self.policy: BasePolicy | None = None
        self.policy_class = policy_class
        self.policy_data = policy_data

        self._policy_version = None

        self._episode_queue = episode_queue
        self._state = state
        self._state_lock = state_lock
        self._stop = stop

        self._logger = logging.getLogger("Worker")

        self._skip_truncated = skip_truncated
        self._queue_put_timeout = queue_put_timeout

    def copy_policy_from_state(self):
        if self.policy is None:
            # noinspection PyArgumentList
            self.policy = self.policy_class(**self.policy_data)

        with self._state_lock:
            version = self._state.version
            if self._policy_version != version:
                # load policy weights to CPU
                weights = torch.load(
                    io.BytesIO(self._state.weights),
                    map_location="cpu",
                    weights_only=True,
                )
                self.policy.load_state_dict(weights)
                # turn off the train mode
                self.policy.set_training_mode(False)
                self._policy_version = version
                self._logger.debug(
                    f"policy loaded from state: version={self._policy_version}"
                )

    def _put_episode_with_timeout(self, episode):
        deadline = time.time() + self._queue_put_timeout

        while time.time() < deadline:
            if self._stop.is_set():
                return

            try:
                self._episode_queue.put(
                    episode,
                    block=True,
                    timeout=min(0.1, deadline - time.time()),
                )
                return
            except queue.Full:
                pass

        try:
            self._episode_queue.get(block=False)
            self._episode_queue.put(episode, block=False)
        except (queue.Full, queue.Empty):
            pass

        self._state.discarded_episodes += 1
        self._logger.info("Dropped episode due to buffer full")

    def run(self):
        self.copy_policy_from_state()

        for episode in self.generate():
            self._state.total_episodes += 1

            if (
                episode[-1].infos[0].get("TimeLimit.truncated", False)
                and self._skip_truncated
            ):
                self._state.discarded_episodes += 1
                continue

            self._put_episode_with_timeout(episode)

            if self._stop.is_set():
                break

        self.env.close()
        self._logger.info("Generator cycle is completed")

    def generate(self) -> Generator[list[Transition], None, None]:
        raise NotImplementedError()
