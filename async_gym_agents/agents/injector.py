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
from types import SimpleNamespace
from typing import Any, Dict, Generator, List, Optional, Type, TypeAlias, cast

import torch
import torch as th
from stable_baselines3.common.base_class import BasePolicy

from async_gym_agents.envs.multi_env import IndexableMultiEnv
from async_gym_agents.profiler import (
    ProfileStats,
    RuntimeProfiler,
    build_profiler_report,
    merge_profile_stats,
)
from async_gym_agents.types import EnvFactory, Transition
from async_gym_agents.utils import make_venv

GenericState: TypeAlias = Namespace | SimpleNamespace
GenericStateLock: TypeAlias = Any
GenericEvent: TypeAlias = MPEvent | threading.Event
GenericQueue: TypeAlias = MPQueue | queue.Queue
GenericUpdateQueue: TypeAlias = MPQueue | queue.Queue
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
        profiler_sync_interval: float = 1.0,
        mp_method: Optional[str] = "spawn",
        **kwargs,
    ):
        """
        :param max_episodes_in_buffer: Max episodes in the buffer before blocking
        :param use_mp: Use processes instead of threads
        :param skip_truncated: Skip episodes with truncated signal
        :param queue_put_timeout: Timeout when putting an episode before dropping
        :param worker_join_timeout: Shutdown time before killing the process
        :param profiler_sync_interval: Worker profiler flush interval in seconds
        :param mp_method: Method to create processes. None for OS default. See https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
        """
        self.max_episodes_in_buffer = max_episodes_in_buffer
        self.use_mp = use_mp

        # noinspection PyTypeChecker
        self.mp_ctx = multiprocessing.get_context(mp_method)

        self._skip_truncated = skip_truncated
        self._queue_put_timeout = queue_put_timeout
        self._worker_join_timeout = worker_join_timeout
        self._profiler_sync_interval = profiler_sync_interval

        # shared memory
        self._episode_queue: GenericQueue | None = None
        self._update_queues: List[GenericUpdateQueue] = []
        self._transitions: List[Transition] = []

        # shared object (!)
        self._manager: Optional[multiprocessing.Manager] = None
        self._state: GenericState | None = None
        self._state_lock: GenericStateLock = (
            self.mp_ctx.Lock() if use_mp else threading.Lock()
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

        self._profiler_main = RuntimeProfiler()
        self._logger = logging.getLogger("async_gym_agents")

    @staticmethod
    def _run_worker(
        worker_class: Type["InjectorWorkerBase"],
        env_func: EnvFactory,
        episode_queue: GenericQueue,
        update_queue: GenericUpdateQueue,
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
            update_queue=update_queue,
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
            update_queue.close()
            update_queue.cancel_join_thread()

    def get_worker_class(self) -> Type["InjectorWorkerBase"]:
        raise NotImplementedError()

    def get_worker_kwargs(self) -> Dict[str, Any]:
        return dict(
            skip_truncated=self._skip_truncated,
            queue_put_timeout=self._queue_put_timeout,
            profiler_sync_interval=self._profiler_sync_interval,
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

        with self._profiler_main.track("syncing"):
            # weights -> bytes
            weights_buf = io.BytesIO()
            th.save(policy.state_dict(), weights_buf)
            weights_bytes = weights_buf.getvalue()

            self._version += 1
            self._push_policy_update(self._version, weights_bytes)

            self._logger.debug(f"update policy to the version: {self._version}")

        with self._profiler_main.track("worker_bootstrap"):
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
        self._update_queues = [
            self.mp_ctx.Queue() if self.use_mp else queue.Queue()
            for _ in self.get_indexable_env().env_fns
        ]

        # Shared state for metrics
        self._manager = self.mp_ctx.Manager() if self.use_mp else None
        self._state = self._manager.Namespace() if self.use_mp else SimpleNamespace()

        self._state.total_episodes = 0
        self._state.discarded_episodes = 0
        self._state.worker_profiler_stats = self._manager.dict() if self.use_mp else {}
        self._state.worker_profiler_last_sync = None

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

        for env_func, update_queue in zip(
            self.get_indexable_env().env_fns, self._update_queues, strict=True
        ):
            worker = (self.mp_ctx.Process if self.use_mp else threading.Thread)(
                target=AsyncAgentInjector._run_worker,
                kwargs=dict(
                    worker_class=self.get_worker_class(),
                    env_func=env_func,
                    episode_queue=self._episode_queue,
                    update_queue=update_queue,
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

            # noinspection PyTypeChecker
            self._workers.append(worker)

        self._initialized_workers = True

    def _excluded_save_params(self) -> List[str]:
        # noinspection PyUnresolvedReferences
        return super()._excluded_save_params() + [
            "_envs",
            "_episode_queue",
            "_update_queues",
            "_transitions",
            "_manager",
            "_state",
            "_state_lock",
            "_version",
            "_stop",
            "_initialized",
            "_initialized_workers",
            "_workers",
            "_profiler_main",
            "_logger",
        ]

    @staticmethod
    def _clear_queue(target_queue: GenericUpdateQueue) -> None:
        while True:
            try:
                target_queue.get_nowait()
            except queue.Empty:
                return

    def _push_policy_update(self, version: int, weights: bytes) -> None:
        for update_queue in self._update_queues:
            self._clear_queue(update_queue)
            update_queue.put((version, weights))

    def _fetch_transitions(self) -> List[Transition]:
        with self._profiler_main.track("transport"):
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
        if self._stop is None:
            return

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
        for update_queue in self._update_queues:
            if self.use_mp:
                update_queue.close()
                update_queue.cancel_join_thread()
        self._update_queues = []

        # release a shared object: manager
        if self._manager is not None:
            worker_profiler_stats = cast(
                ProfileStats, dict(self._state.worker_profiler_stats)
            )
            self._state = SimpleNamespace(
                total_episodes=self._state.total_episodes,
                discarded_episodes=self._state.discarded_episodes,
                worker_profiler_stats=worker_profiler_stats,
                worker_profiler_last_sync=self._state.worker_profiler_last_sync,
            )
            self._manager.shutdown()
            self._manager = None

        self._initialized = False
        self._initialized_workers = False

        self._logger.info("Stopped manager")

    def train(self, *args, **kwargs):
        with self._profiler_main.track("training"):
            # noinspection PyUnresolvedReferences
            return super().train(*args, **kwargs)

    def get_profiler_report(self) -> Dict[str, Any]:
        return build_profiler_report(
            self._profiler_main.snapshot(),
            self._get_worker_profiler_stats(),
            worker_last_sync_time=(
                None
                if self._state is None
                else getattr(self._state, "worker_profiler_last_sync", None)
            ),
            buffer_utilization=self.buffer_utilization,
            buffer_emptiness=self.buffer_emptyness,
            discarded_episodes_fraction=self.discarded_episodes_fraction,
        )

    def _get_worker_profiler_stats(self) -> ProfileStats:
        if self._state is None or not hasattr(self._state, "worker_profiler_stats"):
            return {}

        return cast(ProfileStats, dict(self._state.worker_profiler_stats))

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
        update_queue: GenericUpdateQueue,
        state: GenericState,
        state_lock: GenericStateLock,
        stop: GenericEvent,
        skip_truncated: bool,
        queue_put_timeout: float,
        profiler_sync_interval: float,
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
        self._update_queue = update_queue
        self._state = state
        self._state_lock = state_lock
        self._stop = stop

        self._logger = logging.getLogger("Worker")

        self._skip_truncated = skip_truncated
        self._queue_put_timeout = queue_put_timeout
        self._profiler = RuntimeProfiler()
        self._profiler_sync_interval = profiler_sync_interval
        self._last_profiler_sync = time.time()

    def copy_policy_from_queue(self, block: bool = False):
        if self.policy is None:
            # noinspection PyArgumentList
            self.policy = self.policy_class(**self.policy_data)

        latest_update = None
        while not self._stop.is_set():
            try:
                if latest_update is None and block:
                    latest_update = self._update_queue.get(timeout=0.1)
                else:
                    latest_update = self._update_queue.get_nowait()
            except queue.Empty:
                break

        if latest_update is None:
            return

        version, weights_bytes = latest_update
        if self._policy_version == version:
            return

        with self._profiler.track("syncing"):
            weights = torch.load(
                io.BytesIO(weights_bytes),
                map_location="cpu",
                weights_only=True,
            )
            self.policy.load_state_dict(weights)
            self.policy.set_training_mode(False)
            self._policy_version = version
            self._logger.debug(
                f"policy loaded from queue: version={self._policy_version}"
            )

    def _put_episode_with_timeout(self, episode):
        start_ns = time.perf_counter_ns()
        deadline = time.time() + self._queue_put_timeout

        while time.time() < deadline:
            if self._stop.is_set():
                self._profiler.record("transport", time.perf_counter_ns() - start_ns)
                return

            try:
                self._episode_queue.put(
                    episode,
                    block=True,
                    timeout=min(0.1, deadline - time.time()),
                )
                self._profiler.record("transport", time.perf_counter_ns() - start_ns)
                return
            except queue.Full:
                pass

        try:
            self._episode_queue.get(block=False)
            self._episode_queue.put(episode, block=False)
        except (queue.Full, queue.Empty):
            pass

        with self._state_lock:
            self._state.discarded_episodes += 1
        self._profiler.record("transport", time.perf_counter_ns() - start_ns)
        self._logger.info("Dropped episode due to buffer full")

    def _flush_profiler(self, force: bool = False):
        now = time.time()
        if not force and now - self._last_profiler_sync < self._profiler_sync_interval:
            return

        delta = self._profiler.drain_pending()
        if not delta:
            self._last_profiler_sync = now
            return

        with self._state_lock:
            merge_profile_stats(self._state.worker_profiler_stats, delta)
            self._state.worker_profiler_last_sync = now

        self._last_profiler_sync = now

    def run(self):
        try:
            self.copy_policy_from_queue(block=True)

            for episode in self.generate():
                with self._state_lock:
                    self._state.total_episodes += 1

                if (
                    episode[-1].infos[0].get("TimeLimit.truncated", False)
                    and self._skip_truncated
                ):
                    with self._state_lock:
                        self._state.discarded_episodes += 1
                    self._flush_profiler()
                    continue

                self._put_episode_with_timeout(episode)
                self._flush_profiler()

                if self._stop.is_set():
                    break
        finally:
            self._flush_profiler(force=True)
            self.env.close()
            self._logger.info("Generator cycle is completed")

    def generate(self) -> Generator[list[Transition], None, None]:
        raise NotImplementedError()
