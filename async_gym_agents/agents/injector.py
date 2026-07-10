import contextlib
import io
import logging
import multiprocessing
import os
import queue
import threading
import time
from collections import deque
from contextlib import contextmanager
from multiprocessing.context import Process as MPProcess
from multiprocessing.managers import Namespace
from multiprocessing.queues import Queue as MPQueue
from multiprocessing.synchronize import Event as MPEvent
from types import SimpleNamespace
from typing import (
    Any,
    Deque,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Type,
    TypeAlias,
    cast,
)

import torch
from stable_baselines3.common.base_class import BasePolicy

from async_gym_agents.constants import (
    NANOSECONDS_PER_SECOND,
    PROFILE_PHASE_POLICY_BROADCAST,
    PROFILE_PHASE_POLICY_LOADING,
    PROFILE_PHASE_POLICY_SERIALIZATION,
    PROFILE_PHASE_TRANSITION_CONSUMPTION,
    PROFILE_PHASE_TRANSITION_RECONSTRUCTION,
    PROFILE_PHASE_TRANSPORT,
    PROFILE_PHASE_WAITING,
    QUEUE_PUT_RETRY_TIMEOUT_SECONDS,
)
from async_gym_agents.envs.multi_env import IndexableMultiEnv
from async_gym_agents.profiler import (
    ProfileStats,
    RuntimeProfiler,
    build_profiler_report,
    merge_profile_stats,
)
from async_gym_agents.transport import Transport
from async_gym_agents.transport.constants import (
    DEFAULT_RING_CAPACITY_MULTIPLIER,
    MIN_RING_CAPACITY,
    RING_FULL_POLL_SECONDS,
    TRANSPORT_CONSUMED_ROWS_KEY,
    TRANSPORT_CONSUMER_BUFFERED_ROWS_KEY,
    TRANSPORT_CONSUMER_MAX_BUFFERED_ROWS_KEY,
    TRANSPORT_DROPPED_ROWS_KEY,
    TRANSPORT_PENDING_ROWS_KEY,
    TRANSPORT_PRODUCED_ROWS_KEY,
    TRANSPORT_RING_ALLOCATED_BYTES_KEY,
    TRANSPORT_RING_CAPACITY_ROWS_KEY,
    TRANSPORT_RING_UTILIZATION_KEY,
    TRANSPORT_TRAIN_ALLOCATED_BYTES_KEY,
)
from async_gym_agents.transport.spsc_ring import resolve_ring
from async_gym_agents.types import EnvFactory, Transition
from async_gym_agents.utils import make_venv

GenericState: TypeAlias = Namespace | SimpleNamespace
GenericStateLock: TypeAlias = Any
GenericEvent: TypeAlias = MPEvent | threading.Event
GenericQueue: TypeAlias = MPQueue | queue.Queue
GenericUpdateQueue: TypeAlias = MPQueue | queue.Queue
GenericWorker: TypeAlias = MPProcess | threading.Thread
EpisodePayload: TypeAlias = Tuple[Optional[int], List[Transition]]


@contextmanager
def patched_env(**updates):
    old = {k: os.environ.get(k) for k in updates}
    try:
        for k, v in updates.items():
            os.environ[k] = str(v)
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class AsyncAgentInjector:
    def __init__(
        self,
        *args,
        max_episodes_in_buffer: int = 8,
        use_mp: bool = False,
        worker_start_interval_seconds: float = 0.0,
        skip_truncated: bool = False,
        queue_put_timeout: float = 60.0,
        worker_join_timeout: float = 120.0,
        profiler_sync_interval: float = 1.0,
        mp_threads: int = 1,
        mp_method: Optional[str] = "spawn",
        **kwargs,
    ):
        """
        :param max_episodes_in_buffer: Max episodes in the buffer before blocking
        :param use_mp: Use processes instead of threads
        :param worker_start_interval_seconds: Delay between parent-side worker starts
        :param skip_truncated: Skip episodes with truncated signal
        :param queue_put_timeout: Timeout when putting an episode before dropping
        :param worker_join_timeout: Shutdown time before killing the process
        :param mp_threads: Cores used for various torch multiprocessing, which for workers should be lowered
        :param profiler_sync_interval: Worker profiler flush interval in seconds
        :param mp_method: Method to create processes. None for OS default. See https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
        """
        self.max_episodes_in_buffer = max_episodes_in_buffer
        self.use_mp = use_mp
        self.worker_start_interval_seconds = worker_start_interval_seconds

        # noinspection PyTypeChecker
        self.mp_ctx = multiprocessing.get_context(mp_method)

        self._skip_truncated = skip_truncated
        self._queue_put_timeout = queue_put_timeout
        self._worker_join_timeout = worker_join_timeout
        self._profiler_sync_interval = profiler_sync_interval
        self.mp_threads = mp_threads

        # shared memory
        self._episode_queue: GenericQueue | None = None
        self._transport: Optional[Transport] = None
        self._update_queues: List[GenericUpdateQueue] = []
        self._transitions: Deque[Transition] = deque()

        # shared object (!)
        self._manager: Optional[multiprocessing.Manager] = None
        self._state: GenericState | None = None
        self._state_lock: GenericStateLock | None = None
        self._version = 0

        self._stop: GenericEvent | None = None

        self._initialized = False
        self._initialized_workers = False
        self._workers: List[GenericWorker] = []

        # Metrics
        self._buffer_utilization = 0.0
        self._buffer_emptiness = 0.0
        self._buffer_stat_count = 0
        self._policy_lag_total = 0
        self._policy_lag_count = 0
        self._policy_lag_max = 0
        self._transition_consumption_total_ns = 0
        self._transition_consumption_count = 0
        self._consumer_max_buffered_rows = 0

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
        use_mp: bool = False,
        mp_threads: int = 1,
        ring_handle=None,
    ):
        if use_mp:
            try:
                torch.set_num_threads(mp_threads)
                torch.set_num_interop_threads(mp_threads)
            except RuntimeError:
                logging.getLogger("async_gym_agents").warning(
                    "Failed to set torch threads, make sure to never call torch.set_num_threads() unconditional!"
                )

        worker = worker_class(
            env_func=env_func,
            episode_queue=episode_queue,
            update_queue=update_queue,
            state=state,
            state_lock=state_lock,
            stop=stop,
            policy_class=policy_class,
            policy_data=policy_data,
            ring_handle=ring_handle,
            **worker_kwargs,
        )

        if use_mp:
            logging.getLogger("async_gym_agents").info(
                f"Worker has {torch.get_num_threads()} threads and {torch.get_num_interop_threads()} interop threads"
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

    def _transport_layout(self) -> Optional[List]:
        """Fixed ring layout for shared-memory transport, or None to use the queue."""
        return None

    def _row_to_transition(self, fields: Dict[str, Any], index: int) -> Transition:
        """Reconstruct a Transition from an assembled ring row. Off-policy only."""
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

        with self._profiler_main.track(PROFILE_PHASE_POLICY_SERIALIZATION):
            weights_buf = io.BytesIO()
            torch.save(policy.state_dict(), weights_buf)
            weights_bytes = weights_buf.getvalue()

        self._version += 1
        with self._profiler_main.track(PROFILE_PHASE_POLICY_BROADCAST):
            self._push_policy_update(self._version, weights_bytes)

        self._logger.debug(f"update policy to the version: {self._version}")

        with self._profiler_main.track("worker_bootstrap"):
            self._init_collect_processes(policy)

    def _init_collect_state(self):
        if self._initialized:
            return

        # SB3 load restores `use_mp` after __init__, so rebuild runtime sync
        # primitives from the current mode before creating worker state.
        self._state_lock = self.mp_ctx.Lock() if self.use_mp else threading.Lock()

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

        # Shared-memory transport for experience (falls back to the queue above
        # when the injector does not provide a fixed layout).
        layout = self._transport_layout()
        if layout is not None:
            n_workers = len(self.get_indexable_env().env_fns)
            ring_capacity = max(
                MIN_RING_CAPACITY,
                self.max_episodes_in_buffer
                * 512
                * DEFAULT_RING_CAPACITY_MULTIPLIER
                // 4,
            )
            self._transport = Transport(
                layout,
                n_workers=n_workers,
                ring_capacity=ring_capacity,
                train_capacity=n_workers * ring_capacity,
                use_mp=self.use_mp,
                mp_ctx=self.mp_ctx,
            )

        # Shared state for metrics
        self._manager = self.mp_ctx.Manager() if self.use_mp else None
        self._state = self._manager.Namespace() if self.use_mp else SimpleNamespace()

        self._state.total_episodes = 0
        self._state.discarded_episodes = 0
        self._state.queue_put_attempts = 0
        self._state.full_queue_put_attempts = 0
        self._state.total_queue_put_wait_ns = 0
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

        policy_class = type(policy)
        # noinspection PyProtectedMember
        policy_data = policy._get_constructor_parameters()

        worker_env = dict(
            OMP_NUM_THREADS=self.mp_threads,
            MKL_NUM_THREADS=self.mp_threads,
            OPENBLAS_NUM_THREADS=self.mp_threads,
            NUMEXPR_NUM_THREADS=self.mp_threads,
            TORCH_NUM_THREADS=self.mp_threads,
            TORCH_NUM_INTEROP_THREADS=self.mp_threads,
            CUDA_VISIBLE_DEVICES="",
        )

        worker_env_fns = self.get_indexable_env().env_fns
        worker_count = len(worker_env_fns)
        for worker_index, (env_func, update_queue) in enumerate(
            zip(worker_env_fns, self._update_queues, strict=True)
        ):
            with patched_env(**worker_env) if self.use_mp else contextlib.nullcontext():
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
                        use_mp=self.use_mp,
                        mp_threads=self.mp_threads,
                        ring_handle=(
                            self._transport.worker_ring_handle(worker_index)
                            if self._transport
                            else None
                        ),
                    ),
                )
                worker.start()

            # noinspection PyTypeChecker
            self._workers.append(worker)

            if (
                self.worker_start_interval_seconds > 0
                and worker_index < worker_count - 1
            ):
                time.sleep(self.worker_start_interval_seconds)

        self._initialized_workers = True

    def _excluded_save_params(self) -> List[str]:
        # noinspection PyUnresolvedReferences
        return super()._excluded_save_params() + [
            "_envs",
            "_episode_queue",
            "_transport",
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
            "_policy_lag_total",
            "_policy_lag_count",
            "_policy_lag_max",
            "_transition_consumption_total_ns",
            "_transition_consumption_count",
            "_consumer_max_buffered_rows",
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

    def _fetch_transitions(self, buffer_was_empty: bool) -> List[Transition]:
        phase = PROFILE_PHASE_WAITING if buffer_was_empty else PROFILE_PHASE_TRANSPORT
        with self._profiler_main.track(phase):
            payload = self._episode_queue.get()

        policy_version, transitions = self._decode_episode_payload(payload)
        self._record_policy_lag(policy_version, len(transitions))
        return transitions

    @staticmethod
    def _decode_episode_payload(payload) -> EpisodePayload:
        if isinstance(payload, tuple) and len(payload) == 2:
            return payload

        return None, payload

    def _record_policy_lag(
        self,
        policy_version: Optional[int],
        transition_count: int,
    ) -> None:
        if policy_version is None or transition_count == 0:
            return

        lag = max(0, self._version - policy_version)
        self._policy_lag_total += lag * transition_count
        self._policy_lag_count += transition_count
        self._policy_lag_max = max(self._policy_lag_max, lag)

    def fetch_transition(self) -> Transition:
        """
        Each transition is returned in per-worker production order. With the
        shared-memory transport the trainer assembles available rows and
        reconstructs Transitions; otherwise it drains the episode queue.
        """
        if self._transport is not None:
            while len(self._transitions) == 0:
                self._refill_from_transport()
            return self._consume_buffered_transition()

        while len(self._transitions) == 0:
            self._buffer_utilization += self._episode_queue.qsize()
            buffer_was_empty = self._episode_queue.empty()
            self._buffer_emptiness += 1 if buffer_was_empty else 0
            self._buffer_stat_count += 1

            self._transitions.extend(self._fetch_transitions(buffer_was_empty))

        return self._consume_buffered_transition()

    def _consume_buffered_transition(self) -> Transition:
        start_ns = time.perf_counter_ns()
        transition = self._transitions.popleft()
        self._transition_consumption_total_ns += time.perf_counter_ns() - start_ns
        self._transition_consumption_count += 1
        return transition

    def _refill_from_transport(self) -> None:
        with self._profiler_main.track(PROFILE_PHASE_TRANSPORT):
            rollout = self._transport.assemble_available()
        if rollout.n_rows == 0:
            with self._profiler_main.track(PROFILE_PHASE_WAITING):
                time.sleep(0.0005)
            return

        reconstruction_start_ns = time.perf_counter_ns()
        fields = rollout.fields
        has_version = "policy_version" in fields
        for index in range(rollout.n_rows):
            self._transitions.append(self._row_to_transition(fields, index))
            if has_version:
                self._record_policy_lag(int(fields["policy_version"][index]), 1)
        self._profiler_main.record(
            PROFILE_PHASE_TRANSITION_RECONSTRUCTION,
            time.perf_counter_ns() - reconstruction_start_ns,
            count=rollout.n_rows,
        )
        self._consumer_max_buffered_rows = max(
            self._consumer_max_buffered_rows,
            len(self._transitions),
        )

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

        if self._transport is not None:
            self._transport.shutdown()
            self._transport = None

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
                queue_put_attempts=self._state.queue_put_attempts,
                full_queue_put_attempts=self._state.full_queue_put_attempts,
                total_queue_put_wait_ns=self._state.total_queue_put_wait_ns,
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
        main_stats = self._profiler_main.snapshot()
        merge_profile_stats(
            main_stats,
            {
                PROFILE_PHASE_TRANSITION_CONSUMPTION: {
                    "total_ns": self._transition_consumption_total_ns,
                    "count": self._transition_consumption_count,
                }
            },
        )
        return build_profiler_report(
            main_stats,
            self._get_worker_profiler_stats(),
            worker_last_sync_time=(
                None
                if self._state is None
                else getattr(self._state, "worker_profiler_last_sync", None)
            ),
            buffer_utilization=self.buffer_utilization,
            buffer_emptiness=self.buffer_emptyness,
            buffer_full_push_fraction=self.buffer_full_push_fraction,
            buffer_avg_push_wait_time=self.buffer_avg_push_wait_time,
            discarded_episodes_fraction=self.discarded_episodes_fraction,
            avg_policy_lag=self.avg_policy_lag,
            max_policy_lag=self.max_policy_lag,
            transport_stats=self._build_transport_report(),
        )

    def _build_transport_report(self) -> Dict[str, float | int]:
        if self._transport is None:
            return {}

        stats = self._transport.collect_stats()
        produced_rows = sum(stats.produced)
        consumed_rows = sum(stats.consumed)
        dropped_rows = sum(stats.dropped)
        pending_rows = produced_rows - consumed_rows
        capacity_rows = self._transport.n_workers * self._transport.ring_capacity
        return {
            TRANSPORT_PRODUCED_ROWS_KEY: produced_rows,
            TRANSPORT_CONSUMED_ROWS_KEY: consumed_rows,
            TRANSPORT_PENDING_ROWS_KEY: pending_rows,
            TRANSPORT_DROPPED_ROWS_KEY: dropped_rows,
            TRANSPORT_RING_CAPACITY_ROWS_KEY: capacity_rows,
            TRANSPORT_RING_UTILIZATION_KEY: pending_rows / capacity_rows,
            TRANSPORT_CONSUMER_BUFFERED_ROWS_KEY: len(self._transitions),
            TRANSPORT_CONSUMER_MAX_BUFFERED_ROWS_KEY: self._consumer_max_buffered_rows,
            TRANSPORT_RING_ALLOCATED_BYTES_KEY: (
                self._transport.calculate_ring_allocated_bytes()
            ),
            TRANSPORT_TRAIN_ALLOCATED_BYTES_KEY: (
                self._transport.calculate_train_allocated_bytes()
            ),
        }

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

    @property
    def buffer_full_push_fraction(self) -> float:
        """
        The fraction of episode push attempts that encountered a full buffer.
        """
        return (
            0
            if self._state is None or self._state.queue_put_attempts == 0
            else self._state.full_queue_put_attempts / self._state.queue_put_attempts
        )

    @property
    def buffer_avg_push_time(self) -> float:
        """
        Backwards-compatible alias for the average enqueue wait time.
        """
        return self.buffer_avg_push_wait_time

    @property
    def buffer_avg_push_wait_time(self) -> float:
        """
        The average time spent waiting to enqueue an episode, in seconds.
        """
        return (
            0
            if self._state is None or self._state.queue_put_attempts == 0
            else self._state.total_queue_put_wait_ns
            / self._state.queue_put_attempts
            / NANOSECONDS_PER_SECOND
        )

    @property
    def avg_policy_lag(self) -> float:
        """
        The average number of policy updates between production and consumption.
        """
        return (
            0
            if self._policy_lag_count == 0
            else self._policy_lag_total / self._policy_lag_count
        )

    @property
    def max_policy_lag(self) -> int:
        """
        The maximum number of policy updates between production and consumption.
        """
        return self._policy_lag_max


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
        ring_handle=None,
        **kwargs,
    ):
        self.env = make_venv(env_func())
        self._ring = resolve_ring(ring_handle) if ring_handle is not None else None

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
                if block and latest_update is None:
                    continue
                break

        if latest_update is None:
            return

        version, weights_bytes = latest_update
        if self._policy_version == version:
            return

        with self._profiler.track(PROFILE_PHASE_POLICY_LOADING):
            weights = torch.load(
                io.BytesIO(weights_bytes),
                map_location="cpu",
                weights_only=True,
            )
            self.policy.load_state_dict(weights)
            self.policy.to("cpu")
            self.policy.set_training_mode(False)
            self._policy_version = version
            self._logger.debug(
                f"policy loaded from queue: version={self._policy_version}"
            )

    def _put_episode_with_timeout(self, episode):
        payload = (self._policy_version, episode)
        deadline_ns = time.perf_counter_ns() + int(
            self._queue_put_timeout * NANOSECONDS_PER_SECOND
        )
        queue_was_full = self._episode_queue.full()
        waiting_ns = 0
        transport_ns = 0

        def record_queue_profile() -> None:
            if waiting_ns > 0:
                with self._state_lock:
                    self._state.total_queue_put_wait_ns += waiting_ns
                self._profiler.record(PROFILE_PHASE_WAITING, waiting_ns)
            if transport_ns > 0:
                self._profiler.record(PROFILE_PHASE_TRANSPORT, transport_ns)

        with self._state_lock:
            self._state.queue_put_attempts += 1
            if queue_was_full:
                self._state.full_queue_put_attempts += 1

        while time.perf_counter_ns() < deadline_ns:
            if self._stop.is_set():
                record_queue_profile()
                return

            try:
                remaining_timeout = min(
                    QUEUE_PUT_RETRY_TIMEOUT_SECONDS,
                    max(
                        0.0,
                        (deadline_ns - time.perf_counter_ns()) / NANOSECONDS_PER_SECOND,
                    ),
                )
                put_start_ns = time.perf_counter_ns()
                queue_is_full = self._episode_queue.full()
                self._episode_queue.put(
                    payload,
                    block=True,
                    timeout=remaining_timeout,
                )
                elapsed_ns = time.perf_counter_ns() - put_start_ns
                if queue_is_full:
                    waiting_ns += elapsed_ns
                else:
                    transport_ns += elapsed_ns
                record_queue_profile()
                return
            except queue.Full:
                waiting_ns += time.perf_counter_ns() - put_start_ns

        replacement_start_ns = time.perf_counter_ns()
        try:
            self._episode_queue.get(block=False)
            self._episode_queue.put(payload, block=False)
        except (queue.Full, queue.Empty):
            pass
        finally:
            transport_ns += time.perf_counter_ns() - replacement_start_ns

        with self._state_lock:
            self._state.discarded_episodes += 1
        record_queue_profile()
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

                if self._ring is not None:
                    for transition in episode:
                        self._write_to_ring(self._transition_to_fields(transition))
                else:
                    self._put_episode_with_timeout(episode)
                self._flush_profiler()

                if self._stop.is_set():
                    break
        finally:
            self._flush_profiler(force=True)
            self.env.close()
            self._logger.info("Generator cycle is completed")

    def _transition_to_fields(self, transition: Transition) -> Dict[str, Any]:
        """Map a transition to ring fields. Implemented by ring-enabled workers."""
        raise NotImplementedError()

    def _write_to_ring(self, fields: Dict[str, Any]) -> None:
        """Backpressure like the queue: block up to queue_put_timeout, then drop."""
        deadline = time.perf_counter() + self._queue_put_timeout
        while not self._stop.is_set():
            if self._ring.try_write(fields):
                return
            if time.perf_counter() >= deadline:
                self._ring.note_drop()
                return
            time.sleep(RING_FULL_POLL_SECONDS)

    def generate(self) -> Generator[list[Transition], None, None]:
        raise NotImplementedError()
