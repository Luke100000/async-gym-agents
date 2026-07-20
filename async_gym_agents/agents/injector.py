import contextlib
import io
import logging
import multiprocessing
import os
import threading
import time
from collections import deque
from contextlib import contextmanager
from multiprocessing.context import Process as MPProcess
from multiprocessing.managers import Namespace
from multiprocessing.synchronize import Event as MPEvent
from types import SimpleNamespace
from typing import Any, Deque, Dict, Generator, List, Optional, Type, TypeAlias, cast

import torch
from stable_baselines3.common.base_class import BasePolicy

from async_gym_agents.constants import (
    BYTES_PER_MEBIBYTE,
    NANOSECONDS_PER_SECOND,
    POLICY_MISSING_INITIAL_SNAPSHOT_ERROR,
    POLICY_PAYLOAD_BYTES_KEY,
    POLICY_PUBLICATION_COUNT_KEY,
    POLICY_PUBLICATION_FAILURES_KEY,
    POLICY_PUBLISHED_VERSION_KEY,
    POLICY_SLOT_CAPACITY_BYTES_KEY,
    PROFILE_PHASE_EPISODE_DESERIALIZATION,
    PROFILE_PHASE_EPISODE_PACKING,
    PROFILE_PHASE_EPISODE_SERIALIZATION,
    PROFILE_PHASE_POLICY_BROADCAST,
    PROFILE_PHASE_POLICY_LOADING,
    PROFILE_PHASE_POLICY_SERIALIZATION,
    PROFILE_PHASE_POLICY_SNAPSHOT_COPY,
    PROFILE_PHASE_POLICY_SNAPSHOT_RETRY,
    PROFILE_PHASE_TRANSITION_RECONSTRUCTION,
    PROFILE_PHASE_TRANSPORT,
    PROFILE_PHASE_WAITING,
    PROFILER_EXCLUDED_OUTPUT_FORMATS,
    PROFILER_LOG_PREFIX,
    TRANSPORT_CAPACITY_EPISODES_KEY,
    TRANSPORT_MAX_PENDING_BYTES_KEY,
    TRANSPORT_MAX_PENDING_EPISODES_KEY,
    TRANSPORT_PAYLOAD_MEBIBYTES_PER_SECOND_KEY,
    TRANSPORT_PAYLOAD_RECEIVE_PROFILE_KEY,
    TRANSPORT_PAYLOAD_RECEIVE_TIMEOUT_PROFILE_KEY,
    TRANSPORT_PENDING_BYTES_KEY,
    TRANSPORT_PENDING_EPISODES_KEY,
    TRANSPORT_PIPE_LATENCY_PROFILE_KEY,
    TRANSPORT_READINESS_TIMEOUT_PROFILE_KEY,
    TRANSPORT_READINESS_WAIT_PROFILE_KEY,
    TRANSPORT_RECEIVE_ATTEMPTS_KEY,
    TRANSPORT_RECEIVE_TIMEOUT_FRACTION_KEY,
    TRANSPORT_RECEIVE_TIMEOUTS_KEY,
    TRANSPORT_RECEIVE_TIMEOUTS_WITH_PENDING_FRACTION_KEY,
    TRANSPORT_RECEIVE_TIMEOUTS_WITH_PENDING_KEY,
    TRANSPORT_RECEIVED_BYTES_KEY,
    TRANSPORT_RECEIVED_EPISODES_KEY,
    TRANSPORT_SENT_BYTES_KEY,
    TRANSPORT_SENT_EPISODES_KEY,
    TRANSPORT_UTILIZATION_KEY,
)
from async_gym_agents.data_classes import (
    EpisodePacket,
    EpisodeSendResult,
    SharedPolicyDescriptor,
)
from async_gym_agents.envs.multi_env import IndexableMultiEnv
from async_gym_agents.episode_codec import (
    decode_episode_packet,
    encode_episode_batch,
    pack_episode,
    unpack_episode,
)
from async_gym_agents.episode_transport import (
    EpisodeFeeder,
    EpisodeSender,
    EpisodeTransport,
)
from async_gym_agents.policy_transport import SharedPolicyReader, SharedPolicyStore
from async_gym_agents.profiler import (
    ProfileStats,
    RuntimeProfiler,
    build_profiler_report,
    iterate_profiler_metrics,
    merge_profile_stats,
    summarize_duration,
)
from async_gym_agents.types import EnvFactory, Transition
from async_gym_agents.utils import make_venv

GenericState: TypeAlias = Namespace | SimpleNamespace
GenericStateLock: TypeAlias = Any
GenericEvent: TypeAlias = MPEvent | threading.Event
GenericWorker: TypeAlias = MPProcess | threading.Thread


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
        queue_put_timeout: Optional[float] = None,
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
        :param queue_put_timeout: Optional timeout before dropping a blocked episode. None applies backpressure until shutdown.
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
        self._episode_transport: EpisodeTransport | None = None
        self._policy_store: Optional[SharedPolicyStore] = None
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
        self._final_transport_report = {}
        self._final_policy_report = {}

        self._profiler_main = RuntimeProfiler()
        self._logger = logging.getLogger("async_gym_agents")

    @staticmethod
    def _run_worker(
        worker_class: Type["InjectorWorkerBase"],
        worker_index: int,
        env_func: EnvFactory,
        episode_sender: EpisodeSender,
        policy_descriptor: SharedPolicyDescriptor,
        state: GenericState,
        state_lock: GenericStateLock,
        stop: GenericEvent,
        worker_kwargs: Dict[str, Any],
        policy_class: BasePolicy,
        policy_data: Dict[str, Any],
        use_mp: bool = False,
        mp_threads: int = 1,
    ):
        if use_mp:
            try:
                torch.set_num_threads(mp_threads)
                torch.set_num_interop_threads(mp_threads)
            except RuntimeError:
                logging.getLogger("async_gym_agents").warning(
                    "Failed to set torch threads, make sure to never call torch.set_num_threads() unconditional!"
                )

        policy_reader = SharedPolicyReader(policy_descriptor)
        try:
            worker = worker_class(
                worker_index=worker_index,
                env_func=env_func,
                episode_sender=episode_sender,
                policy_reader=policy_reader,
                state=state,
                state_lock=state_lock,
                stop=stop,
                policy_class=policy_class,
                policy_data=policy_data,
                **worker_kwargs,
            )

            if use_mp:
                logging.getLogger("async_gym_agents").info(
                    f"Worker has {torch.get_num_threads()} threads and {torch.get_num_interop_threads()} interop threads"
                )

            worker.run()
        finally:
            policy_reader.close()
            if use_mp:
                episode_sender.close()

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

        with self._profiler_main.track(PROFILE_PHASE_POLICY_SERIALIZATION):
            weights_buf = io.BytesIO()
            torch.save(policy.state_dict(), weights_buf)
            weights_bytes = weights_buf.getvalue()

        next_version = self._version + 1
        with self._profiler_main.track(PROFILE_PHASE_POLICY_BROADCAST):
            if self._policy_store is None:
                self._policy_store = SharedPolicyStore.create(
                    initial_version=next_version,
                    initial_payload=weights_bytes,
                    mp_ctx=self.mp_ctx,
                )
            else:
                self._policy_store.publish(next_version, weights_bytes)
        self._version = next_version

        self._logger.debug(f"update policy to the version: {self._version}")

        with self._profiler_main.track("worker_bootstrap"):
            self._init_collect_processes(policy)

    def _init_collect_state(self):
        if self._initialized:
            return

        # SB3 load restores `use_mp` after __init__, so rebuild runtime sync
        # primitives from the current mode before creating worker state.
        self._state_lock = self.mp_ctx.Lock() if self.use_mp else threading.Lock()

        worker_count = len(self.get_indexable_env().env_fns)
        self._episode_transport = EpisodeTransport(
            worker_count=worker_count,
            max_pending_episodes=self.max_episodes_in_buffer,
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
        if self._policy_store is None:
            raise RuntimeError(POLICY_MISSING_INITIAL_SNAPSHOT_ERROR)
        policy_descriptor = self._policy_store.get_descriptor()
        for worker_index, env_func in enumerate(worker_env_fns):
            with patched_env(**worker_env) if self.use_mp else contextlib.nullcontext():
                worker = (self.mp_ctx.Process if self.use_mp else threading.Thread)(
                    target=AsyncAgentInjector._run_worker,
                    kwargs=dict(
                        worker_class=self.get_worker_class(),
                        worker_index=worker_index,
                        env_func=env_func,
                        episode_sender=self._episode_transport.get_sender(worker_index),
                        policy_descriptor=policy_descriptor,
                        state=self._state,
                        state_lock=self._state_lock,
                        stop=self._stop,
                        worker_kwargs=self.get_worker_kwargs(),
                        policy_class=policy_class,
                        policy_data=policy_data,
                        use_mp=self.use_mp,
                        mp_threads=self.mp_threads,
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

        self._episode_transport.close_parent_senders()

        self._initialized_workers = True

    def _excluded_save_params(self) -> List[str]:
        # noinspection PyUnresolvedReferences
        return super()._excluded_save_params() + [
            "_envs",
            "_episode_transport",
            "_policy_store",
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
            "_final_transport_report",
            "_final_policy_report",
        ]

    def _fetch_transitions(self, buffer_was_empty: bool) -> List[Transition]:
        phase = PROFILE_PHASE_WAITING if buffer_was_empty else PROFILE_PHASE_TRANSPORT
        with self._profiler_main.track(phase):
            packet: EpisodePacket = self._episode_transport.receive()

        self._record_policy_lag(packet.policy_version, packet.transition_count)
        with self._profiler_main.track(PROFILE_PHASE_EPISODE_DESERIALIZATION):
            episode_batch = decode_episode_packet(packet)

        reconstruction_start_ns = time.perf_counter_ns()
        transitions = unpack_episode(episode_batch)
        self._profiler_main.record(
            PROFILE_PHASE_TRANSITION_RECONSTRUCTION,
            time.perf_counter_ns() - reconstruction_start_ns,
            count=packet.transition_count,
        )
        return transitions

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
        Each episode is returned as a sequence of transitions, in order, complete,
        and not interleaved with episodes from other workers.
        """
        while len(self._transitions) == 0:
            transport_stats = self._episode_transport.get_stats()
            self._buffer_utilization += transport_stats.pending_episodes
            buffer_was_empty = transport_stats.pending_episodes == 0
            self._buffer_emptiness += 1 if buffer_was_empty else 0
            self._buffer_stat_count += 1

            self._transitions.extend(self._fetch_transitions(buffer_was_empty))

        return self._transitions.popleft()

    def shutdown(self):
        if self._stop is None:
            return

        self._logger.info("Send stop event to all processes")
        self._stop.set()
        if self._episode_transport is not None:
            self._episode_transport.interrupt()
        self._stop = None

        for worker in self._workers:
            if not worker.is_alive():
                continue

            worker.join(timeout=self._worker_join_timeout)

            if not worker.is_alive():
                continue

            if self.use_mp:
                try:
                    worker.kill()
                    worker.join(timeout=self._worker_join_timeout)
                except PermissionError:
                    self._logger.warning("cannot kill process due to permission error")

        if self._episode_transport is not None:
            self._final_transport_report = self._build_transport_report()
            self._episode_transport.shutdown()
            self._episode_transport = None
        if self._policy_store is not None:
            self._final_policy_report = self._build_policy_report()
            self._policy_store.close()
            self._policy_store.unlink()
            self._policy_store = None

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
            buffer_full_push_fraction=self.buffer_full_push_fraction,
            buffer_avg_push_wait_time=self.buffer_avg_push_wait_time,
            discarded_episodes_fraction=self.discarded_episodes_fraction,
            avg_policy_lag=self.avg_policy_lag,
            max_policy_lag=self.max_policy_lag,
            transport_stats=self._build_transport_report(),
            assembly_stats=self._build_assembly_report(),
            policy_stats=self._build_policy_report(),
        )

    def record_profiler_metrics(self) -> None:
        """Record profiler report leaves through the Stable Baselines logger."""
        for metric_name, value in iterate_profiler_metrics(self.get_profiler_report()):
            self.logger.record(
                f"{PROFILER_LOG_PREFIX}/{metric_name}",
                value,
                exclude=PROFILER_EXCLUDED_OUTPUT_FORMATS,
            )

    def _build_transport_report(self) -> Dict[str, Any]:
        if self._episode_transport is None:
            return dict(self._final_transport_report)

        stats = self._episode_transport.get_stats()
        capacity = self._episode_transport.max_pending_episodes
        successful_payload_count = (
            stats.payload_receive_count - stats.payload_receive_timeouts
        )
        successful_payload_ns = (
            stats.payload_receive_ns - stats.payload_receive_timeout_ns
        )
        payload_seconds = successful_payload_ns / NANOSECONDS_PER_SECOND
        return {
            TRANSPORT_PENDING_EPISODES_KEY: stats.pending_episodes,
            TRANSPORT_MAX_PENDING_EPISODES_KEY: stats.max_pending_episodes,
            TRANSPORT_CAPACITY_EPISODES_KEY: capacity,
            TRANSPORT_UTILIZATION_KEY: stats.pending_episodes / capacity,
            TRANSPORT_PENDING_BYTES_KEY: stats.pending_bytes,
            TRANSPORT_MAX_PENDING_BYTES_KEY: stats.max_pending_bytes,
            TRANSPORT_SENT_EPISODES_KEY: stats.sent_episodes,
            TRANSPORT_SENT_BYTES_KEY: stats.sent_bytes,
            TRANSPORT_RECEIVED_EPISODES_KEY: stats.received_episodes,
            TRANSPORT_RECEIVED_BYTES_KEY: stats.received_bytes,
            TRANSPORT_RECEIVE_ATTEMPTS_KEY: stats.receive_attempts,
            TRANSPORT_RECEIVE_TIMEOUTS_KEY: stats.receive_timeouts,
            TRANSPORT_RECEIVE_TIMEOUT_FRACTION_KEY: (
                0.0
                if stats.receive_attempts == 0
                else stats.receive_timeouts / stats.receive_attempts
            ),
            TRANSPORT_RECEIVE_TIMEOUTS_WITH_PENDING_KEY: (
                stats.receive_timeouts_with_pending
            ),
            TRANSPORT_RECEIVE_TIMEOUTS_WITH_PENDING_FRACTION_KEY: (
                0.0
                if stats.receive_timeouts == 0
                else stats.receive_timeouts_with_pending / stats.receive_timeouts
            ),
            TRANSPORT_READINESS_WAIT_PROFILE_KEY: summarize_duration(
                stats.readiness_wait_ns,
                stats.readiness_wait_count,
            ),
            TRANSPORT_READINESS_TIMEOUT_PROFILE_KEY: summarize_duration(
                stats.readiness_timeout_ns,
                stats.readiness_timeouts,
            ),
            TRANSPORT_PAYLOAD_RECEIVE_PROFILE_KEY: summarize_duration(
                successful_payload_ns,
                successful_payload_count,
            ),
            TRANSPORT_PAYLOAD_RECEIVE_TIMEOUT_PROFILE_KEY: summarize_duration(
                stats.payload_receive_timeout_ns,
                stats.payload_receive_timeouts,
            ),
            TRANSPORT_PAYLOAD_MEBIBYTES_PER_SECOND_KEY: (
                0.0
                if payload_seconds == 0.0
                else stats.received_bytes / BYTES_PER_MEBIBYTE / payload_seconds
            ),
            TRANSPORT_PIPE_LATENCY_PROFILE_KEY: summarize_duration(
                stats.pipe_latency_ns,
                stats.pipe_latency_count,
            ),
        }

    def _build_assembly_report(self) -> Dict[str, float | int]:
        return {}

    def _build_policy_report(self) -> Dict[str, int]:
        if self._policy_store is None:
            return dict(self._final_policy_report)

        stats = self._policy_store.get_stats()
        return {
            POLICY_PUBLISHED_VERSION_KEY: stats.published_version,
            POLICY_PAYLOAD_BYTES_KEY: stats.payload_bytes,
            POLICY_SLOT_CAPACITY_BYTES_KEY: stats.slot_capacity_bytes,
            POLICY_PUBLICATION_COUNT_KEY: stats.publication_count,
            POLICY_PUBLICATION_FAILURES_KEY: stats.publication_failures,
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
        """Return the average enqueue wait through the legacy property name."""
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
        """Return the transition-weighted average policy update lag."""
        return (
            0
            if self._policy_lag_count == 0
            else self._policy_lag_total / self._policy_lag_count
        )

    @property
    def max_policy_lag(self) -> int:
        """Return the largest observed policy update lag."""
        return self._policy_lag_max


class InjectorWorkerBase:
    def __init__(
        self,
        worker_index: int,
        env_func: EnvFactory,
        episode_sender: EpisodeSender,
        policy_reader: SharedPolicyReader,
        state: GenericState,
        state_lock: GenericStateLock,
        stop: GenericEvent,
        skip_truncated: bool,
        queue_put_timeout: Optional[float],
        profiler_sync_interval: float,
        policy_class: BasePolicy,
        policy_data: Dict[str, Any],
        **kwargs,
    ):
        self.env = make_venv(env_func())
        self.worker_index = worker_index

        self.policy: BasePolicy | None = None
        self.policy_class = policy_class
        self.policy_data = policy_data

        self._policy_version = None

        self._episode_sender = episode_sender
        self._policy_reader = policy_reader
        self._state = state
        self._state_lock = state_lock
        self._stop = stop

        self._logger = logging.getLogger("Worker")

        self._skip_truncated = skip_truncated
        self._queue_put_timeout = queue_put_timeout
        self._profiler = RuntimeProfiler()
        self._profiler_sync_interval = profiler_sync_interval
        self._last_profiler_sync = time.time()
        self._episode_feeder = EpisodeFeeder(
            sender=self._episode_sender,
            stop=self._stop,
            on_send_complete=self._record_episode_send_result,
        )

    def copy_policy_from_store(self) -> None:
        if self.policy is None:
            # noinspection PyArgumentList
            self.policy = self.policy_class(**self.policy_data)

        retries_before_copy = self._policy_reader.retry_count
        with self._profiler.track(PROFILE_PHASE_POLICY_SNAPSHOT_COPY):
            snapshot = self._policy_reader.read_if_new(self._policy_version)
        retry_count = self._policy_reader.retry_count - retries_before_copy
        if retry_count > 0:
            self._profiler.record(
                PROFILE_PHASE_POLICY_SNAPSHOT_RETRY,
                0,
                count=retry_count,
            )

        if snapshot is None:
            return

        with self._profiler.track(PROFILE_PHASE_POLICY_LOADING):
            weights = torch.load(
                io.BytesIO(snapshot.payload),
                map_location="cpu",
                weights_only=True,
            )
            self.policy.load_state_dict(weights)
            self.policy.to("cpu")
            self.policy.set_training_mode(False)
            self._policy_version = snapshot.version
            self._logger.debug(
                f"policy loaded from shared store: version={self._policy_version}"
            )

    def _put_episode_with_timeout(self, episode):
        with self._profiler.track(PROFILE_PHASE_EPISODE_PACKING):
            episode_batch = pack_episode(episode)
        with self._profiler.track(PROFILE_PHASE_EPISODE_SERIALIZATION):
            packet = encode_episode_batch(
                self.worker_index,
                self._policy_version,
                episode_batch,
            )
        submission = self._episode_feeder.submit(
            packet,
            self._queue_put_timeout,
        )
        with self._state_lock:
            self._state.queue_put_attempts += 1
            self._state.total_queue_put_wait_ns += submission.waiting_ns
            if submission.waiting_ns > 0:
                self._state.full_queue_put_attempts += 1
            if not submission:
                self._state.discarded_episodes += 1
        if submission.waiting_ns > 0:
            self._profiler.record(
                PROFILE_PHASE_WAITING,
                submission.waiting_ns,
            )
        if not submission and not self._stop.is_set():
            self._logger.info("Dropped episode after transport timeout")

    def _record_episode_send_result(self, result: EpisodeSendResult) -> None:
        self._profiler.record(PROFILE_PHASE_TRANSPORT, result.transport_ns)
        if result or self._stop.is_set():
            return
        with self._state_lock:
            self._state.discarded_episodes += 1

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
            self.copy_policy_from_store()

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
            try:
                self._episode_feeder.shutdown()
            finally:
                self._flush_profiler(force=True)
                self.env.close()
                self._logger.info("Generator cycle is completed")

    def generate(self) -> Generator[list[Transition], None, None]:
        raise NotImplementedError()
