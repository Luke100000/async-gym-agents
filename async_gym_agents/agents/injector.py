import io
import logging
import multiprocessing
import queue
import threading
from functools import partial
from multiprocessing.context import Process as MPProcess
from multiprocessing.managers import Namespace
from multiprocessing.queues import Queue as MPQueue
from multiprocessing.synchronize import Event as MPEvent
from types import SimpleNamespace
from typing import Any, Dict, Generator, List, Optional, Type, TypeAlias

import torch
import torch as th
from stable_baselines3.common.base_class import BasePolicy

from async_gym_agents.envs.multi_env import IndexableMultiEnv
from async_gym_agents.types import EnvFactory, EnvFactoryList, Transition
from async_gym_agents.utils import identity

logger = logging.getLogger("async_gym_agents")


GenericState: TypeAlias = SimpleNamespace | Namespace
GenericEvent: TypeAlias = MPEvent | threading.Event
GenericQueue: TypeAlias = MPQueue | queue.Queue
GenericWorker: TypeAlias = MPProcess | threading.Thread


class AsyncAgentInjector:
    def __init__(
        self,
        envs: Optional[EnvFactoryList],
        max_episodes_in_buffer: int = 8,
        use_mp: bool = False,
    ):
        self._envs = envs
        self.max_episodes_in_buffer = max_episodes_in_buffer
        self.use_mp = use_mp

        # shared memory
        self._episode_queue: GenericQueue | None = None
        self._transitions: Optional[List[Transition]] = []

        # shared object (!)
        self._manager: Optional[multiprocessing.Manager] = None
        self._state: GenericState | None = None
        self._version = 0

        self._stop: GenericEvent | None = None

        self._initialized = False
        self._workers: List[GenericWorker] = []

        # 1 minute wait for a new message in the queue
        self._queue_get_timeout = 60.0
        # 2-minute wait before try to kill the process
        self._worker_join_timeout = 120.0

        # Metrics
        self._buffer_utilization = 0.0
        self._buffer_emptiness = 0.0
        self._buffer_stat_count = 0

    def _episode_generator(self, index: int):
        raise NotImplementedError()

    @staticmethod
    def _run_worker(
        worker_class: Type["InjectorWorkerBase"],
        env_func: EnvFactory,
        episode_queue: GenericQueue,
        state: GenericState,
        stop: GenericEvent,
        worker_kwargs: Dict[str, Any],
    ):
        worker = worker_class(
            env_func=env_func,
            episode_queue=episode_queue,
            state=state,
            stop=stop,
            **worker_kwargs,
        )
        worker.run()

        # Close the queue
        episode_queue.close()
        episode_queue.cancel_join_thread()

    def get_worker_class(self) -> Type["InjectorWorkerBase"]:
        raise NotImplementedError()

    def get_worker_kwargs(self) -> Dict[str, Any]:
        raise NotImplementedError()

    def init_collect_process(self):
        if self._initialized:
            return

        env_funcs = self._envs
        # noinspection PyUnresolvedReferences
        if env_funcs is None and isinstance(self.env, IndexableMultiEnv):
            # noinspection PyUnresolvedReferences
            env_funcs = [partial(identity, e) for e in self.env.envs]

        if env_funcs is None:
            raise ValueError(
                "Multi-processed injectors must have the envs constructor set."
            )

        # Environment queue
        if self._episode_queue is None:
            self._episode_queue = multiprocessing.Queue(
                maxsize=self.max_episodes_in_buffer
            )

        # Shared state for policy and metrics
        if self._state is None:
            self._manager = multiprocessing.Manager() if self.use_mp else None
            self._state = (
                self._manager.Namespace() if self.use_mp else SimpleNamespace()
            )

            self._state.total_episodes = 0
            self._state.discarded_episodes = 0

        # Stop signal
        if self._stop is None:
            self._stop = multiprocessing.Event() if self.use_mp else threading.Event()

        # Start workers
        for env_func in env_funcs:
            worker = (multiprocessing.Process if self.use_mp else threading.Thread)(
                target=AsyncAgentInjector._run_worker,
                kwargs=dict(
                    worker_class=self.get_worker_class(),
                    env_func=env_func,
                    episode_queue=self._episode_queue,
                    state=self._state,
                    stop=self._stop,
                    worker_kwargs=self.get_worker_kwargs(),
                ),
            )
            worker.start()

            self._workers.append(worker)

        self._initialized = True

    def _excluded_save_params(self) -> List[str]:
        # noinspection PyUnresolvedReferences
        return super()._excluded_save_params() + [
            "_envs",
            "_episode_queue",
            "_transitions",
            "_manager",
            "_state",
            "_version",
            "_stop",
            "_initialized",
            "_workers",
        ]

    def _fetch_transitions(self) -> List[Transition]:
        try:
            return self._episode_queue.get(timeout=self._queue_get_timeout)
        except queue.Empty:
            return []

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

    def pre_collect_preparation(self, policy: BasePolicy):
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

    def shutdown(self):
        logger.info("send stop event to all processes")
        self._stop.set()
        self._stop = None

        for proc in self._workers:
            if not proc.is_alive():
                continue

            proc.join(timeout=self._worker_join_timeout)

            try:
                proc.kill()
            except PermissionError:
                logger.warning("cannot kill process due to permission error")

        # close the queue
        if self._episode_queue is not None:
            self._episode_queue.close()
            self._episode_queue.cancel_join_thread()
            self._episode_queue = None

        # release a shared object: manager
        if self._manager is not None:
            self._manager.shutdown()
            self._manager = None
            self._state = None

        self._initialized = False

        logger.info("stop manager")

    @property
    def buffer_utilization(self) -> float:
        return (
            0
            if self._state.buffer_stat_count == 0
            else self._state.buffer_utilization / self._state.buffer_stat_count
        )

    @property
    def buffer_emptyness(self) -> float:
        return (
            0
            if self._state.buffer_stat_count == 0
            else self._state.buffer_emptiness / self._state.buffer_stat_count
        )

    @property
    def discarded_episodes_fraction(self) -> float:
        return (
            0
            if self._state.total_episodes == 0
            else self._state.discarded_episodes / self._state.total_episodes
        )


class InjectorWorkerBase:
    def __init__(
        self,
        env_func: EnvFactory,
        episode_queue: multiprocessing.Queue,
        state: Namespace,
        stop: multiprocessing.Event,
        **kwargs,
    ):
        self.env = IndexableMultiEnv._make_venv(env_func())

        self.policy = None
        self._policy_version = None

        self._episode_queue = episode_queue
        self._state = state
        self._stop = stop

        self._logger = logging.getLogger("Worker")

        self._skip_truncated = True  # skip_truncated # TODO
        self._timeout = 1.0  # timeout # TODO

    def copy_policy_from_state(self):
        version = self._state
        if self._policy_version != version:
            # load state
            data = io.BytesIO(self._state.policy)
            self.policy = torch.load(data, weights_only=False, map_location="cpu")

            # turn off the train mode
            self.policy.set_training_mode(False)

            self._policy_version = version

    def run(self):
        self.copy_policy_from_state()

        for episode in self.generate():
            # Keeps track of truncated episodes and optionally removes them
            self._state.total_episodes += 1
            if episode[-1].infos[0]["TimeLimit.truncated"] and self._skip_truncated:
                self._state.discarded_episodes += 1
                continue

            # Feeds the episodes into the queue
            try:
                self._episode_queue.put(episode, block=True, timeout=self._timeout)
            except queue.Full:
                try:
                    # Try to drop from the start to keep the more recent episodes
                    self._episode_queue.get(block=False)
                    self._episode_queue.put(episode, block=False)
                except queue.Full:
                    pass
                self._state.discarded_episodes += 1
                logger.info("Dropped episode due to buffer full")

            # Shut down
            if self._stop.is_set():
                break

        self._logger.info("Generator cycle is completed")

        # stop environment
        self.env.close()

    def generate(self) -> Generator[list[Transition], None, None]:
        raise NotImplementedError()
