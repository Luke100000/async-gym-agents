import io
import logging
import queue
import threading
from queue import Queue
from typing import Dict, List

import torch as th
from stable_baselines3.common.base_class import BasePolicy

from async_gym_agents.envs.multi_env import IndexableMultiEnv

logger = logging.getLogger("async_gym_agents")


class AsyncAgentInjector:
    def __init__(
        self,
        max_steps_in_buffer: int,
        skip_truncated: bool = False,
        timeout: float = 1.0,
    ):
        self._buffer_utilization = 0.0
        self._buffer_emptiness = 0.0
        self._buffer_stat_count = 0

        self.running = True
        self.initialized = False
        self.threads = []
        self.thread_lookup = {}

        self.total_episodes = 0
        self.discarded_episodes = 0
        self.skip_truncated = skip_truncated
        self.timeout = timeout

        # The larger the queue, the less wait times, but the more outdated the policies training data is
        self.queue = Queue(max_steps_in_buffer)
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
            "episode_lock",
            "policy_lock",
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

    def _initialize_threads(self):
        self.threads = []
        for index in range(self.get_indexable_env().real_n_envs):
            thread = threading.Thread(
                target=self._collector_loop,
                args=(index,),
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
                self.queue.get(block=False)
                self.queue.put(episode, block=False)
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
