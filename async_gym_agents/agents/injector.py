import queue
import threading
from queue import Queue
from typing import List

import torch as th
from async_gym_agents.envs.multi_env import IndexableMultiEnv
from stable_baselines3.common.base_class import BasePolicy


class AsyncAgentInjector:
    def __init__(self, max_steps_in_buffer: int = 8, skip_truncated: bool = False):
        self._buffer_utilization = 0.0
        self._buffer_emptiness = 0.0
        self._buffer_stat_count = 0

        self.running = True
        self.initialized = False
        self.threads = []
        self.thread_lookup = {}

        self.total_episodes = 0
        self.skipped_episodes = 0
        self.skip_truncated = skip_truncated

        # The larger the queue, the less wait times, but the more outdated the policies training data is
        self.queue = Queue(max_steps_in_buffer)
        self.episode_lock = threading.Lock()

        # The policy itself is rarely thread-safe
        self.policy_lock = threading.Lock()

        # One lock per agent, co-indexed with self.threads
        self.rollout_policy_locks: List[Lock] = []

        # One policy per agent, copied from self.policy after each training
        self.rollout_policies: List[BasePolicy] = []

        self.training_policy = getattr(self, "policy") if hasattr(self, "policy") else None


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
        if self.rollout_policy_locks and not self.rollout_policies:
            self._copy_rollout_policies_from_training_policy()

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
            self.rollout_policy_locks.append(threading.Lock())
            if self.training_policy:
                self._copy_rollout_policies_from_training_policy()
            thread = threading.Thread(
                target=self._collector_loop,
                args=(index,),
            )
            self.thread_lookup[thread.name] = index
            self.threads.append(thread)
            self.threads[index].start()

    def _copy_rollout_policies_from_training_policy(self):
        th.save(self.training_policy, "temp_policy.pth")
        self.rollout_policies = [
            th.load("temp_policy.pth", weights_only=False) for _ in self.rollout_policy_locks
        ]

    def fetch_transition(self):
        self._buffer_utilization += self.queue.qsize()
        self._buffer_emptiness += 1 if self.queue.empty() else 0
        self._buffer_stat_count += 1
        return self.queue.get()

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
    def truncated_episodes_fraction(self) -> float:
        return (
            0
            if self.total_episodes == 0
            else self.skipped_episodes / self.total_episodes
        )

    def _episode_generator(self, index: int):
        raise NotImplementedError()

    def _collector_loop(
        self,
        index: int,
    ):
        """
        Batch-inserts transitions whenever a episode is done.
        """
        for episode in self._episode_generator(index):
            # Keeps track of truncated episodes and optionally removes them
            self.total_episodes += 1
            if episode[-1].infos[0]["TimeLimit.truncated"]:
                self.skipped_episodes += 1
                if self.skip_truncated:
                    continue

            # Feeds the episodes into the queue
            with self.episode_lock:
                for transition in episode:
                    while self.running:
                        try:
                            self.queue.put(transition, block=True, timeout=1)
                            break
                        except queue.Full:
                            pass

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
