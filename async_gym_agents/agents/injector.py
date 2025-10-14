import queue
import threading
from queue import Queue
from typing import Dict, List

import io
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
        self.training_policy_lock = threading.Lock()
        self.training_policy = getattr(self, "policy") if hasattr(self, "policy") else None

        self.rollout_policies: Dict[int, BasePolicy] = {}

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

    def copy_training_policy_to_rollout_policy_completely(self, index: int):
        buffer = io.BytesIO()
        th.save(self.training_policy, buffer)
        buffer.seek(0)
        self.rollout_policies[index] = th.load(buffer, weights_only=False)

    def copy_training_policy_to_rollout_policy_only_weights(self, index: int):
        self.rollout_policies[index].load_state_dict(self.training_policy.state_dict())

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
            self.rollout_policies[index] = None
            self.thread_lookup[thread.name] = index
            self.threads.append(thread)
            self.threads[index].start()

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
