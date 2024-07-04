import queue
import threading
from queue import Queue

from async_gym_agents.envs.multi_env import IndexableMultiEnv


class AsyncAgentInjector:

    def __init__(self, max_steps_in_buffer: int = 8):
        self._buffer_utilization = 0.0
        self._buffer_emptiness = 0.0
        self._buffer_stat_count = 0

        self.running = True
        self.initialized = False

        # The larger the queue, the less wait times, but the more outdated the policies training data is
        self.queue = Queue(max_steps_in_buffer)
        self.lock = threading.Lock()

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
            self.threads.append(thread)
            self.threads[index].start()

    def fetch_transition(self):
        self._buffer_utilization += self.queue.qsize()
        self._buffer_emptiness += 1 if self.queue.empty() else 0
        self._buffer_stat_count += 1
        return self.queue.get()

    @property
    def buffer_utilization(self) -> float:
        return self._buffer_utilization / self._buffer_stat_count

    @property
    def buffer_emptyness(self) -> float:
        return self._buffer_emptiness / self._buffer_stat_count

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
            with self.lock:
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
