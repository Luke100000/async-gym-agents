import io
import logging
import multiprocessing
import queue
import threading
from types import SimpleNamespace

import torch

from async_gym_agents.agents.injector import AsyncAgentInjector, InjectorWorkerBase
from async_gym_agents.profiler import RuntimeProfiler


class DummyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.0]))

    def set_training_mode(self, mode: bool):
        self.train(mode)


def dump_weights(value: float) -> bytes:
    policy = DummyPolicy()
    with torch.no_grad():
        policy.weight.fill_(value)

    buffer = io.BytesIO()
    torch.save(policy.state_dict(), buffer)
    return buffer.getvalue()


def build_worker() -> InjectorWorkerBase:
    worker = InjectorWorkerBase.__new__(InjectorWorkerBase)
    worker.policy = None
    worker.policy_class = DummyPolicy
    worker.policy_data = {}
    worker._policy_version = None
    worker._episode_queue = queue.Queue()
    worker._update_queue = queue.Queue()
    worker._state = SimpleNamespace(
        total_episodes=0,
        discarded_episodes=0,
        queue_put_attempts=0,
        full_queue_put_attempts=0,
        total_queue_put_wait_ns=0,
        policy_sync_count=0,
    )
    worker._state_lock = threading.Lock()
    worker._stop = threading.Event()
    worker._logger = logging.getLogger("test")
    worker._skip_truncated = False
    worker._queue_put_timeout = 1.0
    worker._profiler = RuntimeProfiler()
    return worker


def test_push_policy_update_clears_stale_entries():
    injector = AsyncAgentInjector.__new__(AsyncAgentInjector)
    update_queue = queue.Queue()
    update_queue.put((1, b"old-1"))
    update_queue.put((2, b"old-2"))
    injector._update_queues = [update_queue]

    injector._push_policy_update(3, b"latest")

    assert update_queue.qsize() == 1
    assert update_queue.get_nowait() == (3, b"latest")


def test_try_fetch_transition_is_non_blocking_and_drains_cached_episode():
    injector = AsyncAgentInjector.__new__(AsyncAgentInjector)
    injector._episode_queue = queue.Queue()
    injector._episode_queue.put(["a", "b"])
    injector._transitions = []
    injector._buffer_utilization = 0.0
    injector._buffer_emptiness = 0.0
    injector._buffer_stat_count = 0
    injector._profiler_main = RuntimeProfiler()

    assert injector.try_fetch_transition() == "a"
    assert injector.try_fetch_transition() == "b"
    assert injector.try_fetch_transition() is None
    assert injector.buffer_emptyness == 1 / 2


def test_copy_policy_from_queue_uses_latest_update():
    worker = build_worker()
    worker._update_queue.put((1, dump_weights(1.0)))
    worker._update_queue.put((2, dump_weights(2.0)))

    worker.copy_policy_from_queue()

    assert worker._policy_version == 2
    assert torch.equal(worker.policy.weight.detach(), torch.tensor([2.0]))
    assert worker.policy.training is False


def test_copy_policy_from_queue_blocks_until_first_update_arrives():
    worker = build_worker()

    def delayed_update():
        threading.Event().wait(0.2)
        worker._update_queue.put((1, dump_weights(1.0)))

    update_thread = threading.Thread(target=delayed_update)
    update_thread.start()
    try:
        worker.copy_policy_from_queue(block=True)
    finally:
        update_thread.join()

    assert worker._policy_version == 1
    assert torch.equal(worker.policy.weight.detach(), torch.tensor([1.0]))
    assert worker.policy.training is False


def test_put_episode_tracks_full_buffer_push_attempts():
    worker = build_worker()
    worker._episode_queue = queue.Queue(maxsize=1)
    worker._episode_queue.put("already-full")
    worker._queue_put_timeout = 0.0

    worker._put_episode_with_timeout("new-episode")

    assert worker._state.queue_put_attempts == 1
    assert worker._state.full_queue_put_attempts == 1
    assert worker._state.discarded_episodes == 1
    assert worker._state.total_queue_put_wait_ns == 0


def test_buffer_full_push_fraction_uses_shared_counters():
    injector = AsyncAgentInjector.__new__(AsyncAgentInjector)
    injector._state = SimpleNamespace(
        queue_put_attempts=4,
        full_queue_put_attempts=3,
        total_queue_put_wait_ns=200_000_000,
    )

    assert injector.buffer_full_push_fraction == 0.75
    assert injector.buffer_avg_push_time == 0.05


def test_init_collect_state_rebuilds_mp_lock_after_load_like_state():
    injector = AsyncAgentInjector.__new__(AsyncAgentInjector)
    injector.max_episodes_in_buffer = 1
    injector.use_mp = True
    injector.mp_ctx = multiprocessing.get_context("spawn")
    injector._episode_queue = None
    injector._update_queues = []
    injector._transitions = []
    injector._manager = None
    injector._state = None
    stale_lock = threading.Lock()
    injector._state_lock = stale_lock
    injector._version = 0
    injector._stop = None
    injector._initialized = False
    injector._initialized_workers = False
    injector._workers = []
    injector._buffer_utilization = 0.0
    injector._buffer_emptiness = 0.0
    injector._buffer_stat_count = 0
    injector._profiler_main = RuntimeProfiler()
    injector._logger = logging.getLogger("test")
    injector.get_indexable_env = lambda: SimpleNamespace(env_fns=[object()])

    injector._init_collect_state()
    try:
        assert injector._state_lock is not stale_lock
        assert injector._state_lock.__class__.__module__.startswith("multiprocessing")
    finally:
        injector.shutdown()
