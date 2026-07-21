import multiprocessing
import queue
import signal
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partial
from unittest.mock import Mock, patch

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from async_gym_agents.agents.async_agent import get_injected_agent
from async_gym_agents.data_classes import OffPolicyTransition, OnPolicyTransition
from async_gym_agents.envs.buggy_lunar_lander import BuggyLunarLander
from async_gym_agents.envs.multi_env import IndexableMultiEnv
from async_gym_agents.episode_codec import encode_episode_batch, pack_episode
from async_gym_agents.episode_transport import EpisodeFeeder, EpisodeTransport
from async_gym_agents.policy_transport import SharedPolicyReader, SharedPolicyStore

PROCESSES = 8
DIRECT_TRANSPORT_PAYLOAD_BYTES = 8 * 1024 * 1024
DIRECT_TRANSPORT_PROCESS_TIMEOUT_SECONDS = 5.0
TEST_EPISODE_SEND_TIMEOUT_SECONDS = 1.0
POLICY_TEST_TIMEOUT_SECONDS = 5.0


class LoggingCallback(BaseCallback):
    """Represent the external logging callback contract used by batching tests."""

    def __init__(self, connector, metric_aggregator):
        super().__init__()
        self.connector = connector
        self.logging_frequency = 1
        self.log_distributions = False
        self.episode_counter = {}
        self.metric_aggregator = metric_aggregator
        self.step_call_count = 0

    def _on_step(self) -> bool:
        self.step_call_count += 1
        return True


class MetricAggregator:
    """Represent the framework metric aggregator state used by callback adapters."""

    def __init__(self, aggregate_distributions=False):
        self.aggregate_distributions = aggregate_distributions
        self.episode_reward = None
        self.episode_rewards = {}
        self.episode_actions = None
        self.episode_step_metrics = {}
        self.episode_end_reasons = {}
        self.aggregate_step = Mock()
        self.log_aggregated_metrics = Mock()
        self.reset_multi_episode_trackers = Mock(
            side_effect=self.reset_aggregated_metrics
        )

    def reset_aggregated_metrics(self, agent_index):
        """Reset the same multi-episode trackers as the framework aggregator."""
        self.episode_rewards[agent_index] = []
        for per_agent_values in self.episode_step_metrics.values():
            per_agent_values[agent_index] = []
        if self.episode_actions:
            self.episode_actions[agent_index] = []


class SavingCallback(BaseCallback):
    """Represent the external checkpoint callback contract used by tests."""

    def __init__(self, agent, connector, checkpoint_frequency):
        super().__init__()
        self.agent = agent
        self.connector = connector
        self.checkpoint_frequency = checkpoint_frequency
        self.next_upload = checkpoint_frequency
        self.step_call_count = 0

    def _on_step(self) -> bool:
        self.step_call_count += 1
        return True


class ExperimentPruningCallback(BaseCallback):
    """Represent the external reward-pruning callback contract used by tests."""

    def __init__(self, episode_reward_threshold, pruning_start_at, reward_window):
        super().__init__()
        self.episode_reward_threshold = episode_reward_threshold
        self.pruning_start_at = pruning_start_at
        self.episode_reward = None
        self.episode_rewards = deque(maxlen=reward_window)
        self.step_call_count = 0

    def _on_step(self) -> bool:
        self.step_call_count += 1
        return True


class ResetInfoCallback(BaseCallback):
    """Represent the external reset-info callback contract used by tests."""

    def __init__(self, connector):
        super().__init__()
        self.connector = connector
        self.episode_counter = {}
        self.first_step_tracker = []
        self.step_call_count = 0

    def _on_step(self) -> bool:
        self.step_call_count += 1
        return True


class AsyncSBUtilizationLoggingCallback(BaseCallback):
    """Represent the external terminal-only utilization callback contract."""

    def __init__(self):
        super().__init__()
        self.logging_frequency = 1
        self.shared_episode_counter = 0
        self.step_call_count = 0
        self.terminal_dones = []

    def _on_step(self) -> bool:
        self.step_call_count += 1
        self.terminal_dones.append(self.locals["dones"].copy())
        return True


class StepCountingCallback(BaseCallback):
    """Represent an unrecognized callback that requires SB3 step dispatch."""

    def __init__(self):
        super().__init__()
        self.step_call_count = 0

    def _on_step(self) -> bool:
        self.step_call_count += 1
        return True


def send_episode_and_signal(
    sender,
    packet,
    stop,
    send_started,
    send_completed,
):
    """Send one episode and expose when its synchronous transfer completes."""
    send_started.set()
    if sender.send(packet, stop, DIRECT_TRANSPORT_PROCESS_TIMEOUT_SECONDS):
        send_completed.set()


def read_policy_snapshot_in_process(descriptor, result_queue):
    """Read one shared policy snapshot in a spawned child process."""
    reader = SharedPolicyReader(descriptor)
    try:
        snapshot = reader.read_if_new(None)
        result_queue.put((snapshot.version, snapshot.payload))
    finally:
        reader.close()


@pytest.fixture
def shared_policy_store():
    """Create a spawn-compatible shared policy store with an initial snapshot."""
    store = SharedPolicyStore.create(
        initial_version=7,
        initial_payload=b"initial-policy-padding",
        mp_ctx=multiprocessing.get_context("spawn"),
    )
    yield store
    store.close()
    store.unlink()


@pytest.fixture
def external_logging_callback():
    """Create the external logging callback shape with mocked output boundaries."""
    return LoggingCallback(
        connector=Mock(),
        metric_aggregator=MetricAggregator(),
    )


@pytest.fixture
def external_legacy_logging_callback():
    """Create a logging callback whose aggregator only accepts individual steps."""
    metric_aggregator = Mock(
        spec=[
            "aggregate_step",
            "log_aggregated_metrics",
            "reset_multi_episode_trackers",
        ]
    )
    return LoggingCallback(
        connector=Mock(),
        metric_aggregator=metric_aggregator,
    )


@pytest.fixture
def external_saving_callback():
    """Create the external checkpoint callback shape with mocked boundaries."""
    return SavingCallback(
        agent=Mock(),
        connector=Mock(),
        checkpoint_frequency=2,
    )


@pytest.fixture
def external_pruning_callback():
    """Create a pruning callback whose second short episode stops training."""
    return ExperimentPruningCallback(
        episode_reward_threshold=4.0,
        pruning_start_at=0,
        reward_window=2,
    )


@pytest.fixture
def external_reset_info_callback():
    """Create the external reset-info callback shape with a mocked connector."""
    return ResetInfoCallback(connector=Mock())


@pytest.fixture
def external_utilization_callback():
    """Create the external terminal-only utilization callback shape."""
    return AsyncSBUtilizationLoggingCallback()


@pytest.fixture
def unrecognized_step_callback():
    """Create a callback that must retain Stable Baselines step semantics."""
    return StepCountingCallback()


@pytest.fixture
def shared_policy_reader(shared_policy_store):
    """Open a worker-style reader for the shared policy store."""
    reader = SharedPolicyReader(shared_policy_store.get_descriptor())
    yield reader
    reader.close()


@pytest.fixture
def paused_policy_copy(shared_policy_store):
    """Pause a reader after metadata capture to force an overwritten-slot retry."""
    copy_started = threading.Event()
    allow_copy = threading.Event()
    original_copy = SharedPolicyReader._copy_payload

    def copy_payload_after_signal(reader, slot_index, payload_size):
        copy_started.set()
        if not allow_copy.wait(POLICY_TEST_TIMEOUT_SECONDS):
            raise TimeoutError("Timed out waiting to resume shared policy copy")
        return original_copy(reader, slot_index, payload_size)

    with patch.object(
        SharedPolicyReader,
        "_copy_payload",
        new=copy_payload_after_signal,
    ):
        reader = SharedPolicyReader(shared_policy_store.get_descriptor())
        with ThreadPoolExecutor(max_workers=2) as executor:
            yield shared_policy_store, reader, copy_started, allow_copy, executor
        reader.close()


@pytest.fixture
def failing_policy_publication(shared_policy_store):
    """Make the inactive shared-memory slot fail during its payload copy."""
    with patch.object(
        SharedPolicyStore,
        "_copy_payload",
        side_effect=OSError("injected shared-memory write failure"),
    ):
        yield shared_policy_store


@pytest.fixture
def taxi_multi_env():
    """Fixture for creating a multi-environment with discrete action space environment."""
    return IndexableMultiEnv([partial(gym.make, "Taxi-v3") for _ in range(PROCESSES)])


@pytest.fixture
def lunar_lander_multi_env():
    """Fixture for creating a multi-environment with continuous action space environment."""
    return IndexableMultiEnv(
        [partial(gym.make, "LunarLanderContinuous-v3") for _ in range(PROCESSES)]
    )


@pytest.fixture
def short_cartpole_multi_env():
    """Create workers whose episodes always end after two transitions."""
    return IndexableMultiEnv(
        [partial(gym.make, "CartPole-v1", max_episode_steps=2) for _ in range(2)]
    )


@pytest.fixture
def short_episode_on_policy_agent(short_cartpole_multi_env):
    """Create an on-policy agent whose workers complete two-transition episodes."""
    agent = get_injected_agent(PPO)(
        "MlpPolicy",
        short_cartpole_multi_env,
        batch_size=2,
        device="cpu",
        n_epochs=1,
        n_steps=3,
    )
    yield agent
    agent.shutdown()


@pytest.fixture
def short_episode_on_policy_mp_agent(short_cartpole_multi_env):
    """Create an on-policy agent with short multiprocessing episodes."""
    agent = get_injected_agent(PPO)(
        "MlpPolicy",
        short_cartpole_multi_env,
        batch_size=2,
        device="cpu",
        n_epochs=1,
        n_steps=3,
        use_mp=True,
    )
    yield agent
    agent.shutdown()


@pytest.fixture
def fixed_terminal_value_policy():
    """Create a policy boundary returning a fixed terminal state value."""
    policy = Mock()
    policy.obs_to_tensor.return_value = (torch.zeros((1, 2)), None)
    policy.predict_values.return_value = torch.tensor([2.0])
    return policy


@pytest.fixture
def initialized_on_policy_agent():
    """Create an on-policy agent with initialized transport state and no workers."""
    env = IndexableMultiEnv([partial(gym.make, "Taxi-v3")])
    agent = get_injected_agent(PPO)(
        "MlpPolicy",
        env,
        batch_size=2,
        device="cpu",
        n_steps=2,
    )
    agent._init_collect_state()
    yield agent
    agent.shutdown()


@pytest.fixture
def initialized_off_policy_agent():
    """Create an off-policy agent with initialized transport and no workers."""
    env = IndexableMultiEnv([partial(gym.make, "Pendulum-v1")])
    agent = get_injected_agent(SAC)(
        "MlpPolicy",
        env,
        device="cpu",
    )
    agent._init_collect_state()
    yield agent
    agent.shutdown()


@pytest.fixture
def on_policy_agent_with_signaled_worker(initialized_on_policy_agent):
    """Attach a worker terminated by a cross-platform fatal signal."""
    worker = Mock()
    worker.exitcode = -signal.SIGTERM
    worker.is_alive.return_value = False
    initialized_on_policy_agent._workers = [worker]
    return initialized_on_policy_agent


@pytest.fixture
def on_policy_episode():
    """Create a complete two-step on-policy episode with sparse terminal info."""
    return [
        OnPolicyTransition(
            actions=np.array([[0]]),
            values=np.array([0.25], dtype=np.float32),
            log_probs=np.array([-0.5], dtype=np.float32),
            last_obs=np.array([[1.0, 2.0]], dtype=np.float32),
            new_obs=np.array([[2.0, 3.0]], dtype=np.float32),
            rewards=np.array([1.0], dtype=np.float32),
            dones=np.array([False]),
            last_dones=np.array([True]),
            infos=[{}],
            reset_infos=[{}],
        ),
        OnPolicyTransition(
            actions=np.array([[1]]),
            values=np.array([0.5], dtype=np.float32),
            log_probs=np.array([-0.25], dtype=np.float32),
            last_obs=np.array([[2.0, 3.0]], dtype=np.float32),
            new_obs=np.array([[0.0, 0.0]], dtype=np.float32),
            rewards=np.array([2.0], dtype=np.float32),
            dones=np.array([True]),
            last_dones=np.array([False]),
            infos=[
                {
                    "TimeLimit.truncated": True,
                    "terminal_observation": np.array([3.0, 4.0], dtype=np.float32),
                }
            ],
            reset_infos=[{"seed": 7}],
        ),
    ]


@pytest.fixture
def on_policy_episode_with_metrics(on_policy_episode):
    """Add step metrics, metadata, and an end reason to a complete episode."""
    episode = list(on_policy_episode)
    episode[0] = replace(
        episode[0],
        infos=[
            {
                "meta_settings": {"map": "test"},
                "step_metric_speed": 2.0,
            }
        ],
    )
    terminal_info = dict(episode[1].infos[0])
    terminal_info.update(
        {
            "episode_end_reason": "TIMEOUT",
            "step_metric_speed": 4.0,
        }
    )
    episode[1] = replace(episode[1], infos=[terminal_info])
    return episode


@pytest.fixture
def on_policy_episode_with_changed_metadata(on_policy_episode_with_metrics):
    """Change one metadata value while preserving the metric-bearing episode."""
    episode = list(on_policy_episode_with_metrics)
    initial_info = dict(episode[0].infos[0])
    initial_info["meta_settings"] = {"map": "changed"}
    episode[0] = replace(episode[0], infos=[initial_info])
    return episode


@pytest.fixture
def off_policy_episode():
    """Create a complete two-step off-policy episode."""
    return [
        OffPolicyTransition(
            buffer_actions=np.array([[0.1]], dtype=np.float32),
            last_obs=np.array([[1.0]], dtype=np.float32),
            new_obs=np.array([[2.0]], dtype=np.float32),
            rewards=np.array([1.0], dtype=np.float32),
            dones=np.array([False]),
            infos=[{}],
            reset_infos=[{}],
        ),
        OffPolicyTransition(
            buffer_actions=np.array([[0.2]], dtype=np.float32),
            last_obs=np.array([[2.0]], dtype=np.float32),
            new_obs=np.array([[3.0]], dtype=np.float32),
            rewards=np.array([2.0], dtype=np.float32),
            dones=np.array([True]),
            infos=[{}],
            reset_infos=[{}],
        ),
    ]


@pytest.fixture
def on_policy_packet(on_policy_episode):
    """Encode a representative on-policy episode for transport tests."""
    return encode_episode_batch(0, 1, pack_episode(on_policy_episode))


@pytest.fixture
def off_policy_packet(off_policy_episode):
    """Encode an off-policy episode without a policy version."""
    return encode_episode_batch(0, None, pack_episode(off_policy_episode))


@pytest.fixture
def active_direct_episode_send(on_policy_packet):
    """Start one process sending a payload larger than an operating-system pipe."""
    context = multiprocessing.get_context("spawn")
    transport = EpisodeTransport(
        worker_count=1,
        max_pending_episodes=1,
        use_mp=True,
        mp_ctx=context,
    )
    stop = context.Event()
    send_started = context.Event()
    send_completed = context.Event()
    packet = replace(
        on_policy_packet,
        payload=bytes(DIRECT_TRANSPORT_PAYLOAD_BYTES),
    )
    process = context.Process(
        target=send_episode_and_signal,
        args=(
            transport.get_sender(0),
            packet,
            stop,
            send_started,
            send_completed,
        ),
    )
    process.start()
    assert send_started.wait(DIRECT_TRANSPORT_PROCESS_TIMEOUT_SECONDS)

    yield transport, packet, process, stop, send_completed

    stop.set()
    transport.shutdown()
    process.join(DIRECT_TRANSPORT_PROCESS_TIMEOUT_SECONDS)
    if process.is_alive():
        process.kill()
        process.join(DIRECT_TRANSPORT_PROCESS_TIMEOUT_SECONDS)


@pytest.fixture
def active_episode_feeder(on_policy_packet):
    """Create a feeder blocked on payloads larger than its operating-system pipe."""
    transport = EpisodeTransport(
        worker_count=1,
        max_pending_episodes=2,
        use_mp=False,
    )
    stop = threading.Event()
    completed_sends = queue.Queue()
    feeder = EpisodeFeeder(
        sender=transport.get_sender(0),
        stop=stop,
        on_send_complete=completed_sends.put,
    )
    packet = replace(
        on_policy_packet,
        payload=bytes(DIRECT_TRANSPORT_PAYLOAD_BYTES),
    )

    yield transport, feeder, packet, stop, completed_sends

    stop.set()
    transport.interrupt()
    feeder.shutdown()
    transport.shutdown()


@pytest.fixture
def on_policy_rollout_buffer():
    """Create a rollout buffer matching the representative on-policy episode."""
    return RolloutBuffer(
        buffer_size=2,
        observation_space=gym.spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(2,),
            dtype=np.float32,
        ),
        action_space=gym.spaces.Discrete(2),
        device="cpu",
        gae_lambda=0.95,
        gamma=0.99,
        n_envs=1,
    )


@pytest.fixture
def enqueue_episode_packet():
    """Return a helper that sends a packet through an initialized agent transport."""

    def enqueue(agent, packet):
        """Send one packet and assert that the bounded transport accepted it."""
        sender = agent._episode_transport.get_sender(packet.worker_index)
        assert sender.send(packet, agent._stop, TEST_EPISODE_SEND_TIMEOUT_SECONDS)

    return enqueue


def get_buggy_env(buggy: bool) -> gym.Env:
    """Environment factory for an env returning truncated episodes."""
    return Monitor(
        BuggyLunarLander(
            crash_probability=0.01 if buggy else 0,
            time_limit=1000,
        )
    )
