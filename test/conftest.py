import copy
import multiprocessing
import queue
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
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from async_gym_agents import constants
from async_gym_agents.agents.async_agent import get_injected_agent
from async_gym_agents.callback_batching import (
    CallbackBatchDispatcher,
    resolve_episode_action_field,
    resolve_episode_reward_field,
)
from async_gym_agents.data_classes import (
    EpisodeCallbackContext,
    OffPolicyTransition,
    OnPolicyTransition,
)
from async_gym_agents.enums import EpisodeKind
from async_gym_agents.envs.buggy_lunar_lander import BuggyLunarLander
from async_gym_agents.envs.multi_env import IndexableMultiEnv
from async_gym_agents.episode_codec import (
    encode_episode_batch,
    get_episode_infos,
    get_episode_reset_infos,
    pack_episode,
    slice_episode_field,
)
from async_gym_agents.episode_transport import EpisodeFeeder, EpisodeTransport
from async_gym_agents.policy_transport import SharedPolicyReader, SharedPolicyStore

PROCESSES = 8
DIRECT_TRANSPORT_PAYLOAD_BYTES = 8 * 1024 * 1024
DIRECT_TRANSPORT_PROCESS_TIMEOUT_SECONDS = 5.0
TEST_EPISODE_SEND_TIMEOUT_SECONDS = 1.0
POLICY_TEST_TIMEOUT_SECONDS = 10.0
WORKER_READY_TIMEOUT_SECONDS = 10.0


class RewardRecordingCallback(BaseCallback):
    """Record each reward exposed through the Stable Baselines callback contract."""

    def __init__(self) -> None:
        super().__init__()
        self.rewards = []

    def _on_step(self) -> bool:
        self.rewards.extend(np.asarray(self.locals["rewards"]).tolist())
        return True


class TwoFeatureDiscreteEnv(gym.Env):
    """Provide a one-step environment matching the packed reward-test episode."""

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        self.observation_space = gym.spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(2,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        return np.ones(2, dtype=np.float32), 0.0, True, False, {}


class WorkerTestError(RuntimeError):
    """Identify an intentional worker failure in lifecycle tests."""


class FailingWorkerEnv(TwoFeatureDiscreteEnv):
    """Fail when a real worker starts its environment lifecycle."""

    def reset(self, *, seed=None, options=None):
        raise WorkerTestError("intentional worker failure")


class BlockingWorkerEnv(TwoFeatureDiscreteEnv):
    """Signal worker readiness and block until the test releases the reset."""

    def __init__(self, ready, release) -> None:
        super().__init__()
        self._ready = ready
        self._release = release

    def reset(self, *, seed=None, options=None):
        self._ready.set()
        if not self._release.wait(WORKER_READY_TIMEOUT_SECONDS):
            raise TimeoutError("Worker reset was not released")
        return super().reset(seed=seed, options=options)


def make_failing_worker_env():
    """Create an environment that fails inside a real collection worker."""
    return FailingWorkerEnv()


def make_blocking_worker_env(ready, release):
    """Create an environment controlled by spawn-compatible readiness events."""
    return BlockingWorkerEnv(ready, release)


def make_worker_test_agent(env_fns, *, use_mp=False):
    """Create a small on-policy agent for real worker lifecycle tests."""
    return get_injected_agent(PPO)(
        "MlpPolicy",
        IndexableMultiEnv(env_fns),
        batch_size=2,
        device="cpu",
        n_steps=2,
        use_mp=use_mp,
        worker_join_timeout=2.0,
    )


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
        self.logged_metadata_by_key = {}

    def _on_step(self) -> bool:
        self.step_call_count += 1
        self.metric_aggregator.aggregate_step(
            self.locals["new_obs"],
            self.locals["actions"],
            self.locals["rewards"],
            self.locals["dones"],
            self.locals["infos"],
        )
        for info in self.locals["infos"]:
            for key, value in info.items():
                if not key.startswith(constants.META_INFO_PREFIX):
                    continue
                if self.logged_metadata_by_key.get(key) == value:
                    continue
                self.connector.log_dict(
                    value if isinstance(value, dict) else {key: value},
                    key,
                )
                self.logged_metadata_by_key[key] = value

        for done_index in np.flatnonzero(self.locals["dones"]):
            self.episode_counter[done_index] = (
                self.episode_counter.get(done_index, 0) + 1
            )
            if self.episode_counter[done_index] % self.logging_frequency == 0:
                self.metric_aggregator.log_aggregated_metrics(
                    agent_index=done_index,
                    num_timesteps=self.num_timesteps,
                    log_distributions=self.log_distributions,
                )
                self.metric_aggregator.reset_multi_episode_trackers(done_index)
        return True


class MetricAggregator:
    """Apply real per-step metric state transitions for callback equivalence."""

    def __init__(self, aggregate_distributions=False):
        self.aggregate_distributions = aggregate_distributions
        self.episode_reward = None
        self.episode_rewards = {}
        self.episode_actions = None
        self.episode_step_metrics = {}
        self.episode_end_reasons = {}
        self.aggregated_steps = []
        self.logged_metrics = []
        self.reset_agent_indices = []
        self._current_actions = {}
        self._current_step_metrics = {}

    def aggregate_step(self, new_obs, actions, rewards, dones, infos):
        """Apply one production-shaped transition to episode-local state."""
        self.aggregated_steps.append(
            {
                "new_obs": copy.deepcopy(new_obs),
                "actions": actions.copy(),
                "rewards": rewards.copy(),
                "dones": dones.copy(),
                "infos": copy.deepcopy(infos),
            }
        )
        if self.episode_reward is None:
            self.episode_reward = np.zeros_like(rewards)
        self.episode_reward += rewards

        for agent_index, info in enumerate(infos):
            if self.aggregate_distributions:
                self._current_actions.setdefault(agent_index, []).append(
                    actions[agent_index].copy()
                )
            for key, value in info.items():
                if key.startswith(constants.STEP_METRIC_INFO_PREFIX):
                    metric_name = key[len(constants.STEP_METRIC_INFO_PREFIX) :]
                    self._current_step_metrics.setdefault(metric_name, {}).setdefault(
                        agent_index,
                        [],
                    ).append(float(value))

        for done_index in np.flatnonzero(dones):
            terminal_info = infos[done_index]
            if not terminal_info.get(constants.DISCARD_INFO_KEY, False):
                self.episode_rewards.setdefault(done_index, []).append(
                    self.episode_reward[done_index]
                )
                if self.aggregate_distributions:
                    if self.episode_actions is None:
                        self.episode_actions = [[] for _ in range(len(infos))]
                    self.episode_actions[done_index].extend(
                        self._current_actions.get(done_index, [])
                    )
                for metric_name, current_by_agent in self._current_step_metrics.items():
                    per_agent_values = self.episode_step_metrics.get(metric_name)
                    if per_agent_values is None:
                        per_agent_values = [[] for _ in range(len(infos))]
                        self.episode_step_metrics[metric_name] = per_agent_values
                    per_agent_values[done_index].extend(
                        current_by_agent.get(done_index, [])
                    )
                end_reason = terminal_info.get(constants.EPISODE_END_REASON_INFO_KEY)
                if end_reason is not None:
                    self.episode_end_reasons.setdefault(
                        done_index,
                        deque(maxlen=constants.EPISODE_END_REASON_WINDOW_SIZE),
                    ).append(end_reason)
            self.episode_reward[done_index] = 0
            self._current_actions.pop(done_index, None)
            for current_by_agent in self._current_step_metrics.values():
                current_by_agent.pop(done_index, None)

    def log_aggregated_metrics(
        self,
        agent_index,
        num_timesteps,
        log_distributions,
    ):
        """Record the exact aggregate snapshot emitted at a logging boundary."""
        self.logged_metrics.append(
            {
                "agent_index": agent_index,
                "num_timesteps": num_timesteps,
                "log_distributions": log_distributions,
                "episode_rewards": copy.deepcopy(
                    self.episode_rewards.get(agent_index, [])
                ),
                "episode_actions": copy.deepcopy(
                    []
                    if self.episode_actions is None
                    else self.episode_actions[agent_index]
                ),
                "episode_step_metrics": {
                    name: copy.deepcopy(values[agent_index])
                    for name, values in self.episode_step_metrics.items()
                },
                "episode_end_reasons": list(
                    self.episode_end_reasons.get(agent_index, [])
                ),
            }
        )

    def reset_multi_episode_trackers(self, agent_index):
        """Reset the same multi-episode trackers as the framework aggregator."""
        self.reset_agent_indices.append(agent_index)
        self.episode_rewards[agent_index] = []
        for per_agent_values in self.episode_step_metrics.values():
            per_agent_values[agent_index] = []
        if self.episode_actions:
            self.episode_actions[agent_index] = []


class LegacyMetricAggregator:
    """Record per-step calls for an aggregator without episode state."""

    def __init__(self) -> None:
        self.aggregated_steps = []
        self.logged_metrics = []
        self.reset_agent_indices = []

    def aggregate_step(self, new_obs, actions, rewards, dones, infos):
        """Record one fallback transition with callback-visible values."""
        self.aggregated_steps.append(
            {
                "new_obs": copy.deepcopy(new_obs),
                "actions": actions.copy(),
                "rewards": rewards.copy(),
                "dones": dones.copy(),
                "infos": copy.deepcopy(infos),
            }
        )

    def log_aggregated_metrics(self, **values):
        """Record one fallback logging boundary."""
        self.logged_metrics.append(values)

    def reset_multi_episode_trackers(self, agent_index):
        """Record one fallback reset boundary."""
        self.reset_agent_indices.append(agent_index)


class RecordingConnector:
    """Record connector writes without replacing callback behavior with mocks."""

    def __init__(self) -> None:
        self.logged_dicts = []
        self.uploads = []

    def log_dict(self, value, key):
        """Record one structured metadata or reset-information write."""
        self.logged_dicts.append((copy.deepcopy(value), key))

    def upload(self, **values):
        """Record one checkpoint upload request."""
        self.uploads.append(values)


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
        if self.num_timesteps > self.next_upload:
            self.connector.upload(
                agent=self.agent,
                checkpoint_id=self.num_timesteps,
            )
            self.next_upload = self.num_timesteps + self.checkpoint_frequency
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
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        infos = self.locals["infos"]
        if self.episode_reward is None:
            self.episode_reward = np.zeros_like(rewards)
        self.episode_reward += rewards
        for done_index in np.flatnonzero(dones):
            if not infos[done_index].get(constants.DISCARD_INFO_KEY, False):
                self.episode_rewards.append(self.episode_reward[done_index].copy())
            self.episode_reward[done_index] = 0

        reward_window_is_full = len(self.episode_rewards) == self.episode_rewards.maxlen
        if self.num_timesteps <= self.pruning_start_at or not reward_window_is_full:
            return True
        return bool(np.mean(self.episode_rewards) >= self.episode_reward_threshold)


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
        self.rewards = []
        self.timesteps = []

    def _on_step(self) -> bool:
        self.step_call_count += 1
        self.rewards.extend(np.asarray(self.locals["rewards"]).tolist())
        self.timesteps.append(self.num_timesteps)
        return True


class CallbackReplayModel:
    """Carry the trainer timestep required by off-policy dispatcher replay."""

    def __init__(self) -> None:
        self.num_timesteps = 0


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
    """Create a stateful production-shaped logging callback."""
    return LoggingCallback(
        connector=RecordingConnector(),
        metric_aggregator=MetricAggregator(),
    )


@pytest.fixture
def external_legacy_logging_callback():
    """Create a logging callback whose aggregator only accepts individual steps."""
    return LoggingCallback(
        connector=RecordingConnector(),
        metric_aggregator=LegacyMetricAggregator(),
    )


@pytest.fixture
def external_saving_callback():
    """Create the external checkpoint callback with a recording connector."""
    return SavingCallback(
        agent=object(),
        connector=RecordingConnector(),
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


@pytest.fixture(name="logging_callback_pair")
def create_logging_callback_pair():
    """Create identical stateful callbacks for reference and batched processing."""
    callbacks = [
        LoggingCallback(
            connector=RecordingConnector(),
            metric_aggregator=MetricAggregator(aggregate_distributions=True),
        )
        for _ in range(2)
    ]
    for callback in callbacks:
        callback.logging_frequency = 2
        callback.log_distributions = True
    return callbacks


@pytest.fixture(name="pruning_callback_pair")
def create_pruning_callback_pair():
    """Create pruning callbacks whose decision distinguishes raw reward from training."""
    return [
        ExperimentPruningCallback(
            episode_reward_threshold=3.5,
            pruning_start_at=0,
            reward_window=2,
        )
        for _ in range(2)
    ]


@pytest.fixture
def replay_callback_by_step():
    """Return a genuine per-step callback replay oracle."""

    def replay(callback, batch, start_timestep=0):
        decisions = []
        action_field = resolve_episode_action_field(batch.episode_kind)
        reward_field = resolve_episode_reward_field(batch.episode_kind)
        for transition_index in range(batch.transition_count):
            callback.locals = {
                "new_obs": slice_episode_field(
                    batch,
                    "new_obs",
                    transition_index,
                ),
                "actions": slice_episode_field(
                    batch,
                    action_field,
                    transition_index,
                ),
                "rewards": slice_episode_field(
                    batch,
                    reward_field,
                    transition_index,
                ),
                "dones": slice_episode_field(
                    batch,
                    "dones",
                    transition_index,
                ),
                "infos": get_episode_infos(batch, transition_index),
                "reset_infos": get_episode_reset_infos(batch, transition_index),
            }
            callback.num_timesteps = start_timestep + transition_index + 1
            callback.n_calls += 1
            decisions.append(callback._on_step())
        return decisions

    return replay


@pytest.fixture
def process_callback_by_episode():
    """Return the production dispatcher path for one complete episode."""

    def process(callback, batch, start_timestep=0):
        dispatcher = CallbackBatchDispatcher(callback, batch.episode_kind)
        if batch.episode_kind is EpisodeKind.OFF_POLICY:
            replay_model = CallbackReplayModel()
            for transition_offset in range(1, batch.transition_count + 1):
                replay_model.num_timesteps = start_timestep + transition_offset
                assert dispatcher.process_step({"self": replay_model})
        return dispatcher.process_episode(
            EpisodeCallbackContext(
                batch=batch,
                start_timestep=start_timestep,
                end_timestep=start_timestep + batch.transition_count,
            )
        )

    return process


@pytest.fixture
def snapshot_logging_callback():
    """Return a normalized view of externally observable logging state."""

    def normalize(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, deque)):
            return [normalize(item) for item in value]
        return value

    def snapshot(callback):
        aggregator = callback.metric_aggregator
        return normalize(
            {
                "episode_counter": callback.episode_counter,
                "n_calls": callback.n_calls,
                "num_timesteps": callback.num_timesteps,
                "episode_reward": aggregator.episode_reward,
                "episode_rewards": aggregator.episode_rewards,
                "episode_actions": aggregator.episode_actions,
                "episode_step_metrics": aggregator.episode_step_metrics,
                "episode_end_reasons": aggregator.episode_end_reasons,
                "logged_metrics": aggregator.logged_metrics,
                "reset_agent_indices": aggregator.reset_agent_indices,
                "logged_dicts": callback.connector.logged_dicts,
            }
        )

    return snapshot


@pytest.fixture
def external_reset_info_callback():
    """Create the external reset-info callback with a recording connector."""
    return ResetInfoCallback(connector=RecordingConnector())


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
def short_pendulum_multi_env():
    """Create continuous-control workers with two-transition episodes."""
    return IndexableMultiEnv(
        [partial(gym.make, "Pendulum-v1", max_episode_steps=2) for _ in range(2)]
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
def short_episode_off_policy_agent(short_pendulum_multi_env):
    """Create an agent that consumes one row per rollout from two-step episodes."""
    agent = get_injected_agent(SAC)(
        "MlpPolicy",
        short_pendulum_multi_env,
        batch_size=2,
        buffer_size=32,
        device="cpu",
        gradient_steps=1,
        learning_starts=100,
        train_freq=1,
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


@pytest.fixture(name="reward_recording_callback")
def create_reward_recording_callback():
    """Create a callback that retains its exact callback-local rewards."""
    return RewardRecordingCallback()


@pytest.fixture(name="thread_failure_agent")
def create_thread_failure_agent():
    """Create an agent whose real thread worker fails during reset."""
    agent = make_worker_test_agent([make_failing_worker_env])
    yield agent, WorkerTestError
    agent.shutdown()


@pytest.fixture(name="process_blocked_agent")
def create_process_blocked_agent():
    """Create an agent with one spawn worker held at a readiness barrier."""
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    env_factory = partial(make_blocking_worker_env, ready, release)
    agent = make_worker_test_agent([env_factory], use_mp=True)
    yield agent, ready
    agent.shutdown()


@pytest.fixture(name="multi_worker_failure_agent")
def create_multi_worker_failure_agent():
    """Create one failing and one blocked thread worker for root attribution."""
    ready = threading.Event()
    release = threading.Event()
    env_factory = partial(make_blocking_worker_env, ready, release)
    agent = make_worker_test_agent([make_failing_worker_env, env_factory])
    yield agent, ready
    release.set()
    agent.shutdown()


@pytest.fixture(name="shutdown_worker_agent")
def create_shutdown_worker_agent():
    """Create a healthy one-step worker for intentional-shutdown coverage."""
    agent = make_worker_test_agent([TwoFeatureDiscreteEnv])
    yield agent
    agent.shutdown()


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
    agent._last_obs = agent.env.reset()
    agent._last_episode_starts = np.ones((agent.env.num_envs,), dtype=bool)
    agent.ep_info_buffer = deque(maxlen=agent._stats_window_size)
    agent.ep_success_buffer = deque(maxlen=agent._stats_window_size)
    yield agent
    agent.shutdown()


@pytest.fixture(name="initialized_reward_test_agent")
def provide_initialized_reward_test_agent():
    """Create an initialized agent whose buffer matches the reward-test packet."""
    agent = make_initialized_reward_test_agent()
    yield agent
    agent.shutdown()


def make_initialized_reward_test_agent():
    """Create one initialized agent matching deterministic reward-test packets."""
    env = IndexableMultiEnv([TwoFeatureDiscreteEnv])
    agent = get_injected_agent(PPO)(
        "MlpPolicy",
        env,
        batch_size=2,
        device="cpu",
        n_steps=2,
    )
    agent._init_collect_state()
    agent._last_obs = agent.env.reset()
    agent._last_episode_starts = np.ones((agent.env.num_envs,), dtype=bool)
    agent.ep_info_buffer = deque(maxlen=agent._stats_window_size)
    agent.ep_success_buffer = deque(maxlen=agent._stats_window_size)
    return agent


@pytest.fixture(name="reward_test_agent_pair")
def create_reward_test_agent_pair():
    """Create two agents for optimized-versus-per-step state comparison."""
    agents = [
        make_initialized_reward_test_agent(),
        make_initialized_reward_test_agent(),
    ]
    yield agents
    for agent in agents:
        agent.shutdown()


@pytest.fixture
def record_policy_noise_resets():
    """Return a recorder for parent-policy exploration-noise resets."""

    def attach(agent):
        reset_timesteps = []

        def record_reset(batch_size=1):
            reset_timesteps.append((agent.num_timesteps, batch_size))

        agent.policy.reset_noise = record_reset
        return reset_timesteps

    return attach


@pytest.fixture
def collect_deterministic_reward_rollout(enqueue_episode_packet):
    """Collect two preloaded episodes before live workers can affect the rollout."""

    def collect(
        agent,
        packet,
        callback,
        *,
        use_sde,
        sde_sample_freq,
    ):
        enqueue_episode_packet(agent, packet)
        enqueue_episode_packet(agent, packet)
        agent.use_sde = use_sde
        agent.sde_sample_freq = sde_sample_freq
        callback.init_callback(agent)
        return agent.collect_rollouts(
            agent.env,
            callback,
            agent.rollout_buffer,
            n_rollout_steps=3,
        )

    return collect


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
def on_policy_episode():
    """Create a complete two-step on-policy episode with sparse terminal info."""
    return [
        OnPolicyTransition(
            actions=np.array([[0]]),
            values=np.array([0.25], dtype=np.float32),
            log_probs=np.array([-0.5], dtype=np.float32),
            last_obs=np.array([[1.0, 2.0]], dtype=np.float32),
            new_obs=np.array([[2.0, 3.0]], dtype=np.float32),
            environment_rewards=np.array([1.0], dtype=np.float32),
            training_rewards=np.array([1.0], dtype=np.float32),
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
            environment_rewards=np.array([2.0], dtype=np.float32),
            training_rewards=np.array([3.0], dtype=np.float32),
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


@pytest.fixture(name="discarded_on_policy_episode")
def create_discarded_on_policy_episode(on_policy_episode_with_metrics):
    """Mark a metric-bearing episode as discarded at its terminal row."""
    episode = list(on_policy_episode_with_metrics)
    terminal_info = dict(episode[-1].infos[0])
    terminal_info[constants.DISCARD_INFO_KEY] = True
    episode[-1] = replace(episode[-1], infos=[terminal_info])
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
def off_policy_episode_with_metrics(off_policy_episode):
    """Add logging metadata and metrics to a complete off-policy episode."""
    episode = list(off_policy_episode)
    episode[0] = replace(
        episode[0],
        infos=[
            {
                "meta_settings": {"map": "test"},
                "step_metric_speed": 2.0,
            }
        ],
    )
    episode[1] = replace(
        episode[1],
        infos=[
            {
                "episode_end_reason": "TIMEOUT",
                "step_metric_speed": 4.0,
            }
        ],
    )
    return episode


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


@pytest.fixture(name="dict_on_policy_episode")
def create_dict_on_policy_episode(on_policy_episode):
    """Create a two-step episode with dictionary observations."""
    observation_rows = (
        {
            "position": np.array([[1.0, 2.0]], dtype=np.float32),
            "velocity": np.array([[0.1]], dtype=np.float32),
        },
        {
            "position": np.array([[2.0, 3.0]], dtype=np.float32),
            "velocity": np.array([[0.2]], dtype=np.float32),
        },
        {
            "position": np.array([[3.0, 4.0]], dtype=np.float32),
            "velocity": np.array([[0.3]], dtype=np.float32),
        },
    )
    return [
        replace(
            on_policy_episode[0],
            last_obs=observation_rows[0],
            new_obs=observation_rows[1],
        ),
        replace(
            on_policy_episode[1],
            last_obs=observation_rows[1],
            new_obs=observation_rows[2],
        ),
    ]


@pytest.fixture(name="dict_on_policy_packet")
def pack_dict_on_policy_packet(dict_on_policy_episode):
    """Encode the dictionary-observation episode for assembly tests."""
    return encode_episode_batch(0, 1, pack_episode(dict_on_policy_episode))


@pytest.fixture(name="dict_on_policy_rollout_buffer")
def create_dict_on_policy_rollout_buffer():
    """Create a rollout buffer with dictionary observation destinations."""
    return DictRolloutBuffer(
        buffer_size=2,
        observation_space=gym.spaces.Dict(
            {
                "position": gym.spaces.Box(
                    low=-10.0,
                    high=10.0,
                    shape=(2,),
                    dtype=np.float32,
                ),
                "velocity": gym.spaces.Box(
                    low=-10.0,
                    high=10.0,
                    shape=(1,),
                    dtype=np.float32,
                ),
            }
        ),
        action_space=gym.spaces.Discrete(2),
        device="cpu",
        gae_lambda=0.95,
        gamma=0.99,
        n_envs=1,
    )


@pytest.fixture
def build_reference_rollout_buffer():
    """Return the reviewed concatenate-then-copy rollout assembly oracle."""

    def copy_concatenated_values(destination, values):
        """Copy one recursively concatenated field into its buffer destination."""
        if isinstance(destination, dict):
            for key, destination_value in destination.items():
                copy_concatenated_values(
                    destination_value,
                    [value[key] for value in values],
                )
            return
        source = np.concatenate(values, axis=0)
        np.copyto(destination, source.reshape(destination.shape))

    def build(template, episodes):
        """Build a reference buffer with the pre-remediation allocation path."""
        transition_count = sum(episode.batch.transition_count for episode in episodes)
        rollout_buffer = copy.copy(template)
        rollout_buffer.buffer_size = transition_count
        rollout_buffer.n_envs = 1
        rollout_buffer.reset()
        field_destinations = {
            "last_obs": rollout_buffer.observations,
            "actions": rollout_buffer.actions,
            constants.ON_POLICY_TRAINING_REWARDS_FIELD: rollout_buffer.rewards,
            "last_dones": rollout_buffer.episode_starts,
            "values": rollout_buffer.values,
            "log_probs": rollout_buffer.log_probs,
        }
        for field_name, destination in field_destinations.items():
            copy_concatenated_values(
                destination,
                [episode.batch.fields[field_name] for episode in episodes],
            )
        final_dones = (
            episodes[-1].batch.fields["dones"][-1:].reshape(rollout_buffer.n_envs)
        )
        rollout_buffer.compute_returns_and_advantage(
            last_values=torch.zeros(rollout_buffer.n_envs),
            dones=final_dones,
        )
        rollout_buffer.pos = transition_count
        rollout_buffer.full = True
        return rollout_buffer

    return build


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
