"""Framework-callback-shaped test doubles for callback-batching tests.

These classes mirror the external contract of reinforcement-learning-framework's
SB3 callbacks (LoggingCallback, SavingCallback, ExperimentPruningCallback,
ResetInfoCallback, AsyncSBUtilizationLoggingCallback) closely enough to exercise
async_gym_agents.callback_batching without depending on that sibling package.
Used exclusively by test_callback_batching.py.
"""

import copy
from collections import deque

import numpy as np
import pytest
from stable_baselines3.common.callbacks import BaseCallback

from async_gym_agents import constants
from async_gym_agents.callback_batching import (
    CallbackBatchDispatcher,
    resolve_episode_action_field,
    resolve_episode_reward_field,
)
from async_gym_agents.data_classes import EpisodeCallbackContext
from async_gym_agents.enums import EpisodeKind
from async_gym_agents.episode_codec import (
    get_episode_infos,
    get_episode_reset_infos,
    slice_episode_field,
)


class EpisodeBatchableCallbackMixin:
    """Advance SB3 bookkeeping without invoking the per-step hook.

    Mirrors the trivial half of what a real framework callback implements to
    satisfy async_gym_agents.callback_batching.EpisodeBatchableCallback; the
    interesting half is each class's own `process_episode`.
    """

    def advance_callback(self, transition_count: int, num_timesteps: int) -> None:
        self.n_calls += transition_count
        self.num_timesteps = num_timesteps


class LoggingCallback(EpisodeBatchableCallbackMixin, BaseCallback):
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

    def process_episode(self, context: EpisodeCallbackContext) -> bool:
        """Aggregate framework metrics directly from a complete episode batch."""
        batch = context.batch
        if self._supports_episode_aggregation(self.metric_aggregator):
            self._aggregate_episode(batch)
        else:
            self._aggregate_episode_by_step(batch)
            self._log_episode_metadata(batch)

        terminal_dones = slice_episode_field(
            batch,
            "dones",
            batch.transition_count - 1,
        )
        for done_index in np.flatnonzero(terminal_dones):
            self.episode_counter[done_index] = (
                self.episode_counter.get(done_index, 0) + 1
            )
            if self.episode_counter[done_index] % self.logging_frequency == 0:
                self.metric_aggregator.log_aggregated_metrics(
                    agent_index=done_index,
                    num_timesteps=context.end_timestep,
                    log_distributions=self.log_distributions,
                )
                self.metric_aggregator.reset_multi_episode_trackers(done_index)

        return True

    def _aggregate_episode_by_step(self, batch) -> None:
        for transition_index in range(batch.transition_count):
            infos = get_episode_infos(batch, transition_index)
            self.metric_aggregator.aggregate_step(
                slice_episode_field(batch, "new_obs", transition_index),
                slice_episode_field(
                    batch,
                    resolve_episode_action_field(batch.episode_kind),
                    transition_index,
                ),
                slice_episode_field(
                    batch,
                    resolve_episode_reward_field(batch.episode_kind),
                    transition_index,
                ),
                slice_episode_field(batch, "dones", transition_index),
                infos,
            )

    def _aggregate_episode(self, batch) -> None:
        aggregator = self.metric_aggregator
        # Each complete environment is packed into its own episode batch.
        episode_agent_index = 0
        terminal_index = batch.transition_count - 1
        terminal_dones = slice_episode_field(batch, "dones", terminal_index)
        terminal_infos = get_episode_infos(batch, terminal_index)
        discarded = any(
            terminal_infos[done_index].get(constants.DISCARD_INFO_KEY, False)
            for done_index in np.flatnonzero(terminal_dones)
        )
        if discarded:
            self._log_episode_metadata(batch)
            if aggregator.episode_reward is None:
                reward_values = batch.fields[
                    resolve_episode_reward_field(batch.episode_kind)
                ]
                aggregator.episode_reward = np.zeros(1, dtype=reward_values.dtype)
            aggregator.episode_reward[episode_agent_index] = 0
            return

        rewards = batch.fields[resolve_episode_reward_field(batch.episode_kind)]
        if aggregator.episode_reward is None:
            aggregator.episode_reward = np.zeros(1, dtype=rewards.dtype)
        aggregator.episode_reward[episode_agent_index] += np.sum(rewards)

        if aggregator.aggregate_distributions:
            if not aggregator.episode_actions:
                aggregator.episode_actions = [[]]
            action_field = resolve_episode_action_field(batch.episode_kind)
            aggregator.episode_actions[episode_agent_index].extend(
                batch.fields[action_field]
            )

        self._aggregate_episode_infos(batch)
        for done_index in np.flatnonzero(terminal_dones):
            terminal_info = terminal_infos[done_index]
            aggregator.episode_rewards.setdefault(done_index, []).append(
                aggregator.episode_reward[done_index]
            )
            end_reason = terminal_info.get(constants.EPISODE_END_REASON_INFO_KEY)
            if end_reason is not None:
                aggregator.episode_end_reasons.setdefault(
                    done_index,
                    deque(maxlen=constants.EPISODE_END_REASON_WINDOW_SIZE),
                ).append(end_reason)
            aggregator.episode_reward[done_index] = 0

    def _aggregate_episode_infos(self, batch) -> None:
        aggregator = self.metric_aggregator
        episode_step_metrics = aggregator.episode_step_metrics
        step_metric_prefix = constants.STEP_METRIC_INFO_PREFIX
        meta_info_prefix = constants.META_INFO_PREFIX
        for infos in batch.infos.values():
            for agent_index, info in enumerate(infos):
                for key, value in info.items():
                    if key.startswith(step_metric_prefix):
                        metric_name = key[len(step_metric_prefix) :]
                        per_agent_values = episode_step_metrics.get(metric_name)
                        if per_agent_values is None:
                            per_agent_values = [[] for _ in range(len(infos))]
                            episode_step_metrics[metric_name] = per_agent_values
                        per_agent_values[agent_index].append(float(value))
                    elif key.startswith(meta_info_prefix):
                        self._log_metadata(key, value)

    def _log_episode_metadata(self, batch) -> None:
        for infos in batch.infos.values():
            for info in infos:
                for key, value in info.items():
                    if key.startswith(constants.META_INFO_PREFIX):
                        self._log_metadata(key, value)

    def _log_metadata(self, key: str, value: object) -> None:
        if key in self.logged_metadata_by_key and self._metadata_matches(
            self.logged_metadata_by_key[key],
            value,
        ):
            return

        if isinstance(value, dict):
            self.connector.log_dict(value, key)
        else:
            self.connector.log_dict({key: value}, key)
        self.logged_metadata_by_key[key] = value

    @staticmethod
    def _metadata_matches(previous_value: object, current_value: object) -> bool:
        try:
            comparison = previous_value == current_value
            if isinstance(comparison, np.ndarray):
                return bool(np.all(comparison))
            return bool(comparison)
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _supports_episode_aggregation(metric_aggregator: object) -> bool:
        if type(metric_aggregator).__name__ != "MetricAggregator":
            return False
        return all(
            hasattr(metric_aggregator, name)
            for name in (
                "aggregate_distributions",
                "episode_actions",
                "episode_end_reasons",
                "episode_reward",
                "episode_rewards",
                "episode_step_metrics",
            )
        )


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


class SavingCallback(EpisodeBatchableCallbackMixin, BaseCallback):
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

    def process_episode(self, context: EpisodeCallbackContext) -> bool:
        """Preserve checkpoint scheduling without checking it every transition."""
        checkpoint_timestep = max(
            context.start_timestep + 1,
            self.next_upload + 1,
        )
        while checkpoint_timestep <= context.end_timestep:
            self.num_timesteps = checkpoint_timestep
            self.connector.upload(
                agent=self.agent,
                checkpoint_id=checkpoint_timestep,
            )
            self.next_upload = checkpoint_timestep + self.checkpoint_frequency
            checkpoint_timestep = max(
                checkpoint_timestep + 1,
                self.next_upload + 1,
            )

        return True


class ExperimentPruningCallback(EpisodeBatchableCallbackMixin, BaseCallback):
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

    def process_episode(self, context: EpisodeCallbackContext) -> bool:
        """Evaluate pruning once when a complete episode changes its reward window."""
        batch = context.batch
        episode_rewards = np.sum(
            batch.fields[resolve_episode_reward_field(batch.episode_kind)],
            axis=0,
            keepdims=True,
        )
        terminal_index = batch.transition_count - 1
        terminal_dones = slice_episode_field(batch, "dones", terminal_index)
        terminal_infos = get_episode_infos(batch, terminal_index)

        for done_index in np.flatnonzero(terminal_dones):
            if not terminal_infos[done_index].get(constants.DISCARD_INFO_KEY, False):
                self.episode_rewards.append(episode_rewards[done_index])

        self.episode_reward = np.zeros_like(episode_rewards)
        reward_window_is_full = (
            len(self.episode_rewards) == self.episode_rewards.maxlen
        )
        if (
            context.end_timestep <= self.pruning_start_at
            or not reward_window_is_full
        ):
            return True
        return bool(np.mean(self.episode_rewards) >= self.episode_reward_threshold)


class ResetInfoCallback(EpisodeBatchableCallbackMixin, BaseCallback):
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

    def process_episode(self, context: EpisodeCallbackContext) -> bool:
        """Log initial and post-terminal reset information once per episode."""
        batch = context.batch
        initial_reset_infos = get_episode_reset_infos(batch, 0)
        for agent_index, reset_info in enumerate(initial_reset_infos):
            if agent_index in self.first_step_tracker:
                continue
            self.episode_counter[agent_index] = 0
            self.first_step_tracker.append(agent_index)
            self._log_reset_info(agent_index, reset_info)

        terminal_index = batch.transition_count - 1
        terminal_dones = slice_episode_field(batch, "dones", terminal_index)
        terminal_reset_infos = get_episode_reset_infos(batch, terminal_index)
        for done_index in np.flatnonzero(terminal_dones):
            self.episode_counter[done_index] = (
                self.episode_counter.get(done_index, 0) + 1
            )
            self._log_reset_info(done_index, terminal_reset_infos[done_index])

        return True

    def _log_reset_info(self, agent_index: int, reset_info: dict) -> None:
        if not reset_info:
            return
        self.connector.log_dict(
            reset_info,
            (
                f"Reset Info - Agent {agent_index} - Episode "
                f"{self.episode_counter[agent_index]}"
            ),
        )


class AsyncSBUtilizationLoggingCallback(EpisodeBatchableCallbackMixin, BaseCallback):
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

    def process_episode(self, context: EpisodeCallbackContext) -> bool:
        """Invoke the terminal-only callback once with the completed episode's last row."""
        batch = context.batch
        terminal_index = batch.transition_count - 1
        self.update_locals(
            {
                "new_obs": slice_episode_field(batch, "new_obs", terminal_index),
                "actions": slice_episode_field(
                    batch,
                    resolve_episode_action_field(batch.episode_kind),
                    terminal_index,
                ),
                "rewards": slice_episode_field(
                    batch,
                    resolve_episode_reward_field(batch.episode_kind),
                    terminal_index,
                ),
                "dones": slice_episode_field(batch, "dones", terminal_index),
                "infos": get_episode_infos(batch, terminal_index),
                "reset_infos": get_episode_reset_infos(batch, terminal_index),
            }
        )
        self.num_timesteps = context.end_timestep
        return self._on_step()


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
