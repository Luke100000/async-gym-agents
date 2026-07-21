from abc import ABC, abstractmethod
from collections import deque
from typing import List

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from async_gym_agents import constants
from async_gym_agents.data_classes import EpisodeBatch, OnPolicyEpisodeCallbackContext
from async_gym_agents.episode_codec import (
    get_episode_infos,
    get_episode_reset_infos,
    slice_episode_field,
)


class EpisodeCallbackPatch(ABC):
    """Process a known callback once per complete episode."""

    def __init__(self, callback: BaseCallback) -> None:
        self.callback = callback

    @abstractmethod
    def process_episode(self, context: OnPolicyEpisodeCallbackContext) -> bool:
        """Process one complete episode and return whether training should continue."""

    def advance_callback(self, context: OnPolicyEpisodeCallbackContext) -> None:
        self.callback.n_calls += context.batch.transition_count
        self.callback.num_timesteps = context.end_timestep


class LoggingCallbackPatch(EpisodeCallbackPatch):
    """Aggregate framework metrics directly from complete episode batches."""

    def __init__(self, callback: BaseCallback) -> None:
        super().__init__(callback)
        logged_metadata = getattr(callback, "async_logged_metadata_by_key", None)
        if logged_metadata is None:
            logged_metadata = {}
            callback.async_logged_metadata_by_key = logged_metadata
        self.logged_metadata = logged_metadata

    def process_episode(self, context: OnPolicyEpisodeCallbackContext) -> bool:
        callback = self.callback
        batch = context.batch
        if self._supports_episode_aggregation(callback.metric_aggregator):
            self._aggregate_episode(callback, batch)
        else:
            self._aggregate_episode_by_step(callback, batch)
            self._log_episode_metadata(callback, batch)

        terminal_dones = slice_episode_field(
            batch,
            "dones",
            batch.transition_count - 1,
        )
        for done_index in np.flatnonzero(terminal_dones):
            callback.episode_counter[done_index] = (
                callback.episode_counter.get(done_index, 0) + 1
            )
            if callback.episode_counter[done_index] % callback.logging_frequency == 0:
                callback.metric_aggregator.log_aggregated_metrics(
                    agent_index=done_index,
                    num_timesteps=context.end_timestep,
                    log_distributions=callback.log_distributions,
                )
                callback.metric_aggregator.reset_multi_episode_trackers(done_index)

        self.advance_callback(context)
        return True

    def _aggregate_episode_by_step(
        self,
        callback: BaseCallback,
        batch: EpisodeBatch,
    ) -> None:
        for transition_index in range(batch.transition_count):
            infos = get_episode_infos(batch, transition_index)
            callback.metric_aggregator.aggregate_step(
                slice_episode_field(batch, "new_obs", transition_index),
                slice_episode_field(batch, "actions", transition_index),
                slice_episode_field(batch, "rewards", transition_index),
                slice_episode_field(batch, "dones", transition_index),
                infos,
            )

    def _aggregate_episode(
        self,
        callback: BaseCallback,
        batch: EpisodeBatch,
    ) -> None:
        aggregator = callback.metric_aggregator
        # Workers pack each completed environment into its own episode batch.
        episode_agent_index = 0
        rewards = batch.fields["rewards"]
        if aggregator.episode_reward is None:
            aggregator.episode_reward = np.zeros(1, dtype=rewards.dtype)
        aggregator.episode_reward[episode_agent_index] += np.sum(rewards)

        if aggregator.aggregate_distributions:
            if not aggregator.episode_actions:
                aggregator.episode_actions = [[]]
            aggregator.episode_actions[episode_agent_index].extend(
                batch.fields["actions"]
            )

        self._aggregate_episode_infos(callback, batch)
        terminal_index = batch.transition_count - 1
        terminal_dones = slice_episode_field(batch, "dones", terminal_index)
        terminal_infos = get_episode_infos(batch, terminal_index)
        for done_index in np.flatnonzero(terminal_dones):
            terminal_info = terminal_infos[done_index]
            if not terminal_info.get(constants.DISCARD_INFO_KEY, False):
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

    def _aggregate_episode_infos(
        self,
        callback: BaseCallback,
        batch: EpisodeBatch,
    ) -> None:
        aggregator = callback.metric_aggregator
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
                        self._log_metadata(callback, key, value)

    def _log_episode_metadata(
        self,
        callback: BaseCallback,
        batch: EpisodeBatch,
    ) -> None:
        for infos in batch.infos.values():
            for info in infos:
                for key, value in info.items():
                    if key.startswith(constants.META_INFO_PREFIX):
                        self._log_metadata(callback, key, value)

    def _log_metadata(self, callback: BaseCallback, key: str, value: object) -> None:
        if key in self.logged_metadata and self._metadata_matches(
            self.logged_metadata[key],
            value,
        ):
            return

        if isinstance(value, dict):
            callback.connector.log_dict(value, key)
        else:
            callback.connector.log_dict({key: value}, key)
        self.logged_metadata[key] = value

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
        return _has_attributes(
            metric_aggregator,
            (
                "aggregate_distributions",
                "episode_actions",
                "episode_end_reasons",
                "episode_reward",
                "episode_rewards",
                "episode_step_metrics",
            ),
        )


class SavingCallbackPatch(EpisodeCallbackPatch):
    """Preserve checkpoint scheduling without checking it every transition."""

    def process_episode(self, context: OnPolicyEpisodeCallbackContext) -> bool:
        callback = self.callback
        checkpoint_timestep = max(
            context.start_timestep + 1,
            callback.next_upload + 1,
        )
        while checkpoint_timestep <= context.end_timestep:
            callback.num_timesteps = checkpoint_timestep
            callback.connector.upload(
                agent=callback.agent,
                checkpoint_id=checkpoint_timestep,
            )
            callback.next_upload = checkpoint_timestep + callback.checkpoint_frequency
            checkpoint_timestep = max(
                checkpoint_timestep + 1,
                callback.next_upload + 1,
            )

        self.advance_callback(context)
        return True


class ExperimentPruningCallbackPatch(EpisodeCallbackPatch):
    """Evaluate pruning once when a complete episode changes its reward window."""

    def process_episode(self, context: OnPolicyEpisodeCallbackContext) -> bool:
        callback = self.callback
        batch = context.batch
        episode_rewards = np.sum(batch.fields["rewards"], axis=0, keepdims=True)
        terminal_index = batch.transition_count - 1
        terminal_dones = slice_episode_field(batch, "dones", terminal_index)
        terminal_infos = get_episode_infos(batch, terminal_index)

        for done_index in np.flatnonzero(terminal_dones):
            if not terminal_infos[done_index].get(constants.DISCARD_INFO_KEY, False):
                callback.episode_rewards.append(episode_rewards[done_index])

        callback.episode_reward = np.zeros_like(episode_rewards)
        self.advance_callback(context)
        reward_window_is_full = (
            len(callback.episode_rewards) == callback.episode_rewards.maxlen
        )
        if (
            context.end_timestep <= callback.pruning_start_at
            or not reward_window_is_full
        ):
            return True
        return bool(
            np.mean(callback.episode_rewards) >= callback.episode_reward_threshold
        )


class ResetInfoCallbackPatch(EpisodeCallbackPatch):
    """Log initial and post-terminal reset information once per episode."""

    def process_episode(self, context: OnPolicyEpisodeCallbackContext) -> bool:
        callback = self.callback
        batch = context.batch
        initial_reset_infos = get_episode_reset_infos(batch, 0)
        for agent_index, reset_info in enumerate(initial_reset_infos):
            if agent_index in callback.first_step_tracker:
                continue
            callback.episode_counter[agent_index] = 0
            callback.first_step_tracker.append(agent_index)
            self._log_reset_info(callback, agent_index, reset_info)

        terminal_index = batch.transition_count - 1
        terminal_dones = slice_episode_field(batch, "dones", terminal_index)
        terminal_reset_infos = get_episode_reset_infos(batch, terminal_index)
        for done_index in np.flatnonzero(terminal_dones):
            callback.episode_counter[done_index] = (
                callback.episode_counter.get(done_index, 0) + 1
            )
            self._log_reset_info(
                callback,
                done_index,
                terminal_reset_infos[done_index],
            )

        self.advance_callback(context)
        return True

    @staticmethod
    def _log_reset_info(
        callback: BaseCallback,
        agent_index: int,
        reset_info: dict,
    ) -> None:
        if not reset_info:
            return
        callback.connector.log_dict(
            reset_info,
            (
                f"Reset Info - Agent {agent_index} - Episode "
                f"{callback.episode_counter[agent_index]}"
            ),
        )


class TerminalStepCallbackPatch(EpisodeCallbackPatch):
    """Invoke a terminal-only callback once with the completed episode's last row."""

    def process_episode(self, context: OnPolicyEpisodeCallbackContext) -> bool:
        callback = self.callback
        batch = context.batch
        terminal_index = batch.transition_count - 1
        callback.update_locals(
            {
                "new_obs": slice_episode_field(batch, "new_obs", terminal_index),
                "actions": slice_episode_field(batch, "actions", terminal_index),
                "rewards": slice_episode_field(batch, "rewards", terminal_index),
                "dones": slice_episode_field(batch, "dones", terminal_index),
                "infos": get_episode_infos(batch, terminal_index),
                "reset_infos": get_episode_reset_infos(batch, terminal_index),
            }
        )
        callback.n_calls += batch.transition_count - 1
        callback.num_timesteps = context.end_timestep
        return callback.on_step()


class CallbackBatchDispatcher:
    """Route known callbacks by episode and preserve per-step compatibility."""

    def __init__(self, callback: BaseCallback) -> None:
        self.callback_lists: List[CallbackList] = []
        self.episode_patches: List[EpisodeCallbackPatch] = []
        self.step_callbacks: List[BaseCallback] = []
        self._classify_callback(callback)

    @property
    def needs_step_callbacks(self) -> bool:
        return bool(self.step_callbacks)

    def process_step(self, callback_locals: dict) -> bool:
        """Dispatch one transition only to callbacks without an episode patch."""
        num_timesteps = callback_locals["self"].num_timesteps
        for callback_list in self.callback_lists:
            callback_list.n_calls += 1
            callback_list.num_timesteps = num_timesteps

        continue_training = True
        for callback in self.step_callbacks:
            callback.update_locals(callback_locals)
            continue_training = callback.on_step() and continue_training
        return continue_training

    def process_episode(self, context: OnPolicyEpisodeCallbackContext) -> bool:
        """Dispatch one complete episode to every installed callback patch."""
        if not self.needs_step_callbacks:
            for callback_list in self.callback_lists:
                callback_list.n_calls += context.batch.transition_count
                callback_list.num_timesteps = context.end_timestep

        continue_training = True
        for patch in self.episode_patches:
            continue_training = patch.process_episode(context) and continue_training
        return continue_training

    def _classify_callback(self, callback: BaseCallback) -> None:
        if isinstance(callback, CallbackList):
            self.callback_lists.append(callback)
            for child_callback in callback.callbacks:
                self._classify_callback(child_callback)
            return

        patch = self._create_episode_patch(callback)
        if patch is None:
            self.step_callbacks.append(callback)
        else:
            self.episode_patches.append(patch)

    @staticmethod
    def _create_episode_patch(
        callback: BaseCallback,
    ) -> EpisodeCallbackPatch | None:
        callback_name = type(callback).__name__
        if callback_name == "LoggingCallback" and _has_attributes(
            callback,
            (
                "connector",
                "episode_counter",
                "log_distributions",
                "logging_frequency",
                "metric_aggregator",
            ),
        ):
            return LoggingCallbackPatch(callback)
        if callback_name == "SavingCallback" and _has_attributes(
            callback,
            ("agent", "checkpoint_frequency", "connector", "next_upload"),
        ):
            return SavingCallbackPatch(callback)
        if callback_name == "ExperimentPruningCallback" and _has_attributes(
            callback,
            (
                "episode_reward_threshold",
                "episode_rewards",
                "pruning_start_at",
            ),
        ):
            return ExperimentPruningCallbackPatch(callback)
        if callback_name == "ResetInfoCallback" and _has_attributes(
            callback,
            ("connector", "episode_counter", "first_step_tracker"),
        ):
            return ResetInfoCallbackPatch(callback)
        if callback_name == "AsyncSBUtilizationLoggingCallback" and (
            _has_attributes(
                callback,
                ("logging_frequency", "shared_episode_counter"),
            )
        ):
            return TerminalStepCallbackPatch(callback)
        return None


def _has_attributes(
    value: object,
    attribute_names: tuple[str, ...],
) -> bool:
    return all(hasattr(value, name) for name in attribute_names)
