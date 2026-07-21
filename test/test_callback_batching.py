from unittest.mock import call

import numpy as np

from async_gym_agents.callback_batching import CallbackBatchDispatcher
from async_gym_agents.data_classes import EpisodeCallbackContext
from async_gym_agents.enums import EpisodeKind
from async_gym_agents.episode_codec import pack_episode


class TestBatchedLoggingCallback:
    """Known logging callbacks consume complete episodes outside SB3's step loop."""

    def test_processes_episodes_without_per_transition_callback_dispatch(
        self,
        short_episode_on_policy_agent,
        external_logging_callback,
    ):
        """A four-transition rollout aggregates two episodes without `_on_step()`."""
        short_episode_on_policy_agent.learn(
            total_timesteps=3,
            callback=external_logging_callback,
        )

        assert external_logging_callback.step_call_count == 0
        assert external_logging_callback.n_calls == 4
        assert (
            external_logging_callback.metric_aggregator.aggregate_step.call_count == 0
        )
        assert (
            external_logging_callback.metric_aggregator.log_aggregated_metrics.call_count
            == 2
        )

    def test_aggregates_complete_episode_arrays_directly(
        self,
        external_logging_callback,
        on_policy_episode_with_metrics,
    ):
        """Rewards, actions, metrics, and end reasons bypass step aggregation."""
        callback = external_logging_callback
        callback.logging_frequency = 2
        callback.log_distributions = True
        callback.metric_aggregator.aggregate_distributions = True
        batch = pack_episode(on_policy_episode_with_metrics)

        assert CallbackBatchDispatcher(callback).process_episode(
            EpisodeCallbackContext(batch, 0, 2)
        )

        aggregator = callback.metric_aggregator
        assert aggregator.aggregate_step.call_count == 0
        assert aggregator.episode_rewards[0] == [3.0]
        assert aggregator.episode_step_metrics["speed"][0] == [2.0, 4.0]
        assert list(aggregator.episode_end_reasons[0]) == ["TIMEOUT"]
        assert [action.tolist() for action in aggregator.episode_actions[0]] == [
            [0],
            [1],
        ]

    def test_logs_unchanged_metadata_only_once(
        self,
        external_logging_callback,
        on_policy_episode_with_metrics,
    ):
        """Repeated episodes do not rewrite an identical metadata configuration."""
        batch = pack_episode(on_policy_episode_with_metrics)

        assert CallbackBatchDispatcher(external_logging_callback).process_episode(
            EpisodeCallbackContext(batch, 0, 2)
        )
        assert CallbackBatchDispatcher(external_logging_callback).process_episode(
            EpisodeCallbackContext(batch, 2, 4)
        )

        external_logging_callback.connector.log_dict.assert_called_once_with(
            {"map": "test"},
            "meta_settings",
        )

    def test_logs_metadata_again_after_its_value_changes(
        self,
        external_logging_callback,
        on_policy_episode_with_metrics,
        on_policy_episode_with_changed_metadata,
    ):
        """A changed metadata value is emitted after an earlier value was cached."""
        initial_batch = pack_episode(on_policy_episode_with_metrics)
        changed_batch = pack_episode(on_policy_episode_with_changed_metadata)

        assert CallbackBatchDispatcher(external_logging_callback).process_episode(
            EpisodeCallbackContext(initial_batch, 0, 2)
        )
        assert CallbackBatchDispatcher(external_logging_callback).process_episode(
            EpisodeCallbackContext(changed_batch, 2, 4)
        )

        assert external_logging_callback.connector.log_dict.call_args_list == [
            call({"map": "test"}, "meta_settings"),
            call({"map": "changed"}, "meta_settings"),
        ]


class TestBatchedPeriodicCallbacks:
    """Periodic framework callbacks evaluate only at relevant episode boundaries."""

    def test_saves_at_the_same_strict_timestep_threshold(
        self,
        short_episode_on_policy_agent,
        external_saving_callback,
    ):
        """A checkpoint scheduled at step two uploads at the original step-three edge."""
        short_episode_on_policy_agent.learn(
            total_timesteps=3,
            callback=external_saving_callback,
        )

        external_saving_callback.connector.upload.assert_called_once_with(
            agent=external_saving_callback.agent,
            checkpoint_id=3,
        )
        assert external_saving_callback.step_call_count == 0
        assert external_saving_callback.n_calls == 4

    def test_prunes_after_the_reward_window_is_complete(
        self,
        short_episode_on_policy_agent,
        external_pruning_callback,
    ):
        """Two low-reward episodes stop training at their shared terminal boundary."""
        short_episode_on_policy_agent.learn(
            total_timesteps=100,
            callback=external_pruning_callback,
        )

        assert short_episode_on_policy_agent.num_timesteps == 4
        assert len(external_pruning_callback.episode_rewards) == 2
        assert all(
            reward < external_pruning_callback.episode_reward_threshold
            for reward in external_pruning_callback.episode_rewards
        )
        assert external_pruning_callback.step_call_count == 0
        assert external_pruning_callback.n_calls == 4


class TestBatchedTerminalCallbacks:
    """Terminal-only callbacks receive one invocation per completed episode."""

    def test_logs_reset_information_at_episode_boundaries(
        self,
        external_reset_info_callback,
        on_policy_episode,
    ):
        """Repeated episodes log each post-reset payload with its episode number."""
        batch = pack_episode(on_policy_episode)
        dispatcher = CallbackBatchDispatcher(external_reset_info_callback)

        assert dispatcher.process_episode(EpisodeCallbackContext(batch, 0, 2))
        assert dispatcher.process_episode(EpisodeCallbackContext(batch, 2, 4))

        assert external_reset_info_callback.connector.log_dict.call_args_list == [
            call({"seed": 7}, "Reset Info - Agent 0 - Episode 1"),
            call({"seed": 7}, "Reset Info - Agent 0 - Episode 2"),
        ]
        assert external_reset_info_callback.first_step_tracker == [0]
        assert external_reset_info_callback.step_call_count == 0
        assert external_reset_info_callback.n_calls == 4

    def test_runs_utilization_logging_only_on_terminal_rows(
        self,
        short_episode_on_policy_agent,
        external_utilization_callback,
    ):
        """Two complete episodes replace four utilization `_on_step()` calls with two."""
        short_episode_on_policy_agent.learn(
            total_timesteps=3,
            callback=external_utilization_callback,
        )

        assert external_utilization_callback.step_call_count == 2
        assert external_utilization_callback.n_calls == 4
        assert all(
            np.array_equal(dones, np.array([True]))
            for dones in external_utilization_callback.terminal_dones
        )


class TestOffPolicyCallbackBatching:
    """Off-policy collection batches terminal work without changing step semantics."""

    def test_exposes_episode_batch_only_with_its_terminal_transition(
        self,
        initialized_off_policy_agent,
        off_policy_packet,
        enqueue_episode_packet,
    ):
        """Incremental replay insertion retains one batch until its final row."""
        enqueue_episode_packet(initialized_off_policy_agent, off_policy_packet)

        _, first_completed_episode = (
            initialized_off_policy_agent.fetch_transition_with_episode()
        )
        _, terminal_completed_episode = (
            initialized_off_policy_agent.fetch_transition_with_episode()
        )

        assert first_completed_episode is None
        assert terminal_completed_episode is not None
        assert terminal_completed_episode.episode_kind is EpisodeKind.OFF_POLICY
        assert terminal_completed_episode.transition_count == 2

    def test_batches_logging_while_preserving_third_party_step_callbacks(
        self,
        short_episode_off_policy_agent,
        external_logging_callback,
        unrecognized_step_callback,
    ):
        """A complete episode bypasses logging `_on_step()` but not unknown callbacks."""
        short_episode_off_policy_agent.learn(
            total_timesteps=3,
            callback=[external_logging_callback, unrecognized_step_callback],
        )

        assert external_logging_callback.step_call_count == 0
        assert external_logging_callback.n_calls == 3
        assert (
            external_logging_callback.metric_aggregator.aggregate_step.call_count == 0
        )
        assert (
            external_logging_callback.metric_aggregator.log_aggregated_metrics.call_count
            == 1
        )
        assert unrecognized_step_callback.step_call_count == 3
        assert unrecognized_step_callback.n_calls == 3

    def test_aggregates_off_policy_metrics_and_buffer_actions(
        self,
        external_logging_callback,
        off_policy_episode_with_metrics,
    ):
        """Packed off-policy fields produce the same logger aggregates as step input."""
        callback = external_logging_callback
        callback.logging_frequency = 2
        callback.log_distributions = True
        callback.metric_aggregator.aggregate_distributions = True
        batch = pack_episode(off_policy_episode_with_metrics)
        dispatcher = CallbackBatchDispatcher(callback, EpisodeKind.OFF_POLICY)

        assert dispatcher.process_episode(EpisodeCallbackContext(batch, 0, 2))

        aggregator = callback.metric_aggregator
        assert aggregator.episode_rewards[0] == [3.0]
        assert aggregator.episode_step_metrics["speed"][0] == [2.0, 4.0]
        assert list(aggregator.episode_end_reasons[0]) == ["TIMEOUT"]
        np.testing.assert_allclose(
            np.concatenate(aggregator.episode_actions[0]),
            np.array([0.1, 0.2], dtype=np.float32),
        )

    def test_preserves_checkpoint_checks_per_transition(
        self,
        short_episode_off_policy_agent,
        external_saving_callback,
    ):
        """A changing off-policy model keeps the original checkpoint step boundary."""
        short_episode_off_policy_agent.learn(
            total_timesteps=3,
            callback=external_saving_callback,
        )

        assert external_saving_callback.step_call_count == 3
        assert external_saving_callback.n_calls == 3
        external_saving_callback.connector.upload.assert_called_once_with(
            agent=external_saving_callback.agent,
            checkpoint_id=3,
        )


class TestCallbackCompatibility:
    """Unrecognized callbacks retain Stable Baselines per-transition behavior."""

    def test_preserves_step_dispatch_in_a_mixed_callback_list(
        self,
        short_episode_on_policy_agent,
        external_logging_callback,
        unrecognized_step_callback,
    ):
        """A third-party callback still sees every transition beside batched logging."""
        short_episode_on_policy_agent.learn(
            total_timesteps=3,
            callback=[external_logging_callback, unrecognized_step_callback],
        )

        assert unrecognized_step_callback.step_call_count == 4
        assert unrecognized_step_callback.n_calls == 4
        assert external_logging_callback.step_call_count == 0
        assert external_logging_callback.n_calls == 4
        assert (
            external_logging_callback.metric_aggregator.aggregate_step.call_count == 0
        )

    def test_preserves_step_aggregation_for_an_unknown_aggregator(
        self,
        short_episode_on_policy_agent,
        external_legacy_logging_callback,
    ):
        """A logging implementation without episode state keeps its step contract."""
        short_episode_on_policy_agent.learn(
            total_timesteps=3,
            callback=external_legacy_logging_callback,
        )

        assert (
            external_legacy_logging_callback.metric_aggregator.aggregate_step.call_count
            == 4
        )
