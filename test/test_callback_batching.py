from unittest.mock import call

import numpy as np

from async_gym_agents.callback_batching import CallbackBatchDispatcher
from async_gym_agents.data_classes import OnPolicyEpisodeCallbackContext
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
            external_logging_callback.metric_aggregator.aggregate_step.call_count == 4
        )
        assert (
            external_logging_callback.metric_aggregator.log_aggregated_metrics.call_count
            == 2
        )


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

        assert dispatcher.process_episode(OnPolicyEpisodeCallbackContext(batch, 0, 2))
        assert dispatcher.process_episode(OnPolicyEpisodeCallbackContext(batch, 2, 4))

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
            external_logging_callback.metric_aggregator.aggregate_step.call_count == 4
        )
