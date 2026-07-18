from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement

from async_gym_agents.constants import (
    BUFFER_AVG_POLICY_LAG_KEY,
    BUFFER_MAX_POLICY_LAG_KEY,
    PROFILE_PHASE_TRANSPORT,
    PROFILER_EXCLUDED_OUTPUT_FORMATS,
    PROFILER_LOG_PREFIX,
    TRANSPORT_MAX_PENDING_BYTES_KEY,
    TRANSPORT_PAYLOAD_RECEIVE_PROFILE_KEY,
    TRANSPORT_PENDING_BYTES_KEY,
    TRANSPORT_PIPE_LATENCY_PROFILE_KEY,
    TRANSPORT_RECEIVE_ATTEMPTS_KEY,
    TRANSPORT_RECEIVED_BYTES_KEY,
)
from async_gym_agents.episode_codec import encode_episode_batch, pack_episode
from async_gym_agents.profiler import iterate_profiler_metrics


class TestPolicyLagProfiling:
    """Policy versions remain observable from production through consumption."""

    def test_weights_episode_lag_by_transition_count(
        self,
        initialized_on_policy_agent,
        on_policy_episode,
        enqueue_episode_packet,
    ):
        """A two-row episode contributes two samples of its policy lag."""
        initialized_on_policy_agent._version = 3
        enqueue_episode_packet(
            initialized_on_policy_agent,
            encode_episode_batch(
                worker_index=0,
                policy_version=1,
                batch=pack_episode(on_policy_episode),
            ),
        )

        initialized_on_policy_agent.fetch_transition()

        report = initialized_on_policy_agent.get_profiler_report()
        assert report["buffer"][BUFFER_AVG_POLICY_LAG_KEY] == 2
        assert report["buffer"][BUFFER_MAX_POLICY_LAG_KEY] == 2

    def test_records_nonblocking_episode_fetch_as_transport(
        self,
        initialized_on_policy_agent,
        on_policy_episode,
        enqueue_episode_packet,
    ):
        """Fetching an already pending episode records transport rather than waiting."""
        enqueue_episode_packet(
            initialized_on_policy_agent,
            encode_episode_batch(
                worker_index=0,
                policy_version=0,
                batch=pack_episode(on_policy_episode[:1]),
            ),
        )

        initialized_on_policy_agent.fetch_transition()

        profile = initialized_on_policy_agent.get_profiler_report()["main"]
        assert profile[PROFILE_PHASE_TRANSPORT]["count"] == 1

    def test_reports_dynamic_transport_memory(
        self,
        initialized_on_policy_agent,
        on_policy_episode,
        enqueue_episode_packet,
    ):
        """Transport reports actual packet bytes instead of fixed info allocation."""
        packet = encode_episode_batch(
            worker_index=0,
            policy_version=0,
            batch=pack_episode(on_policy_episode),
        )
        enqueue_episode_packet(initialized_on_policy_agent, packet)

        transport = initialized_on_policy_agent.get_profiler_report()["transport"]

        assert transport[TRANSPORT_PENDING_BYTES_KEY] == len(packet.payload)
        assert transport[TRANSPORT_MAX_PENDING_BYTES_KEY] == len(packet.payload)

    def test_reports_receive_delivery_timings(
        self,
        initialized_on_policy_agent,
        on_policy_packet,
        enqueue_episode_packet,
    ):
        """Delivered episodes expose payload timing, bytes, and pipe latency."""
        enqueue_episode_packet(initialized_on_policy_agent, on_policy_packet)

        initialized_on_policy_agent.fetch_transition()

        transport = initialized_on_policy_agent.get_profiler_report()["transport"]
        assert transport[TRANSPORT_RECEIVE_ATTEMPTS_KEY] == 1
        assert transport[TRANSPORT_RECEIVED_BYTES_KEY] == len(on_policy_packet.payload)
        assert transport[TRANSPORT_PAYLOAD_RECEIVE_PROFILE_KEY]["count"] == 1
        assert transport[TRANSPORT_PIPE_LATENCY_PROFILE_KEY]["count"] == 1

    def test_flattens_report_into_stable_scalar_names(
        self,
        logged_on_policy_agent,
        on_policy_episode,
        enqueue_episode_packet,
    ):
        """Profiler leaves are available to ClearML through the SB3 logger."""
        packet = encode_episode_batch(
            worker_index=0,
            policy_version=0,
            batch=pack_episode(on_policy_episode),
        )
        enqueue_episode_packet(logged_on_policy_agent, packet)
        logged_on_policy_agent.fetch_transition()

        logged_on_policy_agent.record_profiler_metrics()

        metric_name = f"{PROFILER_LOG_PREFIX}/buffer/{BUFFER_AVG_POLICY_LAG_KEY}"
        assert metric_name in logged_on_policy_agent.logger.name_to_value
        flattened_metrics = dict(
            iterate_profiler_metrics(logged_on_policy_agent.get_profiler_report())
        )
        assert f"transport/{TRANSPORT_MAX_PENDING_BYTES_KEY}" in flattened_metrics
        assert (
            f"transport/{TRANSPORT_PAYLOAD_RECEIVE_PROFILE_KEY}/avg_milliseconds"
            in flattened_metrics
        )


class TestProfilerOutputRouting:
    """Profiler scalars avoid SB3's width-limited console table."""

    def test_long_callback_metrics_do_not_collide_in_console_output(
        self,
        initialized_on_policy_agent,
        human_output_profiler_logger,
    ):
        """Long callback metric suffixes remain logged without console truncation."""
        logger, output = human_output_profiler_logger
        initialized_on_policy_agent.set_logger(logger)
        initialized_on_policy_agent._callback_profiler.instrument(
            StopTrainingOnNoModelImprovement(
                max_no_improvement_evals=1,
                min_evals=1,
            )
        )

        initialized_on_policy_agent.record_profiler_metrics()
        callback_metric_name = (
            f"{PROFILER_LOG_PREFIX}/callbacks/StopTrainingOnNoModelImprovement/count"
        )
        assert (
            logger.name_to_excluded[callback_metric_name]
            == PROFILER_EXCLUDED_OUTPUT_FORMATS
        )
        logger.record("time/fps", 1)
        logger.dump(step=1)

        assert "fps" in output.getvalue()
