from async_gym_agents import constants
from async_gym_agents.episode_codec import encode_episode_batch, pack_episode
from async_gym_agents.profiler import (
    iterate_profiler_metrics,
    render_profiler_report,
)


class TestPolicyLagReporting:
    """Profiler reports retain transition-weighted policy lag."""

    def test_weights_episode_lag_by_transition_count(
        self,
        initialized_off_policy_agent,
        off_policy_episode,
        enqueue_episode_packet,
    ):
        """A two-row episode contributes two samples of its policy lag."""
        initialized_off_policy_agent._version = 3
        enqueue_episode_packet(
            initialized_off_policy_agent,
            encode_episode_batch(
                worker_index=0,
                policy_version=1,
                batch=pack_episode(off_policy_episode),
            ),
        )

        initialized_off_policy_agent.fetch_transition()

        report = initialized_off_policy_agent.get_profiler_report()
        assert report["buffer"][constants.BUFFER_AVG_POLICY_LAG_KEY] == 2
        assert report["buffer"][constants.BUFFER_MAX_POLICY_LAG_KEY] == 2


class TestTransportReporting:
    """Profiler reports follow the lifecycle of bounded episode payloads."""

    def test_reports_pending_and_received_payload_bytes(
        self,
        initialized_off_policy_agent,
        off_policy_episode,
        enqueue_episode_packet,
    ):
        """A delivered packet moves from pending memory into received totals."""
        packet = encode_episode_batch(
            worker_index=0,
            policy_version=0,
            batch=pack_episode(off_policy_episode),
        )
        enqueue_episode_packet(initialized_off_policy_agent, packet)

        transport = initialized_off_policy_agent.get_profiler_report()["transport"]

        assert transport["pending_bytes"] == len(packet.payload)
        assert transport["max_pending_bytes"] == len(packet.payload)

        initialized_off_policy_agent.fetch_transition()

        transport = initialized_off_policy_agent.get_profiler_report()["transport"]
        assert transport["pending_bytes"] == 0
        assert transport["received_bytes"] == len(packet.payload)

    def test_reports_current_transport_utilization_only(
        self,
        initialized_off_policy_agent,
        off_policy_packet,
        enqueue_episode_packet,
    ):
        """Occupancy is live under transport and queue-era buffer leaves are absent."""
        empty_report = initialized_off_policy_agent.get_profiler_report()
        enqueue_episode_packet(initialized_off_policy_agent, off_policy_packet)
        occupied_report = initialized_off_policy_agent.get_profiler_report()
        expected_utilization = 1 / initialized_off_policy_agent.max_episodes_in_buffer

        assert empty_report["transport"]["utilization"] == 0.0
        assert occupied_report["transport"]["utilization"] == expected_utilization
        assert "utilization" not in occupied_report["buffer"]
        assert "emptiness" not in occupied_report["buffer"]
        assert "avg_push_time_seconds" not in occupied_report["buffer"]

    def test_renders_utilization_as_transport_state(
        self,
        initialized_off_policy_agent,
        off_policy_packet,
        enqueue_episode_packet,
    ):
        """Human-readable output labels occupancy under the transport section."""
        enqueue_episode_packet(initialized_off_policy_agent, off_policy_packet)

        rendered = render_profiler_report(
            initialized_off_policy_agent.get_profiler_report()
        )

        expected_utilization = 1 / initialized_off_policy_agent.max_episodes_in_buffer
        assert f"Transport: util={expected_utilization:.2f}" in rendered
        assert "Buffer: util=" not in rendered


class TestProfilerMetricIteration:
    """Nested profiler reports expose stable scalar names for external logging."""

    def test_flattens_report_into_stable_scalar_names(
        self,
        initialized_off_policy_agent,
        off_policy_episode,
        enqueue_episode_packet,
    ):
        """Profiler leaves expose stable names for external logging callbacks."""
        packet = encode_episode_batch(
            worker_index=0,
            policy_version=0,
            batch=pack_episode(off_policy_episode),
        )
        enqueue_episode_packet(initialized_off_policy_agent, packet)
        initialized_off_policy_agent.fetch_transition()

        flattened_metrics = dict(
            iterate_profiler_metrics(initialized_off_policy_agent.get_profiler_report())
        )
        assert f"buffer/{constants.BUFFER_AVG_POLICY_LAG_KEY}" in flattened_metrics
        assert "transport/max_pending_bytes" in flattened_metrics
        assert "transport/received_bytes" in flattened_metrics

    def test_emits_one_named_push_wait_metric(
        self,
        initialized_off_policy_agent,
    ):
        """Metric enumeration exposes one unambiguous episode-send wait leaf."""
        metric_names = [
            name
            for name, _ in iterate_profiler_metrics(
                initialized_off_policy_agent.get_profiler_report()
            )
        ]

        expected_name = f"buffer/{constants.BUFFER_AVG_PUSH_WAIT_SECONDS_KEY}"
        assert metric_names.count(expected_name) == 1
        assert "buffer/avg_push_time_seconds" not in metric_names
