from async_gym_agents import constants
from async_gym_agents.episode_codec import encode_episode_batch, pack_episode
from async_gym_agents.profiler import iterate_profiler_metrics


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
