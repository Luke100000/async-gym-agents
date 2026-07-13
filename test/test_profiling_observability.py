from async_gym_agents.constants import (
    BUFFER_AVG_POLICY_LAG_KEY,
    BUFFER_MAX_POLICY_LAG_KEY,
    PROFILE_PHASE_TRANSPORT,
)
from async_gym_agents.episode_codec import encode_episode_batch, pack_episode


class TestPolicyLagProfiling:
    """Policy versions remain observable from production through consumption."""

    def test_weights_episode_lag_by_transition_count(
        self,
        initialized_on_policy_agent,
        on_policy_episode,
    ):
        """A two-row episode contributes two samples of its policy lag."""
        initialized_on_policy_agent._version = 3
        initialized_on_policy_agent._episode_queue.put(
            encode_episode_batch(
                worker_index=0,
                policy_version=1,
                batch=pack_episode(on_policy_episode),
            )
        )

        initialized_on_policy_agent.fetch_transition()

        report = initialized_on_policy_agent.get_profiler_report()
        assert report["buffer"][BUFFER_AVG_POLICY_LAG_KEY] == 2
        assert report["buffer"][BUFFER_MAX_POLICY_LAG_KEY] == 2

    def test_records_nonblocking_episode_fetch_as_transport(
        self,
        initialized_on_policy_agent,
        on_policy_episode,
    ):
        """Fetching an already queued episode records transport rather than waiting."""
        initialized_on_policy_agent._episode_queue.put(
            encode_episode_batch(
                worker_index=0,
                policy_version=0,
                batch=pack_episode(on_policy_episode[:1]),
            )
        )

        initialized_on_policy_agent.fetch_transition()

        profile = initialized_on_policy_agent.get_profiler_report()["main"]
        assert profile[PROFILE_PHASE_TRANSPORT]["count"] == 1
