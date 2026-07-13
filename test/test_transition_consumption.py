from collections import deque

import numpy as np

from async_gym_agents.episode_codec import encode_episode_batch, pack_episode


class TestTransitionConsumption:
    """Episode transitions are consumed in order without shifting a Python list."""

    def test_consumes_episode_from_deque(
        self,
        initialized_on_policy_agent,
        on_policy_episode,
        enqueue_episode_packet,
    ):
        """Fetching the first row keeps the remaining episode in a deque."""
        enqueue_episode_packet(
            initialized_on_policy_agent,
            encode_episode_batch(
                worker_index=0,
                policy_version=0,
                batch=pack_episode(on_policy_episode),
            ),
        )

        fetched_transition = initialized_on_policy_agent.fetch_transition()

        np.testing.assert_array_equal(
            fetched_transition.actions,
            on_policy_episode[0].actions,
        )
        assert isinstance(initialized_on_policy_agent._transitions, deque)
        assert len(initialized_on_policy_agent._transitions) == 1
