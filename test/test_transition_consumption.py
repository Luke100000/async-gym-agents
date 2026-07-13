from collections import deque


class TestTransitionConsumption:
    """Episode transitions are consumed in order without shifting a Python list."""

    def test_consumes_episode_from_deque(self, initialized_on_policy_agent):
        """Fetching the first row keeps the remaining episode in a deque."""
        first_transition = object()
        second_transition = object()
        initialized_on_policy_agent._episode_queue.put(
            [first_transition, second_transition]
        )

        fetched_transition = initialized_on_policy_agent.fetch_transition()

        assert fetched_transition is first_transition
        assert initialized_on_policy_agent._transitions == deque([second_transition])
