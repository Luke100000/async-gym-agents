import async_gym_agents.agents.on_policy_injector as on_policy_injector

EXPECTED_THROUGHPUT_EPISODE_REPETITIONS = 100


class TestThroughputEpisodeReplay:
    """The profiling branch multiplies completed on-policy worker episodes."""

    def test_repeats_completed_episode_one_hundred_times(
        self,
        on_policy_episode,
    ):
        """One environment episode is yielded one hundred times without copying it."""
        repeated_episodes = list(
            on_policy_injector.repeat_episode_for_throughput_benchmark(
                on_policy_episode
            )
        )

        assert len(repeated_episodes) == EXPECTED_THROUGHPUT_EPISODE_REPETITIONS
        assert all(episode is on_policy_episode for episode in repeated_episodes)
