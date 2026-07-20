import async_gym_agents.agents.on_policy_injector as on_policy_injector

EXPECTED_THROUGHPUT_EPISODE_REPETITIONS = 10


class TestPpoThroughputEpisodeReplay:
    """The comparison branch multiplies completed PPO worker episodes."""

    def test_repeats_the_same_completed_episode_ten_times(
        self,
        completed_on_policy_episode,
    ):
        """Repetitions protect their list while reusing the same transitions."""
        repeated_episodes = list(
            on_policy_injector.repeat_episode_for_throughput_benchmark(
                completed_on_policy_episode
            )
        )

        assert len(repeated_episodes) == EXPECTED_THROUGHPUT_EPISODE_REPETITIONS
        assert all(
            episode is not completed_on_policy_episode
            and episode[0] is completed_on_policy_episode[0]
            for episode in repeated_episodes
        )
