import threading
from dataclasses import replace

from stable_baselines3 import PPO

from async_gym_agents.agents.async_agent import get_injected_agent
from async_gym_agents.enums import EpisodeKind
from async_gym_agents.episode_assembler import AsyncEpisodeAssembler
from async_gym_agents.episode_transport import EpisodeTransport
from async_gym_agents.profiler import RuntimeProfiler

ASSEMBLY_TEST_TIMEOUT_SECONDS = 1.0


class TestAsyncEpisodeAssembler:
    """Buffer B is filled from complete episodes while buffer A is in use."""

    def test_keeps_the_final_episode_complete(self, on_policy_packet):
        """Assembly exceeds its row target instead of splitting the final episode."""
        transport = EpisodeTransport(
            worker_count=2,
            max_pending_episodes=2,
            use_mp=False,
        )
        stop = threading.Event()
        worker_one_packet = replace(on_policy_packet, worker_index=1)
        assert transport.get_sender(0).send(
            on_policy_packet,
            stop,
            ASSEMBLY_TEST_TIMEOUT_SECONDS,
        )
        assert transport.get_sender(1).send(
            worker_one_packet,
            stop,
            ASSEMBLY_TEST_TIMEOUT_SECONDS,
        )
        assembler = AsyncEpisodeAssembler(
            transport=transport,
            target_transition_count=3,
            expected_episode_kind=EpisodeKind.ON_POLICY,
            profiler=RuntimeProfiler(),
        )
        assembler.start()

        assembly = assembler.acquire(ASSEMBLY_TEST_TIMEOUT_SECONDS)

        assert assembly.transition_count == 4
        assert [episode.packet.transition_count for episode in assembly.episodes] == [
            2,
            2,
        ]
        assembler.shutdown()
        transport.shutdown()

    def test_refills_after_the_active_buffer_is_released(self, on_policy_packet):
        """Releasing A lets the background thread build the next B immediately."""
        transport = EpisodeTransport(
            worker_count=1,
            max_pending_episodes=1,
            use_mp=False,
        )
        stop = threading.Event()
        assembler = AsyncEpisodeAssembler(
            transport=transport,
            target_transition_count=2,
            expected_episode_kind=EpisodeKind.ON_POLICY,
            profiler=RuntimeProfiler(),
        )
        assembler.start()
        assert transport.get_sender(0).send(
            on_policy_packet,
            stop,
            ASSEMBLY_TEST_TIMEOUT_SECONDS,
        )
        first_assembly = assembler.acquire(ASSEMBLY_TEST_TIMEOUT_SECONDS)
        assembler.release()
        assert transport.get_sender(0).send(
            on_policy_packet,
            stop,
            ASSEMBLY_TEST_TIMEOUT_SECONDS,
        )

        second_assembly = assembler.acquire(ASSEMBLY_TEST_TIMEOUT_SECONDS)

        assert first_assembly.transition_count == 2
        assert second_assembly.transition_count == 2
        assembler.shutdown()
        transport.shutdown()


class TestOnPolicyCompleteEpisodeAssembly:
    """PPO consumes assembled episodes without introducing partial trajectories."""

    def test_resizes_rollout_buffer_to_complete_episode_total(
        self,
        short_cartpole_multi_env,
    ):
        """A three-row PPO target is rounded up to two complete two-row episodes."""
        model = get_injected_agent(PPO)(
            "MlpPolicy",
            short_cartpole_multi_env,
            batch_size=2,
            device="cpu",
            n_epochs=1,
            n_steps=3,
        )

        model.learn(total_timesteps=3)

        assert model.rollout_buffer.buffer_size == 4
        assert model.rollout_buffer.episode_starts[:, 0].tolist() == [
            1.0,
            0.0,
            1.0,
            0.0,
        ]
        model.shutdown()
