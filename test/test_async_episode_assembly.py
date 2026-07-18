import threading
from dataclasses import replace

import numpy as np
from stable_baselines3 import PPO

from async_gym_agents.agents.async_agent import get_injected_agent
from async_gym_agents.agents.on_policy_injector import bootstrap_truncated_rewards
from async_gym_agents.episode_transport import EpisodeTransport
from async_gym_agents.on_policy_rollout_assembler import (
    AsyncOnPolicyRolloutAssembler,
)
from async_gym_agents.profiler import RuntimeProfiler

ASSEMBLY_TEST_TIMEOUT_SECONDS = 1.0


class TestAsyncOnPolicyRolloutAssembler:
    """Buffer B is filled from complete episodes while buffer A is in use."""

    def test_prepares_a_full_rollout_buffer_before_acquisition(
        self,
        on_policy_packet,
        on_policy_rollout_buffer,
    ):
        """The acquired buffer is immediately ready for PPO training."""
        transport = EpisodeTransport(
            worker_count=1,
            max_pending_episodes=1,
            use_mp=False,
        )
        stop = threading.Event()
        assembler = AsyncOnPolicyRolloutAssembler(
            transport=transport,
            target_transition_count=2,
            profiler=RuntimeProfiler(),
            rollout_buffer_template=on_policy_rollout_buffer,
        )
        assembler.start()
        assert transport.get_sender(0).send(
            on_policy_packet,
            stop,
            ASSEMBLY_TEST_TIMEOUT_SECONDS,
        )

        prepared_rollout = assembler.acquire(ASSEMBLY_TEST_TIMEOUT_SECONDS)

        assert prepared_rollout.rollout_buffer.full is True
        assert prepared_rollout.rollout_buffer.pos == 2
        assert prepared_rollout.rollout_buffer.observations[:, 0].tolist() == [
            [1.0, 2.0],
            [2.0, 3.0],
        ]
        assert prepared_rollout.rollout_buffer.episode_starts[:, 0].tolist() == [
            1.0,
            0.0,
        ]
        assembler.shutdown()
        transport.shutdown()

    def test_keeps_the_final_episode_complete(
        self,
        on_policy_packet,
        on_policy_rollout_buffer,
    ):
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
        assembler = AsyncOnPolicyRolloutAssembler(
            transport=transport,
            target_transition_count=3,
            profiler=RuntimeProfiler(),
            rollout_buffer_template=on_policy_rollout_buffer,
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

    def test_refills_while_the_acquired_buffer_remains_in_use(
        self,
        on_policy_packet,
        on_policy_rollout_buffer,
    ):
        """Acquiring A lets the background thread build B without a release."""
        transport = EpisodeTransport(
            worker_count=1,
            max_pending_episodes=1,
            use_mp=False,
        )
        stop = threading.Event()
        assembler = AsyncOnPolicyRolloutAssembler(
            transport=transport,
            target_transition_count=2,
            profiler=RuntimeProfiler(),
            rollout_buffer_template=on_policy_rollout_buffer,
        )
        assembler.start()
        assert transport.get_sender(0).send(
            on_policy_packet,
            stop,
            ASSEMBLY_TEST_TIMEOUT_SECONDS,
        )
        first_assembly = assembler.acquire(ASSEMBLY_TEST_TIMEOUT_SECONDS)
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

    def test_streams_complete_episodes_from_process_workers(
        self,
        short_episode_on_policy_mp_agent,
    ):
        """Multiprocessing workers stream complete episodes into the PPO buffer."""
        short_episode_on_policy_mp_agent.learn(total_timesteps=3)

        assert short_episode_on_policy_mp_agent.rollout_buffer.buffer_size == 4
        assert short_episode_on_policy_mp_agent.rollout_buffer.full is True


class TestTruncatedRewardBootstrap:
    """Time-limit rewards are finalized before background buffer construction."""

    def test_uses_the_rollout_policy_only_for_truncated_episodes(
        self,
        fixed_terminal_value_policy,
    ):
        """Only a truncated terminal reward receives the discounted value."""
        rewards = np.array([1.0, 3.0], dtype=np.float32)
        dones = np.array([True, True])
        infos = [
            {
                "TimeLimit.truncated": True,
                "terminal_observation": np.array([1.0, 2.0], dtype=np.float32),
            },
            {
                "TimeLimit.truncated": False,
                "terminal_observation": np.array([3.0, 4.0], dtype=np.float32),
            },
        ]

        bootstrap_truncated_rewards(
            fixed_terminal_value_policy,
            0.5,
            rewards,
            dones,
            infos,
        )

        assert rewards.tolist() == [2.0, 3.0]
        fixed_terminal_value_policy.predict_values.assert_called_once()
