from functools import partial

import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback

from async_gym_agents.agents.async_agent import get_injected_agent
from async_gym_agents.envs.multi_env import IndexableMultiEnv
from async_gym_agents.episode_codec import encode_episode_batch, pack_episode

WORKERS = 2
SUB_ENVIRONMENTS = 2


class EpisodeOriginRecorder(BaseCallback):
    """Episode-batchable callback which records the origin of every processed episode."""

    def __init__(self):
        super().__init__()
        self.origins = []

    def _on_step(self) -> bool:
        return True

    def advance_callback(self, transition_count: int, num_timesteps: int) -> None:
        self.n_calls += transition_count
        self.num_timesteps = num_timesteps

    def process_episode(self, context) -> bool:
        self.origins.append((context.worker_index, context.env_index))
        return True


def make_vectorized_short_episodes(env_id):
    """Each worker runs a vectorized environment of two sub-environments with two-transition episodes."""
    return [gym.make(env_id, max_episode_steps=2) for _ in range(SUB_ENVIRONMENTS)]


class TestEpisodeOrigin:
    """Episode callbacks learn which worker and sub-environment played an episode."""

    def test_packet_keeps_the_sub_environment_index(self, on_policy_episode):
        batch = pack_episode(on_policy_episode)

        assert encode_episode_batch(7, 3, batch).env_index == 0
        assert encode_episode_batch(7, 3, batch, env_index=2).env_index == 2

    def test_on_policy_callbacks_receive_worker_and_sub_environment(self):
        agent = get_injected_agent(PPO)(
            "MlpPolicy",
            IndexableMultiEnv(
                [
                    partial(make_vectorized_short_episodes, "CartPole-v1")
                    for _ in range(WORKERS)
                ]
            ),
            batch_size=8,
            device="cpu",
            n_epochs=1,
            n_steps=8,
        )
        recorder = EpisodeOriginRecorder()
        try:
            agent.learn(total_timesteps=64, callback=recorder)
        finally:
            agent.shutdown()

        expected = {
            (worker, env)
            for worker in range(WORKERS)
            for env in range(SUB_ENVIRONMENTS)
        }
        assert set(recorder.origins) == expected

    def test_off_policy_callbacks_receive_worker_and_sub_environment(self):
        agent = get_injected_agent(SAC)(
            "MlpPolicy",
            IndexableMultiEnv(
                [
                    partial(make_vectorized_short_episodes, "Pendulum-v1")
                    for _ in range(WORKERS)
                ]
            ),
            batch_size=2,
            buffer_size=128,
            device="cpu",
            learning_starts=1000,
            train_freq=1,
        )
        recorder = EpisodeOriginRecorder()
        try:
            agent.learn(total_timesteps=64, callback=recorder)
        finally:
            agent.shutdown()

        expected = {
            (worker, env)
            for worker in range(WORKERS)
            for env in range(SUB_ENVIRONMENTS)
        }
        assert set(recorder.origins) == expected
