from functools import partial

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from async_gym_agents.agents.async_agent import get_injected_agent
from async_gym_agents.data_classes import OffPolicyTransition, OnPolicyTransition
from async_gym_agents.envs.buggy_lunar_lander import BuggyLunarLander
from async_gym_agents.envs.multi_env import IndexableMultiEnv
from async_gym_agents.episode_codec import encode_episode_batch, pack_episode

PROCESSES = 8
TEST_EPISODE_SEND_TIMEOUT_SECONDS = 1.0


@pytest.fixture
def taxi_multi_env():
    """Fixture for creating a multi-environment with discrete action space environment."""
    return IndexableMultiEnv([partial(gym.make, "Taxi-v3") for _ in range(PROCESSES)])


@pytest.fixture
def lunar_lander_multi_env():
    """Fixture for creating a multi-environment with continuous action space environment."""
    return IndexableMultiEnv(
        [partial(gym.make, "LunarLanderContinuous-v3") for _ in range(PROCESSES)]
    )


@pytest.fixture
def initialized_on_policy_agent():
    """Create an on-policy agent with initialized transport state and no workers."""
    env = IndexableMultiEnv([partial(gym.make, "Taxi-v3")])
    agent = get_injected_agent(PPO)(
        "MlpPolicy",
        env,
        batch_size=2,
        device="cpu",
        n_steps=2,
    )
    agent._init_collect_state()
    yield agent
    agent.shutdown()


@pytest.fixture
def on_policy_episode():
    """Create a complete two-step on-policy episode with sparse terminal info."""
    return [
        OnPolicyTransition(
            actions=np.array([[0]]),
            values=np.array([0.25], dtype=np.float32),
            log_probs=np.array([-0.5], dtype=np.float32),
            last_obs=np.array([[1.0, 2.0]], dtype=np.float32),
            new_obs=np.array([[2.0, 3.0]], dtype=np.float32),
            rewards=np.array([1.0], dtype=np.float32),
            dones=np.array([False]),
            last_dones=np.array([True]),
            infos=[{}],
            reset_infos=[{}],
        ),
        OnPolicyTransition(
            actions=np.array([[1]]),
            values=np.array([0.5], dtype=np.float32),
            log_probs=np.array([-0.25], dtype=np.float32),
            last_obs=np.array([[2.0, 3.0]], dtype=np.float32),
            new_obs=np.array([[0.0, 0.0]], dtype=np.float32),
            rewards=np.array([2.0], dtype=np.float32),
            dones=np.array([True]),
            last_dones=np.array([False]),
            infos=[
                {
                    "TimeLimit.truncated": True,
                    "terminal_observation": np.array([3.0, 4.0], dtype=np.float32),
                }
            ],
            reset_infos=[{"seed": 7}],
        ),
    ]


@pytest.fixture
def off_policy_episode():
    """Create a complete two-step off-policy episode."""
    return [
        OffPolicyTransition(
            buffer_actions=np.array([[0.1]], dtype=np.float32),
            last_obs=np.array([[1.0]], dtype=np.float32),
            new_obs=np.array([[2.0]], dtype=np.float32),
            rewards=np.array([1.0], dtype=np.float32),
            dones=np.array([False]),
            infos=[{}],
            reset_infos=[{}],
        ),
        OffPolicyTransition(
            buffer_actions=np.array([[0.2]], dtype=np.float32),
            last_obs=np.array([[2.0]], dtype=np.float32),
            new_obs=np.array([[3.0]], dtype=np.float32),
            rewards=np.array([2.0], dtype=np.float32),
            dones=np.array([True]),
            infos=[{}],
            reset_infos=[{}],
        ),
    ]


@pytest.fixture
def on_policy_packet(on_policy_episode):
    """Encode a representative on-policy episode for transport tests."""
    return encode_episode_batch(0, 1, pack_episode(on_policy_episode))


@pytest.fixture
def enqueue_episode_packet():
    """Return a helper that sends a packet through an initialized agent transport."""

    def enqueue(agent, packet):
        sender = agent._episode_transport.get_sender(packet.worker_index)
        assert sender.send(packet, agent._stop, TEST_EPISODE_SEND_TIMEOUT_SECONDS)

    return enqueue


def get_buggy_env(buggy: bool) -> gym.Env:
    """Environment factory for an env returning truncated episodes."""
    return Monitor(
        BuggyLunarLander(
            crash_probability=0.01 if buggy else 0,
            time_limit=1000,
        )
    )
