import copy
import pickle

import numpy as np

from async_gym_agents import constants
from async_gym_agents.enums import EpisodeKind
from async_gym_agents.episode_codec import (
    decode_episode_packet,
    encode_episode_batch,
    pack_episode,
    unpack_episode,
)


class TestEpisodeCodec:
    """Complete episodes round-trip through one compact transport packet."""

    def test_round_trips_on_policy_episode(self, on_policy_episode):
        """On-policy arrays and sparse terminal metadata retain their values."""
        batch = pack_episode(on_policy_episode)
        packet = encode_episode_batch(7, 3, batch)

        decoded_batch = decode_episode_packet(packet)
        decoded_episode = unpack_episode(decoded_batch)

        assert packet.worker_index == 7
        assert packet.policy_version == 3
        assert packet.transition_count == 2
        assert packet.episode_kind is EpisodeKind.ON_POLICY
        np.testing.assert_array_equal(
            decoded_episode[0].last_obs,
            on_policy_episode[0].last_obs,
        )
        np.testing.assert_array_equal(
            decoded_episode[1].infos[0]["terminal_observation"],
            on_policy_episode[1].infos[0]["terminal_observation"],
        )
        assert decoded_episode[0].infos == [{}]
        assert decoded_episode[1].reset_infos == [{"seed": 7}]
        np.testing.assert_array_equal(
            decoded_episode[1].environment_rewards,
            np.array([2.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            decoded_episode[1].training_rewards,
            np.array([3.0], dtype=np.float32),
        )
        assert not np.shares_memory(
            decoded_episode[1].environment_rewards,
            decoded_episode[1].training_rewards,
        )

    def test_packs_reward_views_under_explicit_field_names(
        self,
        on_policy_episode,
    ):
        """Packed on-policy episodes retain independent callback and training views."""
        batch = pack_episode(on_policy_episode)

        assert constants.ON_POLICY_ENVIRONMENT_REWARDS_FIELD in batch.fields
        assert constants.ON_POLICY_TRAINING_REWARDS_FIELD in batch.fields
        assert constants.OFF_POLICY_REWARDS_FIELD not in batch.fields
        np.testing.assert_array_equal(
            batch.fields[constants.ON_POLICY_ENVIRONMENT_REWARDS_FIELD],
            np.array([1.0, 2.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            batch.fields[constants.ON_POLICY_TRAINING_REWARDS_FIELD],
            np.array([1.0, 3.0], dtype=np.float32),
        )

    def test_round_trips_off_policy_episode(self, off_policy_episode):
        """Off-policy observations, actions, rewards, and done flags retain shape."""
        batch = pack_episode(off_policy_episode)
        packet = encode_episode_batch(2, 5, batch)

        decoded_episode = unpack_episode(decode_episode_packet(packet))

        np.testing.assert_array_equal(
            decoded_episode[1].buffer_actions,
            off_policy_episode[1].buffer_actions,
        )
        np.testing.assert_array_equal(
            decoded_episode[1].new_obs,
            off_policy_episode[1].new_obs,
        )
        assert decoded_episode[1].dones.tolist() == [True]

    def test_compacts_repeated_transition_objects(self, on_policy_episode):
        """Packing one episode uses less pickle data than separate row arrays."""
        long_episode = [
            copy.deepcopy(on_policy_episode[index % len(on_policy_episode)])
            for index in range(128)
        ]
        raw_payload = pickle.dumps(long_episode, protocol=pickle.HIGHEST_PROTOCOL)

        packet = encode_episode_batch(0, 1, pack_episode(long_episode))

        assert len(packet.payload) < len(raw_payload)
