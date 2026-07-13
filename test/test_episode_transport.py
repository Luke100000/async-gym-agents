import threading
from dataclasses import replace

from async_gym_agents.episode_transport import EpisodeTransport

TRANSPORT_TEST_TIMEOUT_SECONDS = 0.02
TRANSPORT_THREAD_JOIN_TIMEOUT_SECONDS = 1.0


class TestEpisodeTransport:
    """Worker channels bound memory and expose complete episodes fairly."""

    def test_selects_ready_workers_in_round_robin_order(self, on_policy_packet):
        """A later notification is selected first when it is next in rotation."""
        transport = EpisodeTransport(
            worker_count=3,
            max_pending_episodes=3,
            use_mp=False,
        )
        stop = threading.Event()
        worker_two_packet = replace(on_policy_packet, worker_index=2)

        assert transport.get_sender(2).send(
            worker_two_packet,
            stop,
            TRANSPORT_TEST_TIMEOUT_SECONDS,
        )
        assert transport.get_sender(0).send(
            on_policy_packet,
            stop,
            TRANSPORT_TEST_TIMEOUT_SECONDS,
        )

        assert transport.receive(TRANSPORT_TEST_TIMEOUT_SECONDS).worker_index == 0
        assert transport.receive(TRANSPORT_TEST_TIMEOUT_SECONDS).worker_index == 2
        transport.shutdown()

    def test_bounds_pending_episodes_globally(self, on_policy_packet):
        """A sender times out when all global episode slots are occupied."""
        transport = EpisodeTransport(
            worker_count=3,
            max_pending_episodes=2,
            use_mp=False,
        )
        stop = threading.Event()
        worker_one_packet = replace(on_policy_packet, worker_index=1)
        worker_two_packet = replace(on_policy_packet, worker_index=2)

        assert transport.get_sender(0).send(
            on_policy_packet,
            stop,
            TRANSPORT_TEST_TIMEOUT_SECONDS,
        )
        assert transport.get_sender(1).send(
            worker_one_packet,
            stop,
            TRANSPORT_TEST_TIMEOUT_SECONDS,
        )
        blocked_result = transport.get_sender(2).send(
            worker_two_packet,
            stop,
            TRANSPORT_TEST_TIMEOUT_SECONDS,
        )
        assert not blocked_result
        assert blocked_result.waiting_ns > 0

        transport.receive(TRANSPORT_TEST_TIMEOUT_SECONDS)
        assert transport.get_sender(2).send(
            worker_two_packet,
            stop,
            TRANSPORT_TEST_TIMEOUT_SECONDS,
        )
        assert transport.get_stats().max_pending_episodes == 2
        assert transport.get_stats().sent_episodes == 3
        transport.shutdown()

    def test_allows_one_pending_episode_per_worker(self, on_policy_packet):
        """One fast worker cannot occupy multiple pending transport slots."""
        transport = EpisodeTransport(
            worker_count=2,
            max_pending_episodes=2,
            use_mp=False,
        )
        stop = threading.Event()

        assert transport.get_sender(0).send(
            on_policy_packet,
            stop,
            TRANSPORT_TEST_TIMEOUT_SECONDS,
        )
        assert not transport.get_sender(0).send(
            on_policy_packet,
            stop,
            TRANSPORT_TEST_TIMEOUT_SECONDS,
        )
        assert transport.get_stats().pending_episodes == 1
        transport.shutdown()

    def test_unblocks_indefinite_backpressure_on_shutdown(self, on_policy_packet):
        """A stop event releases a sender waiting without a drop timeout."""
        transport = EpisodeTransport(
            worker_count=1,
            max_pending_episodes=1,
            use_mp=False,
        )
        stop = threading.Event()
        assert transport.get_sender(0).send(
            on_policy_packet,
            stop,
            TRANSPORT_TEST_TIMEOUT_SECONDS,
        )
        results = []

        def send_blocked_episode():
            results.append(transport.get_sender(0).send(on_policy_packet, stop, None))

        sender_thread = threading.Thread(target=send_blocked_episode)
        sender_thread.start()

        stop.set()
        sender_thread.join(TRANSPORT_THREAD_JOIN_TIMEOUT_SECONDS)

        assert not sender_thread.is_alive()
        assert not results[0]
        transport.shutdown()
