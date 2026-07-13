import queue
import threading
from dataclasses import replace

import pytest

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
            """Capture the result of a sender blocked by transport capacity."""
            results.append(transport.get_sender(0).send(on_policy_packet, stop, None))

        sender_thread = threading.Thread(target=send_blocked_episode)
        sender_thread.start()

        stop.set()
        sender_thread.join(TRANSPORT_THREAD_JOIN_TIMEOUT_SECONDS)

        assert not sender_thread.is_alive()
        assert not results[0]
        transport.shutdown()


class TestEpisodeTransportProfiling:
    """Receive metrics distinguish delivery from notification and timeout waits."""

    def test_records_successful_payload_delivery(self, on_policy_packet):
        """A delivered packet records payload bytes, latency, and receive time."""
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

        transport.receive(TRANSPORT_TEST_TIMEOUT_SECONDS)

        stats = transport.get_stats()
        assert stats.receive_attempts == 1
        assert stats.receive_timeouts == 0
        assert stats.received_bytes == len(on_policy_packet.payload)
        assert stats.payload_receive_count == 1
        assert stats.payload_receive_timeouts == 0
        assert stats.payload_receive_ns >= 0
        assert stats.queue_latency_count == 1
        assert stats.queue_latency_ns >= 0
        transport.shutdown()

    def test_attributes_empty_transport_timeout_to_notification_wait(self):
        """An empty transport times out before any worker notification arrives."""
        transport = EpisodeTransport(
            worker_count=1,
            max_pending_episodes=1,
            use_mp=False,
        )

        with pytest.raises(queue.Empty):
            transport.receive(TRANSPORT_TEST_TIMEOUT_SECONDS)

        stats = transport.get_stats()
        assert stats.receive_attempts == 1
        assert stats.receive_timeouts == 1
        assert stats.receive_timeouts_with_pending == 0
        assert stats.ready_notification_count == 1
        assert stats.ready_notification_timeouts == 1
        assert stats.ready_notification_ns > 0
        assert stats.ready_notification_timeout_ns > 0
        assert stats.payload_receive_count == 0
        transport.shutdown()

    def test_attributes_announced_packet_timeout_to_payload_wait(self):
        """A notification without delivered bytes is reported as a payload timeout."""
        transport = EpisodeTransport(
            worker_count=1,
            max_pending_episodes=1,
            use_mp=False,
        )
        with transport._pending_episodes.get_lock():
            transport._pending_episodes.value = 1
        transport._ready_queue.put(0)

        with pytest.raises(queue.Empty):
            transport.receive(TRANSPORT_TEST_TIMEOUT_SECONDS)

        stats = transport.get_stats()
        assert stats.receive_attempts == 1
        assert stats.receive_timeouts == 1
        assert stats.receive_timeouts_with_pending == 1
        assert stats.ready_notification_timeouts == 0
        assert stats.payload_receive_count == 1
        assert stats.payload_receive_timeouts == 1
        assert stats.payload_receive_ns > 0
        assert stats.payload_receive_timeout_ns > 0
        transport.shutdown()
