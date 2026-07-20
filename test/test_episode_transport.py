import queue
import threading
from dataclasses import replace

import pytest

from async_gym_agents.episode_transport import EpisodeTransport

TRANSPORT_TEST_TIMEOUT_SECONDS = 0.02
TRANSPORT_THREAD_JOIN_TIMEOUT_SECONDS = 1.0
TRANSPORT_PROCESS_WAIT_SECONDS = 0.5
TRANSPORT_PROCESS_COMPLETION_TIMEOUT_SECONDS = 5.0


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

    def test_one_worker_can_use_available_global_capacity(self, on_policy_packet):
        """A worker may reserve multiple slots until global capacity is full."""
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
        assert transport.get_sender(0).send(
            on_policy_packet,
            stop,
            TRANSPORT_TEST_TIMEOUT_SECONDS,
        )
        blocked_result = transport.get_sender(0).send(
            on_policy_packet,
            stop,
            TRANSPORT_TEST_TIMEOUT_SECONDS,
        )

        assert not blocked_result
        assert blocked_result.waiting_ns > 0
        assert transport.get_stats().pending_episodes == 2
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

    def test_streams_large_payload_before_sender_returns(
        self,
        active_direct_episode_send,
    ):
        """A worker remains in the send until the trainer drains its large payload."""
        transport, expected_packet, process, _, send_completed = (
            active_direct_episode_send
        )

        assert not send_completed.wait(TRANSPORT_PROCESS_WAIT_SECONDS)

        received_packet = transport.receive(TRANSPORT_PROCESS_WAIT_SECONDS)
        assert received_packet.payload == expected_packet.payload
        assert send_completed.wait(TRANSPORT_PROCESS_COMPLETION_TIMEOUT_SECONDS)
        process.join(TRANSPORT_PROCESS_COMPLETION_TIMEOUT_SECONDS)
        assert process.exitcode == 0

    def test_preserves_off_policy_packet_metadata(self, off_policy_packet):
        """Pipe framing preserves the episode kind and absent policy version."""
        transport = EpisodeTransport(
            worker_count=1,
            max_pending_episodes=1,
            use_mp=False,
        )
        stop = threading.Event()
        assert transport.get_sender(0).send(
            off_policy_packet,
            stop,
            TRANSPORT_TEST_TIMEOUT_SECONDS,
        )

        received_packet = transport.receive(TRANSPORT_TEST_TIMEOUT_SECONDS)

        assert received_packet.policy_version is None
        assert received_packet.episode_kind is off_policy_packet.episode_kind
        assert received_packet.transition_count == off_policy_packet.transition_count
        assert received_packet.payload == off_policy_packet.payload
        transport.shutdown()

    def test_interrupts_in_flight_payload_on_shutdown(
        self,
        active_direct_episode_send,
    ):
        """Closing trainer pipe endpoints stops a worker blocked in a large send."""
        transport, _, process, stop, send_completed = active_direct_episode_send
        assert not send_completed.wait(TRANSPORT_PROCESS_WAIT_SECONDS)

        stop.set()
        transport.interrupt()
        process.join(TRANSPORT_PROCESS_COMPLETION_TIMEOUT_SECONDS)

        assert not process.is_alive()
        assert process.exitcode == 0
        assert not send_completed.is_set()


class TestEpisodeFeeder:
    """Worker feeders overlap bounded episode delivery with environment rollout."""

    def test_accepts_episodes_until_global_capacity_is_full(
        self,
        active_episode_feeder,
    ):
        """Large sends return to the worker while globally reserved packets wait."""
        transport, feeder, packet, _, completed_sends = active_episode_feeder

        first_submission = feeder.submit(packet, TRANSPORT_TEST_TIMEOUT_SECONDS)
        second_submission = feeder.submit(packet, TRANSPORT_TEST_TIMEOUT_SECONDS)
        blocked_submission = feeder.submit(packet, TRANSPORT_TEST_TIMEOUT_SECONDS)

        assert first_submission
        assert second_submission
        assert not blocked_submission
        assert blocked_submission.waiting_ns > 0
        assert completed_sends.empty()
        assert transport.get_stats().pending_episodes == 2

        transport.receive(TRANSPORT_PROCESS_COMPLETION_TIMEOUT_SECONDS)
        transport.receive(TRANSPORT_PROCESS_COMPLETION_TIMEOUT_SECONDS)
        first_send = completed_sends.get(
            timeout=TRANSPORT_PROCESS_COMPLETION_TIMEOUT_SECONDS
        )
        second_send = completed_sends.get(
            timeout=TRANSPORT_PROCESS_COMPLETION_TIMEOUT_SECONDS
        )

        assert first_send
        assert second_send
        assert transport.get_stats().pending_episodes == 0


class TestEpisodeTransportProfiling:
    """Receive metrics distinguish payload delivery from pipe-readiness waits."""

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

        received_packet = transport.receive(TRANSPORT_TEST_TIMEOUT_SECONDS)

        stats = transport.get_stats()
        assert received_packet.worker_index == on_policy_packet.worker_index
        assert received_packet.policy_version == on_policy_packet.policy_version
        assert received_packet.episode_kind is on_policy_packet.episode_kind
        assert received_packet.transition_count == on_policy_packet.transition_count
        assert received_packet.payload == on_policy_packet.payload
        assert stats.receive_attempts == 1
        assert stats.receive_timeouts == 0
        assert stats.received_bytes == len(on_policy_packet.payload)
        assert stats.payload_receive_count == 1
        assert stats.payload_receive_timeouts == 0
        assert stats.payload_receive_ns >= 0
        assert stats.pipe_latency_count == 1
        assert stats.pipe_latency_ns >= 0
        transport.shutdown()

    def test_attributes_empty_transport_timeout_to_readiness_wait(self):
        """An empty transport times out while waiting for a readable worker pipe."""
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
        assert stats.readiness_wait_count == 1
        assert stats.readiness_timeouts == 1
        assert stats.readiness_wait_ns > 0
        assert stats.readiness_timeout_ns > 0
        assert stats.payload_receive_count == 0
        transport.shutdown()

    def test_attributes_pending_pipe_timeout_to_readiness_wait(self):
        """A reserved packet without readable bytes remains a readiness timeout."""
        transport = EpisodeTransport(
            worker_count=1,
            max_pending_episodes=1,
            use_mp=False,
        )
        with transport._pending_episodes.get_lock():
            transport._pending_episodes.value = 1

        with pytest.raises(queue.Empty):
            transport.receive(TRANSPORT_TEST_TIMEOUT_SECONDS)

        stats = transport.get_stats()
        assert stats.receive_attempts == 1
        assert stats.receive_timeouts == 1
        assert stats.receive_timeouts_with_pending == 1
        assert stats.readiness_timeouts == 1
        assert stats.payload_receive_count == 0
        transport.shutdown()
