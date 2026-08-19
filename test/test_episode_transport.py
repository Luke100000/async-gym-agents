import queue
import threading
import time
from dataclasses import replace

import pytest

from async_gym_agents.episode_transport import (
    EpisodeTransport,
    WorkerTransportClosedError,
    _encode_episode_packet_header,
)

TRANSPORT_TEST_TIMEOUT_SECONDS = 0.02
TRANSPORT_THREAD_JOIN_TIMEOUT_SECONDS = 1.0
TRANSPORT_PROCESS_WAIT_SECONDS = 0.5
TRANSPORT_PROCESS_COMPLETION_TIMEOUT_SECONDS = 5.0


class _SlowPayloadConnection:
    """Return a header immediately, then block a released event before the payload."""

    def __init__(self, header: bytes, payload: bytes, release: threading.Event):
        self._header = header
        self._payload = payload
        self._release = release
        self.recv_calls = 0

    def recv_bytes(self) -> bytes:
        self.recv_calls += 1
        if self.recv_calls == 1:
            return self._header
        self._release.wait()
        return self._payload

    def close(self) -> None:
        pass


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

    def test_honors_the_deadline_while_a_payload_is_still_arriving(
        self,
        on_policy_packet,
    ):
        """A slow-arriving payload no longer blocks receive() past its timeout."""
        transport = EpisodeTransport(
            worker_count=1,
            max_pending_episodes=1,
            use_mp=False,
        )
        release = threading.Event()
        transport._receive_connections[0] = _SlowPayloadConnection(
            _encode_episode_packet_header(on_policy_packet),
            on_policy_packet.payload,
            release,
        )

        started = time.monotonic()
        with pytest.raises(queue.Empty):
            transport._receive_packet(
                0, time.monotonic() + TRANSPORT_TEST_TIMEOUT_SECONDS
            )
        elapsed = time.monotonic() - started

        assert elapsed < TRANSPORT_PROCESS_WAIT_SECONDS

        release.set()
        packet = transport._receive_packet(
            0, time.monotonic() + TRANSPORT_PROCESS_COMPLETION_TIMEOUT_SECONDS
        )

        assert packet.payload == on_policy_packet.payload
        # The resumed read reused the in-flight background read instead of
        # starting the payload over from scratch.
        assert transport._receive_connections[0].recv_calls == 2
        transport.shutdown()

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

    def test_attributes_a_closed_sender_to_its_worker(self):
        """Receiver closure errors identify the worker endpoint that disappeared."""
        transport = EpisodeTransport(
            worker_count=1,
            max_pending_episodes=1,
            use_mp=False,
        )
        transport.get_sender(0).close()

        with pytest.raises(
            WorkerTransportClosedError,
            match="worker 0",
        ) as error:
            transport.receive(TRANSPORT_TEST_TIMEOUT_SECONDS)

        assert error.value.worker_index == 0
        transport.shutdown()

    def test_reports_pending_state_across_packet_lifecycle(self, on_policy_packet):
        """The narrow pending query changes only while an episode is queued."""
        transport = EpisodeTransport(
            worker_count=1,
            max_pending_episodes=1,
            use_mp=False,
        )
        stop = threading.Event()

        assert transport.has_pending_episodes() is False
        assert transport.get_sender(0).send(
            on_policy_packet,
            stop,
            TRANSPORT_TEST_TIMEOUT_SECONDS,
        )
        assert transport.has_pending_episodes() is True

        transport.receive(TRANSPORT_TEST_TIMEOUT_SECONDS)

        assert transport.has_pending_episodes() is False
        transport.shutdown()


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
