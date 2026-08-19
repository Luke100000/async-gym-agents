import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from conftest import POLICY_TEST_TIMEOUT_SECONDS, read_policy_snapshot_in_process

from async_gym_agents.policy_transport import SharedPolicyReader


class _CountingLock:
    """Wrap a lock while counting how many times it is actually entered."""

    def __init__(self, lock):
        self._lock = lock
        self.enter_count = 0

    def __enter__(self):
        self.enter_count += 1
        return self._lock.__enter__()

    def __exit__(self, *args):
        return self._lock.__exit__(*args)


class TestSharedPolicySnapshots:
    """Workers observe complete latest-only policy snapshots."""

    def test_reads_initial_and_newer_snapshots(
        self,
        shared_policy_store,
        shared_policy_reader,
    ):
        """A reader receives the initial publication and the next policy version."""
        initial_snapshot = shared_policy_reader.read_if_new(None)

        shared_policy_store.publish(8, b"updated-policy")
        updated_snapshot = shared_policy_reader.read_if_new(initial_snapshot.version)

        assert initial_snapshot.version == 7
        assert initial_snapshot.payload == b"initial-policy-padding"
        assert updated_snapshot.version == 8
        assert updated_snapshot.payload == b"updated-policy"

    def test_returns_none_for_current_version(
        self,
        shared_policy_reader,
    ):
        """A worker avoids copying bytes when its local policy is current."""
        assert shared_policy_reader.read_if_new(7) is None

    def test_returns_none_for_current_version_without_taking_the_lock(
        self,
        shared_policy_reader,
    ):
        """The common no-op case never contends on the shared metadata lock."""
        counting_lock = _CountingLock(shared_policy_reader._descriptor.metadata_lock)
        shared_policy_reader._descriptor = replace(
            shared_policy_reader._descriptor, metadata_lock=counting_lock
        )

        assert shared_policy_reader.read_if_new(7) is None
        assert counting_lock.enter_count == 0

    def test_takes_the_lock_only_when_a_copy_is_needed(
        self,
        shared_policy_store,
        shared_policy_reader,
    ):
        """A genuinely newer snapshot still goes through the locked, seqlock path."""
        shared_policy_store.publish(8, b"updated-policy")
        counting_lock = _CountingLock(shared_policy_reader._descriptor.metadata_lock)
        shared_policy_reader._descriptor = replace(
            shared_policy_reader._descriptor, metadata_lock=counting_lock
        )

        snapshot = shared_policy_reader.read_if_new(7)

        assert snapshot.version == 8
        assert snapshot.payload == b"updated-policy"
        assert counting_lock.enter_count >= 1

    def test_concurrent_readers_receive_identical_publication(
        self,
        shared_policy_store,
    ):
        """Independent readers copy the same immutable version and payload."""
        shared_policy_store.publish(8, b"concurrent-policy")
        readers = [
            SharedPolicyReader(shared_policy_store.get_descriptor()) for _ in range(8)
        ]
        try:
            with ThreadPoolExecutor(max_workers=len(readers)) as executor:
                futures = [
                    executor.submit(reader.read_if_new, None) for reader in readers
                ]
                snapshots = [
                    future.result(timeout=POLICY_TEST_TIMEOUT_SECONDS)
                    for future in futures
                ]
        finally:
            for reader in readers:
                reader.close()

        assert {snapshot.version for snapshot in snapshots} == {8}
        assert {snapshot.payload for snapshot in snapshots} == {b"concurrent-policy"}

    def test_retries_when_a_copied_slot_is_overwritten(
        self,
        paused_policy_copy,
    ):
        """A reader rejects bytes whose slot changed during its private copy."""
        store, reader, copy_started, allow_copy, executor = paused_policy_copy
        read_future = executor.submit(reader.read_if_new, None)
        assert copy_started.wait(POLICY_TEST_TIMEOUT_SECONDS)

        first_publication = executor.submit(store.publish, 8, b"intermediate-policy")
        first_publication.result(timeout=POLICY_TEST_TIMEOUT_SECONDS)
        second_publication = executor.submit(store.publish, 9, b"latest-policy")
        second_publication.result(timeout=POLICY_TEST_TIMEOUT_SECONDS)
        allow_copy.set()

        snapshot = read_future.result(timeout=POLICY_TEST_TIMEOUT_SECONDS)
        assert snapshot.version == 9
        assert snapshot.payload == b"latest-policy"
        assert reader.retry_count >= 1


class TestSharedPolicyPublication:
    """The trainer publishes monotonically versioned snapshots without reader waits."""

    def test_rejects_oversized_payload_without_changing_active_snapshot(
        self,
        shared_policy_store,
        shared_policy_reader,
    ):
        """Capacity errors leave the previously published policy readable."""
        oversized_payload = bytes(shared_policy_store.slot_capacity + 1)

        with pytest.raises(ValueError, match="exceeds shared policy slot capacity"):
            shared_policy_store.publish(8, oversized_payload)

        snapshot = shared_policy_reader.read_if_new(None)
        assert snapshot.version == 7
        assert snapshot.payload == b"initial-policy-padding"

    def test_rejects_non_increasing_policy_version(self, shared_policy_store):
        """A trainer cannot replace the latest snapshot with an older version."""
        with pytest.raises(ValueError, match="newer than published version"):
            shared_policy_store.publish(7, b"duplicate-version")

    def test_failed_write_leaves_previous_snapshot_readable(
        self,
        failing_policy_publication,
    ):
        """An inactive-slot write failure never changes active metadata."""
        store = failing_policy_publication

        with pytest.raises(OSError, match="injected shared-memory write failure"):
            store.publish(8, b"unpublished-policy")

        reader = SharedPolicyReader(store.get_descriptor())
        try:
            snapshot = reader.read_if_new(None)
        finally:
            reader.close()
        assert snapshot.version == 7
        assert snapshot.payload == b"initial-policy-padding"


class TestSharedPolicyLifecycle:
    """Shared policy handles work across spawn and close in owner order."""

    def test_descriptor_is_readable_in_spawned_process(self, shared_policy_store):
        """A spawned worker maps and reads the trainer's initial snapshot."""
        mp_ctx = multiprocessing.get_context("spawn")
        result_queue = mp_ctx.Queue()
        process = mp_ctx.Process(
            target=read_policy_snapshot_in_process,
            args=(shared_policy_store.get_descriptor(), result_queue),
        )

        process.start()
        process.join(POLICY_TEST_TIMEOUT_SECONDS)

        assert process.exitcode == 0
        assert result_queue.get(timeout=POLICY_TEST_TIMEOUT_SECONDS) == (
            7,
            b"initial-policy-padding",
        )
        result_queue.close()
        result_queue.cancel_join_thread()

    def test_unlink_prevents_new_worker_mappings(self, shared_policy_store):
        """The trainer removes the segment after all worker readers have closed."""
        descriptor = shared_policy_store.get_descriptor()

        shared_policy_store.close()
        shared_policy_store.unlink()

        with pytest.raises(FileNotFoundError):
            SharedPolicyReader(descriptor)
