import threading
from multiprocessing.context import BaseContext
from multiprocessing.shared_memory import SharedMemory
from typing import Optional

from async_gym_agents import constants
from async_gym_agents.data_classes import (
    PolicySnapshot,
    SharedPolicyDescriptor,
    SharedPolicyStats,
)


def _calculate_slot_capacity(payload_size: int) -> int:
    return max(1, 1 << (payload_size - 1).bit_length())


def _calculate_write_sequence(current_sequence: int) -> int:
    if current_sequence % 2 == 0:
        return current_sequence + 1
    return current_sequence + 2


class SharedPolicyStore:
    def __init__(
        self,
        shared_memory: SharedMemory,
        descriptor: SharedPolicyDescriptor,
    ) -> None:
        self._shared_memory = shared_memory
        self._descriptor = descriptor
        self._writer_lock = threading.Lock()
        self._publication_count = 1
        self._publication_failures = 0
        self._closed = False
        self._unlinked = False

    @classmethod
    def create(
        cls,
        initial_version: int,
        initial_payload: bytes,
        mp_ctx: BaseContext,
    ) -> "SharedPolicyStore":
        """Create and publish the first shared policy snapshot."""
        slot_capacity = _calculate_slot_capacity(len(initial_payload))
        shared_memory = SharedMemory(
            create=True,
            size=constants.POLICY_SNAPSHOT_SLOT_COUNT * slot_capacity,
        )
        descriptor = SharedPolicyDescriptor(
            shared_memory_name=shared_memory.name,
            slot_capacity=slot_capacity,
            active_slot=mp_ctx.Value(
                constants.SHARED_SLOT_INDEX_TYPE_CODE,
                constants.POLICY_INITIAL_SLOT_INDEX,
                lock=False,
            ),
            published_version=mp_ctx.Value(
                constants.SHARED_COUNTER_TYPE_CODE,
                initial_version,
                lock=False,
            ),
            slot_sizes=mp_ctx.Array(
                constants.SHARED_SIZE_TYPE_CODE,
                constants.POLICY_SNAPSHOT_SLOT_COUNT,
                lock=False,
            ),
            slot_versions=mp_ctx.Array(
                constants.SHARED_COUNTER_TYPE_CODE,
                [initial_version, constants.POLICY_UNPUBLISHED_VERSION],
                lock=False,
            ),
            slot_sequences=mp_ctx.Array(
                constants.SHARED_COUNTER_TYPE_CODE,
                [2, 0],
                lock=False,
            ),
            metadata_lock=mp_ctx.Lock(),
        )
        store = cls(shared_memory, descriptor)
        try:
            store._copy_payload(constants.POLICY_INITIAL_SLOT_INDEX, initial_payload)
            descriptor.slot_sizes[constants.POLICY_INITIAL_SLOT_INDEX] = len(
                initial_payload
            )
        except BaseException:
            shared_memory.close()
            shared_memory.unlink()
            raise
        return store

    @property
    def slot_capacity(self) -> int:
        return self._descriptor.slot_capacity

    def get_descriptor(self) -> SharedPolicyDescriptor:
        """Return the spawn-picklable handles workers need to open the store."""
        return self._descriptor

    def get_stats(self) -> SharedPolicyStats:
        """Return current publication and capacity counters."""
        with self._descriptor.metadata_lock:
            active_slot = self._descriptor.active_slot.value
            published_version = self._descriptor.published_version.value
            payload_bytes = self._descriptor.slot_sizes[active_slot]
        return SharedPolicyStats(
            published_version=published_version,
            payload_bytes=payload_bytes,
            slot_capacity_bytes=self._descriptor.slot_capacity,
            publication_count=self._publication_count,
            publication_failures=self._publication_failures,
        )

    def publish(self, version: int, payload: bytes) -> None:
        """Copy a policy into the inactive slot and atomically publish it."""
        if self._closed:
            raise RuntimeError("Cannot publish to a closed shared policy store")
        if len(payload) > self._descriptor.slot_capacity:
            self._publication_failures += 1
            raise ValueError(
                f"Policy payload size {len(payload)} exceeds shared policy slot "
                f"capacity {self._descriptor.slot_capacity}"
            )

        with self._writer_lock:
            with self._descriptor.metadata_lock:
                published_version = self._descriptor.published_version.value
                if version <= published_version:
                    self._publication_failures += 1
                    raise ValueError(
                        f"Policy version {version} must be newer than published "
                        f"version {published_version}"
                    )
                inactive_slot = (
                    constants.POLICY_SNAPSHOT_SLOT_COUNT
                    - 1
                    - self._descriptor.active_slot.value
                )
                write_sequence = _calculate_write_sequence(
                    self._descriptor.slot_sequences[inactive_slot]
                )
                self._descriptor.slot_sequences[inactive_slot] = write_sequence

            try:
                self._copy_payload(inactive_slot, payload)
            except BaseException:
                self._publication_failures += 1
                raise

            with self._descriptor.metadata_lock:
                self._descriptor.slot_sizes[inactive_slot] = len(payload)
                self._descriptor.slot_versions[inactive_slot] = version
                self._descriptor.slot_sequences[inactive_slot] = write_sequence + 1
                self._descriptor.active_slot.value = inactive_slot
                self._descriptor.published_version.value = version
                self._publication_count += 1

    def close(self) -> None:
        """Close the trainer's mapping without removing the shared segment."""
        if self._closed:
            return
        self._closed = True
        self._shared_memory.close()

    def unlink(self) -> None:
        """Remove the trainer-owned segment after every worker mapping is closed."""
        if self._unlinked:
            return
        self._unlinked = True
        try:
            self._shared_memory.unlink()
        except FileNotFoundError:
            pass

    def _copy_payload(self, slot_index: int, payload: bytes) -> None:
        offset = slot_index * self._descriptor.slot_capacity
        self._shared_memory.buf[offset : offset + len(payload)] = payload


class SharedPolicyReader:
    def __init__(self, descriptor: SharedPolicyDescriptor) -> None:
        self._descriptor = descriptor
        self._shared_memory = SharedMemory(name=descriptor.shared_memory_name)
        self._retry_count = 0
        self._closed = False

    @property
    def retry_count(self) -> int:
        return self._retry_count

    def read_if_new(self, local_version: Optional[int]) -> Optional[PolicySnapshot]:
        """Copy and validate the latest snapshot when the worker is behind."""
        if self._closed:
            raise RuntimeError("Cannot read from a closed shared policy reader")

        # Lock-free fast path for the common "nothing new" case: published_version
        # is a lock=False Value, so this single-field read never contends with
        # other workers or the writer. Only fall through to the locked, seqlock
        # path below when there is actually a newer snapshot to copy.
        if self._descriptor.published_version.value == local_version:
            return None

        while True:
            with self._descriptor.metadata_lock:
                active_slot = self._descriptor.active_slot.value
                published_version = self._descriptor.published_version.value
                if published_version == local_version:
                    return None
                payload_size = self._descriptor.slot_sizes[active_slot]
                slot_version = self._descriptor.slot_versions[active_slot]
                sequence = self._descriptor.slot_sequences[active_slot]

            if sequence % 2 != 0:
                self._retry_count += 1
                continue

            payload = self._copy_payload(active_slot, payload_size)

            with self._descriptor.metadata_lock:
                snapshot_is_stable = (
                    self._descriptor.active_slot.value == active_slot
                    and self._descriptor.published_version.value == published_version
                    and self._descriptor.slot_sizes[active_slot] == payload_size
                    and self._descriptor.slot_versions[active_slot] == slot_version
                    and self._descriptor.slot_sequences[active_slot] == sequence
                    and sequence % 2 == 0
                )

            if snapshot_is_stable:
                return PolicySnapshot(version=slot_version, payload=payload)
            self._retry_count += 1

    def close(self) -> None:
        """Release this worker's read-only shared-memory mapping."""
        if self._closed:
            return
        self._closed = True
        self._shared_memory.close()

    def _copy_payload(self, slot_index: int, payload_size: int) -> bytes:
        offset = slot_index * self._descriptor.slot_capacity
        return bytes(self._shared_memory.buf[offset : offset + payload_size])
