"""Single-producer / single-consumer ring over fixed-shape numeric fields.

The producer (a worker) writes into slot ``produced % capacity`` and only then
advances ``produced``; the consumer (the trainer) reads slots in
``[consumed, produced)`` and advances ``consumed`` once it has copied them out.
Producer and consumer touch disjoint markers, so the only synchronized state is
the two counters. On a full ring the producer drops the *new* transition (never
blocks, never overwrites a slot the consumer might be reading) and counts it.
"""

import threading
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any, Dict, List, Protocol

import numpy as np

from async_gym_agents.transport.data_classes import FieldSpec


class Marker(Protocol):
    """A shared integer counter with barrier-providing access."""

    def get(self) -> int: ...
    def set(self, value: int) -> None: ...
    def add(self, delta: int) -> None: ...


class ThreadMarker:
    """Marker backed by a plain int + lock, for thread-mode workers."""

    def __init__(self) -> None:
        self._value = 0
        self._lock = threading.Lock()

    def get(self) -> int:
        with self._lock:
            return self._value

    def set(self, value: int) -> None:
        with self._lock:
            self._value = value

    def add(self, delta: int) -> None:
        with self._lock:
            self._value += delta


class MpMarker:
    """Marker backed by a shared ``multiprocessing.Value`` (its own lock)."""

    def __init__(self, value):
        self._value = value

    def get(self) -> int:
        with self._value.get_lock():
            return self._value.value

    def set(self, value: int) -> None:
        with self._value.get_lock():
            self._value.value = value

    def add(self, delta: int) -> None:
        with self._value.get_lock():
            self._value.value += delta


class RingBuffer:
    """SPSC ring over pre-created per-field arrays and markers.

    The arrays/markers are injected so the same logic serves both thread-mode
    (plain numpy + ThreadMarker) and process-mode (shared-memory numpy views +
    shared-value markers) backends.
    """

    def __init__(
        self,
        capacity: int,
        fields: List[FieldSpec],
        arrays: Dict[str, np.ndarray],
        produced: Marker,
        consumed: Marker,
        dropped: Marker,
    ) -> None:
        self.capacity = capacity
        self.fields = fields
        self.arrays = arrays
        self._produced = produced
        self._consumed = consumed
        self._dropped = dropped

    def try_write(self, values: Dict[str, np.ndarray]) -> bool:
        """Publish one transition; return False (without counting a drop) if full."""
        produced = self._produced.get()
        consumed = self._consumed.get()
        if produced - consumed >= self.capacity:
            return False

        idx = produced % self.capacity
        for name, array in self.arrays.items():
            array[idx] = values[name]
        # Publish only after every field is written (no torn reads).
        self._produced.set(produced + 1)
        return True

    def write(self, values: Dict[str, np.ndarray]) -> bool:
        """Publish one transition; count a drop and return False if full."""
        if self.try_write(values):
            return True
        self._dropped.add(1)
        return False

    def note_drop(self) -> None:
        self._dropped.add(1)

    def snapshot(self) -> tuple[int, int]:
        """Return (consumed, produced) markers at this instant; never blocks."""
        return self._consumed.get(), self._produced.get()

    def read(self, consumed: int, produced: int) -> tuple[Dict[str, np.ndarray], int]:
        """Copy rows ``[consumed, produced)`` out; handles wrap-around."""
        count = produced - consumed
        if count <= 0:
            return {}, 0

        start = consumed % self.capacity
        if start + count <= self.capacity:
            index = slice(start, start + count)
        else:
            index = [(start + offset) % self.capacity for offset in range(count)]

        rows = {name: array[index].copy() for name, array in self.arrays.items()}
        return rows, count

    def release(self, up_to: int) -> None:
        """Mark rows consumed up to (exclusive) ``up_to``, freeing the slots."""
        self._consumed.set(up_to)

    @property
    def produced_count(self) -> int:
        return self._produced.get()

    @property
    def consumed_count(self) -> int:
        return self._consumed.get()

    @property
    def dropped_count(self) -> int:
        return self._dropped.get()


def create_thread_ring(capacity: int, fields: List[FieldSpec]) -> RingBuffer:
    """Build an in-process ring (numpy arrays + thread markers)."""
    arrays = {
        field.name: np.zeros((capacity, *field.shape), dtype=field.dtype)
        for field in fields
    }
    return RingBuffer(
        capacity,
        fields,
        arrays,
        produced=ThreadMarker(),
        consumed=ThreadMarker(),
        dropped=ThreadMarker(),
    )


@dataclass
class RingHandle:
    """Picklable description a spawned worker uses to attach to its ring.

    Segments are opened by name (never created or unlinked worker-side); the
    shared ``multiprocessing.Value`` markers are passed through directly.
    """

    capacity: int
    fields: List[FieldSpec]
    segment_names: Dict[str, str]
    produced: Any
    consumed: Any
    dropped: Any

    def attach(self) -> RingBuffer:
        segments = []
        arrays = {}
        for field in self.fields:
            segment = shared_memory.SharedMemory(name=self.segment_names[field.name])
            segments.append(segment)
            arrays[field.name] = np.ndarray(
                (self.capacity, *field.shape), dtype=field.dtype, buffer=segment.buf
            )
        ring = RingBuffer(
            self.capacity,
            self.fields,
            arrays,
            MpMarker(self.produced),
            MpMarker(self.consumed),
            MpMarker(self.dropped),
        )
        # Keep segment handles alive so the numpy views stay valid.
        ring._attached_segments = segments
        return ring


def resolve_ring(ring_or_handle) -> RingBuffer:
    """Return a usable ring from either a direct ring (threads) or a handle."""
    if isinstance(ring_or_handle, RingHandle):
        return ring_or_handle.attach()
    return ring_or_handle
