import multiprocessing
import queue
from multiprocessing import shared_memory
from typing import Iterable, List, Optional

import numpy as np

from async_gym_agents.transport.constants import (
    MP_COUNTER_TYPE_CODE,
    MP_READINESS_FLAG_TYPE_CODE,
)
from async_gym_agents.transport.data_classes import (
    AssembledRollout,
    FieldSpec,
    TransportStats,
)
from async_gym_agents.transport.spsc_ring import (
    MpMarker,
    RingBuffer,
    RingHandle,
    RingReadiness,
    create_thread_ring,
)


class Transport:
    def __init__(
        self,
        layout: List[FieldSpec],
        n_workers: int,
        ring_capacity: int,
        train_capacity: int,
        use_mp: bool = False,
        mp_ctx: Optional[multiprocessing.context.BaseContext] = None,
    ) -> None:
        assert n_workers >= 1
        self.layout = layout
        self.n_workers = n_workers
        self.ring_capacity = ring_capacity
        self.train_capacity = train_capacity
        self.use_mp = use_mp
        self._mp_ctx = mp_ctx or multiprocessing.get_context()
        self._segments: list[shared_memory.SharedMemory] = []
        self._ready_workers = self._mp_ctx.Queue() if self.use_mp else queue.Queue()

        self.rings: List[RingBuffer] = []
        self._handles: list = []
        for worker_index in range(n_workers):
            ring, handle = self._make_ring(worker_index)
            self.rings.append(ring)
            self._handles.append(handle)

        # Preallocated training buffer the trainer assembles into (one move).
        self._train_arrays = {
            field.name: np.zeros((train_capacity, *field.shape), dtype=field.dtype)
            for field in layout
        }

    def _make_ring(self, worker_index: int) -> tuple[RingBuffer, object]:
        if not self.use_mp:
            ring = create_thread_ring(
                self.ring_capacity,
                self.layout,
                worker_index=worker_index,
                ready_queue=self._ready_workers,
            )
            return ring, ring  # thread workers use the ring object directly

        arrays = {}
        segment_names = {}
        for field in self.layout:
            count = self.ring_capacity * int(np.prod(field.shape, dtype=np.int64))
            nbytes = int(count * field.dtype.itemsize)
            segment = shared_memory.SharedMemory(create=True, size=max(nbytes, 1))
            self._segments.append(segment)
            segment_names[field.name] = segment.name
            arrays[field.name] = np.ndarray(
                (self.ring_capacity, *field.shape),
                dtype=field.dtype,
                buffer=segment.buf,
            )
        values = [self._mp_ctx.Value(MP_COUNTER_TYPE_CODE, 0) for _ in range(3)]
        ready_value = self._mp_ctx.RawValue(MP_READINESS_FLAG_TYPE_CODE, 0)
        readiness = RingReadiness(
            worker_index,
            self._ready_workers,
            ready_value,
        )
        ring = RingBuffer(
            self.ring_capacity,
            self.layout,
            arrays,
            *(MpMarker(value) for value in values),
            readiness=readiness,
        )
        handle = RingHandle(
            self.ring_capacity,
            self.layout,
            segment_names,
            *values,
            ready_value,
            self._ready_workers,
            worker_index,
        )
        return ring, handle

    def worker_ring(self, worker_index: int) -> RingBuffer:
        return self.rings[worker_index]

    def worker_ring_handle(self, worker_index: int):
        """Return what to hand a worker: the ring (threads) or a RingHandle (mp)."""
        return self._handles[worker_index]

    def assemble_available(self) -> AssembledRollout:
        """Move all currently-available rows into the training buffer and release.

        Never waits for any worker. Releases each ring's consumed slots as it
        copies them, so workers can refill during the caller's train cycle.
        """
        return self._assemble_workers(range(self.n_workers))

    def wait_for_ready_workers(self, timeout: Optional[float] = None) -> List[int]:
        """Block until a worker has rows, then include other queued workers."""
        first_worker = self._ready_workers.get(block=True, timeout=timeout)
        ready_workers = [first_worker]
        observed_workers = {first_worker}

        while True:
            try:
                worker_index = self._ready_workers.get_nowait()
            except queue.Empty:
                return ready_workers
            if worker_index not in observed_workers:
                observed_workers.add(worker_index)
                ready_workers.append(worker_index)

    def assemble_ready_workers(self, worker_indices: Iterable[int]) -> AssembledRollout:
        """Move rows only from workers that notified the trainer."""
        return self._assemble_workers(worker_indices, requeue_unassembled=True)

    def _assemble_workers(
        self,
        worker_indices: Iterable[int],
        requeue_unassembled: bool = False,
    ) -> AssembledRollout:
        """Copy selected worker rows into the preallocated training buffer."""
        selected_workers = (
            list(worker_indices) if requeue_unassembled else worker_indices
        )
        offset = 0
        segments: list[tuple[int, int]] = []
        for selected_index, worker_index in enumerate(selected_workers):
            if offset >= self.train_capacity:
                if requeue_unassembled:
                    self._requeue_ready_workers(selected_workers[selected_index:])
                break
            ring = self.rings[worker_index]
            consumed, produced = ring.snapshot()
            available = produced - consumed
            if available <= 0:
                continue
            take = min(available, self.train_capacity - offset)
            rows, _ = ring.read(consumed, consumed + take)
            for name in self._train_arrays:
                self._train_arrays[name][offset : offset + take] = rows[name][:take]
            ring.release(consumed + take)
            segments.append((worker_index, take))
            offset += take

        fields = {name: array[:offset] for name, array in self._train_arrays.items()}
        return AssembledRollout(fields=fields, n_rows=offset, segments=segments)

    def _requeue_ready_workers(self, worker_indices: Iterable[int]) -> None:
        """Preserve notifications that did not fit in this assembled rollout."""
        for worker_index in worker_indices:
            self._ready_workers.put(worker_index)

    def collect_stats(self) -> TransportStats:
        return TransportStats(
            produced=[ring.produced_count for ring in self.rings],
            consumed=[ring.consumed_count for ring in self.rings],
            dropped=[ring.dropped_count for ring in self.rings],
        )

    def calculate_ring_allocated_bytes(self) -> int:
        """Return bytes reserved by all worker ring arrays."""
        return sum(
            array.nbytes for ring in self.rings for array in ring.arrays.values()
        )

    def calculate_train_allocated_bytes(self) -> int:
        """Return bytes reserved by the assembled training arrays."""
        return sum(array.nbytes for array in self._train_arrays.values())

    def shutdown(self) -> None:
        """Release every shared segment; idempotent, owner unlinks."""
        for segment in self._segments:
            try:
                segment.close()
                segment.unlink()
            except FileNotFoundError:
                pass
        self._segments = []
        if self.use_mp and self._ready_workers is not None:
            self._ready_workers.close()
            self._ready_workers.join_thread()
            self._ready_workers = None
