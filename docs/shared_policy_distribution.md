# Shared Latest-Policy Distribution

Status: deferred design; not implemented by the episode-feeder change.

## Context

The trainer currently distributes every policy update through one
`multiprocessing.Queue` per worker. With 128 workers, `_push_policy_update()`
clears and writes 128 queues from the trainer thread. The policy is serialized
only once, but each queue has its own feeder and pipe, so publication performs
O(worker count) queue operations and copies the same payload into every worker
channel.

Profiling from the PPO experiments measured approximately:

- 5 ms to serialize the policy once;
- 0.5 seconds to broadcast the serialized policy to 128 worker queues; and
- 5-6 ms per worker to deserialize and load a policy update on CPU.

Workers already use latest-only semantics. At an episode boundary, each worker
drains its update queue, discards superseded versions, and loads only the newest
one. Episode packets retain the policy version used to generate them, and the
trainer calculates policy lag from that version.

The replacement should preserve these semantics while removing synchronous
O(worker count) publication from the trainer.

## Requirements

The implementation must:

1. Serialize each trainer policy once.
2. Publish one immutable snapshot shared by every worker.
3. Keep the trainer independent of worker scheduling and worker failures.
4. Prevent workers from observing partially written serialized bytes.
5. Let workers copy the snapshot without holding a long global lock.
6. Preserve monotonically increasing policy versions and episode policy-lag
   reporting.
7. Keep worker policy instances on CPU and trainer policy instances on their
   configured device.
8. Work with both `fork` and `spawn` multiprocessing start methods.
9. Close worker mappings before the trainer unlinks the shared-memory segment.

It does not need to share live PyTorch tensors. Workers may continue using
`torch.load(..., map_location="cpu", weights_only=True)` after copying stable
serialized bytes into process-local memory.

## Recommended architecture

Use a single shared-memory segment containing two fixed-capacity slots, A and
B. A small set of multiprocessing metadata values identifies the active slot.
Per-slot sequence counters provide seqlock-style validation, so the trainer can
overwrite an inactive slot even if a slow worker was copying that slot from an
older publication. The worker discards a copy whenever the sequence changed
during the copy.

This provides non-blocking publication without reader counts, reader locks, or
128 acknowledgement messages.

### Production types and placement

Add the following modules and types when implementing the design:

- `async_gym_agents/policy_transport.py`
  - `SharedPolicyStore`: trainer-owned writer and lifecycle owner.
  - `SharedPolicyReader`: worker-owned read-only mapping.
- `async_gym_agents/data_classes.py`
  - `SharedPolicyDescriptor`: picklable shared-memory name, slot capacity, and
    multiprocessing metadata handles passed to workers.
  - `PolicySnapshot`: immutable `version` and private `payload` returned to a
    worker.
- `async_gym_agents/constants.py`
  - `POLICY_SNAPSHOT_SLOT_COUNT = 2`.
  - Shared counter type codes used by the descriptor.

The store API should be:

```python
from typing import Optional


class SharedPolicyStore:
    @classmethod
    def create(cls, initial_version: int, initial_payload: bytes, mp_ctx): ...

    def get_descriptor(self) -> SharedPolicyDescriptor: ...

    def publish(self, version: int, payload: bytes) -> None: ...

    def close(self) -> None: ...

    def unlink(self) -> None: ...


class SharedPolicyReader:
    def read_if_new(
        self,
        local_version: Optional[int],
    ) -> Optional[PolicySnapshot]: ...

    def close(self) -> None: ...
```

`SharedPolicyStore` is the only writer. Each worker creates one
`SharedPolicyReader` from the descriptor after process startup.

### Shared-memory layout

Allocate one `multiprocessing.shared_memory.SharedMemory` segment:

```text
offset 0                                      slot_capacity
  |---------------- slot A ----------------------|
  |---------------- slot B ----------------------|
slot_capacity                              2 * slot_capacity
```

Calculate `slot_capacity` as the next power of two greater than or equal to the
initial serialized payload size. The policy topology is fixed after worker
startup, so later payloads should fit the same capacity. `publish()` must raise
a descriptive error before modifying metadata if a payload exceeds the slot
capacity. Resizing requires stopping workers and constructing a new store.

The descriptor contains these multiprocessing primitives:

- `active_slot`: index of the currently published slot;
- `published_version`: version in the active slot;
- `slot_sizes[2]`: valid byte count for each slot;
- `slot_versions[2]`: policy version stored in each slot;
- `slot_sequences[2]`: monotonically increasing write sequence per slot; and
- `metadata_lock`: protects metadata snapshots and publication swaps.

All counters must use fixed-width multiprocessing values. Versions and
sequences should use signed or unsigned 64-bit storage. Slot sizes must use an
unsigned type capable of representing `slot_capacity`.

## Publication protocol

The trainer writes only the inactive slot. It never waits for readers.

Given `inactive_slot = 1 - active_slot`, `publish(version, payload)` performs:

1. Validate that `version` is newer than `published_version`.
2. Validate that `len(payload) <= slot_capacity`.
3. Acquire `metadata_lock`.
4. Re-read `active_slot` and choose the other slot.
5. Advance the inactive slot sequence to a new odd value, marking a write in
   progress.
6. Release `metadata_lock`.
7. Copy the payload into the inactive slot without holding the lock.
8. Acquire `metadata_lock` again.
9. Store the slot size and slot version.
10. Advance the slot sequence to the next even value, marking a complete
    snapshot.
11. Set `active_slot` and `published_version` to the new publication.
12. Release `metadata_lock`.

The active slot remains immutable until step 11. A worker still copying the old
slot when a later publication begins overwriting it will observe a changed or
odd sequence and retry before deserializing anything.

If writing fails after the sequence becomes odd, leave the active slot and
published version unchanged. The next write to that inactive slot must advance
its sequence to a new odd value before copying. An incomplete inactive slot is
never published.

## Worker read protocol

`read_if_new(local_version)` loops until it obtains a stable latest snapshot or
finds that the local version is current:

1. Acquire `metadata_lock`.
2. Snapshot `active_slot`, `published_version`, the active slot size, and its
   sequence.
3. Return `None` if `published_version == local_version`.
4. Release `metadata_lock`.
5. Retry if the sequence is odd.
6. Copy exactly the snapshotted number of bytes into a process-local `bytes`
   object.
7. Acquire `metadata_lock` again.
8. Confirm that all of the following still match the first snapshot:
   `active_slot`, `published_version`, slot size, and slot sequence.
9. Release `metadata_lock`.
10. Retry from step 1 if any value changed or the sequence is odd.
11. Return `PolicySnapshot(version, payload)`.

The worker calls `torch.load` only after this validation. No shared-memory view
may escape `read_if_new()`.

The metadata lock is held only for small scalar snapshots. The potentially
large byte copy and `torch.load` happen without the lock. Therefore, 128
workers may copy concurrently and cannot block trainer publication.

## Agent integration

Replace the current policy queues in these steps:

1. In `pre_collect_preparation()`, serialize the trainer state dictionary as it
   does today.
2. On the first call, create `SharedPolicyStore` with that payload before
   workers start. On later calls, publish into the inactive slot.
3. Increment the trainer policy version once for each successfully published
   trainer state and store that exact version in the shared metadata.
4. Pass `SharedPolicyDescriptor`, rather than one update queue, to every worker.
5. Replace `copy_policy_from_queue()` with `copy_policy_from_store()`.
6. Keep the existing episode-boundary update point. The worker compares its
   local version, copies the latest stable snapshot, loads it on CPU, and then
   updates `_policy_version`.
7. Remove `_update_queues`, `_clear_queue()`, `_push_policy_update()`, and their
   save-exclusion and shutdown handling.
8. Close each worker reader in the worker `finally` block.
9. During trainer shutdown, stop and join workers first, close the trainer
   mapping second, and unlink the shared-memory segment last.

The initial snapshot must be published before workers enter their blocking
initial policy load. A worker should not need a notification primitive: it
checks the shared published version at the same episode boundaries where it
currently polls its queue.

## Version and on-policy semantics

The version stored in an episode packet remains the version actually loaded by
that worker. The trainer's `_version` remains the version of the latest
published policy. Existing transition-weighted average and maximum policy-lag
metrics therefore remain valid.

Publishing one shared latest snapshot does not increase the intended lag. It
removes per-worker queue delivery delay and naturally discards intermediate
versions when a worker completes a long episode.

## Profiling

Keep policy serialization and publication separate in the main profiler:

- `policy_serialization`: `torch.save` into process-local bytes;
- `policy_publication`: inactive-slot copy plus metadata swap;
- `policy_snapshot_copy`: worker stable shared-memory copy;
- `policy_loading`: worker `torch.load` and `load_state_dict`.

Also report:

- published policy version;
- payload bytes and slot capacity;
- worker snapshot retries caused by concurrent publication; and
- publication failures or oversized payload errors.

The current `policy_broadcast` metric may remain as a compatibility alias for
`policy_publication` for one release, but it must no longer include worker-side
copy or loading time.

## Tests required before rollout

Add behavior-focused tests proving:

1. Every worker reader obtains the initial snapshot.
2. A newer publication replaces the active slot and version.
3. `read_if_new()` returns `None` for the current version.
4. A worker copying an old slot while it is overwritten detects the sequence
   change and retries without deserializing torn bytes.
5. Concurrent readers obtain identical payloads and versions.
6. Publication does not wait for a paused reader.
7. An oversized payload fails before active metadata changes.
8. A failed inactive-slot write leaves the prior snapshot readable.
9. The descriptor and reader work under the `spawn` start method.
10. Worker shutdown closes mappings and trainer shutdown unlinks the segment.
11. Episode policy versions and existing policy-lag metrics remain correct.

The integration benchmark should compare the old 128-queue publication with
the shared store using the real serialized policy size. Success means trainer
publication time no longer grows linearly with worker count, workers load the
same version, and training policy-lag behavior does not regress.

## Operational risks

- Shared-memory segments must be unlinked exactly once by the trainer owner.
- Windows and Python resource trackers require explicit close/unlink ordering.
- A policy topology change can exceed fixed slot capacity and must fail loudly.
- Sequence validation is mandatory; copying from an inactive slot without it
  can produce a torn pickle.
- Workers must copy to private bytes before `torch.load`; deserializing directly
  from a mutable shared view is unsafe.
