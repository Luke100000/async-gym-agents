import argparse
import json
import multiprocessing
import time
from typing import Any, Dict, Optional

from async_gym_agents.constants import BYTES_PER_MEBIBYTE
from async_gym_agents.policy_transport import SharedPolicyStore
from benchmarks.ipc_transport_benchmark import (
    join_processes,
    start_processes,
    stop_processes,
    wait_for_processes_ready,
)

DEFAULT_POLICY_PAYLOAD_MEBIBYTES = 2.5
DEFAULT_POLICY_PUBLICATION_COUNT = 10
DEFAULT_POLICY_WORKER_COUNT = 128
POLICY_QUEUE_RECEIVE_TIMEOUT_SECONDS = 60.0
BENCHMARK_DIMENSION_ERROR = "Benchmark {name} must be positive"


def receive_queue_publications(
    update_queue: Any,
    publication_count: int,
    ready: Any,
    start: Any,
) -> None:
    """Drain every policy version from one legacy worker queue."""
    ready.release()
    start.wait()
    for expected_version in range(1, publication_count + 1):
        version, _ = update_queue.get(timeout=POLICY_QUEUE_RECEIVE_TIMEOUT_SECONDS)
        if version != expected_version:
            raise RuntimeError(
                f"Expected policy version {expected_version}, received {version}"
            )


def summarize_publications(
    method: str,
    worker_count: int,
    publication_count: int,
    payload_bytes: int,
    total_seconds: float,
    trainer_copies_per_publication: int,
) -> Dict[str, float | int | str]:
    """Return comparable trainer-side policy publication measurements."""
    return {
        "method": method,
        "workers": worker_count,
        "publication_count": publication_count,
        "payload_bytes": payload_bytes,
        "trainer_copies_per_publication": trainer_copies_per_publication,
        "trainer_bytes_per_publication": (
            payload_bytes * trainer_copies_per_publication
        ),
        "total_seconds": total_seconds,
        "average_milliseconds": total_seconds * 1_000 / publication_count,
    }


def measure_queue_fan_out(
    context: multiprocessing.context.BaseContext,
    worker_count: int,
    publication_count: int,
    payload: bytes,
) -> Dict[str, float | int | str]:
    """Measure trainer time spent putting each policy into every worker queue."""
    update_queues = [context.Queue() for _ in range(worker_count)]
    ready = context.Semaphore(0)
    start = context.Event()
    processes = [
        context.Process(
            target=receive_queue_publications,
            args=(update_queue, publication_count, ready, start),
        )
        for update_queue in update_queues
    ]

    try:
        start_processes(processes)
        wait_for_processes_ready(ready, worker_count)
        start_time = time.perf_counter()
        start.set()
        for version in range(1, publication_count + 1):
            for update_queue in update_queues:
                update_queue.put((version, payload))
        total_seconds = time.perf_counter() - start_time
        join_processes(processes)
        return summarize_publications(
            method="queue_fan_out",
            worker_count=worker_count,
            publication_count=publication_count,
            payload_bytes=len(payload),
            total_seconds=total_seconds,
            trainer_copies_per_publication=worker_count,
        )
    finally:
        start.set()
        stop_processes(processes)
        for update_queue in update_queues:
            update_queue.close()
            update_queue.cancel_join_thread()


def measure_shared_snapshot(
    context: multiprocessing.context.BaseContext,
    worker_count: int,
    publication_count: int,
    payload: bytes,
) -> Dict[str, float | int | str]:
    """Measure trainer time spent publishing one shared A/B snapshot."""
    start_time = time.perf_counter()
    store = SharedPolicyStore.create(
        initial_version=1,
        initial_payload=payload,
        mp_ctx=context,
    )
    try:
        for version in range(2, publication_count + 1):
            store.publish(version, payload)
        total_seconds = time.perf_counter() - start_time
        result = summarize_publications(
            method="shared_snapshot",
            worker_count=worker_count,
            publication_count=publication_count,
            payload_bytes=len(payload),
            total_seconds=total_seconds,
            trainer_copies_per_publication=1,
        )
        result["slot_capacity_bytes"] = store.slot_capacity
        return result
    finally:
        store.close()
        store.unlink()


def run_benchmark(
    worker_count: int,
    publication_count: int,
    payload_bytes: int,
    start_method: Optional[str],
) -> Dict[str, Any]:
    """Compare legacy queue fan-out with shared latest-policy publication."""
    validate_benchmark_dimensions(
        worker_count,
        publication_count,
        payload_bytes,
    )
    context = multiprocessing.get_context(start_method)
    payload = bytes(payload_bytes)
    queue_result = measure_queue_fan_out(
        context,
        worker_count,
        publication_count,
        payload,
    )
    shared_result = measure_shared_snapshot(
        context,
        worker_count,
        publication_count,
        payload,
    )
    return {
        "configuration": {
            "start_method": context.get_start_method(),
            "workers": worker_count,
            "publication_count": publication_count,
            "payload_bytes": payload_bytes,
        },
        "queue_fan_out": queue_result,
        "shared_snapshot": shared_result,
        "comparison": {
            "trainer_publication_speedup": (
                queue_result["total_seconds"] / shared_result["total_seconds"]
            ),
            "trainer_copy_reduction": worker_count,
        },
    }


def validate_benchmark_dimensions(
    worker_count: int,
    publication_count: int,
    payload_bytes: int,
) -> None:
    """Reject dimensions that cannot produce a policy publication measurement."""
    dimensions = {
        "workers": worker_count,
        "publications": publication_count,
        "payload bytes": payload_bytes,
    }
    for name, value in dimensions.items():
        if value <= 0:
            raise ValueError(BENCHMARK_DIMENSION_ERROR.format(name=name))


def parse_arguments() -> argparse.Namespace:
    """Parse policy topology and payload dimensions."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare trainer-side per-worker queue fan-out with one shared "
            "latest-policy snapshot."
        )
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_POLICY_WORKER_COUNT)
    parser.add_argument(
        "--publications",
        type=int,
        default=DEFAULT_POLICY_PUBLICATION_COUNT,
    )
    parser.add_argument(
        "--payload-mib",
        type=float,
        default=DEFAULT_POLICY_PAYLOAD_MEBIBYTES,
    )
    parser.add_argument(
        "--start-method",
        choices=multiprocessing.get_all_start_methods(),
    )
    return parser.parse_args()


def main() -> None:
    """Run both policy transports and print machine-readable results."""
    arguments = parse_arguments()
    result = run_benchmark(
        worker_count=arguments.workers,
        publication_count=arguments.publications,
        payload_bytes=int(arguments.payload_mib * BYTES_PER_MEBIBYTE),
        start_method=arguments.start_method,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
