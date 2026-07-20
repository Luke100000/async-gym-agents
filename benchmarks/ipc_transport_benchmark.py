import argparse
import json
import multiprocessing
import queue
import time
from dataclasses import replace
from multiprocessing.connection import Connection, wait
from typing import Any, Dict, List, Optional

from async_gym_agents.constants import (
    ASSEMBLER_RECEIVE_TIMEOUT_SECONDS,
    BYTES_PER_MEBIBYTE,
    MILLISECONDS_PER_SECOND,
    NANOSECONDS_PER_SECOND,
)
from async_gym_agents.data_classes import EpisodePacket
from async_gym_agents.enums import EpisodeKind
from async_gym_agents.episode_transport import (
    EpisodeFeeder,
    EpisodeSender,
    EpisodeTransport,
)

DEFAULT_MAX_PENDING_EPISODES = 120
DEFAULT_PACKETS_PER_WORKER = 2
DEFAULT_PAYLOAD_MEBIBYTES = 2.5
DEFAULT_WORKER_COUNT = 128
PROCESS_JOIN_TIMEOUT_SECONDS = 30.0
BENCHMARK_STALL_TIMEOUT_SECONDS = 60.0


def send_production_packets(
    sender: EpisodeSender,
    packet: EpisodePacket,
    packet_count: int,
    ready: Any,
    start: Any,
    stop: Any,
) -> None:
    """Send representative packets through the production pipe transport."""
    feeder = EpisodeFeeder(sender=sender, stop=stop)
    try:
        ready.release()
        start.wait()
        for _ in range(packet_count):
            if not feeder.submit(packet, timeout=None):
                raise RuntimeError("Pipe benchmark feeder stopped before completion")
    finally:
        feeder.shutdown()


def send_pipe_payloads(
    connection: Connection,
    payload: bytes,
    packet_count: int,
    ready: Any,
    start: Any,
) -> None:
    """Send representative payloads directly through one worker pipe."""
    ready.release()
    start.wait()
    try:
        for _ in range(packet_count):
            connection.send_bytes(payload)
    finally:
        connection.close()


def summarize_benchmark(
    method: str,
    worker_count: int,
    packet_count: int,
    payload_bytes: int,
    elapsed_seconds: float,
    consumer_cpu_seconds: float,
) -> Dict[str, float | int | str]:
    """Return comparable throughput fields for one IPC implementation."""
    total_packets = worker_count * packet_count
    total_bytes = total_packets * payload_bytes
    return {
        "method": method,
        "workers": worker_count,
        "packets": total_packets,
        "payload_bytes": payload_bytes,
        "total_bytes": total_bytes,
        "elapsed_seconds": elapsed_seconds,
        "consumer_cpu_seconds": consumer_cpu_seconds,
        "consumer_cpu_fraction": consumer_cpu_seconds / elapsed_seconds,
        "packets_per_second": total_packets / elapsed_seconds,
        "mib_per_second": total_bytes / BYTES_PER_MEBIBYTE / elapsed_seconds,
    }


def measure_production_transport(
    context: multiprocessing.context.BaseContext,
    worker_count: int,
    packet_count: int,
    payload: bytes,
    max_pending_episodes: int,
) -> Dict[str, Any]:
    """Measure the production framed-payload pipe implementation."""
    transport = EpisodeTransport(
        worker_count=worker_count,
        max_pending_episodes=max_pending_episodes,
        use_mp=True,
        mp_ctx=context,
    )
    start = context.Event()
    stop = context.Event()
    ready = context.Semaphore(0)
    base_packet = EpisodePacket(
        worker_index=0,
        policy_version=0,
        episode_kind=EpisodeKind.ON_POLICY,
        transition_count=1,
        payload=payload,
    )
    processes = [
        context.Process(
            target=send_production_packets,
            args=(
                transport.get_sender(worker_index),
                replace(base_packet, worker_index=worker_index),
                packet_count,
                ready,
                start,
                stop,
            ),
        )
        for worker_index in range(worker_count)
    ]

    try:
        start_processes(processes)
        transport.close_parent_senders()
        wait_for_processes_ready(ready, worker_count)
        start_time = time.perf_counter()
        consumer_cpu_start = time.process_time()
        start.set()
        received_packet_count = 0
        expected_packet_count = worker_count * packet_count
        while received_packet_count < expected_packet_count:
            try:
                transport.receive(ASSEMBLER_RECEIVE_TIMEOUT_SECONDS)
            except queue.Empty:
                if time.perf_counter() - start_time > BENCHMARK_STALL_TIMEOUT_SECONDS:
                    raise TimeoutError(
                        "Production pipe benchmark stopped making progress"
                    )
                continue
            received_packet_count += 1
        elapsed_seconds = time.perf_counter() - start_time
        consumer_cpu_seconds = time.process_time() - consumer_cpu_start
        join_processes(processes)

        result = summarize_benchmark(
            "production_pipe",
            worker_count,
            packet_count,
            len(payload),
            elapsed_seconds,
            consumer_cpu_seconds,
        )
        stats = transport.get_stats()
        successful_payload_count = (
            stats.payload_receive_count - stats.payload_receive_timeouts
        )
        successful_payload_ns = (
            stats.payload_receive_ns - stats.payload_receive_timeout_ns
        )
        result.update(
            {
                "receive_attempts": stats.receive_attempts,
                "receive_timeouts": stats.receive_timeouts,
                "receive_timeouts_with_pending": (stats.receive_timeouts_with_pending),
                "payload_receive_timeouts": stats.payload_receive_timeouts,
                "payload_receive_avg_ms": calculate_average_milliseconds(
                    successful_payload_ns,
                    successful_payload_count,
                ),
                "payload_timeout_avg_ms": calculate_average_milliseconds(
                    stats.payload_receive_timeout_ns,
                    stats.payload_receive_timeouts,
                ),
                "pipe_latency_avg_ms": calculate_average_milliseconds(
                    stats.pipe_latency_ns,
                    stats.pipe_latency_count,
                ),
            }
        )
        return result
    finally:
        stop.set()
        start.set()
        transport.shutdown()
        stop_processes(processes)


def measure_pipe_transport(
    context: multiprocessing.context.BaseContext,
    worker_count: int,
    packet_count: int,
    payload: bytes,
) -> Dict[str, Any]:
    """Measure one raw unidirectional pipe per worker with direct byte sends."""
    connections = [context.Pipe(duplex=False) for _ in range(worker_count)]
    receive_connections = [connection[0] for connection in connections]
    send_connections = [connection[1] for connection in connections]
    start = context.Event()
    ready = context.Semaphore(0)
    processes = [
        context.Process(
            target=send_pipe_payloads,
            args=(connection, payload, packet_count, ready, start),
        )
        for connection in send_connections
    ]

    try:
        start_processes(processes)
        for connection in send_connections:
            connection.close()
        wait_for_processes_ready(ready, worker_count)
        start_time = time.perf_counter()
        consumer_cpu_start = time.process_time()
        start.set()
        receive_pipe_payloads(
            receive_connections,
            worker_count * packet_count,
        )
        elapsed_seconds = time.perf_counter() - start_time
        consumer_cpu_seconds = time.process_time() - consumer_cpu_start
        join_processes(processes)
        return summarize_benchmark(
            "raw_pipe",
            worker_count,
            packet_count,
            len(payload),
            elapsed_seconds,
            consumer_cpu_seconds,
        )
    finally:
        start.set()
        stop_processes(processes)
        for connection in [*receive_connections, *send_connections]:
            connection.close()


def receive_pipe_payloads(
    receive_connections: List[Connection],
    expected_packet_count: int,
) -> None:
    """Drain pipe frames fairly until every expected payload arrives."""
    active_connections = set(receive_connections)
    received_packet_count = 0
    while received_packet_count < expected_packet_count:
        ready_connections = wait(
            active_connections,
            timeout=BENCHMARK_STALL_TIMEOUT_SECONDS,
        )
        if not ready_connections:
            raise queue.Empty("Timed out waiting for pipe benchmark payload")
        for connection in ready_connections:
            try:
                connection.recv_bytes()
            except EOFError:
                active_connections.remove(connection)
                continue
            received_packet_count += 1


def calculate_average_milliseconds(total_ns: int, count: int) -> float:
    """Return average milliseconds for a cumulative nanosecond measurement."""
    if count == 0:
        return 0.0
    return total_ns / count / (NANOSECONDS_PER_SECOND / MILLISECONDS_PER_SECOND)


def start_processes(processes: List[multiprocessing.Process]) -> None:
    """Start every benchmark producer before the timed start barrier opens."""
    for process in processes:
        process.start()


def wait_for_processes_ready(ready: Any, process_count: int) -> None:
    """Exclude producer startup and imports from the timed benchmark region."""
    for _ in range(process_count):
        if not ready.acquire(timeout=BENCHMARK_STALL_TIMEOUT_SECONDS):
            raise TimeoutError("Benchmark producer did not reach the start barrier")


def join_processes(processes: List[multiprocessing.Process]) -> None:
    """Require every benchmark producer to exit successfully."""
    for process in processes:
        process.join(PROCESS_JOIN_TIMEOUT_SECONDS)
        if process.is_alive():
            raise TimeoutError("Benchmark producer did not exit")
        if process.exitcode != 0:
            raise RuntimeError(
                f"Benchmark producer exited with code {process.exitcode}"
            )


def stop_processes(processes: List[multiprocessing.Process]) -> None:
    """Stop only producers left alive after normal benchmark cleanup."""
    for process in processes:
        if not process.is_alive():
            continue
        process.kill()
        process.join(PROCESS_JOIN_TIMEOUT_SECONDS)


def run_benchmark(
    worker_count: int,
    packet_count: int,
    payload_bytes: int,
    max_pending_episodes: int,
    start_method: Optional[str],
) -> Dict[str, Any]:
    """Compare framed production pipes against raw payload pipe throughput."""
    validate_benchmark_dimensions(
        worker_count,
        packet_count,
        payload_bytes,
        max_pending_episodes,
    )
    context = multiprocessing.get_context(start_method)
    payload = bytes(payload_bytes)
    production_result = measure_production_transport(
        context,
        worker_count,
        packet_count,
        payload,
        max_pending_episodes,
    )
    raw_pipe_result = measure_pipe_transport(
        context,
        worker_count,
        packet_count,
        payload,
    )
    return {
        "configuration": {
            "start_method": context.get_start_method(),
            "workers": worker_count,
            "packets_per_worker": packet_count,
            "payload_bytes": payload_bytes,
            "max_pending_episodes": max_pending_episodes,
        },
        "production_pipe": production_result,
        "raw_pipe": raw_pipe_result,
        "comparison": {
            "raw_pipe_throughput_ratio": (
                raw_pipe_result["mib_per_second"] / production_result["mib_per_second"]
            ),
            "production_elapsed_time_ratio": (
                production_result["elapsed_seconds"]
                / raw_pipe_result["elapsed_seconds"]
            ),
        },
    }


def validate_benchmark_dimensions(
    worker_count: int,
    packet_count: int,
    payload_bytes: int,
    max_pending_episodes: int,
) -> None:
    """Reject dimensions that cannot produce a meaningful IPC measurement."""
    dimensions = {
        "workers": worker_count,
        "packets per worker": packet_count,
        "payload bytes": payload_bytes,
        "max pending episodes": max_pending_episodes,
    }
    for name, value in dimensions.items():
        if value <= 0:
            raise ValueError(f"Benchmark {name} must be positive")


def parse_arguments() -> argparse.Namespace:
    """Parse benchmark topology and payload dimensions."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare the framed production episode pipes with raw payload pipes. "
            "Process startup is excluded from the timed region."
        )
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKER_COUNT)
    parser.add_argument(
        "--packets-per-worker",
        type=int,
        default=DEFAULT_PACKETS_PER_WORKER,
    )
    parser.add_argument(
        "--payload-mib",
        type=float,
        default=DEFAULT_PAYLOAD_MEBIBYTES,
    )
    parser.add_argument(
        "--max-pending-episodes",
        type=int,
        default=DEFAULT_MAX_PENDING_EPISODES,
    )
    parser.add_argument(
        "--start-method",
        choices=multiprocessing.get_all_start_methods(),
    )
    return parser.parse_args()


def main() -> None:
    """Run both IPC transports and print machine-readable results."""
    arguments = parse_arguments()
    results = run_benchmark(
        worker_count=arguments.workers,
        packet_count=arguments.packets_per_worker,
        payload_bytes=int(arguments.payload_mib * BYTES_PER_MEBIBYTE),
        max_pending_episodes=arguments.max_pending_episodes,
        start_method=arguments.start_method,
    )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
