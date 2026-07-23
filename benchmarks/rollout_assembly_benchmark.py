import json
import statistics
import time
import tracemalloc

import numpy as np

from async_gym_agents.on_policy_rollout_assembler import (
    AsyncOnPolicyRolloutAssembler,
)

EPISODE_COUNT = 8
TRANSITIONS_PER_EPISODE = 32
OBSERVATION_HEIGHT = 84
OBSERVATION_WIDTH = 84
OBSERVATION_CHANNELS = 4
WARMUP_COUNT = 3
MEASUREMENT_COUNT = 11
MAX_OPTIMIZED_TO_BASELINE_RATIO = 1.05


def build_observation_chunks() -> list[np.ndarray]:
    """Create the representative large-observation rollout workload."""
    random_generator = np.random.default_rng(7)
    shape = (
        TRANSITIONS_PER_EPISODE,
        OBSERVATION_HEIGHT,
        OBSERVATION_WIDTH,
        OBSERVATION_CHANNELS,
    )
    return [
        random_generator.random(shape, dtype=np.float32) for _ in range(EPISODE_COUNT)
    ]


def assemble_with_baseline(
    destination: np.ndarray,
    chunks: list[np.ndarray],
) -> None:
    """Run the reviewed concatenate-then-copy implementation."""
    concatenated = np.concatenate(chunks, axis=0)
    np.copyto(destination, concatenated.reshape(destination.shape))


def assemble_with_destination_fill(
    destination: np.ndarray,
    chunks: list[np.ndarray],
) -> None:
    """Run the production destination-filled implementation."""
    AsyncOnPolicyRolloutAssembler.fill_concatenated_values(
        destination,
        chunks,
    )


def measure_peak_bytes(operation, destination, chunks) -> int:
    """Measure peak traced temporary memory for one assembly operation."""
    tracemalloc.start()
    operation(destination, chunks)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak_bytes


def measure_median_seconds(chunks) -> tuple[float, float]:
    """Measure alternating baseline and optimized samples and return medians."""
    shape = (
        EPISODE_COUNT * TRANSITIONS_PER_EPISODE,
        OBSERVATION_HEIGHT,
        OBSERVATION_WIDTH,
        OBSERVATION_CHANNELS,
    )
    baseline_destination = np.empty(shape, dtype=np.float32)
    optimized_destination = np.empty(shape, dtype=np.float32)
    for _ in range(WARMUP_COUNT):
        assemble_with_baseline(baseline_destination, chunks)
        assemble_with_destination_fill(optimized_destination, chunks)

    baseline_samples = []
    optimized_samples = []
    operations = (
        (assemble_with_baseline, baseline_destination, baseline_samples),
        (
            assemble_with_destination_fill,
            optimized_destination,
            optimized_samples,
        ),
    )
    for iteration in range(MEASUREMENT_COUNT):
        ordered_operations = operations if iteration % 2 == 0 else operations[::-1]
        for operation, destination, samples in ordered_operations:
            started_at = time.perf_counter()
            operation(destination, chunks)
            samples.append(time.perf_counter() - started_at)

    return (
        statistics.median(baseline_samples),
        statistics.median(optimized_samples),
    )


def run_benchmark() -> dict[str, float | int | bool]:
    """Compare value equality, temporary memory, and median assembly duration."""
    chunks = build_observation_chunks()
    destination_shape = (
        EPISODE_COUNT * TRANSITIONS_PER_EPISODE,
        OBSERVATION_HEIGHT,
        OBSERVATION_WIDTH,
        OBSERVATION_CHANNELS,
    )
    baseline_destination = np.empty(destination_shape, dtype=np.float32)
    optimized_destination = np.empty(destination_shape, dtype=np.float32)
    baseline_peak_bytes = measure_peak_bytes(
        assemble_with_baseline,
        baseline_destination,
        chunks,
    )
    optimized_peak_bytes = measure_peak_bytes(
        assemble_with_destination_fill,
        optimized_destination,
        chunks,
    )
    baseline_seconds, optimized_seconds = measure_median_seconds(chunks)
    payload_bytes = sum(chunk.nbytes for chunk in chunks)
    ratio = optimized_seconds / baseline_seconds
    equal_values = bool(np.array_equal(baseline_destination, optimized_destination))
    traced_bytes_removed = baseline_peak_bytes - optimized_peak_bytes
    bytes_removed = payload_bytes
    return {
        "equal_values": equal_values,
        "payload_bytes": payload_bytes,
        "baseline_peak_bytes": baseline_peak_bytes,
        "optimized_peak_bytes": optimized_peak_bytes,
        "bytes_removed": bytes_removed,
        "traced_bytes_removed": traced_bytes_removed,
        "baseline_median_seconds": baseline_seconds,
        "optimized_median_seconds": optimized_seconds,
        "optimized_to_baseline_ratio": ratio,
        "meets_memory_target": bytes_removed >= payload_bytes,
        "meets_throughput_target": ratio <= MAX_OPTIMIZED_TO_BASELINE_RATIO,
    }


def main() -> None:
    """Print machine-readable benchmark acceptance results."""
    result = run_benchmark()
    print(json.dumps(result, sort_keys=True))
    if not (
        result["equal_values"]
        and result["meets_memory_target"]
        and result["meets_throughput_target"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
