import argparse
import multiprocessing
import pickle
import time
from typing import Callable, Sequence

import numpy as np

from async_gym_agents.data_classes import OnPolicyTransition
from async_gym_agents.episode_codec import (
    decode_episode_packet,
    encode_episode_batch,
    pack_episode,
)

DEFAULT_EPISODE_LENGTH = 4_096
DEFAULT_ITERATIONS = 20
DEFAULT_OBSERVATION_SIZE = 64
MILLISECONDS_PER_SECOND = 1_000


def build_episode(
    episode_length: int,
    observation_size: int,
) -> list[OnPolicyTransition]:
    """Build a representative on-policy episode with independently owned rows."""
    episode = []
    for index in range(episode_length):
        done = index == episode_length - 1
        observation = np.full((1, observation_size), index, dtype=np.float32)
        episode.append(
            OnPolicyTransition(
                actions=np.array([[index % 4]], dtype=np.int64),
                values=np.array([index / episode_length], dtype=np.float32),
                log_probs=np.array([-0.5], dtype=np.float32),
                last_obs=observation,
                new_obs=observation + 1,
                rewards=np.array([1.0], dtype=np.float32),
                dones=np.array([done]),
                last_dones=np.array([index == 0]),
                infos=[{}],
                reset_infos=[{}],
            )
        )
    return episode


def measure_average_seconds(operation: Callable[[], object], iterations: int) -> float:
    """Return the mean wall time for one operation."""
    start_time = time.perf_counter()
    for _ in range(iterations):
        operation()
    return (time.perf_counter() - start_time) / iterations


def measure_queue_seconds(payload: object, iterations: int) -> float:
    """Measure one multiprocessing queue put/get cycle including outer pickle."""
    context = multiprocessing.get_context()
    transport_queue = context.Queue(maxsize=1)
    try:
        return measure_average_seconds(
            lambda: (transport_queue.put(payload), transport_queue.get()),
            iterations,
        )
    finally:
        transport_queue.close()
        transport_queue.join_thread()


def benchmark_episode_transport(
    episode: Sequence[OnPolicyTransition],
    iterations: int,
) -> dict[str, float | int]:
    """Compare legacy object pickling with packed whole-episode transfer."""
    raw_payload = pickle.dumps(episode, protocol=pickle.HIGHEST_PROTOCOL)
    packed_batch = pack_episode(episode)
    packed_packet = encode_episode_batch(0, 1, packed_batch)

    raw_roundtrip_seconds = measure_average_seconds(
        lambda: pickle.loads(pickle.dumps(episode, protocol=pickle.HIGHEST_PROTOCOL)),
        iterations,
    )
    packed_worker_seconds = measure_average_seconds(
        lambda: encode_episode_batch(0, 1, pack_episode(episode)),
        iterations,
    )
    packed_trainer_seconds = measure_average_seconds(
        lambda: decode_episode_packet(packed_packet),
        iterations,
    )
    legacy_queue_seconds = measure_queue_seconds(episode, iterations)
    packed_queue_seconds = measure_queue_seconds(packed_packet, iterations)

    return {
        "transitions": len(episode),
        "legacy_payload_bytes": len(raw_payload),
        "packed_payload_bytes": len(packed_packet.payload),
        "payload_ratio": len(packed_packet.payload) / len(raw_payload),
        "legacy_roundtrip_ms": raw_roundtrip_seconds * MILLISECONDS_PER_SECOND,
        "legacy_queue_ms": legacy_queue_seconds * MILLISECONDS_PER_SECOND,
        "packed_worker_ms": packed_worker_seconds * MILLISECONDS_PER_SECOND,
        "packed_queue_ms": packed_queue_seconds * MILLISECONDS_PER_SECOND,
        "packed_trainer_ms": packed_trainer_seconds * MILLISECONDS_PER_SECOND,
        "packed_roundtrip_ms": (packed_worker_seconds + packed_trainer_seconds)
        * MILLISECONDS_PER_SECOND,
        "packed_end_to_end_ms": (
            packed_worker_seconds + packed_queue_seconds + packed_trainer_seconds
        )
        * MILLISECONDS_PER_SECOND,
    }


def parse_arguments() -> argparse.Namespace:
    """Parse benchmark dimensions from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-length", type=int, default=DEFAULT_EPISODE_LENGTH)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument(
        "--observation-size", type=int, default=DEFAULT_OBSERVATION_SIZE
    )
    return parser.parse_args()


def main() -> None:
    """Run the transport benchmark and print stable key/value output."""
    arguments = parse_arguments()
    episode = build_episode(
        arguments.episode_length,
        arguments.observation_size,
    )
    results = benchmark_episode_transport(episode, arguments.iterations)
    for name, value in results.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
