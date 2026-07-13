import pytest

from benchmarks.ipc_transport_benchmark import (
    summarize_benchmark,
    validate_benchmark_dimensions,
)


class TestIpcTransportBenchmark:
    """IPC benchmark results remain comparable and reject invalid workloads."""

    def test_summarizes_payload_throughput(self):
        """Two one-MiB packets over one second report two MiB per second."""
        result = summarize_benchmark(
            method="queue",
            worker_count=2,
            packet_count=1,
            payload_bytes=1_048_576,
            elapsed_seconds=1.0,
            consumer_cpu_seconds=0.5,
        )

        assert result["packets"] == 2
        assert result["mib_per_second"] == 2.0
        assert result["consumer_cpu_fraction"] == 0.5

    @pytest.mark.parametrize(
        "worker_count, packet_count, payload_bytes, max_pending_episodes",
        [
            (0, 1, 1, 1),
            (1, 0, 1, 1),
            (1, 1, 0, 1),
            (1, 1, 1, 0),
        ],
    )
    def test_rejects_nonpositive_dimensions(
        self,
        worker_count,
        packet_count,
        payload_bytes,
        max_pending_episodes,
    ):
        """Every topology and payload dimension must be positive."""
        with pytest.raises(ValueError, match="must be positive"):
            validate_benchmark_dimensions(
                worker_count,
                packet_count,
                payload_bytes,
                max_pending_episodes,
            )
