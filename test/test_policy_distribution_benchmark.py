import pytest

from benchmarks.policy_distribution_benchmark import run_benchmark


class TestPolicyDistributionBenchmark:
    """Policy publication benchmarks compare queue fan-out with shared snapshots."""

    def test_reports_both_publication_methods(self):
        """A small spawn run returns comparable timing and topology fields."""
        result = run_benchmark(
            worker_count=2,
            publication_count=2,
            payload_bytes=1_024,
            start_method="spawn",
        )

        assert result["configuration"]["workers"] == 2
        assert result["queue_fan_out"]["publication_count"] == 2
        assert result["shared_snapshot"]["publication_count"] == 2
        assert result["queue_fan_out"]["total_seconds"] > 0
        assert result["shared_snapshot"]["total_seconds"] > 0

    def test_rejects_non_positive_dimensions(self):
        """Zero-sized benchmark dimensions fail before allocating processes."""
        with pytest.raises(ValueError, match="workers must be positive"):
            run_benchmark(
                worker_count=0,
                publication_count=1,
                payload_bytes=1,
                start_method="spawn",
            )
