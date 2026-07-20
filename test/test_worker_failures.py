import pytest


class TestWorkerFailureAttribution:
    """Trainer failures identify workers terminated outside Python."""

    def test_reports_worker_index_and_fatal_signal(
        self,
        on_policy_agent_with_signaled_worker,
    ):
        """A signaled worker is reported instead of a generic pipe EOF."""
        with pytest.raises(RuntimeError, match="worker 0 exited with SIGTERM"):
            on_policy_agent_with_signaled_worker.raise_for_failed_workers()
