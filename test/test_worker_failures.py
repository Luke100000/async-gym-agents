import signal
import time

import pytest

from async_gym_agents.agents.injector import AsyncWorkerFailureError

FAILURE_PROPAGATION_TIMEOUT_SECONDS = 5.0
WORKER_READY_TIMEOUT_SECONDS = 10.0


class TestWorkerFailureAttribution:
    """Trainer acquisition reports real thread and process worker failures."""

    def test_reports_real_thread_failure_with_original_cause(
        self,
        thread_failure_agent,
    ):
        """A thread exception reaches the trainer promptly with its exact cause."""
        started_at = time.monotonic()
        agent, expected_error_type = thread_failure_agent
        agent.pre_collect_preparation(agent.policy)
        agent._initialize_rollout_assembler(2)

        with pytest.raises(
            AsyncWorkerFailureError,
            match="worker 0",
        ) as error:
            agent._acquire_prepared_assembly()

        assert time.monotonic() - started_at < FAILURE_PROPAGATION_TIMEOUT_SECONDS
        assert len(error.value.failures) == 1
        assert isinstance(error.value.__cause__, expected_error_type)

    def test_reports_real_process_exit_reason(
        self,
        process_blocked_agent,
    ):
        """A killed process reaches the trainer promptly with platform attribution."""
        agent, ready = process_blocked_agent
        agent.pre_collect_preparation(agent.policy)
        assert ready.wait(WORKER_READY_TIMEOUT_SECONDS)
        agent._initialize_rollout_assembler(2)
        worker = agent._workers[0]
        started_at = time.monotonic()
        worker.kill()

        with pytest.raises(
            AsyncWorkerFailureError,
            match="worker 0",
        ) as error:
            agent._acquire_prepared_assembly()

        elapsed = time.monotonic() - started_at
        reason = error.value.failures[0].reason
        assert elapsed < FAILURE_PROPAGATION_TIMEOUT_SECONDS
        if worker.exitcode > 0:
            assert reason == f"exit code {worker.exitcode}"
        else:
            assert reason == signal.Signals(-worker.exitcode).name

    def test_reports_only_the_root_failing_worker(
        self,
        multi_worker_failure_agent,
    ):
        """A healthy sibling stopped during cleanup is not another root failure."""
        agent, ready = multi_worker_failure_agent
        agent.pre_collect_preparation(agent.policy)
        assert ready.wait(WORKER_READY_TIMEOUT_SECONDS)
        agent._initialize_rollout_assembler(2)

        with pytest.raises(AsyncWorkerFailureError) as error:
            agent._acquire_prepared_assembly()

        assert [failure.worker_index for failure in error.value.failures] == [0]

    def test_does_not_report_intentional_shutdown(
        self,
        shutdown_worker_agent,
    ):
        """Parent-initiated shutdown leaves no unexpected worker failure."""
        shutdown_worker_agent.pre_collect_preparation(shutdown_worker_agent.policy)

        shutdown_worker_agent.shutdown()

        shutdown_worker_agent.raise_for_failed_workers()
