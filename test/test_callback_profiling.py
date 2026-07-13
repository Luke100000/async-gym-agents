from async_gym_agents.callback_profiler import CallbackRuntimeProfiler
from async_gym_agents.constants import (
    PROFILE_PHASE_ASSEMBLER_ACQUIRE,
    PROFILE_PHASE_CALLBACK_DEBUG_PRINT,
    PROFILE_PHASE_LOGGER_DUMP,
    PROFILE_PHASE_PROFILER_REPORTING,
)


class TestCallbackRuntimeProfiler:
    """Callback timings identify expensive SB3 callback implementations."""

    def test_accumulates_callback_time_and_count(
        self,
        single_convert_callback_list,
        deterministic_callback_clock,
    ):
        """Two lifecycle calls produce cumulative and per-call timings."""
        profiler = CallbackRuntimeProfiler(clock=deterministic_callback_clock)
        profiler.instrument(single_convert_callback_list)

        single_convert_callback_list.on_rollout_start()
        single_convert_callback_list.on_rollout_end()

        assert profiler.get_report() == {
            "ConvertCallback": {
                "total_seconds": 0.000003,
                "count": 2,
                "avg_milliseconds": 0.0015,
            }
        }

    def test_prints_callback_report_after_each_ppo_train(
        self,
        short_episode_on_policy_agent,
        single_convert_callback_list,
        capsys,
    ):
        """Every PPO update prints the callback timing summary for debugging."""
        short_episode_on_policy_agent.learn(
            total_timesteps=3,
            callback=single_convert_callback_list,
        )

        output = capsys.readouterr().out
        assert "Callback profiler" in output
        assert "ConvertCallback" in output


class TestPpoBoundaryProfiling:
    """PPO timings expose work outside processing and training phases."""

    def test_records_each_unaccounted_iteration_boundary(
        self,
        short_episode_on_policy_agent,
        single_convert_callback_list,
    ):
        """One PPO update records each previously unmeasured boundary once."""
        short_episode_on_policy_agent.learn(
            total_timesteps=3,
            callback=single_convert_callback_list,
        )

        main_report = short_episode_on_policy_agent.get_profiler_report()["main"]
        expected_phases = {
            PROFILE_PHASE_ASSEMBLER_ACQUIRE,
            PROFILE_PHASE_CALLBACK_DEBUG_PRINT,
            PROFILE_PHASE_LOGGER_DUMP,
            PROFILE_PHASE_PROFILER_REPORTING,
        }
        assert expected_phases <= main_report.keys()
        assert {main_report[phase]["count"] for phase in expected_phases} == {1}
