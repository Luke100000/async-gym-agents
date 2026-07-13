from async_gym_agents.callback_profiler import CallbackRuntimeProfiler
from async_gym_agents.constants import (
    CALLBACK_PROFILER_REPORT_KEY,
    PROFILE_PHASE_ASSEMBLER_ACQUIRE,
    PROFILE_PHASE_CALLBACK_PROCESSING,
    PROFILE_PHASE_LOGGER_DUMP,
    PROFILE_PHASE_PROFILER_REPORTING,
    PROFILE_PHASE_ROLLOUT_BUFFER_BUILDING,
    PROFILER_LOG_PREFIX,
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

    def test_reports_callback_timings_without_printing(
        self,
        short_episode_on_policy_agent,
        single_convert_callback_list,
        capsys,
    ):
        """Callback timings use profiler metrics without writing debug output."""
        short_episode_on_policy_agent.learn(
            total_timesteps=3,
            callback=single_convert_callback_list,
        )

        output = capsys.readouterr().out
        assert "Callback profiler" not in output
        callback_report = short_episode_on_policy_agent.get_profiler_report()[
            CALLBACK_PROFILER_REPORT_KEY
        ]
        assert callback_report["ConvertCallback"]["count"] == 8

        short_episode_on_policy_agent.record_profiler_metrics()
        metric_name = (
            f"{PROFILER_LOG_PREFIX}/{CALLBACK_PROFILER_REPORT_KEY}/"
            "ConvertCallback/total_seconds"
        )
        assert metric_name in short_episode_on_policy_agent.logger.name_to_value


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
            PROFILE_PHASE_LOGGER_DUMP,
            PROFILE_PHASE_PROFILER_REPORTING,
        }
        assert expected_phases <= main_report.keys()
        assert {main_report[phase]["count"] for phase in expected_phases} == {1}

    def test_separates_background_building_from_trainer_callback_work(
        self,
        short_episode_on_policy_agent,
        single_convert_callback_list,
    ):
        """One PPO update reports buffer building and callback replay separately."""
        short_episode_on_policy_agent.learn(
            total_timesteps=3,
            callback=single_convert_callback_list,
        )

        main_report = short_episode_on_policy_agent.get_profiler_report()["main"]
        assert main_report[PROFILE_PHASE_ROLLOUT_BUFFER_BUILDING]["count"] >= 1
        assert main_report[PROFILE_PHASE_CALLBACK_PROCESSING]["count"] == 1
