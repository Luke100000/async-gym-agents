from functools import wraps
from time import perf_counter_ns
from typing import Any, Callable, Dict

from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from async_gym_agents.constants import (
    CALLBACK_PROFILE_HOOK_NAMES,
    MILLISECONDS_PER_SECOND,
    NANOSECONDS_PER_SECOND,
)


class _CallbackTiming:
    __slots__ = ("count", "total_ns")

    def __init__(self) -> None:
        self.total_ns = 0
        self.count = 0


class CallbackRuntimeProfiler:
    """Measure leaf callback runtime without adding profiler locks per call."""

    def __init__(self, clock: Callable[[], int] = perf_counter_ns) -> None:
        self._clock = clock
        self._timings: Dict[str, _CallbackTiming] = {}
        self._instrumented_callbacks: Dict[BaseCallback, str] = {}
        self._callback_type_counts: Dict[str, int] = {}

    def instrument(self, callback: BaseCallback) -> None:
        """Instrument every leaf callback in an initialized SB3 callback tree."""
        if isinstance(callback, CallbackList):
            for child_callback in callback.callbacks:
                self.instrument(child_callback)
            return

        if callback in self._instrumented_callbacks:
            return

        callback_name = self._reserve_callback_name(callback)
        timing = _CallbackTiming()
        self._timings[callback_name] = timing
        self._instrumented_callbacks[callback] = callback_name
        for hook_name in CALLBACK_PROFILE_HOOK_NAMES:
            self._instrument_hook(callback, hook_name, timing)

    def get_report(self) -> Dict[str, Dict[str, float | int]]:
        """Return cumulative runtime, invocation count, and mean time."""
        report = {}
        for callback_name, timing in self._timings.items():
            total_seconds = timing.total_ns / NANOSECONDS_PER_SECOND
            avg_milliseconds = (
                0.0
                if timing.count == 0
                else total_seconds / timing.count * MILLISECONDS_PER_SECOND
            )
            report[callback_name] = {
                "total_seconds": total_seconds,
                "count": timing.count,
                "avg_milliseconds": avg_milliseconds,
            }
        return report

    def _reserve_callback_name(self, callback: BaseCallback) -> str:
        callback_type = type(callback).__name__
        type_count = self._callback_type_counts.get(callback_type, 0) + 1
        self._callback_type_counts[callback_type] = type_count
        if type_count == 1:
            return callback_type
        return f"{callback_type} #{type_count}"

    def _instrument_hook(
        self,
        callback: BaseCallback,
        hook_name: str,
        timing: _CallbackTiming,
    ) -> None:
        original_hook = getattr(callback, hook_name)
        clock = self._clock

        @wraps(original_hook)
        def measured_hook(*args: Any, **kwargs: Any) -> Any:
            start_ns = clock()
            try:
                return original_hook(*args, **kwargs)
            finally:
                timing.total_ns += max(0, clock() - start_ns)
                timing.count += 1

        setattr(callback, hook_name, measured_hook)
