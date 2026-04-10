from contextlib import contextmanager
from time import perf_counter_ns, time
from typing import Any, Dict, Iterator, Mapping, MutableMapping, Optional

ProfileStats = Dict[str, Dict[str, int]]


def _copy_stats(stats: Mapping[str, Mapping[str, int]]) -> ProfileStats:
    return {
        phase: {
            "total_ns": int(values.get("total_ns", 0)),
            "count": int(values.get("count", 0)),
        }
        for phase, values in stats.items()
    }


def merge_profile_stats(
    target: MutableMapping[str, Dict[str, int]],
    delta: Mapping[str, Mapping[str, int]],
) -> None:
    for phase, values in delta.items():
        current = target.get(phase, {"total_ns": 0, "count": 0})
        target[phase] = {
            "total_ns": int(current.get("total_ns", 0))
            + int(values.get("total_ns", 0)),
            "count": int(current.get("count", 0)) + int(values.get("count", 0)),
        }


class RuntimeProfiler:
    def __init__(self) -> None:
        self._stats: ProfileStats = {}
        self._pending: ProfileStats = {}

    @contextmanager
    def track(self, phase: str) -> Iterator[None]:
        start_ns = perf_counter_ns()
        try:
            yield
        finally:
            self.record(phase, perf_counter_ns() - start_ns)

    def record(self, phase: str, duration_ns: int, count: int = 1) -> None:
        update = {"total_ns": max(0, int(duration_ns)), "count": int(count)}
        merge_profile_stats(self._stats, {phase: update})
        merge_profile_stats(self._pending, {phase: update})

    def snapshot(self) -> ProfileStats:
        return _copy_stats(self._stats)

    def drain_pending(self) -> ProfileStats:
        pending = self._pending
        self._pending = {}
        return _copy_stats(pending)


def build_profiler_report(
    main_stats: Mapping[str, Mapping[str, int]],
    worker_stats: Mapping[str, Mapping[str, int]],
    *,
    worker_last_sync_time: Optional[float],
    buffer_utilization: float,
    buffer_emptiness: float,
    discarded_episodes_fraction: float,
) -> Dict[str, object]:
    return {
        "main": _summarize_stats(main_stats),
        "worker": _summarize_stats(worker_stats),
        "buffer": {
            "utilization": buffer_utilization,
            "emptiness": buffer_emptiness,
            "discarded_episodes_fraction": discarded_episodes_fraction,
        },
        "worker_sync": {
            "last_sync_unix_time": worker_last_sync_time,
            "seconds_since_last_sync": None
            if worker_last_sync_time is None
            else max(0.0, time() - worker_last_sync_time),
        },
    }


def render_profiler_report(report: Mapping[str, Any]) -> str:
    lines = ["Profiler report"]
    lines.extend(_render_profile_section("Main", report.get("main", {})))
    lines.extend(_render_profile_section("Worker", report.get("worker", {})))

    buffer = report.get("buffer", {})
    lines.append(
        "Buffer: "
        f"util={buffer.get('utilization', 0.0):.2f}, "
        f"empty={buffer.get('emptiness', 0.0):.2f}, "
        f"dropped={buffer.get('discarded_episodes_fraction', 0.0):.2f}"
    )

    worker_sync = report.get("worker_sync", {})
    seconds_since_last_sync = worker_sync.get("seconds_since_last_sync")
    sync_age = (
        "n/a"
        if seconds_since_last_sync is None
        else f"{seconds_since_last_sync:.2f}s ago"
    )
    lines.append(f"Worker sync: {sync_age}")

    return "\n".join(lines)


def _summarize_stats(
    stats: Mapping[str, Mapping[str, int]],
) -> Dict[str, Dict[str, float | int]]:
    summarized: Dict[str, Dict[str, float | int]] = {}
    total_ns = sum(int(values.get("total_ns", 0)) for values in stats.values())

    for phase, values in sorted(stats.items()):
        phase_total_ns = int(values.get("total_ns", 0))
        count = int(values.get("count", 0))
        summarized[phase] = {
            "total_seconds": phase_total_ns / 1_000_000_000,
            "count": count,
            "avg_milliseconds": 0.0
            if count == 0
            else phase_total_ns / count / 1_000_000,
            "share": 0.0 if total_ns == 0 else phase_total_ns / total_ns,
        }

    return summarized


def _render_profile_section(title: str, stats: Mapping[str, Any]) -> list[str]:
    lines = [f"{title}:"]
    if not stats:
        lines.append("  (no samples)")
        return lines

    sorted_stats = sorted(
        stats.items(),
        key=lambda item: item[1].get("total_seconds", 0.0),
        reverse=True,
    )
    for phase, values in sorted_stats:
        lines.append(
            "  "
            f"{phase:<20} "
            f"{values.get('total_seconds', 0.0):>8.3f}s "
            f"{int(values.get('count', 0)):>6}x "
            f"{values.get('avg_milliseconds', 0.0):>8.3f}ms "
            f"{values.get('share', 0.0):>6.1%}"
        )
    return lines
