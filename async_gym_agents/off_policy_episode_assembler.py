import time
from typing import cast

from async_gym_agents.data_classes import (
    AssembledEpisode,
    OffPolicyTransition,
    PreparedOffPolicyEpisode,
)
from async_gym_agents.enums import EpisodeKind
from async_gym_agents.episode_assembler import AsyncEpisodeAssembler
from async_gym_agents.episode_codec import unpack_episode
from async_gym_agents.episode_transport import EpisodeTransport
from async_gym_agents.profiler import RuntimeProfiler


class AsyncOffPolicyEpisodeAssembler(AsyncEpisodeAssembler[PreparedOffPolicyEpisode]):
    """Decode and reconstruct one off-policy episode ahead of the trainer."""

    def __init__(
        self,
        transport: EpisodeTransport,
        profiler: RuntimeProfiler,
    ) -> None:
        super().__init__(
            transport=transport,
            episode_kind=EpisodeKind.OFF_POLICY,
            target_transition_count=1,
            profiler=profiler,
            thread_name="off-policy-episode-assembler",
        )

    def _build_assembly(
        self,
        episodes: list[AssembledEpisode],
        transition_count: int,
    ) -> PreparedOffPolicyEpisode:
        if len(episodes) != 1:
            raise RuntimeError("Off-policy assembly must contain exactly one episode")

        reconstruction_start_ns = time.perf_counter_ns()
        transitions = cast(
            list[OffPolicyTransition],
            unpack_episode(episodes[0].batch),
        )
        self._profiler.record(
            "transition_reconstruction",
            time.perf_counter_ns() - reconstruction_start_ns,
            count=transition_count,
        )
        return PreparedOffPolicyEpisode(
            episode=episodes[0],
            transitions=transitions,
        )
