from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs

from async_gym_agents.enums import EpisodeKind


@dataclass
class OnPolicyTransition:
    actions: np.ndarray
    values: np.ndarray
    log_probs: np.ndarray
    last_obs: VecEnvObs
    new_obs: VecEnvObs
    rewards: np.ndarray
    dones: np.ndarray
    last_dones: np.ndarray
    infos: List[Dict]
    reset_infos: List[Dict]


@dataclass
class OffPolicyTransition:
    buffer_actions: np.ndarray
    last_obs: VecEnvObs
    new_obs: VecEnvObs
    rewards: np.ndarray
    dones: np.ndarray
    infos: List[Dict]
    reset_infos: List[Dict]


@dataclass(frozen=True)
class EpisodeBatch:
    episode_kind: EpisodeKind
    transition_count: int
    fields: Dict[str, Any]
    infos: Dict[int, List[Dict]]
    reset_infos: Dict[int, List[Dict]]


@dataclass(frozen=True)
class EpisodePacket:
    worker_index: int
    policy_version: Optional[int]
    episode_kind: EpisodeKind
    transition_count: int
    payload: bytes


@dataclass(frozen=True)
class EpisodeTransportStats:
    pending_episodes: int
    max_pending_episodes: int
    pending_bytes: int
    max_pending_bytes: int
    sent_episodes: int
    sent_bytes: int
    received_episodes: int


@dataclass(frozen=True)
class AssembledEpisode:
    packet: EpisodePacket
    batch: EpisodeBatch


@dataclass(frozen=True)
class PreparedOnPolicyRollout:
    rollout_buffer: RolloutBuffer
    episodes: List[AssembledEpisode]
    transition_count: int
    payload_bytes: int


@dataclass(frozen=True)
class EpisodeSendResult:
    sent: bool
    waiting_ns: int
    transport_ns: int

    def __bool__(self) -> bool:
        return self.sent


@dataclass(frozen=True)
class EpisodeAssemblerStats:
    target_transition_count: int
    filling_transition_count: int
    completed_assemblies: int
    last_transition_count: int
    max_transition_count: int
    last_payload_bytes: int
    max_payload_bytes: int
