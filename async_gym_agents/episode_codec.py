import pickle
from dataclasses import fields
from typing import Any, Dict, List, Optional, Sequence, Type

import numpy as np

from async_gym_agents.data_classes import (
    EpisodeBatch,
    EpisodePacket,
    OffPolicyTransition,
    OnPolicyTransition,
)
from async_gym_agents.enums import EpisodeKind

INFO_FIELD_NAME = "infos"
RESET_INFO_FIELD_NAME = "reset_infos"
SPARSE_INFO_FIELD_NAMES = {INFO_FIELD_NAME, RESET_INFO_FIELD_NAME}


def pack_episode(
    transitions: Sequence[OnPolicyTransition | OffPolicyTransition],
) -> EpisodeBatch:
    """Pack one complete episode into contiguous arrays and sparse info maps."""
    if not transitions:
        raise ValueError("Cannot pack an empty episode")

    transition_type, episode_kind = _resolve_episode_type(transitions[0])
    packed_fields = {
        field.name: _concatenate_values(
            [getattr(transition, field.name) for transition in transitions]
        )
        for field in fields(transition_type)
        if field.name not in SPARSE_INFO_FIELD_NAMES
    }
    return EpisodeBatch(
        episode_kind=episode_kind,
        transition_count=len(transitions),
        fields=packed_fields,
        infos=_pack_sparse_infos([transition.infos for transition in transitions]),
        reset_infos=_pack_sparse_infos(
            [transition.reset_infos for transition in transitions]
        ),
    )


def encode_episode_batch(
    worker_index: int,
    policy_version: Optional[int],
    batch: EpisodeBatch,
) -> EpisodePacket:
    """Serialize one packed episode into a queue-friendly byte payload."""
    return EpisodePacket(
        worker_index=worker_index,
        policy_version=policy_version,
        episode_kind=batch.episode_kind,
        transition_count=batch.transition_count,
        payload=pickle.dumps(batch, protocol=pickle.HIGHEST_PROTOCOL),
    )


def decode_episode_packet(packet: EpisodePacket) -> EpisodeBatch:
    """Deserialize and validate one complete episode packet."""
    batch = pickle.loads(packet.payload)
    if not isinstance(batch, EpisodeBatch):
        raise TypeError("Episode payload did not contain an EpisodeBatch")
    if batch.episode_kind is not packet.episode_kind:
        raise ValueError("Episode payload kind does not match its packet metadata")
    if batch.transition_count != packet.transition_count:
        raise ValueError("Episode payload length does not match its packet metadata")
    return batch


def unpack_episode(
    batch: EpisodeBatch,
) -> List[OnPolicyTransition | OffPolicyTransition]:
    """Reconstruct transition objects for consumers that still process rows."""
    transition_type = _resolve_transition_class(batch.episode_kind)
    transitions = []
    for index in range(batch.transition_count):
        values = {
            name: _slice_value(value, index) for name, value in batch.fields.items()
        }
        values[INFO_FIELD_NAME] = batch.infos.get(index, [{}])
        values[RESET_INFO_FIELD_NAME] = batch.reset_infos.get(index, [{}])
        transitions.append(transition_type(**values))
    return transitions


def slice_episode_field(batch: EpisodeBatch, field_name: str, index: int) -> Any:
    """Return one row from a packed episode field with its environment axis."""
    return _slice_value(batch.fields[field_name], index)


def get_episode_infos(batch: EpisodeBatch, index: int) -> List[Dict]:
    """Return one row's sparse info list."""
    return batch.infos.get(index, [{}])


def get_episode_reset_infos(batch: EpisodeBatch, index: int) -> List[Dict]:
    """Return one row's sparse reset-info list."""
    return batch.reset_infos.get(index, [{}])


def _resolve_episode_type(
    transition: OnPolicyTransition | OffPolicyTransition,
) -> tuple[Type[OnPolicyTransition | OffPolicyTransition], EpisodeKind]:
    if isinstance(transition, OnPolicyTransition):
        return OnPolicyTransition, EpisodeKind.ON_POLICY
    if isinstance(transition, OffPolicyTransition):
        return OffPolicyTransition, EpisodeKind.OFF_POLICY
    raise TypeError(f"Unsupported transition type: {type(transition)!r}")


def _resolve_transition_class(
    episode_kind: EpisodeKind,
) -> Type[OnPolicyTransition | OffPolicyTransition]:
    if episode_kind is EpisodeKind.ON_POLICY:
        return OnPolicyTransition
    if episode_kind is EpisodeKind.OFF_POLICY:
        return OffPolicyTransition
    raise ValueError(f"Unsupported episode kind: {episode_kind!r}")


def _concatenate_values(values: Sequence[Any]) -> Any:
    first_value = values[0]
    if isinstance(first_value, np.ndarray):
        return np.concatenate(values, axis=0)
    if isinstance(first_value, dict):
        return {
            key: _concatenate_values([value[key] for value in values])
            for key in first_value
        }
    if isinstance(first_value, tuple):
        return tuple(
            _concatenate_values([value[index] for value in values])
            for index in range(len(first_value))
        )
    raise TypeError(f"Cannot pack field value of type {type(first_value)!r}")


def _slice_value(value: Any, index: int) -> Any:
    if isinstance(value, np.ndarray):
        return value[index : index + 1]
    if isinstance(value, dict):
        return {key: _slice_value(item, index) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_slice_value(item, index) for item in value)
    raise TypeError(f"Cannot slice packed field value of type {type(value)!r}")


def _pack_sparse_infos(values: Sequence[List[Dict]]) -> Dict[int, List[Dict]]:
    return {
        index: list(infos)
        for index, infos in enumerate(values)
        if any(info for info in infos)
    }
