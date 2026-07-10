"""Serialize per-transition info dicts into a fixed-size ring slot.

Info dicts (terminal_observation, TimeLimit.truncated, episode stats, custom
keys) cannot be a fixed numeric layout, so they ride two extra ring fields:
``info_len`` (how many bytes are valid) and ``info`` (a capped byte buffer).
Empty infos — the common per-step case — serialize to nothing, so the hot path
pays no serialization; only non-empty infos (rare, mostly episode boundaries)
are pickled. Trust boundary is the same as the queue this replaces: the bytes
come only from our own worker processes.
"""

import pickle
from typing import Any, Dict, List

import numpy as np

from async_gym_agents.transport.constants import INFO_BLOB_CAPACITY
from async_gym_agents.transport.data_classes import FieldSpec


def build_info_fields(capacity: int = INFO_BLOB_CAPACITY) -> List[FieldSpec]:
    return [
        FieldSpec("info_len", (1,), np.dtype(np.int32)),
        FieldSpec("info", (capacity,), np.dtype(np.uint8)),
    ]


def encode_info(info: Dict[str, Any]) -> bytes:
    if not info:
        return b""
    return pickle.dumps(info, protocol=pickle.HIGHEST_PROTOCOL)


def decode_info(blob: bytes) -> Dict[str, Any]:
    if len(blob) == 0:
        return {}
    return pickle.loads(bytes(blob))


def write_info(values: Dict[str, np.ndarray], info: Dict[str, Any]) -> None:
    """Fill the ``info_len``/``info`` slot fields for one transition."""
    blob = encode_info(info)
    capacity = values["info"].shape[0]
    if len(blob) > capacity:
        raise ValueError(
            f"Serialized info ({len(blob)} bytes) exceeds INFO_BLOB_CAPACITY "
            f"({capacity}); raise the cap or reduce info payload."
        )
    values["info_len"][0] = len(blob)
    values["info"][: len(blob)] = np.frombuffer(blob, dtype=np.uint8)


def read_info(row_len: np.ndarray, row_blob: np.ndarray) -> Dict[str, Any]:
    """Decode one assembled row's info from its ``info_len``/``info`` fields."""
    length = int(row_len[0])
    if length == 0:
        return {}
    return decode_info(row_blob[:length])
