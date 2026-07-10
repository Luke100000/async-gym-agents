from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class FieldSpec:
    name: str
    shape: Tuple[int, ...]
    dtype: np.dtype


@dataclass
class AssembledRollout:
    """Rollout assembled from the worker rings in one move."""

    fields: Dict[str, np.ndarray]
    n_rows: int
    segments: List[Tuple[int, int]]


@dataclass
class TransportStats:
    """Per-worker transport counters, indexed by worker."""

    produced: List[int]
    consumed: List[int]
    dropped: List[int]
