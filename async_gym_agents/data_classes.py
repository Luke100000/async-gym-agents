from dataclasses import dataclass
from typing import Generic, List, Optional, TypeVar

TransitionType = TypeVar("TransitionType")


@dataclass(frozen=True)
class EpisodeEnvelope(Generic[TransitionType]):
    worker_index: int
    policy_version: Optional[int]
    transitions: List[TransitionType]
