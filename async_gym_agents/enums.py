from enum import Enum


class EpisodeKind(str, Enum):
    ON_POLICY = "on_policy"
    OFF_POLICY = "off_policy"
