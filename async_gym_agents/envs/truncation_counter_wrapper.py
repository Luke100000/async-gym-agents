from typing import Any, Dict, List, SupportsFloat, Tuple

import gymnasium as gym
from gymnasium.core import ActType, ObsType


class TruncationCounterWrapper(gym.Wrapper[ObsType, ActType, ObsType, ActType]):
    def __init__(self, env: gym.Env):
        super().__init__(env=env)

        self.terminated = 0
        self.truncated = 0

    def step(
        self, action: ActType
    ) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)

        if terminated:
            self.terminated += 1
        elif truncated:
            self.truncated += 1

        return observation, reward, terminated, truncated, info

    def get_truncation_factor(self) -> List[float]:
        return self.truncated / (self.truncated + self.terminated)
