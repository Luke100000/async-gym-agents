from typing import Optional

import numpy as np
from gymnasium.envs.box2d import LunarLander


class BuggyLunarLander(LunarLander):
    """
    This environment fakes being buggy by randomly setting truncated to True.
    """

    def __init__(
        self,
        crash_probability: float = 0.01,
        time_limit: int = 999999,
        render_mode: Optional[str] = None,
    ):
        super().__init__(render_mode)

        self.tick = 0
        self.tick_until_crash = 0
        self.crash_probability = crash_probability
        self.time_limit = time_limit

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.tick = 0
        self.tick_until_crash = (
            99999
            if self.crash_probability <= 0
            else np.random.geometric(self.crash_probability)
        )

        return super().reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        self.tick += 1

        if self.tick >= self.tick_until_crash and not terminated:
            truncated = True

        if self.tick >= self.time_limit and not truncated:
            terminated = True

        return obs, reward, terminated, truncated, info
