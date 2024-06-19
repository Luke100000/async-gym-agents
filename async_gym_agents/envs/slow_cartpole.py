import random
import time
from typing import Optional

from gymnasium.envs.classic_control import CartPoleEnv


class SlowCartPoleEnv(CartPoleEnv):
    """
    This environment fakes being slow.
    """

    def __init__(
        self,
        min_sleep: float = 0.01,
        max_sleep: float = 0.1,
        render_mode: Optional[str] = None,
    ):
        super().__init__(render_mode)

        self.min_sleep = min_sleep
        self.max_sleep = max_sleep

    def step(self, action):
        t = self.min_sleep + random.random() * (self.max_sleep - self.min_sleep)
        time.sleep(t)
        return super().step(action)
