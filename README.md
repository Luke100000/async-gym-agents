# Async Gym Agents

Wrapper environments and agent injectors for drop-in asynchronous Stable
Baselines 3 training.

Workers send complete episodes through dedicated unidirectional pipes. Numeric
transition fields are packed into contiguous arrays, empty `info` dictionaries
are omitted, and a feeder thread overlaps pipe delivery with the next episode.
`max_episodes_in_buffer` applies globally to queued and in-flight episodes, so
it bounds transport memory and applies backpressure across all workers.

PPO uses a background assembler to build the next rollout buffer while the
current buffer trains. Episodes are never split, so a rollout may exceed
`n_steps` by the final complete episode. Workers load the latest policy at
episode boundaries from a versioned shared-memory snapshot; the trainer writes
each policy only once, regardless of worker count.

```python
from functools import partial

import gymnasium as gym
from stable_baselines3 import TD3

from async_gym_agents.agents.async_agent import get_injected_agent
from async_gym_agents.envs.multi_env import IndexableMultiEnv

env = IndexableMultiEnv(
    [partial(gym.make, "Pendulum-v1") for _ in range(8)]
)
model = get_injected_agent(TD3)(
    "MlpPolicy",
    env,
    use_mp=False,
    max_episodes_in_buffer=8,
)

model.learn(total_timesteps=10)
model.shutdown()
```

`queue_put_timeout` defaults to `None`, which keeps workers under backpressure
until capacity becomes available or shutdown starts. Set a finite timeout only
when dropping an episode is preferable to waiting.

`get_profiler_report()` exposes main and worker timings, policy lag, bounded
transport usage, rollout assembly progress, and shared-policy publication
statistics for external logging callbacks.
