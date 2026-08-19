# Async Gym Agents

Drop-in asynchronous data collection for Stable Baselines 3 agents.

Known framework callbacks are processed from complete episode batches when their
timing permits. Off-policy checkpointing and unrecognized Stable Baselines
callbacks retain their normal per-step behavior.

## Usage

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

Workers send complete episodes, so on-policy rollouts may exceed `n_steps` by
the final episode. `max_episodes_in_buffer` limits buffered episodes across all
workers. `queue_put_timeout` defaults to `None`; set a finite timeout to allow
episode drops instead of waiting for buffer capacity.

Episodes are decoded in a background assembler. On-policy agents prepare the
next rollout buffer, while off-policy agents prepare one episode ahead and keep
replay-buffer insertion on the trainer thread.

On-policy episode packets retain both the environment reward and the
time-limit-bootstrapped training reward. Stable Baselines callbacks, batched
logging, and pruning receive the environment reward; rollout-buffer returns and
advantages use the training reward. Off-policy episodes retain their existing
single reward view.

Use `get_profiler_report()` to inspect trainer, worker, buffer, transport, and
policy statistics. `transport.utilization` is the current fraction of bounded
episode slots in use. Episode-send backpressure is reported once as
`buffer.avg_push_wait_seconds`; the buffer section does not contain historical
queue utilization or emptiness estimates.
