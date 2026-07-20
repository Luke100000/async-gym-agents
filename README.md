# Async Gym Agents

Wrapper environments and agent injectors to allow for drop-in async training.

This comparison branch deliberately yields every completed PPO episode ten
times before advancing its worker environment. Each repetition shallow-copies
the episode list because main's thread-mode consumer mutates that container; the
underlying transition objects are reused. This invalidates training data but
isolates downstream throughput from environment-step latency under the
main-branch queue transport.

```py
import gymnasium as gym
from functools import partial
from stable_baselines3 import TD3

from async_gym_agents.agents.async_agent import get_injected_agent
from async_gym_agents.envs.multi_env import IndexableMultiEnv
from async_gym_agents.profiler import render_profiler_report

# Create env with 8 parallel envs (Also supports VecEnvs)
env = IndexableMultiEnv([partial(gym.make, "Pendulum-v1") for i in range(8)])

# Create the model, injected with async capabilities
model = get_injected_agent(TD3)("MlpPolicy", env, use_mp=False)

# Train the model
model.learn(total_timesteps=10)

# Inspect main-thread and worker timing
report = model.get_profiler_report()
print(render_profiler_report(report))

# Shut down workers
model.shutdown()
```
