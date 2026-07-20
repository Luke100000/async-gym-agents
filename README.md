# Async Gym Agents

Wrapper environments and agent injectors to allow for drop-in async training.

Workers build complete episodes, compact their numeric fields into contiguous
arrays, and serialize each episode once. Empty per-step `info` dictionaries are
not stored. Each worker reserves global capacity before handing an episode to a
local feeder thread, so pipe delivery overlaps the worker's next rollout. The
shared `max_episodes_in_buffer` limit bounds all locally queued and in-flight
episodes, so transport memory grows with real payloads rather than a fixed
transition ring.

For PPO and other on-policy algorithms, a background assembler decodes complete
episodes, bulk-populates the inactive Stable Baselines rollout buffer B, and
computes its advantages while Stable Baselines trains buffer A. Acquiring the
prepared B immediately starts construction of its replacement; no trainer-side
transition insertion or return pass is required. The final episode is never
split, so an `n_steps` target may produce a slightly larger rollout buffer.
Episode-start markers still separate trajectories for GAE. Each feeder streams
through its worker's unidirectional pipe. A global capacity semaphore provides
backpressure after B is full and prevents workers from running arbitrarily far
ahead of the trainer.

The trainer publishes each serialized policy once into an A/B shared-memory
snapshot. Workers check its version at episode boundaries, copy only a stable
latest snapshot, and load it into their CPU policy. This removes the previous
per-worker policy queues and their O(worker count) trainer-side broadcast.

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
model = get_injected_agent(TD3)(
    "MlpPolicy",
    env,
    use_mp=False,
    max_episodes_in_buffer=8,
)

# Train the model
model.learn(total_timesteps=10)

# Inspect main-thread and worker timing
report = model.get_profiler_report()
print(render_profiler_report(report))

# Shut down workers
model.shutdown()
```

`queue_put_timeout` defaults to `None`, which keeps workers blocked under
backpressure until capacity becomes available or shutdown begins. Set a finite
timeout only when intentionally dropping episodes is preferable to waiting.

Profiler values are recorded automatically through the Stable Baselines logger
under `profiler/`. Useful series include:

- `profiler/buffer/avg_policy_lag` and `profiler/buffer/max_policy_lag`
- `profiler/transport/pending_episodes` and
  `profiler/transport/max_pending_bytes`
- `profiler/assembly/filling_transitions` and
  `profiler/assembly/last_transitions`
- `profiler/policy/published_version`, `profiler/policy/payload_bytes`, and
  `profiler/policy/publication_failures`
- phase timings such as
  `profiler/main/rollout_buffer_building/avg_milliseconds`,
  `profiler/main/callback_processing/avg_milliseconds`, and
  `profiler/worker/episode_packing/avg_milliseconds`
- leaf callback totals under `profiler/callbacks/<callback>/`, without
  synchronous console reporting
- transport attribution under `profiler/transport/`, including
  `receive_timeout_fraction`, `receive_timeouts_with_pending_fraction`,
  `readiness_wait/avg_milliseconds`,
  `readiness_timeout/avg_milliseconds`,
  `payload_receive/avg_milliseconds`,
  `payload_receive_timeout/avg_milliseconds`, `payload_mib_per_second`, and
  `pipe_latency/avg_milliseconds`

A high `receive_timeouts_with_pending_fraction` means workers have reserved
transport capacity but no complete pipe frame became readable. Compare
readiness time, payload-receive time, and end-to-end pipe latency to distinguish
worker starvation from payload transfer delays. Worker-side `transport` timing
covers local feeder queueing and pipe delivery, while worker-side `waiting`
covers global-capacity backpressure.

The shared latest-policy transport and its concurrency protocol are documented
in [docs/shared_policy_distribution.md](docs/shared_policy_distribution.md).

The standalone payload benchmark compares legacy transition-object pickling
with the packed whole-episode path:

```shell
python benchmarks/episode_transport_benchmark.py --episode-length 4096
```

The IPC benchmark reproduces the training topology with 128 producers,
approximately 2.5 MiB per episode, and the production pending-episode bound.
It compares the framed production transport with raw payload throughput over
the same one-pipe-per-worker topology. Process startup is excluded from the
timed region:

```shell
python benchmarks/ipc_transport_benchmark.py
```

Use smaller dimensions for a quick smoke test:

```shell
python benchmarks/ipc_transport_benchmark.py --workers 4 --packets-per-worker 1 --payload-mib 0.25
```

The policy-distribution benchmark compares legacy queue fan-out with the
shared snapshot using the production default of 128 workers:

```shell
python benchmarks/policy_distribution_benchmark.py
```
