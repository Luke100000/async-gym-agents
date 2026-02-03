# Async Gym Agents

Wrapper environments and agent injectors to allow for drop-in async training.

```py
import gymnasium as gym
from stable_baselines3 import TD3

from async_gym_agents.agents.async_agent import get_injected_agent
from async_gym_agents.envs.multi_env import IndexableMultiEnv

# Create env with 8 parallel envs
env = IndexableMultiEnv([lambda: gym.make("Pendulum-v1") for i in range(8)])

# Create the model, injected with async capabilities
model = get_injected_agent(TD3)("MlpPolicy", env)

# Train the model
model.learn(total_timesteps=10)

# Shut down workers
model.shutdown()
```

## Multiprocessing

AsyncGymAgents is primarily designed for IO/Networking heavy situations and uses threads.
For CPU-constrained applications, multiprocessing can be enabled using `use_mp=True`:

```py
import gymnasium as gym
from async_gym_agents.agents.async_agent import get_injected_agent
from stable_baselines3 import PPO


def env_func():
    # Can be one or a list of either gym.Env or VecEnv
    # return [gym.make("Pendulum-v1") for _ in range(4)]
    # return [DummyVecEnv([partial(gym.make, "Pendulum-v1") for _ in range(4)]) for _ in range(4)]
    # return DummyVecEnv([partial(gym.make, "Pendulum-v1")
    return gym.make("Pendulum-v1")


# Create env to define spaces, it will not be used otherwise in MP mode.
env = gym.make("Pendulum-v1")

# Create the model, injected with async capabilities
model = get_injected_agent(PPO, use_mp=True)("MlpPolicy", env, envs=[env_func for _ in range(8)])
```

Since not all envs can be transferred to processes, a constructor `envs` is required.
This constructor allows returning a list of processes, run in threads within a single process.
This allows, e.g., balancing the tradeoff between GIL and memory usage.
