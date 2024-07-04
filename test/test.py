import gymnasium as gym
from stable_baselines3 import TD3

from async_gym_agents.agents.async_agent import get_injected_agent
from async_gym_agents.envs.multi_env import IndexableMultiEnv

# Create env with 8 parallel envs
env = IndexableMultiEnv([lambda: gym.make("Pendulum-v1") for i in range(8)])

# Create the model, injected with async capabilities
model = get_injected_agent(TD3)("MlpPolicy", env)

# Train the model
model.learn(total_timesteps=1000)

model.shutdown()

print("Done")
