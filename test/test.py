import gymnasium as gym
from stable_baselines3 import SAC, TD3

from async_gym_agents.agents.async_agent import get_injected_agent
from async_gym_agents.envs.multi_env import IndexableMultiEnv


def test_on_policy():
    # Create env with 8 parallel envs
    env = IndexableMultiEnv([lambda: gym.make("Pendulum-v1") for _ in range(8)])

    # Create the model, injected with async capabilities
    model = get_injected_agent(TD3)("MlpPolicy", env)

    # Train the model
    model.learn(total_timesteps=1000)

    model.shutdown()


def test_off_policy():
    # Create env with 8 parallel envs
    env = IndexableMultiEnv(
        [lambda: gym.make("LunarLanderContinuous-v2") for _ in range(8)]
    )

    # Create the model, injected with async capabilities
    model = get_injected_agent(SAC)("MlpPolicy", env)

    # Train the model
    model.learn(total_timesteps=1000)

    model.shutdown()


if __name__ == "__main__":
    test_on_policy()
    test_off_policy()
    print("Done")
