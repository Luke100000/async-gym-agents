from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

from async_gym_agents.agents.off_policy_injector import OffPolicyAlgorithmInjector
from async_gym_agents.agents.on_policy_injector import OnPolicyAlgorithmInjector


def get_injected_agent(clazz: BaseAlgorithm):
    if issubclass(clazz, OnPolicyAlgorithm):

        class AsyncAgent(OnPolicyAlgorithmInjector, clazz):
            pass

        return AsyncAgent

    elif issubclass(clazz, OffPolicyAlgorithm):

        class AsyncAgent(OffPolicyAlgorithmInjector, clazz):
            pass

        return AsyncAgent

    else:
        raise ValueError(f"Unknown agent class {clazz}!")
