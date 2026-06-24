from typing import Type, TypeVar, Union

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

from async_gym_agents.agents.fast_off_policy_injector import (
    FastOffPolicyAlgorithmInjector,
)
from async_gym_agents.agents.fast_on_policy_injector import (
    FastOnPolicyAlgorithmInjector,
)
from async_gym_agents.agents.injector import AsyncAgentInjector
from async_gym_agents.agents.off_policy_injector import OffPolicyAlgorithmInjector
from async_gym_agents.agents.on_policy_injector import OnPolicyAlgorithmInjector

T = TypeVar("T", bound=BaseAlgorithm)


def get_injected_agent(clazz: Type[T]) -> Union[Type[T], Type[AsyncAgentInjector]]:
    if issubclass(clazz, OnPolicyAlgorithm):
        injector_class = OnPolicyAlgorithmInjector
    elif issubclass(clazz, OffPolicyAlgorithm):
        injector_class = OffPolicyAlgorithmInjector
    else:
        raise ValueError(f"Unknown agent class {clazz}!")

    class AsyncAgent(injector_class, clazz):
        pass

    return AsyncAgent


def get_fast_injected_agent(clazz: Type[T]) -> Union[Type[T], Type[AsyncAgentInjector]]:
    if issubclass(clazz, OnPolicyAlgorithm):
        injector_class = FastOnPolicyAlgorithmInjector
    elif issubclass(clazz, OffPolicyAlgorithm):
        injector_class = FastOffPolicyAlgorithmInjector
    else:
        raise ValueError(f"Unknown agent class {clazz}!")

    class AsyncAgent(injector_class, clazz):
        pass

    return AsyncAgent
