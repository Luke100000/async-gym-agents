from typing import Type, TypeVar, Union

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

from async_gym_agents.agents.injector import AsyncAgentInjectorBase
from async_gym_agents.agents.off_policy_injector import (
    OffPolicyAlgorithmInjector,
    OffPolicyAlgorithmInjectorMP,
)
from async_gym_agents.agents.on_policy_injector import (
    OnPolicyAlgorithmInjector,
    OnPolicyAlgorithmInjectorMP,
)

T = TypeVar("T", bound=BaseAlgorithm)


def get_injected_agent(
    clazz: Type[T], use_mp: bool = False
) -> Union[Type[T], Type[AsyncAgentInjectorBase]]:
    if issubclass(clazz, OnPolicyAlgorithm):
        injector_class = (
            OnPolicyAlgorithmInjectorMP if use_mp else OnPolicyAlgorithmInjector
        )
    elif issubclass(clazz, OffPolicyAlgorithm):
        injector_class = (
            OffPolicyAlgorithmInjectorMP if use_mp else OffPolicyAlgorithmInjector
        )
    else:
        raise ValueError(f"Unknown agent class {clazz}!")

    class AsyncAgent(injector_class, clazz):
        pass

    return AsyncAgent
