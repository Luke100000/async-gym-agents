from typing import List, Protocol, runtime_checkable

from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from async_gym_agents import constants
from async_gym_agents.data_classes import EpisodeCallbackContext
from async_gym_agents.enums import EpisodeKind


@runtime_checkable
class EpisodeBatchableCallback(Protocol):
    """A callback that can process a complete episode instead of every transition.

    Framework callbacks (e.g. in reinforcement-learning-framework) implement
    these two methods directly on themselves. async-gym-agents only detects
    the shape structurally and dispatches to it, without knowing about any
    specific callback class.
    """

    def process_episode(self, context: EpisodeCallbackContext) -> bool:
        """Process one complete episode and return whether training should continue."""
        ...

    def advance_callback(self, transition_count: int, num_timesteps: int) -> None:
        """Advance SB3 callback counters without invoking its per-step hook."""
        ...


class CallbackBatchDispatcher:
    """Route episode-batchable callbacks by episode and preserve per-step compatibility."""

    def __init__(
        self,
        callback: BaseCallback,
        episode_kind: EpisodeKind = EpisodeKind.ON_POLICY,
    ) -> None:
        self.episode_kind = episode_kind
        self.callback_lists: List[CallbackList] = []
        self.episode_adapters: List[EpisodeBatchableCallback] = []
        self.step_callbacks: List[BaseCallback] = []
        self._classify_callback(callback)

    def process_step(self, callback_locals: dict) -> bool:
        """Advance one transition and invoke callbacks without an episode adapter."""
        num_timesteps = callback_locals["self"].num_timesteps
        for callback_list in self.callback_lists:
            callback_list.n_calls += 1
            callback_list.num_timesteps = num_timesteps

        if self.episode_kind is EpisodeKind.OFF_POLICY:
            for adapter in self.episode_adapters:
                adapter.advance_callback(1, num_timesteps)

        continue_training = True
        for callback in self.step_callbacks:
            callback.update_locals(callback_locals)
            continue_training = callback.on_step() and continue_training
        return continue_training

    def process_episode(self, context: EpisodeCallbackContext) -> bool:
        """Dispatch one complete episode to every episode-batchable callback."""
        if self.episode_kind is EpisodeKind.ON_POLICY:
            for adapter in self.episode_adapters:
                adapter.advance_callback(
                    context.batch.transition_count,
                    context.end_timestep,
                )

        continue_training = True
        for adapter in self.episode_adapters:
            continue_training = adapter.process_episode(context) and continue_training
        return continue_training

    def _classify_callback(self, callback: BaseCallback) -> None:
        if isinstance(callback, CallbackList):
            self.callback_lists.append(callback)
            for child_callback in callback.callbacks:
                self._classify_callback(child_callback)
            return

        if isinstance(callback, EpisodeBatchableCallback):
            self.episode_adapters.append(callback)
        else:
            self.step_callbacks.append(callback)


def resolve_episode_action_field(episode_kind: EpisodeKind) -> str:
    """Return the packed action field consumed by framework logging."""
    if episode_kind is EpisodeKind.ON_POLICY:
        return constants.ON_POLICY_ACTIONS_FIELD
    if episode_kind is EpisodeKind.OFF_POLICY:
        return constants.OFF_POLICY_ACTIONS_FIELD
    raise ValueError(f"Unsupported episode kind: {episode_kind!r}")


def resolve_episode_reward_field(episode_kind: EpisodeKind) -> str:
    """Return the callback-visible reward field for an episode kind."""
    if episode_kind is EpisodeKind.ON_POLICY:
        return constants.ON_POLICY_ENVIRONMENT_REWARDS_FIELD
    if episode_kind is EpisodeKind.OFF_POLICY:
        return constants.OFF_POLICY_REWARDS_FIELD
    raise ValueError(f"Unsupported episode kind: {episode_kind!r}")
