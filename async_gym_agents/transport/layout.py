from typing import Sequence, Tuple

import numpy as np
from gymnasium import spaces

from async_gym_agents.transport.data_classes import FieldSpec


def resolve_space_layout(space: spaces.Space) -> Tuple[Tuple[int, ...], np.dtype]:
    if isinstance(space, spaces.Box):
        return tuple(space.shape), np.dtype(space.dtype)
    if isinstance(space, spaces.Discrete):
        return (1,), np.dtype(np.int64)
    if isinstance(space, (spaces.MultiDiscrete, spaces.MultiBinary)):
        return tuple(space.shape), np.dtype(np.int64)
    raise ValueError(
        f"Shared-memory transport requires fixed-shape spaces; "
        f"{type(space).__name__} is unsupported (dict/variable observations are "
        f"out of scope). Use the queue transport for this space."
    )


def build_numeric_layout(
    observation_space: spaces.Space,
    action_space: spaces.Space,
    extra_fields: Sequence[FieldSpec] = (),
) -> list[FieldSpec]:
    """Derive the fixed numeric slot layout from the spaces.

    ``extra_fields`` carries algorithm-specific numeric fields (e.g. on-policy
    ``value``/``log_prob``); info dicts are handled separately.
    """
    obs_shape, obs_dtype = resolve_space_layout(observation_space)
    act_shape, act_dtype = resolve_space_layout(action_space)
    fields = [
        FieldSpec("obs", obs_shape, obs_dtype),
        FieldSpec("next_obs", obs_shape, obs_dtype),
        FieldSpec("action", act_shape, act_dtype),
        # Scalars per env: reconstructing a row slice [i:i+1] restores shape (1,).
        FieldSpec("reward", (), np.dtype(np.float32)),
        FieldSpec("done", (), np.dtype(np.float32)),
    ]
    fields.extend(extra_fields)

    names = [field.name for field in fields]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate field names in transition layout: {names}")
    return fields
