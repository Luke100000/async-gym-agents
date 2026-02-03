from typing import Any, Tuple, TypeVar

T = TypeVar("T")


def single_slice(x: T, idx: int) -> T | Tuple[Any, ...]:
    if isinstance(x, tuple):
        return x[idx : idx + 1]
    return x[idx : idx + 1]


def identity(x):
    return x
