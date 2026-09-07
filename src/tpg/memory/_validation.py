"""Shared strict validation for memory values."""

from __future__ import annotations

import math
from collections.abc import Sequence

from tpg.memory.errors import MemoryConfigurationError, MemoryStateError


def positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise MemoryConfigurationError(f"{name} must be a positive integer")
    return value


def non_negative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MemoryConfigurationError(f"{name} must be a non-negative integer")
    return value


def finite_values(
    values: Sequence[float],
    expected_size: int,
    name: str,
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise MemoryStateError(f"{name} must be a numeric sequence")
    try:
        items = tuple(values)
    except TypeError as error:
        raise MemoryStateError(f"{name} must be a sequence") from error
    if len(items) != expected_size:
        raise MemoryStateError(
            f"{name} length {len(items)} does not match expected size {expected_size}"
        )
    normalized: list[float] = []
    for index, value in enumerate(items):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MemoryStateError(
                f"{name}[{index}] must be a finite real number, got {value!r}"
            )
        converted = float(value)
        if not math.isfinite(converted):
            raise MemoryStateError(f"{name}[{index}] must be finite, got {value!r}")
        normalized.append(converted)
    return tuple(normalized)
