"""Public protocols and immutable values for stateful memory."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from tpg.memory.errors import MemoryStateError
from tpg.runtime import TraversalResult


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    """Immutable memory values and their component-local revision."""

    values: tuple[float, ...]
    revision: int

    def __post_init__(self) -> None:
        if not isinstance(self.values, tuple):
            raise MemoryStateError("memory snapshot values must be a tuple")
        normalized: list[float] = []
        for index, value in enumerate(self.values):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise MemoryStateError(
                    f"memory snapshot value {index} must be a finite real number"
                )
            converted = float(value)
            if not math.isfinite(converted):
                raise MemoryStateError(f"memory snapshot value {index} must be finite")
            normalized.append(converted)
        object.__setattr__(self, "values", tuple(normalized))
        if (
            isinstance(self.revision, bool)
            or not isinstance(self.revision, int)
            or self.revision < 0
        ):
            raise MemoryStateError("memory snapshot revision must be non-negative")

    @property
    def size(self) -> int:
        """Return the number of exposed memory values."""

        return len(self.values)


@dataclass(frozen=True, slots=True)
class MemoryUpdate:
    """One successful graph decision supplied to a memory component."""

    observation: tuple[float, ...]
    action_id: int
    traversal: TraversalResult

    def __post_init__(self) -> None:
        if not isinstance(self.observation, tuple):
            raise MemoryStateError("memory update observation must be a tuple")
        normalized: list[float] = []
        for index, value in enumerate(self.observation):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise MemoryStateError(
                    f"memory update observation {index} must be a finite real number"
                )
            if not math.isfinite(float(value)):
                raise MemoryStateError(
                    f"memory update observation {index} must be finite"
                )
            normalized.append(float(value))
        object.__setattr__(self, "observation", tuple(normalized))
        if (
            isinstance(self.action_id, bool)
            or not isinstance(self.action_id, int)
            or self.action_id < 0
        ):
            raise MemoryStateError("memory update action_id must be non-negative")
        if not isinstance(self.traversal, TraversalResult):
            raise MemoryStateError("memory update traversal is invalid")
        if self.traversal.action_id != self.action_id:
            raise MemoryStateError(
                "memory update action_id does not match its traversal result"
            )


class Memory(Protocol):
    """Fixed-width, episode-scoped state composed around graph traversal."""

    @property
    def size(self) -> int: ...

    def reset(self) -> MemorySnapshot: ...

    def snapshot(self) -> MemorySnapshot: ...

    def restore(self, snapshot: MemorySnapshot) -> MemorySnapshot: ...

    def update(self, context: MemoryUpdate) -> MemorySnapshot: ...


class MemoryFactory(Protocol):
    """Create independent memory for one graph evaluation or agent."""

    def __call__(self) -> Memory: ...


class MemoryUpdater(Protocol):
    """Derive a complete next register state without mutating the old state."""

    def __call__(
        self,
        previous: MemorySnapshot,
        context: MemoryUpdate,
    ) -> Sequence[float]: ...


__all__ = [
    "Memory",
    "MemoryFactory",
    "MemorySnapshot",
    "MemoryUpdate",
    "MemoryUpdater",
]
