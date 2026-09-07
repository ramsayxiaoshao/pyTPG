"""Observation-history memory built by composing fixed registers."""

from __future__ import annotations

from tpg.memory._validation import positive_integer
from tpg.memory.base import MemorySnapshot, MemoryUpdate
from tpg.memory.errors import MemoryStateError
from tpg.memory.register import RegisterMemory


class ObservationHistoryMemory:
    """Expose prior raw observations from oldest to most recent."""

    __slots__ = ("_registers", "depth", "observation_size")

    def __init__(self, observation_size: int, depth: int = 1) -> None:
        self.observation_size = positive_integer(observation_size, "observation_size")
        self.depth = positive_integer(depth, "history depth")
        self._registers = RegisterMemory(
            self.observation_size * self.depth,
            updater=self._shift_observation,
        )

    @property
    def size(self) -> int:
        return self._registers.size

    def reset(self) -> MemorySnapshot:
        return self._registers.reset()

    def snapshot(self) -> MemorySnapshot:
        return self._registers.snapshot()

    def restore(self, snapshot: MemorySnapshot) -> MemorySnapshot:
        return self._registers.restore(snapshot)

    def update(self, context: MemoryUpdate) -> MemorySnapshot:
        return self._registers.update(context)

    def _shift_observation(
        self,
        previous: MemorySnapshot,
        context: MemoryUpdate,
    ) -> tuple[float, ...]:
        if len(context.observation) != self.observation_size:
            raise MemoryStateError(
                f"observation length {len(context.observation)} does not match "
                f"history width {self.observation_size}"
            )
        return previous.values[self.observation_size :] + context.observation


__all__ = ["ObservationHistoryMemory"]
