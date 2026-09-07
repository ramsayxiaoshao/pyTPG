"""Finite fixed-width memory with atomic whole-state updates."""

from __future__ import annotations

from collections.abc import Sequence

from tpg.memory._validation import finite_values, positive_integer
from tpg.memory.base import MemorySnapshot, MemoryUpdate, MemoryUpdater
from tpg.memory.errors import MemoryConfigurationError, MemoryStateError


class RegisterMemory:
    """Own finite registers and optionally update them after each decision."""

    __slots__ = ("_initial_values", "_revision", "_updater", "_values")

    def __init__(
        self,
        size: int,
        *,
        initial_values: Sequence[float] | None = None,
        updater: MemoryUpdater | None = None,
    ) -> None:
        width = positive_integer(size, "memory size")
        self._initial_values = (
            (0.0,) * width
            if initial_values is None
            else finite_values(initial_values, width, "initial memory values")
        )
        if updater is not None and not callable(updater):
            raise MemoryConfigurationError("memory updater must be callable or None")
        self._updater = updater
        self._values = self._initial_values
        self._revision = 0

    @property
    def size(self) -> int:
        return len(self._initial_values)

    def reset(self) -> MemorySnapshot:
        """Restore configured initial values and revision zero."""

        self._values = self._initial_values
        self._revision = 0
        return self.snapshot()

    def snapshot(self) -> MemorySnapshot:
        return MemorySnapshot(self._values, self._revision)

    def restore(self, snapshot: MemorySnapshot) -> MemorySnapshot:
        """Restore an exact compatible snapshot without changing its revision."""

        if not isinstance(snapshot, MemorySnapshot):
            raise MemoryStateError("restore requires a MemorySnapshot")
        if snapshot.size != self.size:
            raise MemoryStateError(
                f"snapshot size {snapshot.size} does not match memory size {self.size}"
            )
        self._values = snapshot.values
        self._revision = snapshot.revision
        return self.snapshot()

    def update(self, context: MemoryUpdate) -> MemorySnapshot:
        """Validate a complete candidate before atomically replacing values."""

        if not isinstance(context, MemoryUpdate):
            raise MemoryStateError("memory update requires a MemoryUpdate")
        previous = self.snapshot()
        candidate = (
            previous.values
            if self._updater is None
            else self._updater(previous, context)
        )
        normalized = finite_values(candidate, self.size, "updated memory values")
        self._values = normalized
        self._revision += 1
        return self.snapshot()

    def read(self, index: int) -> float:
        """Read one register using strict non-wrapping bounds."""

        self._check_index(index)
        return self._values[index]

    def write(self, values: Sequence[float]) -> MemorySnapshot:
        """Atomically replace all registers and advance the revision."""

        normalized = finite_values(values, self.size, "memory values")
        self._values = normalized
        self._revision += 1
        return self.snapshot()

    def _check_index(self, index: int) -> None:
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < self.size
        ):
            raise MemoryStateError(
                f"memory register index {index!r} is outside [0, {self.size})"
            )


__all__ = ["RegisterMemory"]
