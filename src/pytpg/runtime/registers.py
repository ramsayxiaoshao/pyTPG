"""Finite, bounds-checked registers for the reference executor."""

import math

from pytpg.runtime.errors import RegisterAccessError


class RegisterFile:
    """A mutable register file owned by one program execution."""

    __slots__ = ("_values",)

    def __init__(self, count: int) -> None:
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            msg = f"register count must be a positive integer, got {count!r}"
            raise RegisterAccessError(msg)
        self._values = [0.0] * count

    @property
    def count(self) -> int:
        """Return the number of addressable registers."""

        return len(self._values)

    def read(self, index: int) -> float:
        """Read a register using strict, non-wrapping bounds."""

        self._check_index(index)
        return self._values[index]

    def write(self, index: int, value: float) -> None:
        """Write a finite real value using strict, non-wrapping bounds."""

        self._check_index(index)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            msg = f"register value must be a real number, got {value!r}"
            raise RegisterAccessError(msg)
        normalized = float(value)
        if not math.isfinite(normalized):
            msg = f"register value must be finite, got {value!r}"
            raise RegisterAccessError(msg)
        self._values[index] = normalized

    def snapshot(self) -> tuple[float, ...]:
        """Return an immutable copy suitable for tests and inspection."""

        return tuple(self._values)

    def _check_index(self, index: int) -> None:
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= self.count
        ):
            msg = f"register index {index!r} is outside [0, {self.count})"
            raise RegisterAccessError(msg)


__all__ = ["RegisterFile"]
