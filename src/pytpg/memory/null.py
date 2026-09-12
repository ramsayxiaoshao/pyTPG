"""Zero-width memory preserving stateless behavior through composition."""

from pytpg.memory.base import MemorySnapshot, MemoryUpdate
from pytpg.memory.errors import MemoryStateError


class NullMemory:
    """A reusable no-state implementation of the memory protocol."""

    __slots__ = ()

    @property
    def size(self) -> int:
        return 0

    def reset(self) -> MemorySnapshot:
        return self.snapshot()

    def snapshot(self) -> MemorySnapshot:
        return MemorySnapshot((), 0)

    def restore(self, snapshot: MemorySnapshot) -> MemorySnapshot:
        if not isinstance(snapshot, MemorySnapshot) or snapshot != self.snapshot():
            raise MemoryStateError("NullMemory can restore only its empty snapshot")
        return self.snapshot()

    def update(self, context: MemoryUpdate) -> MemorySnapshot:
        if not isinstance(context, MemoryUpdate):
            raise MemoryStateError("memory update requires a MemoryUpdate")
        return self.snapshot()


__all__ = ["NullMemory"]
