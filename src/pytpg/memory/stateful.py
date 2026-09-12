"""Episode-scoped memory composition around the unchanged graph runtime."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pytpg.core import TPGGraph
from pytpg.memory._validation import finite_values, non_negative_integer
from pytpg.memory.base import Memory, MemorySnapshot, MemoryUpdate
from pytpg.memory.errors import MemoryConfigurationError, MemoryStateError
from pytpg.runtime import GraphRuntime, TraversalResult


@dataclass(frozen=True, slots=True)
class StatefulTraversalResult:
    """A graph traversal together with its exact memory transition."""

    traversal: TraversalResult
    augmented_observation: tuple[float, ...]
    memory_before: MemorySnapshot
    memory_after: MemorySnapshot

    @property
    def action_id(self) -> int:
        return self.traversal.action_id


class StatefulGraphRuntime:
    """Append memory inputs and update memory once per successful decision."""

    __slots__ = ("_episode_active", "memory", "observation_size", "runtime")

    def __init__(
        self,
        runtime: GraphRuntime,
        memory: Memory,
        observation_size: int,
    ) -> None:
        if not isinstance(runtime, GraphRuntime):
            raise MemoryConfigurationError("runtime must be a GraphRuntime")
        self.observation_size = non_negative_integer(
            observation_size, "observation_size"
        )
        memory_size = non_negative_integer(memory.size, "memory size")
        expected_input_size = self.observation_size + memory_size
        if runtime.config.input_size != expected_input_size:
            raise MemoryConfigurationError(
                f"runtime input_size {runtime.config.input_size} does not equal "
                f"observation_size + memory_size ({expected_input_size})"
            )
        self.runtime = runtime
        self.memory = memory
        self._episode_active = False
        self._validate_snapshot(memory.snapshot())

    @property
    def episode_active(self) -> bool:
        return self._episode_active

    def reset_episode(self) -> MemorySnapshot:
        """Reset memory and make the controller ready for decisions."""

        snapshot = self.memory.reset()
        self._validate_snapshot(snapshot)
        self._episode_active = True
        return snapshot

    def restore_episode(self, snapshot: MemorySnapshot) -> MemorySnapshot:
        """Restore memory and make the controller ready for decisions."""

        restored = self.memory.restore(snapshot)
        self._validate_snapshot(restored)
        self._episode_active = True
        return restored

    def end_episode(self) -> MemorySnapshot:
        """End the episode without discarding its inspectable final memory."""

        if not self._episode_active:
            raise MemoryStateError("no memory episode is active")
        self._episode_active = False
        snapshot = self.memory.snapshot()
        self._validate_snapshot(snapshot)
        return snapshot

    def traverse(
        self,
        graph: TPGGraph,
        observation: Sequence[float],
        *,
        root_team_id: int | None = None,
    ) -> StatefulTraversalResult:
        """Traverse with prior memory, then apply one atomic memory update."""

        if not self._episode_active:
            raise MemoryStateError("reset_episode is required before traversal")
        normalized = finite_values(
            observation,
            self.observation_size,
            "stateful observation",
        )
        before = self.memory.snapshot()
        self._validate_snapshot(before)
        augmented = normalized + before.values
        traversal = self.runtime.traverse(
            graph,
            augmented,
            root_team_id=root_team_id,
        )
        after = self.memory.update(
            MemoryUpdate(normalized, traversal.action_id, traversal)
        )
        self._validate_snapshot(after)
        return StatefulTraversalResult(traversal, augmented, before, after)

    def act(
        self,
        graph: TPGGraph,
        observation: Sequence[float],
        *,
        root_team_id: int | None = None,
    ) -> int:
        """Return only the atomic action while retaining the memory update."""

        return self.traverse(
            graph,
            observation,
            root_team_id=root_team_id,
        ).action_id

    def _validate_snapshot(self, snapshot: MemorySnapshot) -> None:
        if not isinstance(snapshot, MemorySnapshot):
            raise MemoryStateError("memory component returned an invalid snapshot")
        if snapshot.size != self.memory.size:
            raise MemoryStateError(
                f"memory snapshot size {snapshot.size} does not match component "
                f"size {self.memory.size}"
            )


__all__ = ["StatefulGraphRuntime", "StatefulTraversalResult"]
