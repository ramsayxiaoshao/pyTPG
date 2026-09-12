"""Unit tests for memory values and reference components."""

from collections.abc import Sequence
from typing import cast

import pytest

from pytpg.memory import (
    MemoryConfigurationError,
    MemorySnapshot,
    MemoryStateError,
    MemoryUpdate,
    NullMemory,
    ObservationHistoryMemory,
    RegisterMemory,
)
from pytpg.runtime import TraversalResult, TraversalStep


def update_context(
    observation: tuple[float, ...] = (2.0,),
    action_id: int = 1,
) -> MemoryUpdate:
    traversal = TraversalResult(
        action_id,
        (
            TraversalStep(
                team_id=0,
                learner_id=0,
                bid=1.0,
                atomic_action_id=action_id,
            ),
        ),
    )
    return MemoryUpdate(observation, action_id, traversal)


def test_snapshot_normalizes_values_and_reports_size() -> None:
    snapshot = MemorySnapshot((1, 2.5), revision=3)

    assert snapshot.values == (1.0, 2.5)
    assert snapshot.size == 2
    assert snapshot.revision == 3


@pytest.mark.parametrize(
    "factory",
    [
        lambda: MemorySnapshot(cast(tuple[float, ...], [1.0]), 0),
        lambda: MemorySnapshot((True,), 0),
        lambda: MemorySnapshot((float("nan"),), 0),
        lambda: MemorySnapshot((), -1),
    ],
)
def test_invalid_snapshots_are_rejected(factory: object) -> None:
    with pytest.raises(MemoryStateError):
        cast(object, factory)()  # type: ignore[operator]


def test_memory_update_requires_action_to_match_traversal() -> None:
    traversal = update_context(action_id=0).traversal

    with pytest.raises(MemoryStateError, match="does not match"):
        MemoryUpdate((1.0,), 1, traversal)


def test_null_memory_has_one_canonical_empty_state() -> None:
    memory = NullMemory()

    assert memory.size == 0
    assert memory.reset() == MemorySnapshot((), 0)
    assert memory.update(update_context()) == MemorySnapshot((), 0)
    assert memory.restore(MemorySnapshot((), 0)) == MemorySnapshot((), 0)
    with pytest.raises(MemoryStateError):
        memory.restore(MemorySnapshot((1.0,), 0))


def test_register_memory_write_reset_read_and_restore() -> None:
    memory = RegisterMemory(2, initial_values=(1, -1))

    assert memory.snapshot() == MemorySnapshot((1.0, -1.0), 0)
    assert memory.write((4, 5)) == MemorySnapshot((4.0, 5.0), 1)
    assert memory.read(1) == 5.0
    assert memory.restore(MemorySnapshot((8.0, 9.0), 7)).revision == 7
    assert memory.reset() == MemorySnapshot((1.0, -1.0), 0)


def test_register_updater_receives_prior_state_and_decision_context() -> None:
    def updater(
        previous: MemorySnapshot,
        context: MemoryUpdate,
    ) -> Sequence[float]:
        return (previous.values[0] + context.observation[0], context.action_id)

    memory = RegisterMemory(2, updater=updater)

    first = memory.update(update_context((2.0,), 1))
    second = memory.update(update_context((3.0,), 0))

    assert first == MemorySnapshot((2.0, 1.0), 1)
    assert second == MemorySnapshot((5.0, 0.0), 2)


def test_invalid_register_update_is_atomic() -> None:
    memory = RegisterMemory(
        1,
        initial_values=(3.0,),
        updater=lambda _previous, _context: (float("nan"),),
    )
    before = memory.snapshot()

    with pytest.raises(MemoryStateError, match="finite"):
        memory.update(update_context())

    assert memory.snapshot() == before


@pytest.mark.parametrize(
    "factory",
    [
        lambda: RegisterMemory(0),
        lambda: RegisterMemory(2, initial_values=(1.0,)),
        lambda: RegisterMemory(1, updater=cast(object, 3)),
    ],
)
def test_invalid_register_memory_configuration_is_rejected(factory: object) -> None:
    with pytest.raises((MemoryConfigurationError, MemoryStateError)):
        cast(object, factory)()  # type: ignore[operator]


def test_register_bounds_and_restore_width_are_strict() -> None:
    memory = RegisterMemory(2)

    with pytest.raises(MemoryStateError, match="outside"):
        memory.read(-1)
    with pytest.raises(MemoryStateError, match="outside"):
        memory.read(2)
    with pytest.raises(MemoryStateError, match="snapshot size"):
        memory.restore(MemorySnapshot((1.0,), 0))


def test_observation_history_is_zero_padded_oldest_to_newest() -> None:
    memory = ObservationHistoryMemory(observation_size=2, depth=3)

    assert memory.reset().values == (0.0,) * 6
    assert memory.update(update_context((1.0, 2.0))).values == (
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        2.0,
    )
    assert memory.update(update_context((3.0, 4.0))).values == (
        0.0,
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
    )
    assert memory.update(update_context((5.0, 6.0))).values == (
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
    )


def test_observation_history_rejects_wrong_observation_width_atomically() -> None:
    memory = ObservationHistoryMemory(observation_size=2)
    before = memory.snapshot()

    with pytest.raises(MemoryStateError, match="history width"):
        memory.update(update_context((1.0,)))

    assert memory.snapshot() == before
