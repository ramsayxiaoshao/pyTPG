"""Regression tests for stateful graph-runtime composition."""

import pytest

from pytpg.core import (
    ActionID,
    AtomicAction,
    ConstantOperand,
    InputIndex,
    InputOperand,
    Instruction,
    Learner,
    LearnerID,
    OperatorName,
    Program,
    ProgramID,
    RegisterIndex,
    Team,
    TeamID,
    TPGGraph,
)
from pytpg.memory import (
    MemoryConfigurationError,
    MemorySnapshot,
    MemoryStateError,
    NullMemory,
    ObservationHistoryMemory,
    StatefulGraphRuntime,
)
from pytpg.runtime import GraphRuntime, GraphValidationError, RuntimeConfig


def memory_choice_graph(memory_input: int = 1) -> TPGGraph:
    negative = Program(
        ProgramID(0),
        (
            Instruction(
                OperatorName("subtract"),
                RegisterIndex(0),
                (
                    ConstantOperand(0.0),
                    InputOperand(InputIndex(memory_input)),
                ),
            ),
        ),
    )
    positive = Program(
        ProgramID(1),
        (
            Instruction(
                OperatorName("identity"),
                RegisterIndex(0),
                (InputOperand(InputIndex(memory_input)),),
            ),
        ),
    )
    team = Team(
        TeamID(0),
        (
            Learner(LearnerID(0), negative, AtomicAction(ActionID(0))),
            Learner(LearnerID(1), positive, AtomicAction(ActionID(1))),
        ),
    )
    return TPGGraph((team,), (team.id,))


def test_stateful_runtime_uses_prior_memory_then_updates_once() -> None:
    graph = memory_choice_graph()
    controller = StatefulGraphRuntime(
        GraphRuntime(RuntimeConfig(input_size=2, register_count=1)),
        ObservationHistoryMemory(1),
        observation_size=1,
    )
    controller.reset_episode()

    signal = controller.traverse(graph, (1.0,))
    recall = controller.traverse(graph, (0.0,))

    assert signal.action_id == 0
    assert signal.augmented_observation == (1.0, 0.0)
    assert signal.memory_before == MemorySnapshot((0.0,), 0)
    assert signal.memory_after == MemorySnapshot((1.0,), 1)
    assert recall.action_id == 1
    assert recall.augmented_observation == (0.0, 1.0)
    assert recall.memory_after == MemorySnapshot((0.0,), 2)


def test_reset_prevents_memory_leakage_between_episodes() -> None:
    graph = memory_choice_graph()
    controller = StatefulGraphRuntime(
        GraphRuntime(RuntimeConfig(2, 1)),
        ObservationHistoryMemory(1),
        observation_size=1,
    )
    controller.reset_episode()
    controller.act(graph, (1.0,))
    controller.end_episode()

    reset = controller.reset_episode()
    first = controller.traverse(graph, (0.0,))

    assert reset == MemorySnapshot((0.0,), 0)
    assert first.memory_before == reset
    assert first.action_id == 0


def test_explicit_episode_lifecycle_is_enforced() -> None:
    graph = memory_choice_graph()
    controller = StatefulGraphRuntime(
        GraphRuntime(RuntimeConfig(2, 1)),
        ObservationHistoryMemory(1),
        observation_size=1,
    )

    with pytest.raises(MemoryStateError, match="reset_episode"):
        controller.act(graph, (0.0,))
    controller.reset_episode()
    assert controller.episode_active is True
    final = controller.end_episode()
    assert final == MemorySnapshot((0.0,), 0)
    assert controller.episode_active is False
    with pytest.raises(MemoryStateError, match="no memory episode"):
        controller.end_episode()


def test_restore_episode_continues_from_exact_snapshot() -> None:
    graph = memory_choice_graph()
    controller = StatefulGraphRuntime(
        GraphRuntime(RuntimeConfig(2, 1)),
        ObservationHistoryMemory(1),
        observation_size=1,
    )

    restored = controller.restore_episode(MemorySnapshot((-1.0,), 9))
    result = controller.traverse(graph, (0.0,))

    assert restored.revision == 9
    assert result.action_id == 0
    assert result.memory_after.revision == 10


def test_failed_traversal_does_not_update_memory() -> None:
    graph = memory_choice_graph(memory_input=2)
    memory = ObservationHistoryMemory(1)
    controller = StatefulGraphRuntime(
        GraphRuntime(RuntimeConfig(2, 1)),
        memory,
        observation_size=1,
    )
    controller.reset_episode()
    before = memory.snapshot()

    with pytest.raises(GraphValidationError, match="invalid_instruction"):
        controller.act(graph, (1.0,))

    assert memory.snapshot() == before


def test_null_memory_matches_stateless_action_semantics() -> None:
    graph = memory_choice_graph(memory_input=0)
    runtime = GraphRuntime(RuntimeConfig(1, 1))
    controller = StatefulGraphRuntime(runtime, NullMemory(), observation_size=1)
    controller.reset_episode()

    assert controller.act(graph, (1.0,)) == runtime.act(graph, (1.0,))


@pytest.mark.parametrize(
    "factory",
    [
        lambda: StatefulGraphRuntime(
            GraphRuntime(RuntimeConfig(2, 1)),
            ObservationHistoryMemory(1),
            observation_size=2,
        ),
        lambda: StatefulGraphRuntime(
            GraphRuntime(RuntimeConfig(1, 1)),
            ObservationHistoryMemory(1),
            observation_size=-1,
        ),
    ],
)
def test_incompatible_stateful_runtime_configuration_is_rejected(
    factory: object,
) -> None:
    with pytest.raises(MemoryConfigurationError):
        factory()  # type: ignore[operator]


def test_stateful_observation_is_exact_and_finite() -> None:
    graph = memory_choice_graph()
    controller = StatefulGraphRuntime(
        GraphRuntime(RuntimeConfig(2, 1)),
        ObservationHistoryMemory(1),
        observation_size=1,
    )
    controller.reset_episode()

    with pytest.raises(MemoryStateError, match="length"):
        controller.act(graph, ())
    with pytest.raises(MemoryStateError, match="finite"):
        controller.act(graph, (float("nan"),))
