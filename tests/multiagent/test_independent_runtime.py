"""Independent, heterogeneous, and parameter-shared controller tests."""

import pytest

from pytpg.core import (
    ActionID,
    AtomicAction,
    ConstantOperand,
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
from pytpg.memory import MemorySnapshot, ObservationHistoryMemory, RegisterMemory
from pytpg.multiagent import (
    AgentID,
    AgentSpec,
    IndependentMultiAgentRuntime,
    MultiAgentConfigurationError,
    MultiAgentObservationError,
    MultiAgentStateError,
    TPGAgentController,
)


def constant_graph(action_id: int) -> TPGGraph:
    program = Program(
        ProgramID(0),
        (
            Instruction(
                OperatorName("identity"),
                RegisterIndex(0),
                (ConstantOperand(1.0),),
            ),
        ),
    )
    learner = Learner(LearnerID(0), program, AtomicAction(ActionID(action_id)))
    team = Team(TeamID(0), (learner,))
    return TPGGraph((team,), (team.id,))


def test_independent_runtime_preserves_roster_order_and_heterogeneous_shapes() -> None:
    scout = TPGAgentController.create(
        AgentSpec(AgentID("scout"), 1, 2), constant_graph(1)
    )
    arm = TPGAgentController.create(AgentSpec(AgentID("arm"), 2, 3), constant_graph(2))
    runtime = IndependentMultiAgentRuntime((scout, arm))
    runtime.reset_episode()

    result = runtime.step({AgentID("arm"): (4.0, 5.0), AgentID("scout"): (3.0,)})

    assert result.actions_by_agent == {
        AgentID("scout"): 1,
        AgentID("arm"): 2,
    }
    assert tuple(action.agent_id for action in result.actions) == (
        AgentID("scout"),
        AgentID("arm"),
    )
    assert result.traversals[0].augmented_observation == (3.0,)
    assert result.traversals[1].augmented_observation == (4.0, 5.0)
    runtime.end_episode()


def test_agents_may_share_immutable_graph_but_keep_separate_memory() -> None:
    graph = constant_graph(0)
    first = TPGAgentController.create(
        AgentSpec(AgentID("first"), 1, 1),
        graph,
        memory=ObservationHistoryMemory(1),
    )
    second = TPGAgentController.create(
        AgentSpec(AgentID("second"), 1, 1),
        graph,
        memory=ObservationHistoryMemory(1),
    )
    runtime = IndependentMultiAgentRuntime((first, second))
    runtime.reset_episode()

    result = runtime.step({AgentID("second"): (2.0,), AgentID("first"): (1.0,)})

    assert first.graph is second.graph
    assert first.memory is not second.memory
    assert result.traversals[0].memory_after == MemorySnapshot((1.0,), 1)
    assert result.traversals[1].memory_after == MemorySnapshot((2.0,), 1)


def test_missing_observation_is_rejected_before_any_agent_updates() -> None:
    first = TPGAgentController.create(
        AgentSpec(AgentID("first"), 1, 1),
        constant_graph(0),
        memory=ObservationHistoryMemory(1),
    )
    second = TPGAgentController.create(
        AgentSpec(AgentID("second"), 1, 1),
        constant_graph(0),
        memory=ObservationHistoryMemory(1),
    )
    runtime = IndependentMultiAgentRuntime((first, second))
    initial = runtime.reset_episode()

    with pytest.raises(MultiAgentObservationError, match="missing"):
        runtime.step({AgentID("first"): (1.0,)})

    assert tuple(item.memory.snapshot() for item in runtime.controllers) == initial


def test_failed_agent_update_rolls_all_memories_back() -> None:
    def fail_update(*args: object) -> tuple[float, ...]:
        raise RuntimeError("injected update failure")

    first = TPGAgentController.create(
        AgentSpec(AgentID("first"), 1, 1),
        constant_graph(0),
        memory=ObservationHistoryMemory(1),
    )
    second = TPGAgentController.create(
        AgentSpec(AgentID("second"), 1, 1),
        constant_graph(0),
        memory=RegisterMemory(1, updater=fail_update),
    )
    runtime = IndependentMultiAgentRuntime((first, second))
    initial = runtime.reset_episode()

    with pytest.raises(RuntimeError, match="injected"):
        runtime.step({AgentID("first"): (1.0,), AgentID("second"): (2.0,)})

    assert tuple(item.memory.snapshot() for item in runtime.controllers) == initial
    assert runtime.episode_active is True


def test_live_memory_owner_cannot_be_shared_between_agents() -> None:
    memory = ObservationHistoryMemory(1)
    first = TPGAgentController.create(
        AgentSpec(AgentID("first"), 1, 1), constant_graph(0), memory=memory
    )
    second = TPGAgentController.create(
        AgentSpec(AgentID("second"), 1, 1), constant_graph(0), memory=memory
    )

    with pytest.raises(MultiAgentConfigurationError, match="live memory"):
        IndependentMultiAgentRuntime((first, second))


def test_episode_lifecycle_and_external_controller_mutation_are_detected() -> None:
    controller = TPGAgentController.create(
        AgentSpec(AgentID("agent"), 0, 1), constant_graph(0)
    )
    runtime = IndependentMultiAgentRuntime((controller,))

    with pytest.raises(MultiAgentStateError, match="reset_episode"):
        runtime.step({AgentID("agent"): ()})
    runtime.reset_episode()
    with pytest.raises(MultiAgentStateError, match="already active"):
        runtime.reset_episode()
    controller.end_episode()
    with pytest.raises(MultiAgentStateError, match="outside"):
        runtime.step({AgentID("agent"): ()})


def test_controller_rejects_graph_actions_outside_agent_space() -> None:
    with pytest.raises(MultiAgentConfigurationError, match="atomic action 2"):
        TPGAgentController.create(AgentSpec(AgentID("agent"), 1, 2), constant_graph(2))
