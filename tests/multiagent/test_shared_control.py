"""Centralized shared-control runtime tests."""

import pytest

from tpg.core import (
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
from tpg.memory import MemorySnapshot, ObservationHistoryMemory
from tpg.multiagent import (
    AgentID,
    AgentSpec,
    MultiAgentConfigurationError,
    MultiAgentObservationError,
    MultiAgentStateError,
    SharedControlRuntime,
)


def sign_graph(negative_action: int, positive_action: int) -> TPGGraph:
    negative = Program(
        ProgramID(0),
        (
            Instruction(
                OperatorName("subtract"),
                RegisterIndex(0),
                (ConstantOperand(0.0), InputOperand(InputIndex(0))),
            ),
        ),
    )
    positive = Program(
        ProgramID(1),
        (
            Instruction(
                OperatorName("identity"),
                RegisterIndex(0),
                (InputOperand(InputIndex(0)),),
            ),
        ),
    )
    team = Team(
        TeamID(0),
        (
            Learner(LearnerID(0), negative, AtomicAction(ActionID(negative_action))),
            Learner(LearnerID(1), positive, AtomicAction(ActionID(positive_action))),
        ),
    )
    return TPGGraph((team,), (team.id,))


def specs() -> tuple[AgentSpec, ...]:
    return (
        AgentSpec(AgentID("scout"), 1, 2),
        AgentSpec(AgentID("arm"), 2, 3),
    )


def test_shared_control_concatenates_in_roster_order_and_decodes_once() -> None:
    runtime = SharedControlRuntime.create(specs(), sign_graph(0, 5))
    runtime.reset_episode()

    result = runtime.step({AgentID("arm"): (8.0, 9.0), AgentID("scout"): (1.0,)})

    assert result.joint_action_id == 5
    assert result.actions_by_agent == {
        AgentID("scout"): 1,
        AgentID("arm"): 2,
    }
    assert result.traversal.augmented_observation == (1.0, 8.0, 9.0)
    assert len(result.traversal.traversal.steps) == 1


def test_shared_memory_updates_once_with_joint_observation() -> None:
    memory = ObservationHistoryMemory(observation_size=3)
    runtime = SharedControlRuntime.create(specs(), sign_graph(0, 5), memory=memory)
    assert runtime.reset_episode() == MemorySnapshot((0.0, 0.0, 0.0), 0)

    result = runtime.step({AgentID("scout"): (-1.0,), AgentID("arm"): (2.0, 3.0)})

    assert result.joint_action_id == 0
    assert result.traversal.memory_after == MemorySnapshot((-1.0, 2.0, 3.0), 1)
    assert runtime.end_episode() == result.traversal.memory_after


def test_shared_control_requires_exact_finite_observations() -> None:
    runtime = SharedControlRuntime.create(specs(), sign_graph(0, 5))
    runtime.reset_episode()

    with pytest.raises(MultiAgentObservationError, match="unexpected"):
        runtime.step(
            {
                AgentID("scout"): (1.0,),
                AgentID("arm"): (2.0, 3.0),
                AgentID("other"): (4.0,),
            }
        )
    with pytest.raises(MultiAgentObservationError, match="length"):
        runtime.step({AgentID("scout"): (), AgentID("arm"): (2.0, 3.0)})
    with pytest.raises(MultiAgentObservationError, match="finite"):
        runtime.step(
            {
                AgentID("scout"): (float("nan"),),
                AgentID("arm"): (2.0, 3.0),
            }
        )


def test_shared_control_rejects_unrepresentable_graph_action() -> None:
    with pytest.raises(MultiAgentConfigurationError, match="atomic action 6"):
        SharedControlRuntime.create(specs(), sign_graph(0, 6))


def test_shared_control_enforces_episode_lifecycle() -> None:
    runtime = SharedControlRuntime.create(specs(), sign_graph(0, 5))
    observations = {
        AgentID("scout"): (1.0,),
        AgentID("arm"): (2.0, 3.0),
    }

    with pytest.raises(MultiAgentStateError, match="reset_episode"):
        runtime.step(observations)
    runtime.reset_episode()
    with pytest.raises(MultiAgentStateError, match="already active"):
        runtime.reset_episode()
    runtime.end_episode()
    with pytest.raises(MultiAgentStateError, match="reset_episode"):
        runtime.step(observations)
