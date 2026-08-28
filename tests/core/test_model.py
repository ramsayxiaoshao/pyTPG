"""Basic construction and context-free invariant tests."""

import pytest

from tpg.core import (
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
    TeamReference,
    TPGGraph,
)


def make_program(program_id: int = 0) -> Program:
    instruction = Instruction(
        OperatorName("identity"),
        RegisterIndex(0),
        (ConstantOperand(0.0),),
    )
    return Program(ProgramID(program_id), (instruction,))


def test_constructs_graph_with_atomic_and_team_reference_actions() -> None:
    atomic = Learner(LearnerID(0), make_program(), AtomicAction(ActionID(1)))
    reference = Learner(
        LearnerID(1),
        make_program(1),
        TeamReference(TeamID(1)),
    )
    root = Team(TeamID(0), (reference,))
    target = Team(TeamID(1), (atomic,))

    graph = TPGGraph((root, target), (root.id,))

    assert graph.teams == (root, target)
    assert graph.root_team_ids == (TeamID(0),)


def test_program_rejects_empty_instruction_tuple() -> None:
    with pytest.raises(ValueError, match="at least one instruction"):
        Program(ProgramID(0), ())


def test_team_rejects_duplicate_learner_id() -> None:
    learner = Learner(LearnerID(0), make_program(), AtomicAction(ActionID(0)))

    with pytest.raises(ValueError, match="repeats learner ID"):
        Team(TeamID(0), (learner, learner))


def test_graph_rejects_dangling_team_reference() -> None:
    learner = Learner(
        LearnerID(0),
        make_program(),
        TeamReference(TeamID(99)),
    )
    team = Team(TeamID(0), (learner,))

    with pytest.raises(ValueError, match="references missing team"):
        TPGGraph((team,), (team.id,))


def test_constant_operand_rejects_non_finite_value() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        ConstantOperand(float("nan"))
