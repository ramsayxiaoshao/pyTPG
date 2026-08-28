"""Tests for learner bidding, stable team selection, and atomic actions."""

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
    RegisterOperand,
    Team,
    TeamID,
    TeamReference,
)
from tpg.runtime import (
    DeterministicRuntime,
    RuntimeConfig,
    TeamReferenceRequiresGraphError,
)


def constant_learner(learner_id: int, bid: float, action_id: int) -> Learner:
    program = Program(
        ProgramID(learner_id),
        (
            Instruction(
                OperatorName("identity"),
                RegisterIndex(0),
                (ConstantOperand(bid),),
            ),
        ),
    )
    return Learner(
        LearnerID(learner_id),
        program,
        AtomicAction(ActionID(action_id)),
    )


def test_team_selects_greatest_raw_bid() -> None:
    low = constant_learner(0, -2.0, 0)
    high = constant_learner(1, 3.0, 1)
    team = Team(TeamID(0), (low, high))
    runtime = DeterministicRuntime(RuntimeConfig(0, 1))

    assert runtime.bids(team, ()) == (-2.0, 3.0)
    assert runtime.select(team, ()) is high
    assert runtime.act(team, ()) == ActionID(1)


def test_tie_breaking_uses_first_team_position_not_lowest_id() -> None:
    first = constant_learner(9, 2.0, 9)
    second = constant_learner(1, 2.0, 1)
    team = Team(TeamID(0), (first, second))
    runtime = DeterministicRuntime(RuntimeConfig(0, 1))

    assert runtime.select(team, ()) is first


def test_learners_do_not_share_register_state_while_bidding() -> None:
    instruction = Instruction(
        OperatorName("add"),
        RegisterIndex(0),
        (RegisterOperand(RegisterIndex(0)), ConstantOperand(1.0)),
    )
    first = Learner(
        LearnerID(0),
        Program(ProgramID(0), (instruction,)),
        AtomicAction(ActionID(0)),
    )
    second = Learner(
        LearnerID(1),
        Program(ProgramID(1), (instruction,)),
        AtomicAction(ActionID(1)),
    )
    team = Team(TeamID(0), (first, second))

    assert DeterministicRuntime(RuntimeConfig(0, 1)).bids(team, ()) == (1.0, 1.0)


def test_winning_team_reference_requires_graph_runtime() -> None:
    program = constant_learner(0, 1.0, 0).program
    learner = Learner(
        LearnerID(0),
        program,
        TeamReference(TeamID(1)),
    )
    team = Team(TeamID(0), (learner,))

    with pytest.raises(TeamReferenceRequiresGraphError, match="Milestone 2"):
        DeterministicRuntime(RuntimeConfig(0, 1)).act(team, ())


def test_team_act_convenience_infers_minimum_register_count() -> None:
    program = Program(
        ProgramID(0),
        (
            Instruction(
                OperatorName("identity"),
                RegisterIndex(2),
                (InputOperand(InputIndex(0)),),
            ),
            Instruction(
                OperatorName("add"),
                RegisterIndex(0),
                (RegisterOperand(RegisterIndex(2)), ConstantOperand(1.0)),
            ),
        ),
    )
    learner = Learner(LearnerID(0), program, AtomicAction(ActionID(7)))
    team = Team(TeamID(0), (learner,))

    assert team.act((4.0,)) == ActionID(7)


def test_team_act_accepts_explicit_runtime_configuration() -> None:
    learner = constant_learner(0, 1.0, 3)
    team = Team(TeamID(0), (learner,))
    runtime = DeterministicRuntime(RuntimeConfig(input_size=2, register_count=4))

    assert team.act((9.0, 8.0), runtime=runtime) == ActionID(3)


def test_fixed_input_program_is_deterministic_across_repeated_calls() -> None:
    program = Program(
        ProgramID(0),
        (
            Instruction(
                OperatorName("multiply"),
                RegisterIndex(0),
                (InputOperand(InputIndex(0)), ConstantOperand(2.5)),
            ),
        ),
    )
    learner = Learner(LearnerID(0), program, AtomicAction(ActionID(4)))
    team = Team(TeamID(0), (learner,))
    runtime = DeterministicRuntime(RuntimeConfig(1, 1))

    results = tuple(runtime.bids(team, (3.0,)) for _ in range(5))

    assert results == ((7.5,),) * 5
