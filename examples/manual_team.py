"""Manually construct and execute a deterministic Milestone 1 team."""

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
)


def learner(learner_id: int, scale: float, action_id: int) -> Learner:
    instruction = Instruction(
        OperatorName("multiply"),
        RegisterIndex(0),
        (InputOperand(InputIndex(0)), ConstantOperand(scale)),
    )
    return Learner(
        LearnerID(learner_id),
        Program(ProgramID(learner_id), (instruction,)),
        AtomicAction(ActionID(action_id)),
    )


positive = learner(0, 1.0, 0)
negative = learner(1, -1.0, 1)
team = Team(TeamID(0), (positive, negative))

assert team.act((0.75,)) == ActionID(0)
assert team.act((-0.75,)) == ActionID(1)

print("positive observation -> action 0")
print("negative observation -> action 1")
