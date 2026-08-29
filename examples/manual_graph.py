"""Build and traverse a small cyclic TPG with deterministic termination."""

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


def learner(
    learner_id: int,
    bid: float,
    action: AtomicAction | TeamReference,
) -> Learner:
    instruction = Instruction(
        OperatorName("identity"),
        RegisterIndex(0),
        (ConstantOperand(bid),),
    )
    return Learner(
        LearnerID(learner_id),
        Program(ProgramID(learner_id), (instruction,)),
        action,
    )


root = Team(
    TeamID(0),
    (
        learner(0, 10.0, TeamReference(TeamID(1))),
        learner(1, 0.0, AtomicAction(ActionID(0))),
    ),
)
child = Team(
    TeamID(1),
    (
        learner(2, 100.0, TeamReference(TeamID(0))),
        learner(3, -1.0, AtomicAction(ActionID(7))),
    ),
)
graph = TPGGraph((root, child), (root.id,))

report = graph.validate()
assert report.is_valid

result = graph.traverse(())
assert result.action_id == ActionID(7)
assert result.visited_team_ids == (0, 1)

print(f"action: {result.action_id}")
print(f"visited teams: {result.visited_team_ids}")
print(f"cyclic teams: {graph.summary().cyclic_team_count}")
