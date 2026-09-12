"""Tests for non-executing graph inspection statistics."""

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
    TeamReference,
    TPGGraph,
)
from pytpg.runtime import cyclic_team_ids, reachable_team_ids, summarize_graph


def learner(
    learner_id: int,
    action: AtomicAction | TeamReference,
) -> Learner:
    program = Program(
        ProgramID(learner_id),
        (
            Instruction(
                OperatorName("identity"),
                RegisterIndex(0),
                (ConstantOperand(float(learner_id)),),
            ),
        ),
    )
    return Learner(LearnerID(learner_id), program, action)


def test_graph_summary_reports_reachability_cycles_and_unique_entities() -> None:
    first = Team(
        TeamID(0),
        (
            learner(0, TeamReference(TeamID(1))),
            learner(1, AtomicAction(ActionID(0))),
        ),
    )
    second = Team(
        TeamID(1),
        (
            learner(2, TeamReference(TeamID(0))),
            learner(3, AtomicAction(ActionID(1))),
        ),
    )
    orphan = Team(
        TeamID(2),
        (learner(4, AtomicAction(ActionID(2))),),
    )
    graph = TPGGraph((first, second, orphan), (first.id,))

    summary = graph.summary()

    assert summary == summarize_graph(graph)
    assert summary.root_team_count == 1
    assert summary.team_count == 3
    assert summary.reachable_team_count == 2
    assert summary.orphan_team_count == 1
    assert summary.cyclic_team_count == 2
    assert summary.learner_count == 5
    assert summary.program_count == 5
    assert summary.instruction_count == 5
    assert summary.mean_program_length == 1.0
    assert summary.atomic_action_learner_count == 3
    assert summary.team_reference_learner_count == 2
    assert reachable_team_ids(graph) == frozenset({0, 1})
    assert cyclic_team_ids(graph) == frozenset({0, 1})


def test_shared_immutable_learner_is_counted_once() -> None:
    shared = learner(0, AtomicAction(ActionID(0)))
    root = Team(TeamID(0), (shared,))
    orphan = Team(TeamID(1), (shared,))
    graph = TPGGraph((root, orphan), (root.id,))

    summary = graph.summary()

    assert summary.learner_count == 1
    assert summary.program_count == 1
    assert summary.atomic_action_learner_count == 1
