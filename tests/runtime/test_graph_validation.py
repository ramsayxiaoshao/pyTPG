"""Graph-wide invariant and diagnostic tests."""

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
    TeamReference,
    TPGGraph,
)
from tpg.runtime import GraphValidationError, RuntimeConfig


def learner(
    learner_id: int,
    action: AtomicAction | TeamReference,
    *,
    operator: str = "identity",
    input_index: int | None = None,
) -> Learner:
    operand = (
        ConstantOperand(1.0)
        if input_index is None
        else InputOperand(InputIndex(input_index))
    )
    program = Program(
        ProgramID(learner_id),
        (
            Instruction(
                OperatorName(operator),
                RegisterIndex(0),
                (operand,),
            ),
        ),
    )
    return Learner(LearnerID(learner_id), program, action)


def atomic(learner_id: int, action_id: int = 0) -> Learner:
    return learner(learner_id, AtomicAction(ActionID(action_id)))


def issue_codes(graph: TPGGraph) -> set[str]:
    return {issue.code for issue in graph.validate().issues}


def test_valid_graph_has_no_diagnostics() -> None:
    team = Team(TeamID(0), (atomic(0),))
    graph = TPGGraph((team,), (team.id,))

    report = graph.validate()

    assert report.is_valid
    assert report.errors == ()
    assert report.warnings == ()
    report.require_valid()


def test_team_without_atomic_action_is_invalid() -> None:
    root = Team(
        TeamID(0),
        (learner(0, TeamReference(TeamID(1))),),
    )
    target = Team(TeamID(1), (atomic(1),))
    graph = TPGGraph((root, target), (root.id,))

    report = graph.validate()

    assert not report.is_valid
    assert "team_without_atomic_action" in {issue.code for issue in report.errors}
    with pytest.raises(GraphValidationError) as captured:
        report.require_valid()
    assert captured.value.report is report


def test_self_reference_is_invalid_even_with_atomic_fallback() -> None:
    self_reference = learner(0, TeamReference(TeamID(0)))
    team = Team(TeamID(0), (self_reference, atomic(1)))
    graph = TPGGraph((team,), (team.id,))

    assert "self_reference" in issue_codes(graph)


def test_orphan_team_is_a_warning_not_an_error() -> None:
    root = Team(TeamID(0), (atomic(0),))
    orphan = Team(TeamID(1), (atomic(1),))
    graph = TPGGraph((root, orphan), (root.id,))

    report = graph.validate()

    assert report.is_valid
    assert tuple(issue.code for issue in report.warnings) == ("orphan_team",)
    assert report.warnings[0].team_id == 1


def test_non_self_cycle_is_valid_and_not_reported_as_an_error() -> None:
    first = Team(
        TeamID(0),
        (learner(0, TeamReference(TeamID(1))), atomic(1)),
    )
    second = Team(
        TeamID(1),
        (learner(2, TeamReference(TeamID(0))), atomic(3)),
    )
    graph = TPGGraph((first, second), (first.id,))

    assert graph.validate().is_valid


def test_explicit_runtime_config_detects_out_of_bounds_input() -> None:
    indexed = learner(
        0,
        AtomicAction(ActionID(0)),
        input_index=2,
    )
    team = Team(TeamID(0), (indexed,))
    graph = TPGGraph((team,), (team.id,))

    inferred_report = graph.validate()
    explicit_report = graph.validate(RuntimeConfig(input_size=2, register_count=1))

    assert inferred_report.is_valid
    assert not explicit_report.is_valid
    issue = explicit_report.errors[0]
    assert issue.code == "invalid_instruction"
    assert issue.team_id == 0
    assert issue.learner_id == 0
    assert issue.program_id == 0
    assert issue.instruction_index == 0


def test_unknown_operator_is_reported_with_location() -> None:
    unknown = learner(
        0,
        AtomicAction(ActionID(0)),
        operator="unknown",
    )
    team = Team(TeamID(0), (unknown,))
    graph = TPGGraph((team,), (team.id,))

    issue = graph.validate().errors[0]

    assert issue.code == "unknown_operator"
    assert issue.team_id == 0
    assert issue.learner_id == 0
    assert issue.program_id == 0
