"""Integration and regression tests for safe deterministic graph traversal."""

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
from tpg.runtime import (
    GraphRuntime,
    GraphValidationError,
    NoEligibleLearnerError,
    RootSelectionError,
    RuntimeConfig,
    RuntimeConfigurationError,
    TraversalLimitExceededError,
    TraversalStep,
)


def learner(
    learner_id: int,
    bid: float,
    action: AtomicAction | TeamReference,
) -> Learner:
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
    return Learner(LearnerID(learner_id), program, action)


def atomic(learner_id: int, bid: float, action_id: int) -> Learner:
    return learner(learner_id, bid, AtomicAction(ActionID(action_id)))


def reference(learner_id: int, bid: float, team_id: int) -> Learner:
    return learner(learner_id, bid, TeamReference(TeamID(team_id)))


def two_team_graph() -> TPGGraph:
    root = Team(
        TeamID(0),
        (reference(0, 10.0, 1), atomic(1, 0.0, 0)),
    )
    target = Team(TeamID(1), (atomic(2, 5.0, 7),))
    return TPGGraph((root, target), (root.id,))


def test_graph_traversal_follows_reference_to_atomic_action() -> None:
    graph = two_team_graph()
    runtime = GraphRuntime(RuntimeConfig(0, 1))

    result = runtime.traverse(graph, ())

    assert result.action_id == 7
    assert result.visited_team_ids == (0, 1)
    assert result.steps == (
        TraversalStep(0, 0, 10.0, referenced_team_id=1),
        TraversalStep(1, 2, 5.0, atomic_action_id=7),
    )
    assert tuple(step.action_kind for step in result.steps) == (
        "team_reference",
        "atomic",
    )
    assert runtime.act(graph, ()) == 7


@pytest.mark.parametrize(
    ("atomic_action_id", "referenced_team_id"),
    [(None, None), (1, 2)],
)
def test_traversal_step_keeps_action_and_team_ids_distinct(
    atomic_action_id: int | None,
    referenced_team_id: int | None,
) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        TraversalStep(
            team_id=0,
            learner_id=0,
            bid=1.0,
            atomic_action_id=atomic_action_id,
            referenced_team_id=referenced_team_id,
        )


def test_graph_convenience_act_and_traverse_infer_dimensions() -> None:
    graph = two_team_graph()

    assert graph.act(()) == ActionID(7)
    assert graph.traverse(()).visited_team_ids == (0, 1)


def test_visited_team_reference_is_excluded_to_break_cycle() -> None:
    first = Team(
        TeamID(0),
        (reference(0, 10.0, 1), atomic(1, 0.0, 0)),
    )
    second = Team(
        TeamID(1),
        (reference(2, 100.0, 0), atomic(3, -5.0, 9)),
    )
    graph = TPGGraph((first, second), (first.id,))

    result = graph.traverse(())

    assert result.action_id == 9
    assert result.visited_team_ids == (0, 1)
    assert result.steps[-1].learner_id == 3


def test_equal_eligible_bids_use_stable_team_order() -> None:
    first = atomic(9, 2.0, 9)
    second = reference(1, 2.0, 1)
    root = Team(TeamID(0), (first, second))
    target = Team(TeamID(1), (atomic(2, 1.0, 1),))
    graph = TPGGraph((root, target), (root.id,))

    assert graph.act(()) == ActionID(9)


def test_multiple_roots_require_explicit_selection() -> None:
    first = Team(TeamID(0), (atomic(0, 1.0, 4),))
    second = Team(TeamID(1), (atomic(1, 1.0, 8),))
    graph = TPGGraph((first, second), (first.id, second.id))
    runtime = GraphRuntime(RuntimeConfig(0, 1))

    with pytest.raises(RootSelectionError, match="required"):
        runtime.act(graph, ())
    assert runtime.act(graph, (), root_team_id=1) == 8


def test_non_root_cannot_be_selected_as_traversal_root() -> None:
    graph = two_team_graph()

    with pytest.raises(RootSelectionError, match="not a declared"):
        GraphRuntime(RuntimeConfig(0, 1)).act(graph, (), root_team_id=1)


def test_graph_is_validated_before_execution_by_default() -> None:
    root = Team(TeamID(0), (reference(0, 1.0, 1),))
    target = Team(TeamID(1), (atomic(1, 1.0, 0),))
    graph = TPGGraph((root, target), (root.id,))

    with pytest.raises(GraphValidationError) as captured:
        GraphRuntime(RuntimeConfig(0, 1)).act(graph, ())
    assert captured.value.report.errors[0].code == "team_without_atomic_action"


def test_unvalidated_dead_end_raises_instead_of_looping() -> None:
    first = Team(TeamID(0), (reference(0, 1.0, 1),))
    second = Team(TeamID(1), (reference(1, 1.0, 0),))
    graph = TPGGraph((first, second), (first.id,))
    runtime = GraphRuntime(
        RuntimeConfig(0, 1),
        validate_before_execution=False,
    )

    with pytest.raises(NoEligibleLearnerError, match="no eligible learner"):
        runtime.act(graph, ())


def test_explicit_step_limit_is_a_second_termination_guard() -> None:
    runtime = GraphRuntime(RuntimeConfig(0, 1), max_steps=1)

    with pytest.raises(TraversalLimitExceededError, match="1-step"):
        runtime.act(two_team_graph(), ())


@pytest.mark.parametrize("max_steps", [0, -1, True, 1.5])
def test_invalid_step_limit_is_rejected(max_steps: object) -> None:
    with pytest.raises(RuntimeConfigurationError, match="max_steps"):
        GraphRuntime(RuntimeConfig(0, 1), max_steps=max_steps)  # type: ignore[arg-type]


def test_fixed_graph_traversal_is_reproducible() -> None:
    graph = two_team_graph()
    runtime = GraphRuntime(RuntimeConfig(0, 1))

    results = tuple(runtime.traverse(graph, ()) for _ in range(5))

    assert results == (results[0],) * 5
