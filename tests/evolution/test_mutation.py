"""Unit and invariant tests for focused immutable mutation operators."""

from dataclasses import dataclass

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
    TPGGraph,
)
from tpg.evolution import (
    AddLearnerMutation,
    AddTeamMutation,
    DeleteInstructionMutation,
    DeleteLearnerMutation,
    DeleteTeamMutation,
    EvolutionConfigurationError,
    GenomeConfig,
    GenomeFactory,
    GraphInitializer,
    InitializationConfig,
    InsertInstructionMutation,
    ModifyInstructionMutation,
    MutateLearnerAction,
    MutationConfig,
    MutationOutcome,
    WeightedMutation,
    create_rng,
    default_mutation,
)
from tpg.runtime import reachable_team_ids


class ZeroRNG:
    """Small deterministic protocol implementation for exact operator tests."""

    def integers(self, high: int) -> int:
        assert high > 0
        return 0

    def uniform(self, low: float, high: float) -> float:
        assert low < high
        return low

    def choice(self, values: int, *, size: int, replace: bool) -> tuple[int, ...]:
        assert not replace
        return tuple(range(size))

    def random(self) -> float:
        return 0.0


def instruction(value: float) -> Instruction:
    return Instruction(
        OperatorName("identity"),
        RegisterIndex(0),
        (ConstantOperand(value),),
    )


def learner(learner_id: int, length: int = 2, action_id: int = 0) -> Learner:
    return Learner(
        LearnerID(learner_id),
        Program(
            ProgramID(learner_id),
            tuple(instruction(float(index + 1)) for index in range(length)),
        ),
        AtomicAction(ActionID(action_id)),
    )


def atomic_graph(*, length: int = 2, learners: int = 2) -> TPGGraph:
    team = Team(
        TeamID(0),
        tuple(learner(index, length, index % 2) for index in range(learners)),
    )
    return TPGGraph((team,), (team.id,))


def factory(*, minimum: int = 1, maximum: int = 4) -> GenomeFactory:
    return GenomeFactory(
        GenomeConfig(
            input_size=1,
            register_count=2,
            n_actions=2,
            min_program_length=minimum,
            max_program_length=maximum,
        )
    )


@pytest.mark.parametrize(
    ("operator_type", "expected_delta"),
    [
        (InsertInstructionMutation, 1),
        (DeleteInstructionMutation, -1),
        (ModifyInstructionMutation, 0),
    ],
)
def test_instruction_mutations_clone_selected_membership(
    operator_type: type[
        InsertInstructionMutation
        | DeleteInstructionMutation
        | ModifyInstructionMutation
    ],
    expected_delta: int,
) -> None:
    source = atomic_graph()
    original = source.teams[0].learners[0]

    outcome = operator_type(factory()).mutate(source, ZeroRNG())
    changed = outcome.graph.teams[0].learners[0]

    assert outcome.applied
    assert outcome.graph is not source
    assert source.teams[0].learners[0] == original
    assert changed.id > max(item.id for item in source.teams[0].learners)
    assert changed.program.id > max(
        item.program.id for item in source.teams[0].learners
    )
    assert len(changed.program.instructions) == (
        len(original.program.instructions) + expected_delta
    )
    assert outcome.graph.validate(factory().runtime_config).is_valid


def test_instruction_bounds_are_explicit_no_ops() -> None:
    bounded = factory(minimum=1, maximum=1)
    source = atomic_graph(length=1)

    inserted = InsertInstructionMutation(bounded).mutate(source, ZeroRNG())
    deleted = DeleteInstructionMutation(bounded).mutate(source, ZeroRNG())

    assert not inserted.applied
    assert inserted.graph is source
    assert not deleted.applied
    assert deleted.graph is source


def test_shared_learner_is_cloned_only_for_selected_team_membership() -> None:
    shared = learner(0)
    first = Team(TeamID(0), (shared, learner(1)))
    second = Team(TeamID(1), (shared, learner(2)))
    source = TPGGraph((first, second), (first.id, second.id))

    outcome = ModifyInstructionMutation(factory()).mutate(source, ZeroRNG())

    assert outcome.graph.teams[0].learners[0].id == 3
    assert outcome.graph.teams[1].learners[0] is shared
    assert source.teams[0].learners[0] is shared


def test_action_mutation_changes_action_and_clones_learner_not_program() -> None:
    source = atomic_graph()
    original = source.teams[0].learners[0]

    outcome = MutateLearnerAction(factory()).mutate(source, ZeroRNG())
    changed = outcome.graph.teams[0].learners[0]

    assert outcome.applied
    assert changed.action == AtomicAction(ActionID(1))
    assert changed.program is original.program
    assert changed.id != original.id


def test_action_mutation_reports_no_op_when_no_alternative_exists() -> None:
    one_action_factory = GenomeFactory(GenomeConfig(0, 1, 1))
    source = TPGGraph(
        (Team(TeamID(0), (learner(0, length=1, action_id=0),)),),
        (TeamID(0),),
    )

    outcome = MutateLearnerAction(one_action_factory).mutate(source, ZeroRNG())

    assert not outcome.applied
    assert outcome.graph is source


def test_add_and_delete_learner_respect_team_size_bounds() -> None:
    source = atomic_graph(learners=2)
    added = AddLearnerMutation(factory(), max_team_size=3, program_length=2).mutate(
        source, ZeroRNG()
    )

    assert added.applied
    assert len(added.graph.teams[0].learners) == 3
    deleted = DeleteLearnerMutation(factory(), min_team_size=2).mutate(
        added.graph, ZeroRNG()
    )
    assert deleted.applied
    assert len(deleted.graph.teams[0].learners) == 2
    assert not DeleteLearnerMutation(factory(), min_team_size=2).mutate(
        source, ZeroRNG()
    ).applied


def test_add_and_delete_team_preserve_references_and_roots() -> None:
    source = atomic_graph(learners=2)

    added = AddTeamMutation(
        factory(), max_team_size=3, new_team_size=2, program_length=2
    ).mutate(source, ZeroRNG())

    assert added.applied
    assert added.graph.root_team_ids == source.root_team_ids
    assert len(added.graph.teams) == 2
    assert reachable_team_ids(added.graph) == {0, 1}
    assert added.graph.validate(factory().runtime_config).is_valid

    deleted = DeleteTeamMutation(factory(), min_team_size=2).mutate(
        added.graph, ZeroRNG()
    )
    assert deleted.applied
    assert deleted.graph == source


def test_delete_team_no_ops_when_incoming_reference_cannot_be_removed() -> None:
    initializer = GraphInitializer(
        InitializationConfig(
            GenomeConfig(1, 2, 2),
            population_size=1,
            team_count=2,
            learners_per_team=2,
            program_length=1,
        )
    )
    source = initializer.initialize(create_rng(2))

    outcome = DeleteTeamMutation(initializer.factory, min_team_size=2).mutate(
        source, ZeroRNG()
    )

    assert not outcome.applied
    assert outcome.graph is source


@dataclass(frozen=True)
class MarkerMutation:
    name: str

    def mutate(self, graph: TPGGraph, rng: object) -> MutationOutcome:
        return MutationOutcome(graph, self.name, False, f"selected {self.name}")


def test_weighted_mutation_uses_relative_interval_boundaries() -> None:
    source = atomic_graph()
    operators = (MarkerMutation("first"), MarkerMutation("second"))
    mutation = WeightedMutation(operators, (1.0, 3.0))

    assert mutation.mutate(source, ZeroRNG()).operator == "first"


@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf"), True])
def test_invalid_mutation_weight_is_rejected(weight: object) -> None:
    with pytest.raises(EvolutionConfigurationError, match="weights"):
        WeightedMutation((MarkerMutation("only"),), (weight,))  # type: ignore[arg-type]


def test_default_mutation_preserves_validity_across_fixed_seed_sequences() -> None:
    config = InitializationConfig(
        GenomeConfig(2, 4, 3, max_program_length=8),
        population_size=1,
        team_count=2,
        learners_per_team=4,
        program_length=3,
    )
    initializer = GraphInitializer(config)
    mutation = default_mutation(initializer.factory)

    for seed in range(12):
        rng = create_rng(seed)
        graph = initializer.initialize(rng)
        for _ in range(80):
            outcome = mutation.mutate(graph, rng)
            graph = outcome.graph
            report = graph.validate(
                initializer.factory.runtime_config,
                operators=initializer.factory.operators,
            )
            assert report.is_valid, (seed, outcome, report.errors)


@pytest.mark.parametrize(
    "values",
    [
        {"min_team_size": 0},
        {"min_team_size": 3, "new_team_size": 2},
    ],
)
def test_invalid_mutation_bounds_are_rejected(values: dict[str, int]) -> None:
    with pytest.raises(EvolutionConfigurationError):
        MutationConfig(**values)
