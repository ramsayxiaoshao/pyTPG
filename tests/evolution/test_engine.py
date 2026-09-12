"""Reproduction, evolution-engine, and toy-task integration tests."""

from collections.abc import Callable
from dataclasses import dataclass

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
from pytpg.evaluation import (
    ContextualBandit,
    ContextualBanditCase,
    SequentialEvaluator,
)
from pytpg.evolution import (
    EvaluatedIndividual,
    EvaluatedPopulation,
    EvolutionConfigurationError,
    EvolutionEngine,
    GenomeConfig,
    GraphInitializer,
    InitializationConfig,
    MutationOutcome,
    Population,
    PopulationInitializer,
    Reproducer,
    ReproductionConfig,
    TournamentSelection,
    create_rng,
    default_mutation,
)


def constant_graph(graph_id: int, action_id: int) -> TPGGraph:
    program = Program(
        ProgramID(graph_id),
        (
            Instruction(
                OperatorName("identity"),
                RegisterIndex(0),
                (ConstantOperand(float(graph_id)),),
            ),
        ),
    )
    learner = Learner(
        LearnerID(graph_id),
        program,
        AtomicAction(ActionID(action_id)),
    )
    team = Team(TeamID(graph_id), (learner,))
    return TPGGraph((team,), (team.id,))


@dataclass(frozen=True)
class NoOpMutation:
    name: str = "noop"

    def mutate(self, graph: TPGGraph, rng: object) -> MutationOutcome:
        return MutationOutcome(graph, self.name, False, "test no-op")


@dataclass(frozen=True)
class FirstSelection:
    def select(
        self,
        population: EvaluatedPopulation,
        count: int,
        rng: object,
    ) -> tuple[EvaluatedIndividual, ...]:
        return (population.individuals[0],) * count


def evaluated_population() -> EvaluatedPopulation:
    graphs = tuple(constant_graph(index, index % 2) for index in range(4))
    fitness = (1.0, 3.0, 3.0, 0.0)
    return EvaluatedPopulation(
        6,
        tuple(
            EvaluatedIndividual(position, graph, value)
            for position, (graph, value) in enumerate(
                zip(graphs, fitness, strict=True)
            )
        ),
    )


def test_reproduction_keeps_stable_elites_and_records_parentage() -> None:
    evaluated = evaluated_population()
    reproducer = Reproducer(
        FirstSelection(),  # type: ignore[arg-type]
        NoOpMutation(),  # type: ignore[arg-type]
        ReproductionConfig(elite_count=2, mutation_steps=2),
    )

    result = reproducer.reproduce(evaluated, create_rng(1))

    assert result.population.generation == 7
    assert len(result.population) == len(evaluated)
    assert result.population.individuals[:2] == (
        evaluated.individuals[1].graph,
        evaluated.individuals[2].graph,
    )
    assert tuple(record.parent_position for record in result.children) == (1, 2, 0, 0)
    assert tuple(record.elite for record in result.children) == (
        True,
        True,
        False,
        False,
    )
    assert tuple(len(record.mutations) for record in result.children) == (0, 0, 2, 2)
    assert evaluated == evaluated_population()


def test_engine_evaluates_generation_zero_and_each_offspring_generation() -> None:
    source = Population(
        0, tuple(constant_graph(index, index % 2) for index in range(3))
    )
    reproducer = Reproducer(
        FirstSelection(),  # type: ignore[arg-type]
        NoOpMutation(),  # type: ignore[arg-type]
        ReproductionConfig(elite_count=1),
    )
    engine = EvolutionEngine(SequentialEvaluator(), reproducer)

    result = engine.run(
        source,
        lambda graph: float(graph.act(())),
        generations=2,
        rng=create_rng(10),
    )

    assert tuple(item.generation for item in result.evaluations) == (0, 1, 2)
    assert len(result.reproductions) == 2
    assert result.final_population.generation == 2
    assert result.best_fitness_history == (1.0, 1.0, 1.0)
    assert result.best.fitness == 1.0


def test_zero_generation_run_only_evaluates_initial_population() -> None:
    source = Population(4, (constant_graph(0, 0),))
    engine = EvolutionEngine(
        SequentialEvaluator(),
        Reproducer(
            FirstSelection(),  # type: ignore[arg-type]
            NoOpMutation(),  # type: ignore[arg-type]
        ),
    )

    result = engine.run(source, lambda graph: 2.0, 0, create_rng(0))

    assert len(result.evaluations) == 1
    assert result.reproductions == ()
    assert result.final_population == source


@pytest.mark.parametrize(
    "config",
    [
        {"elite_count": -1},
        {"mutation_steps": 0},
    ],
)
def test_invalid_reproduction_config_is_rejected(config: dict[str, int]) -> None:
    with pytest.raises(EvolutionConfigurationError):
        ReproductionConfig(**config)


def test_elite_count_cannot_exceed_population() -> None:
    reproducer = Reproducer(
        FirstSelection(),  # type: ignore[arg-type]
        NoOpMutation(),  # type: ignore[arg-type]
        ReproductionConfig(elite_count=5),
    )

    with pytest.raises(EvolutionConfigurationError, match="elite_count"):
        reproducer.reproduce(evaluated_population(), create_rng(0))


def test_contextual_bandit_scores_exact_accuracy() -> None:
    task = ContextualBandit.signed_binary()

    always_zero = constant_graph(0, 0)

    assert task(always_zero) == 0.5
    assert task.input_size == 1
    assert task.n_actions == 2


def test_invalid_toy_case_is_rejected() -> None:
    with pytest.raises(EvolutionConfigurationError, match="finite"):
        ContextualBanditCase((float("nan"),), ActionID(0))
    with pytest.raises(EvolutionConfigurationError, match="expected action"):
        ContextualBanditCase((0.0,), ActionID(-1))


@pytest.mark.parametrize(
    "task",
    [
        lambda: ContextualBandit(True, 2, ContextualBandit.signed_binary().cases),
        lambda: ContextualBandit(1, 0, ContextualBandit.signed_binary().cases),
        lambda: ContextualBandit(1, 2, ()),
        lambda: ContextualBandit(
            2,
            2,
            (ContextualBanditCase((0.0,), ActionID(0)),),
        ),
        lambda: ContextualBandit(
            1,
            2,
            (ContextualBanditCase((0.0,), ActionID(2)),),
        ),
    ],
)
def test_invalid_toy_task_is_rejected(
    task: Callable[[], ContextualBandit],
) -> None:
    with pytest.raises(EvolutionConfigurationError):
        task()


def run_tiny_evolution(seed: int) -> tuple[tuple[float, ...], Population]:
    rng = create_rng(seed)
    initializer = GraphInitializer(
        InitializationConfig(
            GenomeConfig(1, 4, 2, max_program_length=8),
            population_size=4,
            learners_per_team=2,
            program_length=2,
        )
    )
    population = PopulationInitializer(initializer).initialize(rng)
    reproducer = Reproducer(
        TournamentSelection(2),
        default_mutation(initializer.factory),
        ReproductionConfig(elite_count=1, mutation_steps=2),
    )
    result = EvolutionEngine(SequentialEvaluator(), reproducer).run(
        population,
        ContextualBandit.signed_binary(),
        generations=20,
        rng=rng,
    )
    return result.best_fitness_history, result.final_population


def test_fixed_seed_toy_evolution_is_reproducible_and_valid() -> None:
    first_history, first_population = run_tiny_evolution(0)
    second_history, second_population = run_tiny_evolution(0)

    assert first_history == second_history
    assert first_population == second_population
    assert first_history == (0.5,) * 13 + (1.0,) * 8
    assert all(
        graph.validate().is_valid for graph in first_population.individuals
    )
