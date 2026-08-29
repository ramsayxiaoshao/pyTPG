"""Population evaluation and parent-selection semantics."""

from collections.abc import Callable

import pytest

from tpg.core import TPGGraph
from tpg.evaluation import SequentialEvaluator
from tpg.evolution import (
    EvaluatedIndividual,
    EvaluatedPopulation,
    EvolutionConfigurationError,
    GenomeConfig,
    GraphInitializer,
    InitializationConfig,
    InvalidFitnessError,
    Population,
    PopulationInitializer,
    TournamentSelection,
    create_rng,
)


def population(size: int = 4) -> Population:
    initializer = PopulationInitializer(
        GraphInitializer(
            InitializationConfig(
                GenomeConfig(1, 2, 2),
                population_size=size,
                learners_per_team=2,
                program_length=1,
            )
        )
    )
    return initializer.initialize(create_rng(5))


def evaluated(values: tuple[float, ...]) -> EvaluatedPopulation:
    source = population(len(values))
    return EvaluatedPopulation(
        source.generation,
        tuple(
            EvaluatedIndividual(position, graph, fitness)
            for position, (graph, fitness) in enumerate(
                zip(source.individuals, values, strict=True)
            )
        ),
    )


class ScriptedChoiceRNG:
    def __init__(self, samples: tuple[tuple[int, ...], ...]) -> None:
        self.samples = list(samples)

    def choice(self, values: int, *, size: int, replace: bool) -> tuple[int, ...]:
        assert not replace
        sample = self.samples.pop(0)
        assert len(sample) == size
        assert all(0 <= value < values for value in sample)
        return sample


def test_sequential_evaluator_calls_once_in_population_order() -> None:
    source = population(3)
    positions = {
        id(graph): position for position, graph in enumerate(source.individuals)
    }
    calls: list[int] = []

    def fitness(graph: TPGGraph) -> float:
        position = positions[id(graph)]
        calls.append(position)
        return float(position)

    result = SequentialEvaluator().evaluate(source, fitness)

    assert calls == [0, 1, 2]
    assert tuple(item.fitness for item in result.individuals) == (0.0, 1.0, 2.0)
    assert tuple(item.graph for item in result.individuals) == source.individuals


def test_best_and_ranking_use_stable_position_ties() -> None:
    result = evaluated((4.0, 9.0, 9.0, 1.0))

    assert result.best.position == 1
    assert tuple(item.position for item in result.ranked()) == (1, 2, 0, 3)


@pytest.mark.parametrize("fitness", [float("nan"), float("inf"), True, "1"])
def test_invalid_fitness_is_rejected(fitness: object) -> None:
    graph = population(1).individuals[0]

    with pytest.raises(InvalidFitnessError, match="fitness"):
        EvaluatedIndividual(0, graph, fitness)  # type: ignore[arg-type]


def test_tournament_samples_without_replacement_and_breaks_ties_stably() -> None:
    source = evaluated((5.0, 8.0, 8.0, 1.0))
    rng = ScriptedChoiceRNG(((2, 1, 3), (0, 3, 2)))

    selected = TournamentSelection(3).select(source, 2, rng)  # type: ignore[arg-type]

    assert tuple(item.position for item in selected) == (1, 2)


def test_tournament_selection_is_reproducible_for_a_fixed_seed() -> None:
    source = evaluated((0.0, 1.0, 2.0, 3.0))

    first = TournamentSelection(2).select(source, 20, create_rng(99))
    second = TournamentSelection(2).select(source, 20, create_rng(99))

    assert tuple(item.position for item in first) == tuple(
        item.position for item in second
    )


@pytest.mark.parametrize(
    "call",
    [
        lambda: Population(-1, population(1).individuals),
        lambda: Population(0, ()),
        lambda: EvaluatedPopulation(0, ()),
        lambda: TournamentSelection(0),
        lambda: TournamentSelection(5).select(evaluated((1.0, 2.0)), 1, create_rng(0)),
        lambda: TournamentSelection(1).select(evaluated((1.0,)), -1, create_rng(0)),
    ],
)
def test_invalid_population_or_selection_configuration_is_rejected(
    call: Callable[[], object],
) -> None:
    with pytest.raises(EvolutionConfigurationError):
        call()
