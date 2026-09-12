"""Stable elitism and mutation-only asexual reproduction."""

from __future__ import annotations

from dataclasses import dataclass

from pytpg.core import TPGGraph
from pytpg.evolution.errors import EvolutionConfigurationError
from pytpg.evolution.mutation import MutationOperator, MutationOutcome
from pytpg.evolution.population import EvaluatedPopulation, Population
from pytpg.evolution.rng import RandomGenerator
from pytpg.evolution.selection import SelectionStrategy


@dataclass(frozen=True, slots=True)
class ReproductionConfig:
    """Reference reproduction policy independent of population size."""

    elite_count: int = 1
    mutation_steps: int = 1

    def __post_init__(self) -> None:
        if (
            isinstance(self.elite_count, bool)
            or not isinstance(self.elite_count, int)
            or self.elite_count < 0
        ):
            raise EvolutionConfigurationError("elite_count must be non-negative")
        if (
            isinstance(self.mutation_steps, bool)
            or not isinstance(self.mutation_steps, int)
            or self.mutation_steps < 1
        ):
            raise EvolutionConfigurationError("mutation_steps must be positive")


@dataclass(frozen=True, slots=True)
class ChildRecord:
    """Parent position and mutation attempts for one next-generation graph."""

    child_position: int
    parent_position: int
    elite: bool
    mutations: tuple[MutationOutcome, ...]


@dataclass(frozen=True, slots=True)
class ReproductionResult:
    """The next population and its complete local reproduction record."""

    population: Population
    children: tuple[ChildRecord, ...]


class Reproducer:
    """Copy stable elites, then select and mutate asexual parents."""

    __slots__ = ("config", "mutation", "selection")

    def __init__(
        self,
        selection: SelectionStrategy,
        mutation: MutationOperator,
        config: ReproductionConfig | None = None,
    ) -> None:
        self.selection = selection
        self.mutation = mutation
        self.config = config or ReproductionConfig()

    def reproduce(
        self,
        evaluated: EvaluatedPopulation,
        rng: RandomGenerator,
    ) -> ReproductionResult:
        """Create one equal-sized next generation with no in-place changes."""

        if self.config.elite_count > len(evaluated):
            raise EvolutionConfigurationError(
                "elite_count cannot exceed evaluated population size"
            )

        graphs: list[TPGGraph] = []
        records: list[ChildRecord] = []
        for elite in evaluated.ranked()[: self.config.elite_count]:
            child_position = len(graphs)
            graphs.append(elite.graph)
            records.append(
                ChildRecord(
                    child_position=child_position,
                    parent_position=elite.position,
                    elite=True,
                    mutations=(),
                )
            )

        while len(graphs) < len(evaluated):
            parent = self.selection.select(evaluated, 1, rng)[0]
            graph = parent.graph
            outcomes: list[MutationOutcome] = []
            for _ in range(self.config.mutation_steps):
                outcome = self.mutation.mutate(graph, rng)
                outcomes.append(outcome)
                graph = outcome.graph
            child_position = len(graphs)
            graphs.append(graph)
            records.append(
                ChildRecord(
                    child_position=child_position,
                    parent_position=parent.position,
                    elite=False,
                    mutations=tuple(outcomes),
                )
            )

        population = Population(evaluated.generation + 1, tuple(graphs))
        return ReproductionResult(population, tuple(records))


__all__ = [
    "ChildRecord",
    "Reproducer",
    "ReproductionConfig",
    "ReproductionResult",
]
