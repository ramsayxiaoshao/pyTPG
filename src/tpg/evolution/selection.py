"""Composable parent-selection strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from tpg.evolution.errors import EvolutionConfigurationError
from tpg.evolution.population import EvaluatedIndividual, EvaluatedPopulation
from tpg.evolution.rng import RandomGenerator


class SelectionStrategy(Protocol):
    """Choose evaluated parents using an explicit RNG."""

    def select(
        self,
        population: EvaluatedPopulation,
        count: int,
        rng: RandomGenerator,
    ) -> tuple[EvaluatedIndividual, ...]: ...


@dataclass(frozen=True, slots=True)
class TournamentSelection:
    """Sample without replacement per tournament and maximize fitness."""

    tournament_size: int = 3

    def __post_init__(self) -> None:
        if (
            isinstance(self.tournament_size, bool)
            or not isinstance(self.tournament_size, int)
            or self.tournament_size < 1
        ):
            raise EvolutionConfigurationError(
                "tournament_size must be a positive integer"
            )

    def select(
        self,
        population: EvaluatedPopulation,
        count: int,
        rng: RandomGenerator,
    ) -> tuple[EvaluatedIndividual, ...]:
        """Select parents; ties favor the earlier population position."""

        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise EvolutionConfigurationError("selection count must be non-negative")
        if self.tournament_size > len(population):
            raise EvolutionConfigurationError(
                "tournament_size cannot exceed evaluated population size"
            )

        selected: list[EvaluatedIndividual] = []
        for _ in range(count):
            positions = tuple(
                int(value)
                for value in rng.choice(
                    len(population),
                    size=self.tournament_size,
                    replace=False,
                )
            )
            candidates = tuple(population.individuals[index] for index in positions)
            selected.append(
                max(
                    candidates,
                    key=lambda individual: (
                        individual.fitness,
                        -individual.position,
                    ),
                )
            )
        return tuple(selected)


__all__ = ["SelectionStrategy", "TournamentSelection"]
