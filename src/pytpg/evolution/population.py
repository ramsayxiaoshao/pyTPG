"""Immutable population and evaluated-population values."""

from __future__ import annotations

import math
from dataclasses import dataclass

from pytpg.core import TPGGraph
from pytpg.evolution.errors import EvolutionConfigurationError, InvalidFitnessError


@dataclass(frozen=True, slots=True)
class Population:
    """An ordered generation of independent graph individuals."""

    generation: int
    individuals: tuple[TPGGraph, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
            or self.generation < 0
        ):
            msg = f"generation must be a non-negative integer, got {self.generation!r}"
            raise EvolutionConfigurationError(msg)
        if not isinstance(self.individuals, tuple) or not self.individuals:
            raise EvolutionConfigurationError("population must contain individuals")
        if not all(isinstance(graph, TPGGraph) for graph in self.individuals):
            raise EvolutionConfigurationError(
                "population individuals must be TPGGraph values"
            )

    def __len__(self) -> int:
        return len(self.individuals)


@dataclass(frozen=True, slots=True)
class EvaluatedIndividual:
    """One graph, its original population position, and finite fitness."""

    position: int
    graph: TPGGraph
    fitness: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.position, bool)
            or not isinstance(self.position, int)
            or self.position < 0
        ):
            msg = f"position must be a non-negative integer, got {self.position!r}"
            raise EvolutionConfigurationError(msg)
        if not isinstance(self.graph, TPGGraph):
            raise EvolutionConfigurationError(
                "evaluated individual graph must be a TPGGraph"
            )
        if isinstance(self.fitness, bool) or not isinstance(self.fitness, (int, float)):
            msg = f"fitness must be a finite real number, got {self.fitness!r}"
            raise InvalidFitnessError(msg)
        normalized = float(self.fitness)
        if not math.isfinite(normalized):
            msg = f"fitness must be finite, got {self.fitness!r}"
            raise InvalidFitnessError(msg)
        object.__setattr__(self, "fitness", normalized)


@dataclass(frozen=True, slots=True)
class EvaluatedPopulation:
    """Fitness values preserving the exact source-population order."""

    generation: int
    individuals: tuple[EvaluatedIndividual, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
            or self.generation < 0
        ):
            msg = f"generation must be a non-negative integer, got {self.generation!r}"
            raise EvolutionConfigurationError(msg)
        if not isinstance(self.individuals, tuple) or not all(
            isinstance(individual, EvaluatedIndividual)
            for individual in self.individuals
        ):
            raise EvolutionConfigurationError(
                "evaluated population must contain evaluated individuals"
            )
        if not self.individuals:
            raise EvolutionConfigurationError("evaluated population cannot be empty")
        expected = tuple(range(len(self.individuals)))
        positions = tuple(individual.position for individual in self.individuals)
        if positions != expected:
            raise EvolutionConfigurationError(
                "evaluated-individual positions must match population order"
            )

    def __len__(self) -> int:
        return len(self.individuals)

    @property
    def best(self) -> EvaluatedIndividual:
        """Return the first maximum-fitness individual in stable order."""

        return max(self.individuals, key=lambda individual: individual.fitness)

    def ranked(self) -> tuple[EvaluatedIndividual, ...]:
        """Return descending fitness with stable source-order tie-breaking."""

        return tuple(
            sorted(
                self.individuals,
                key=lambda individual: (-individual.fitness, individual.position),
            )
        )


__all__ = [
    "EvaluatedIndividual",
    "EvaluatedPopulation",
    "Population",
]
