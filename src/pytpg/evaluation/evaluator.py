"""Fitness evaluation abstraction and sequential reference implementation."""

from __future__ import annotations

from typing import Protocol

from pytpg.core import TPGGraph
from pytpg.evolution.population import (
    EvaluatedIndividual,
    EvaluatedPopulation,
    Population,
)


class FitnessFunction(Protocol):
    """A deterministic or explicitly stateful scalar graph objective."""

    def __call__(self, graph: TPGGraph) -> float: ...


class Evaluator(Protocol):
    """Evaluation strategy independent of evolution and parallelization."""

    def evaluate(
        self,
        population: Population,
        fitness_function: FitnessFunction,
    ) -> EvaluatedPopulation: ...


class SequentialEvaluator:
    """Evaluate graphs in stable population order in the current process."""

    __slots__ = ()

    def evaluate(
        self,
        population: Population,
        fitness_function: FitnessFunction,
    ) -> EvaluatedPopulation:
        """Evaluate once per graph and reject invalid fitness values."""

        return EvaluatedPopulation(
            generation=population.generation,
            individuals=tuple(
                EvaluatedIndividual(position, graph, fitness_function(graph))
                for position, graph in enumerate(population.individuals)
            ),
        )


__all__ = ["Evaluator", "FitnessFunction", "SequentialEvaluator"]
