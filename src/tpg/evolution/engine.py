"""Small orchestration layer composing evaluation and reproduction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tpg.evolution.errors import EvolutionConfigurationError
from tpg.evolution.population import (
    EvaluatedIndividual,
    EvaluatedPopulation,
    Population,
)
from tpg.evolution.reproduction import Reproducer, ReproductionResult
from tpg.evolution.rng import RandomGenerator

if TYPE_CHECKING:
    from tpg.evaluation.evaluator import Evaluator, FitnessFunction


@dataclass(frozen=True, slots=True)
class EvolutionRunResult:
    """Immutable evaluations and reproduction records for a bounded run."""

    evaluations: tuple[EvaluatedPopulation, ...]
    reproductions: tuple[ReproductionResult, ...]

    @property
    def final_population(self) -> Population:
        """Return the population corresponding to the final evaluation."""

        final = self.evaluations[-1]
        return Population(
            generation=final.generation,
            individuals=tuple(item.graph for item in final.individuals),
        )

    @property
    def best_fitness_history(self) -> tuple[float, ...]:
        """Return one stable best fitness per evaluated generation."""

        return tuple(evaluation.best.fitness for evaluation in self.evaluations)

    @property
    def best(self) -> EvaluatedIndividual:
        """Return the stable best individual from the final generation."""

        return self.evaluations[-1].best


class EvolutionEngine:
    """Compose evaluator and reproducer without owning config or global RNG."""

    __slots__ = ("evaluator", "reproducer")

    def __init__(self, evaluator: Evaluator, reproducer: Reproducer) -> None:
        self.evaluator = evaluator
        self.reproducer = reproducer

    def run(
        self,
        population: Population,
        fitness_function: FitnessFunction,
        generations: int,
        rng: RandomGenerator,
    ) -> EvolutionRunResult:
        """Evaluate generation zero and apply exactly `generations` reproductions."""

        if (
            isinstance(generations, bool)
            or not isinstance(generations, int)
            or generations < 0
        ):
            raise EvolutionConfigurationError("generations must be non-negative")

        current = population
        evaluations: list[EvaluatedPopulation] = []
        reproductions: list[ReproductionResult] = []
        for _ in range(generations):
            evaluated = self.evaluator.evaluate(current, fitness_function)
            evaluations.append(evaluated)
            reproduction = self.reproducer.reproduce(evaluated, rng)
            reproductions.append(reproduction)
            current = reproduction.population
        evaluations.append(self.evaluator.evaluate(current, fitness_function))
        return EvolutionRunResult(tuple(evaluations), tuple(reproductions))


__all__ = ["EvolutionEngine", "EvolutionRunResult"]
