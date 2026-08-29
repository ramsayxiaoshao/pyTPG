"""Small orchestration layer composing evaluation and reproduction."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tpg.callbacks.base import EvolutionCallback, EvolutionEvent, dispatch_event
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

    def __post_init__(self) -> None:
        if not self.evaluations:
            raise EvolutionConfigurationError("run result requires evaluations")
        if len(self.reproductions) + 1 != len(self.evaluations):
            raise EvolutionConfigurationError(
                "run result must contain one more evaluation than reproduction"
            )
        for index, reproduction in enumerate(self.reproductions):
            current = self.evaluations[index]
            following = self.evaluations[index + 1]
            if reproduction.population.generation != current.generation + 1:
                raise EvolutionConfigurationError(
                    "reproduction generation does not follow its evaluation"
                )
            if following.generation != reproduction.population.generation:
                raise EvolutionConfigurationError(
                    "evaluation generation does not follow reproduction"
                )
            evaluated_graphs = tuple(item.graph for item in following.individuals)
            if evaluated_graphs != reproduction.population.individuals:
                raise EvolutionConfigurationError(
                    "evaluated graphs do not match reproduced population"
                )

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
        *,
        callbacks: Iterable[EvolutionCallback] = (),
    ) -> EvolutionRunResult:
        """Evaluate generation zero and apply exactly `generations` reproductions."""

        self._validate_generations(generations)
        selected_callbacks = tuple(callbacks)
        dispatch_event(
            selected_callbacks,
            EvolutionEvent(
                "run_started",
                population.generation,
                len(population),
            ),
        )
        evaluated = self.evaluator.evaluate(population, fitness_function)
        return self._continue_from_evaluated(
            evaluated,
            fitness_function,
            generations,
            rng,
            selected_callbacks,
            resumed=False,
        )

    def resume(
        self,
        evaluated: EvaluatedPopulation,
        fitness_function: FitnessFunction,
        generations: int,
        rng: RandomGenerator,
        *,
        callbacks: Iterable[EvolutionCallback] = (),
    ) -> EvolutionRunResult:
        """Continue from an evaluated checkpoint without evaluating it again."""

        self._validate_generations(generations)
        selected_callbacks = tuple(callbacks)
        dispatch_event(
            selected_callbacks,
            EvolutionEvent(
                "run_started",
                evaluated.generation,
                len(evaluated),
                resumed=True,
            ),
        )
        return self._continue_from_evaluated(
            evaluated,
            fitness_function,
            generations,
            rng,
            selected_callbacks,
            resumed=True,
        )

    def _continue_from_evaluated(
        self,
        initial: EvaluatedPopulation,
        fitness_function: FitnessFunction,
        generations: int,
        rng: RandomGenerator,
        callbacks: tuple[EvolutionCallback, ...],
        *,
        resumed: bool,
    ) -> EvolutionRunResult:
        evaluations = [initial]
        reproductions: list[ReproductionResult] = []
        self._notify_evaluated(callbacks, initial, resumed=resumed)
        for _ in range(generations):
            reproduction = self.reproducer.reproduce(evaluations[-1], rng)
            reproductions.append(reproduction)
            attempts = tuple(
                mutation
                for child in reproduction.children
                for mutation in child.mutations
            )
            dispatch_event(
                callbacks,
                EvolutionEvent(
                    "generation_reproduced",
                    reproduction.population.generation,
                    len(reproduction.population),
                    resumed=resumed,
                    elite_count=sum(child.elite for child in reproduction.children),
                    mutation_attempts=len(attempts),
                    applied_mutations=sum(outcome.applied for outcome in attempts),
                ),
            )
            evaluated = self.evaluator.evaluate(
                reproduction.population, fitness_function
            )
            evaluations.append(evaluated)
            self._notify_evaluated(callbacks, evaluated, resumed=resumed)
        result = EvolutionRunResult(tuple(evaluations), tuple(reproductions))
        dispatch_event(
            callbacks,
            EvolutionEvent(
                "run_finished",
                result.evaluations[-1].generation,
                len(result.evaluations[-1]),
                resumed=resumed,
                best_fitness=result.best.fitness,
            ),
        )
        return result

    @staticmethod
    def _notify_evaluated(
        callbacks: tuple[EvolutionCallback, ...],
        evaluated: EvaluatedPopulation,
        *,
        resumed: bool,
    ) -> None:
        dispatch_event(
            callbacks,
            EvolutionEvent(
                "generation_evaluated",
                evaluated.generation,
                len(evaluated),
                resumed=resumed,
                best_fitness=evaluated.best.fitness,
            ),
        )

    @staticmethod
    def _validate_generations(generations: int) -> None:
        if (
            isinstance(generations, bool)
            or not isinstance(generations, int)
            or generations < 0
        ):
            raise EvolutionConfigurationError("generations must be non-negative")


__all__ = ["EvolutionEngine", "EvolutionRunResult"]
