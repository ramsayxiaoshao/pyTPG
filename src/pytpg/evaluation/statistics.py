"""Deterministic fitness and mutation summaries for completed run data."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from pytpg.evolution import EvaluatedPopulation, EvolutionRunResult, ReproductionResult
from pytpg.evolution.errors import EvolutionConfigurationError


@dataclass(frozen=True, slots=True)
class OperatorStatistics:
    """Attempt and applied counts for one named mutation operator."""

    operator: str
    attempts: int
    applied: int

    def __post_init__(self) -> None:
        if not isinstance(self.operator, str) or not self.operator.strip():
            raise EvolutionConfigurationError("operator statistic name cannot be empty")
        if (
            isinstance(self.attempts, bool)
            or not isinstance(self.attempts, int)
            or self.attempts < 1
        ):
            raise EvolutionConfigurationError(
                "operator statistic attempts must be positive"
            )
        if (
            isinstance(self.applied, bool)
            or not isinstance(self.applied, int)
            or not 0 <= self.applied <= self.attempts
        ):
            raise EvolutionConfigurationError(
                "operator statistic applied count is invalid"
            )


@dataclass(frozen=True, slots=True)
class GenerationStatistics:
    """Stable scalar summary for one evaluated population and reproduction."""

    generation: int
    population_size: int
    minimum_fitness: float
    maximum_fitness: float
    mean_fitness: float
    median_fitness: float
    population_stddev: float
    best_position: int
    elite_count: int = 0
    mutation_attempts: int = 0
    applied_mutations: int = 0
    operators: tuple[OperatorStatistics, ...] = ()

    def __post_init__(self) -> None:
        for name, value, minimum in (
            ("generation", self.generation, 0),
            ("population_size", self.population_size, 1),
            ("best_position", self.best_position, 0),
            ("elite_count", self.elite_count, 0),
            ("mutation_attempts", self.mutation_attempts, 0),
            ("applied_mutations", self.applied_mutations, 0),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < minimum
            ):
                raise EvolutionConfigurationError(
                    f"generation statistic {name} must be at least {minimum}"
                )
        if self.best_position >= self.population_size:
            raise EvolutionConfigurationError(
                "best_position must be inside the population"
            )
        fitness = (
            self.minimum_fitness,
            self.maximum_fitness,
            self.mean_fitness,
            self.median_fitness,
            self.population_stddev,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in fitness
        ):
            raise EvolutionConfigurationError(
                "generation fitness statistics must be finite"
            )
        if self.minimum_fitness > self.maximum_fitness:
            raise EvolutionConfigurationError(
                "minimum fitness cannot exceed maximum fitness"
            )
        if not (
            self.minimum_fitness <= self.mean_fitness <= self.maximum_fitness
            and self.minimum_fitness <= self.median_fitness <= self.maximum_fitness
        ):
            raise EvolutionConfigurationError(
                "mean and median fitness must lie within the fitness range"
            )
        if self.population_stddev < 0.0:
            raise EvolutionConfigurationError(
                "population standard deviation cannot be negative"
            )
        if not 0 <= self.elite_count <= self.population_size:
            raise EvolutionConfigurationError("elite count is outside population")
        if not 0 <= self.applied_mutations <= self.mutation_attempts:
            raise EvolutionConfigurationError("applied mutation count is invalid")
        if not isinstance(self.operators, tuple) or any(
            not isinstance(item, OperatorStatistics) for item in self.operators
        ):
            raise EvolutionConfigurationError(
                "operator statistics must be a tuple of OperatorStatistics"
            )
        if sum(item.attempts for item in self.operators) != self.mutation_attempts:
            raise EvolutionConfigurationError(
                "operator attempts do not match total mutation attempts"
            )
        if sum(item.applied for item in self.operators) != self.applied_mutations:
            raise EvolutionConfigurationError(
                "operator applied counts do not match total applied mutations"
            )

    @classmethod
    def calculate(
        cls,
        evaluated: EvaluatedPopulation,
        reproduction: ReproductionResult | None = None,
    ) -> GenerationStatistics:
        """Calculate without changing evaluation order or consuming randomness."""

        if reproduction is not None and (
            reproduction.population.generation != evaluated.generation + 1
            or len(reproduction.population) != len(evaluated)
        ):
            raise EvolutionConfigurationError(
                "reproduction does not follow the evaluated population"
            )
        fitness = tuple(item.fitness for item in evaluated.individuals)
        elite_count = 0
        attempts: list[str] = []
        applied: list[str] = []
        if reproduction is not None:
            elite_count = sum(child.elite for child in reproduction.children)
            for child in reproduction.children:
                for outcome in child.mutations:
                    attempts.append(outcome.operator)
                    if outcome.applied:
                        applied.append(outcome.operator)
        operator_names = tuple(dict.fromkeys(attempts))
        operator_statistics = tuple(
            OperatorStatistics(
                operator=name,
                attempts=attempts.count(name),
                applied=applied.count(name),
            )
            for name in operator_names
        )
        return cls(
            generation=evaluated.generation,
            population_size=len(evaluated),
            minimum_fitness=min(fitness),
            maximum_fitness=max(fitness),
            mean_fitness=statistics.fmean(fitness),
            median_fitness=statistics.median(fitness),
            population_stddev=statistics.pstdev(fitness),
            best_position=evaluated.best.position,
            elite_count=elite_count,
            mutation_attempts=len(attempts),
            applied_mutations=len(applied),
            operators=operator_statistics,
        )


@dataclass(frozen=True, slots=True)
class RunStatistics:
    """One ordered statistics record per evaluated generation."""

    generations: tuple[GenerationStatistics, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.generations, tuple) or not self.generations:
            raise EvolutionConfigurationError("run statistics cannot be empty")
        if not all(
            isinstance(item, GenerationStatistics) for item in self.generations
        ):
            raise EvolutionConfigurationError(
                "run statistics must contain GenerationStatistics values"
            )
        generation_ids = tuple(item.generation for item in self.generations)
        expected = tuple(
            range(generation_ids[0], generation_ids[0] + len(generation_ids))
        )
        if generation_ids != expected:
            raise EvolutionConfigurationError(
                "run statistics generations must be consecutive"
            )

    @classmethod
    def from_run(cls, result: EvolutionRunResult) -> RunStatistics:
        """Summarize every evaluation and its following reproduction, if any."""

        return cls(
            tuple(
                GenerationStatistics.calculate(
                    evaluated,
                    result.reproductions[index]
                    if index < len(result.reproductions)
                    else None,
                )
                for index, evaluated in enumerate(result.evaluations)
            )
        )


__all__ = ["GenerationStatistics", "OperatorStatistics", "RunStatistics"]
