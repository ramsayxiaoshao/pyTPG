"""Mutation protocols, outcomes, and weighted composition."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from tpg.core import TPGGraph
from tpg.evolution.errors import EvolutionConfigurationError
from tpg.evolution.rng import RandomGenerator


@dataclass(frozen=True, slots=True)
class MutationOutcome:
    """One inspectable mutation attempt and its immutable graph result."""

    graph: TPGGraph
    operator: str
    applied: bool
    description: str

    def __post_init__(self) -> None:
        if not isinstance(self.graph, TPGGraph):
            raise EvolutionConfigurationError("mutation graph must be a TPGGraph")
        if not isinstance(self.operator, str) or not self.operator.strip():
            raise EvolutionConfigurationError("mutation operator name cannot be empty")
        if not isinstance(self.applied, bool):
            raise EvolutionConfigurationError("mutation applied flag must be boolean")
        if not isinstance(self.description, str) or not self.description.strip():
            raise EvolutionConfigurationError("mutation description cannot be empty")


class MutationOperator(Protocol):
    """Focused immutable graph transformation using an explicit RNG."""

    @property
    def name(self) -> str: ...

    def mutate(self, graph: TPGGraph, rng: RandomGenerator) -> MutationOutcome: ...


@dataclass(frozen=True, slots=True)
class WeightedMutation:
    """Choose exactly one mutation category according to relative weights."""

    operators: tuple[MutationOperator, ...]
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.operators:
            raise EvolutionConfigurationError("weighted mutation requires operators")
        if len(self.operators) != len(self.weights):
            raise EvolutionConfigurationError(
                "mutation operators and weights must have equal length"
            )
        if any(
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or weight < 0.0
            for weight in self.weights
        ):
            raise EvolutionConfigurationError(
                "mutation weights must be finite non-negative real numbers"
            )
        if sum(self.weights) <= 0.0:
            raise EvolutionConfigurationError(
                "at least one mutation weight must be positive"
            )

    @property
    def name(self) -> str:
        return "weighted"

    def mutate(self, graph: TPGGraph, rng: RandomGenerator) -> MutationOutcome:
        """Select one operator without relying on NumPy's probability API."""

        threshold = float(rng.uniform(0.0, sum(self.weights)))
        cumulative = 0.0
        selected = self.operators[-1]
        for operator, weight in zip(self.operators, self.weights, strict=True):
            cumulative += weight
            if threshold < cumulative:
                selected = operator
                break
        return selected.mutate(graph, rng)


__all__ = ["MutationOperator", "MutationOutcome", "WeightedMutation"]
